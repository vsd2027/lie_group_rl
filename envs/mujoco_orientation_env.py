"""
mujoco_orientation_env.py — End-Effector Orientation Control via Task-Space Actions

The policy outputs an orientation delta in the chosen action_repr. An IK controller
converts that to joint commands. This is the correct setup for testing the paper's
claim: Lie algebra in the ACTION space is what matters.

Pipeline:
    1. Policy outputs action vector (3D for lie_algebra/euler, 4D for quat)
    2. Convert to rotation matrix: R_delta = Exp(action) or from_euler(action) etc.
    3. Desired EE orientation: R_desired = R_current_ee · R_delta
    4. Compute orientation error in axis-angle: e = Log(R_desired · R_current_ee^T)
    5. Use rotational Jacobian J_r to get joint velocities: dq = J_r^† · e
    6. Apply as delta joint positions through MuJoCo position actuators

This means changing action_repr changes what the network must learn to output.
Lie algebra: output IS axis-angle, goes straight through Exp — clean.
Euler angles: output goes through gimbal-lock-prone conversion — network must
              learn to avoid singularities.
Quaternions: output gets normalized, double-cover ambiguity — network wastes capacity.

Requirements: pip install mujoco gymnasium
"""

import numpy as np
from typing import Optional, Dict
import os
import sys

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.lie_utils import (
    exp_so3_np, log_so3_np, geodesic_distance_np,
    sample_uniform_quaternion_np, quat_to_rotmat_np,
    OrientationRepresentation
)


FRANKA_XML = """
<mujoco model="franka_ee_orient">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <default>
    <joint armature="0.1" damping="1.0"/>
    <geom condim="3" friction="1 0.5 0.01"/>
  </default>
  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.3 0.4 0.5" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom type="plane" size="2 2 0.1" material="grid"/>
    <body name="link0" pos="0 0 0">
      <geom type="cylinder" size="0.06 0.05" rgba="0.9 0.9 0.9 1" mass="0.5"/>
      <body name="link1" pos="0 0 0.333">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
        <geom type="cylinder" size="0.06 0.15" rgba="0.9 0.9 0.9 1" mass="2.0"/>
        <body name="link2" pos="0 0 0.15" quat="0.707 0.707 0 0">
          <joint name="joint2" type="hinge" axis="0 0 1" range="-1.7628 1.7628"/>
          <geom type="cylinder" size="0.06 0.15" rgba="0.9 0.9 0.9 1" mass="2.0"/>
          <body name="link3" pos="0 0 0.15" quat="0.707 -0.707 0 0">
            <joint name="joint3" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
            <geom type="cylinder" size="0.06 0.12" rgba="0.9 0.9 0.9 1" mass="1.5"/>
            <body name="link4" pos="0.0825 0 0.12" quat="0.707 0.707 0 0">
              <joint name="joint4" type="hinge" axis="0 0 1" range="-3.0718 -0.0698"/>
              <geom type="cylinder" size="0.06 0.12" rgba="0.9 0.9 0.9 1" mass="1.5"/>
              <body name="link5" pos="-0.0825 0 0.12" quat="0.707 -0.707 0 0">
                <joint name="joint5" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                <geom type="cylinder" size="0.04 0.06" rgba="0.9 0.9 0.9 1" mass="1.0"/>
                <body name="link6" pos="0 0 0.06" quat="0.707 0.707 0 0">
                  <joint name="joint6" type="hinge" axis="0 0 1" range="-0.0175 3.7525"/>
                  <geom type="cylinder" size="0.04 0.06" rgba="0.9 0.9 0.9 1" mass="1.0"/>
                  <body name="link7" pos="0.088 0 0.06" quat="0.707 0.707 0 0">
                    <joint name="joint7" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                    <geom type="cylinder" size="0.04 0.04" rgba="0.7 0.3 0.3 1" mass="0.5"/>
                    <site name="ee_site" pos="0 0 0.107" size="0.02" rgba="1 0 0 1"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act1" joint="joint1" kp="300" ctrlrange="-2.8973 2.8973"/>
    <position name="act2" joint="joint2" kp="300" ctrlrange="-1.7628 1.7628"/>
    <position name="act3" joint="joint3" kp="300" ctrlrange="-2.8973 2.8973"/>
    <position name="act4" joint="joint4" kp="300" ctrlrange="-3.0718 -0.0698"/>
    <position name="act5" joint="joint5" kp="300" ctrlrange="-2.8973 2.8973"/>
    <position name="act6" joint="joint6" kp="300" ctrlrange="-0.0175 3.7525"/>
    <position name="act7" joint="joint7" kp="300" ctrlrange="-2.8973 2.8973"/>
  </actuator>
</mujoco>
"""


class FrankaOrientationEnv(gym.Env):
    """
    EE orientation control with task-space orientation actions + IK controller.

    The policy outputs a desired rotation delta in action_repr.
    A damped-pseudoinverse IK controller converts it to joint commands.

    This is the correct experiment: the action representation determines
    what mathematical path the network output takes to become a rotation.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(
        self,
        state_repr: str = 'lie_algebra',
        action_repr: str = 'lie_algebra',
        max_angle: float = 0.1 * np.pi,
        threshold: float = 0.15,
        max_steps: int = 100,
        dense_reward: bool = True,
        ctrl_cost: float = 0.001,
        ik_damping: float = 0.05,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        import mujoco

        self.state_repr = state_repr
        self.action_repr = action_repr
        self.max_angle = max_angle
        self.threshold = threshold
        self.max_steps = max_steps
        self.dense_reward = dense_reward
        self.ctrl_cost = ctrl_cost
        self.ik_damping = ik_damping
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        self.model = mujoco.MjModel.from_xml_string(FRANKA_XML)
        self.data = mujoco.MjData(self.model)
        self.n_joints = 7
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )

        # Observation: joints + ee orient + goal orient
        orient_dim = OrientationRepresentation.REPR_DIMS[state_repr]
        self.obs_dim = self.n_joints * 2 + orient_dim * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Action: orientation delta in chosen representation
        action_dim = OrientationRepresentation.REPR_DIMS[action_repr]
        if action_repr in ('lie_algebra', 'euler'):
            bound = max_angle
        else:
            bound = 1.0
        self.action_space = spaces.Box(
            low=-bound, high=bound, shape=(action_dim,), dtype=np.float32
        )

        self.goal_R = None
        self.step_count = 0
        self._initial_distance = 0.0
        self.q_init = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785], dtype=np.float64)
        self._viewer = None
        self._jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self._jacr = np.zeros((3, self.model.nv), dtype=np.float64)

    def _get_ee_rotmat(self):
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy().astype(np.float32)

    def _get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy().astype(np.float32)

    def _get_obs(self):
        qpos = self.data.qpos[:self.n_joints].astype(np.float32)
        qvel = self.data.qvel[:self.n_joints].astype(np.float32)
        ee_repr = OrientationRepresentation.to_network_input(self._get_ee_rotmat(), self.state_repr)
        goal_repr = OrientationRepresentation.to_network_input(self.goal_R, self.state_repr)
        return np.concatenate([qpos, qvel, ee_repr, goal_repr]).astype(np.float32)

    def _get_distance(self):
        return float(geodesic_distance_np(self._get_ee_rotmat(), self.goal_R))

    def _action_to_rotation(self, action):
        """Convert policy output to rotation matrix. THIS is representation-dependent."""
        if self.action_repr == 'lie_algebra':
            angle = np.linalg.norm(action)
            if angle > self.max_angle:
                action = action * (self.max_angle / angle)
            return exp_so3_np(action.astype(np.float32))
        else:
            return OrientationRepresentation.action_to_rotation(
                action.astype(np.float32), self.action_repr
            )

    def _ik_orientation(self, R_desired):
        """Damped pseudoinverse IK: orientation error → joint deltas."""
        import mujoco

        R_current = self._get_ee_rotmat()
        R_error = R_desired @ R_current.T
        omega_error = log_so3_np(R_error.astype(np.float32))

        self._jacp[:] = 0
        self._jacr[:] = 0
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.ee_site_id)
        Jr = self._jacr[:, :self.n_joints]

        lam = self.ik_damping
        JJT = Jr @ Jr.T + lam**2 * np.eye(3)
        dq = Jr.T @ np.linalg.solve(JJT, omega_error.astype(np.float64))
        return dq

    def reset(self, *, seed=None, options=None):
        import mujoco
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:self.n_joints] = self.q_init
        self.data.ctrl[:self.n_joints] = self.q_init
        mujoco.mj_forward(self.model, self.data)

        ee_R = self._get_ee_rotmat()
        axis = self.rng.standard_normal(3).astype(np.float32)
        axis /= (np.linalg.norm(axis) + 1e-8)
        angle = self.rng.uniform(0.3, np.pi / 2)
        self.goal_R = (ee_R @ exp_so3_np((axis * angle).astype(np.float32))).astype(np.float32)

        self.step_count = 0
        self._initial_distance = self._get_distance()
        return self._get_obs(), {
            'distance': self._initial_distance,
            'initial_distance': self._initial_distance,
            'ee_pos': self._get_ee_pos(),
        }

    def step(self, action):
        import mujoco
        self.step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Policy output → rotation (representation-dependent)
        R_delta = self._action_to_rotation(action)

        # Desired EE orientation = current · delta
        R_desired = (self._get_ee_rotmat() @ R_delta).astype(np.float32)

        # IK → joint deltas
        dq = self._ik_orientation(R_desired)

        # Apply to actuators
        new_ctrl = self.data.ctrl[:self.n_joints].copy() + dq
        for i in range(self.n_joints):
            new_ctrl[i] = np.clip(new_ctrl[i], self.model.jnt_range[i, 0], self.model.jnt_range[i, 1])
        self.data.ctrl[:self.n_joints] = new_ctrl

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        distance = self._get_distance()
        if self.dense_reward:
            reward = -distance - self.ctrl_cost * np.linalg.norm(dq)
        else:
            reward = 0.0 if distance <= self.threshold else -1.0

        return self._get_obs(), reward, False, self.step_count >= self.max_steps, {
            'distance': distance,
            'success': distance <= self.threshold,
            'initial_distance': self._initial_distance,
            'ee_pos': self._get_ee_pos(),
        }

    def render(self):
        import mujoco
        if self.render_mode == 'human':
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


def make_franka_orientation_env(state_repr='lie_algebra', action_repr='lie_algebra', **kwargs):
    return FrankaOrientationEnv(state_repr=state_repr, action_repr=action_repr, **kwargs)


# ============================================================================
# GoalEnv wrapper for SB3 HER
# ============================================================================

class FrankaOrientationGoalEnv(gym.Env):
    """
    GoalEnv wrapper around FrankaOrientationEnv for use with SB3's HerReplayBuffer.

    Reformats observations into the dict format HER expects:
        observation:   joint_pos (7) + joint_vel (7)
        achieved_goal: current EE orientation in action_repr
        desired_goal:  target orientation in action_repr

    Goals are in action_repr so HER relabeling is representation-consistent —
    achieved_goal lives in the same space as the policy output.

    Reward: sparse (-1/0) — HER handles the learning signal.
    """

    def __init__(
        self,
        state_repr: str = 'lie_algebra',
        action_repr: str = 'lie_algebra',
        max_angle: float = 0.1 * np.pi,
        threshold: float = 0.15,
        max_steps: int = 100,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        # Inner env does all the physics + IK
        self.inner = FrankaOrientationEnv(
            state_repr=state_repr,
            action_repr=action_repr,
            max_angle=max_angle,
            threshold=threshold,
            max_steps=max_steps,
            dense_reward=False,  # HER needs sparse reward
            seed=seed,
            **kwargs,
        )
        self.action_repr = action_repr
        self.threshold = threshold

        # Obs: joint state only (orientations go in achieved/desired goal)
        n_joints = self.inner.n_joints
        goal_dim = OrientationRepresentation.REPR_DIMS[action_repr]

        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, shape=(n_joints * 2,), dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(goal_dim,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=(goal_dim,), dtype=np.float32),
        })
        self.action_space = self.inner.action_space

    def _ee_to_goal_repr(self, R: np.ndarray) -> np.ndarray:
        return OrientationRepresentation.to_network_input(R, self.action_repr).astype(np.float32)

    def _make_obs(self):
        qpos = self.inner.data.qpos[:self.inner.n_joints].astype(np.float32)
        qvel = self.inner.data.qvel[:self.inner.n_joints].astype(np.float32)
        achieved = self._ee_to_goal_repr(self.inner._get_ee_rotmat())
        desired = self._ee_to_goal_repr(self.inner.goal_R)
        return {
            'observation': np.concatenate([qpos, qvel]),
            'achieved_goal': achieved,
            'desired_goal': desired,
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Vectorized sparse reward for HER. Handles batched (N, dim) inputs."""
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal[np.newaxis]
            desired_goal = desired_goal[np.newaxis]
            squeeze = True
        else:
            squeeze = False

        rewards = np.zeros(len(achieved_goal), dtype=np.float32)
        for i in range(len(achieved_goal)):
            R_a = OrientationRepresentation.action_to_rotation(achieved_goal[i], self.action_repr)
            R_d = OrientationRepresentation.action_to_rotation(desired_goal[i], self.action_repr)
            dist = geodesic_distance_np(R_a, R_d)
            rewards[i] = 0.0 if dist <= self.threshold else -1.0

        return rewards[0] if squeeze else rewards

    def reset(self, **kwargs):
        _, info = self.inner.reset(**kwargs)
        return self._make_obs(), {'distance': info['distance'], 'is_success': False}

    def step(self, action):
        _, _, terminated, truncated, info = self.inner.step(action)
        obs = self._make_obs()
        dist = info['distance']
        reward = 0.0 if dist <= self.threshold else -1.0
        info['is_success'] = info['success']
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.inner.render()

    def close(self):
        self.inner.close()


def make_franka_goal_env(state_repr='lie_algebra', action_repr='lie_algebra', **kwargs):
    return FrankaOrientationGoalEnv(state_repr=state_repr, action_repr=action_repr, **kwargs)


if __name__ == "__main__":
    import mujoco
    print(f"MuJoCo {mujoco.__version__}\n")

    print("--- Flat env (for PPO) ---")
    for ar in ['lie_algebra', 'euler']:
        env = make_franka_orientation_env(action_repr=ar, seed=42)
        obs, info = env.reset()
        print(f"  action={ar:12s} | obs={obs.shape} act={env.action_space.shape} | dist={info['distance']:.2f}")
        env.close()

    print("\n--- GoalEnv (for DDPG+HER) ---")
    for ar in ['lie_algebra', 'euler', 'quat']:
        env = make_franka_goal_env(action_repr=ar, seed=42)
        obs, info = env.reset()
        print(f"  action={ar:12s} | obs={obs['observation'].shape} "
              f"goal={obs['achieved_goal'].shape} act={env.action_space.shape} | "
              f"dist={info['distance']:.2f}")
        total_r = 0.0
        for _ in range(100):
            obs, r, term, trunc, info = env.step(env.action_space.sample() * 0.3)
            total_r += r
            if term or trunc:
                break
        print(f"    → reward={total_r:.1f} success={info['is_success']}")
        env.close()

    print("\n=== OK ===")