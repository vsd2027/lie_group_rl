"""
ddpg_her_lie_group.py — DDPG + HER with Lie Group Orientations

This matches the paper's actual algorithm: DDPG with Hindsight Experience Replay.
Uses Stable Baselines3's built-in HerReplayBuffer.

The environment must follow the GoalEnv interface:
    obs = {
        'observation': current state features,
        'achieved_goal': current orientation in chosen repr,
        'desired_goal': target orientation in chosen repr,
    }

Usage:
    pip install stable-baselines3

    # Paper's method
    python ddpg_her_lie_group.py --state-repr lie_algebra --action-repr lie_algebra

    # Comparison
    python ddpg_her_lie_group.py --state-repr euler --action-repr euler
    python ddpg_her_lie_group.py --state-repr quat --action-repr quat

    # With config
    python ddpg_her_lie_group.py --config configs/lie_algebra_1M.yaml
"""

import argparse
import os
import sys
import time
import numpy as np

import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.lie_utils import (
    exp_so3_np, log_so3_np, geodesic_distance_np,
    sample_uniform_quaternion_np, quat_to_rotmat_np,
    OrientationRepresentation
)


# ============================================================================
# GoalEnv wrapper for orientation control
# ============================================================================

class OrientationGoalEnv(gym.Env):
    """
    Direct orientation control as a GoalEnv for HER.

    obs dict:
        'observation':   Log(R_current) or other repr — what the network sees as "state"
        'achieved_goal': R_current in action_repr — what we achieved (same space as actions)
        'desired_goal':  R_goal in action_repr — what we want

    HER relabels desired_goal with achieved_goal from future steps.
    For this to work, achieved_goal and desired_goal must be in the same space.
    We use action_repr for goals so the relabeling is representation-consistent.

    Reward: sparse — 0 if geodesic distance < threshold, -1 otherwise.
    (HER is designed for sparse reward.)
    """

    def __init__(
        self,
        state_repr: str = 'lie_algebra',
        action_repr: str = 'lie_algebra',
        max_angle: float = 0.1 * np.pi,
        threshold: float = 0.1,
        max_steps: int = 50,
        seed: int = None,
    ):
        super().__init__()
        self.state_repr = state_repr
        self.action_repr = action_repr
        self.max_angle = max_angle
        self.threshold = threshold
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        state_dim = OrientationRepresentation.REPR_DIMS[state_repr]
        action_dim = OrientationRepresentation.REPR_DIMS[action_repr]
        goal_dim = action_dim  # goals live in action representation space

        # GoalEnv dict observation space
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(goal_dim,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=(goal_dim,), dtype=np.float32),
        })

        # Action: orientation delta in action_repr
        if action_repr in ('lie_algebra', 'euler'):
            bound = max_angle
        else:
            bound = 1.0
        self.action_space = spaces.Box(-bound, bound, shape=(action_dim,), dtype=np.float32)

        self.current_R = None
        self.goal_R = None
        self.step_count = 0

    def _rotmat_to_goal_repr(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to the goal representation (= action_repr)."""
        return OrientationRepresentation.to_network_input(R, self.action_repr).astype(np.float32)

    def _get_obs(self):
        state_vec = OrientationRepresentation.to_network_input(
            self.current_R, self.state_repr
        ).astype(np.float32)
        achieved = self._rotmat_to_goal_repr(self.current_R)
        desired = self._rotmat_to_goal_repr(self.goal_R)
        return {
            'observation': state_vec,
            'achieved_goal': achieved,
            'desired_goal': desired,
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Vectorized reward computation (required by HER).

        Must handle batched inputs: (batch_size, goal_dim) arrays.
        Fully vectorized — no Python for-loops.
        """
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal[np.newaxis]
            desired_goal = desired_goal[np.newaxis]
            squeeze = True
        else:
            squeeze = False

        if self.action_repr == 'lie_algebra':
            # Fast path: goals are axis-angle vectors.
            # Convert to rotation matrices in batch, compute batch geodesic distance.
            R_a = exp_so3_np(achieved_goal.astype(np.float32))  # (B, 3, 3)
            R_d = exp_so3_np(desired_goal.astype(np.float32))   # (B, 3, 3)
            # R_diff = R_a^T @ R_d  (batch matmul)
            R_diff = np.einsum('bij,bjk->bik', R_a.transpose(0, 2, 1), R_d)
            # Geodesic distance = arccos((tr(R_diff) - 1) / 2)
            trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
            cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
            dist = np.arccos(cos_angle)
        elif self.action_repr == 'euler':
            # Convert euler to rotation matrices in batch via scipy
            from scipy.spatial.transform import Rotation
            R_a = Rotation.from_euler('xyz', achieved_goal).as_matrix()
            R_d = Rotation.from_euler('xyz', desired_goal).as_matrix()
            R_diff = np.einsum('bij,bjk->bik', R_a.transpose(0, 2, 1), R_d)
            trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
            cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
            dist = np.arccos(cos_angle)
        elif self.action_repr in ('quat', 'quat_pos'):
            # Quaternion distance: d = 2 * arccos(|q1 · q2|)
            q_a = achieved_goal / (np.linalg.norm(achieved_goal, axis=-1, keepdims=True) + 1e-8)
            q_d = desired_goal / (np.linalg.norm(desired_goal, axis=-1, keepdims=True) + 1e-8)
            dot = np.abs(np.sum(q_a * q_d, axis=-1))
            dot = np.clip(dot, 0.0, 1.0)
            dist = 2.0 * np.arccos(dot)
        else:
            # Fallback: per-element (slow, but correct for any repr)
            dist = np.zeros(len(achieved_goal), dtype=np.float32)
            for i in range(len(achieved_goal)):
                R_a = OrientationRepresentation.action_to_rotation(achieved_goal[i], self.action_repr)
                R_d = OrientationRepresentation.action_to_rotation(desired_goal[i], self.action_repr)
                dist[i] = geodesic_distance_np(R_a, R_d)

        rewards = np.where(dist <= self.threshold, 0.0, -1.0).astype(np.float32)
        if squeeze:
            rewards = rewards[0]
        return rewards

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        q_init = sample_uniform_quaternion_np(self.rng)
        q_goal = sample_uniform_quaternion_np(self.rng)
        self.current_R = quat_to_rotmat_np(q_init).astype(np.float32)
        self.goal_R = quat_to_rotmat_np(q_goal).astype(np.float32)
        self.step_count = 0

        return self._get_obs(), {'distance': geodesic_distance_np(self.current_R, self.goal_R)}

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)

        # Action → rotation (representation-dependent)
        if self.action_repr == 'lie_algebra':
            angle = np.linalg.norm(action)
            if angle > self.max_angle:
                action = action * (self.max_angle / angle)
            R_action = exp_so3_np(action)
        else:
            R_action = OrientationRepresentation.action_to_rotation(action, self.action_repr)

        # Compose
        self.current_R = (self.current_R @ R_action).astype(np.float32)
        # Re-orthogonalize
        U, _, Vt = np.linalg.svd(self.current_R)
        self.current_R = (U @ Vt).astype(np.float32)

        obs = self._get_obs()
        dist = geodesic_distance_np(self.current_R, self.goal_R)
        reward = 0.0 if dist <= self.threshold else -1.0

        terminated = False
        truncated = (self.step_count >= self.max_steps)

        info = {
            'distance': dist,
            'is_success': dist <= self.threshold,
        }
        return obs, reward, terminated, truncated, info


# ============================================================================
# Training
# ============================================================================

def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    config_defaults = {}
    if pre_args.config:
        import yaml
        with open(pre_args.config) as f:
            config_defaults = yaml.safe_load(f)
        config_defaults = {k.replace('-', '_'): v for k, v in config_defaults.items()}

    parser = argparse.ArgumentParser(description="DDPG+HER with Lie Group Orientations")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--env", type=str, default="orientation",
        choices=["orientation", "franka"],
        help="'orientation' = pure rotation (Task A), 'franka' = MuJoCo arm (Task B)")
    parser.add_argument("--state-repr", type=str, default="lie_algebra",
        choices=["lie_algebra", "rotmat", "rotmat_6d", "quat", "quat_pos", "euler"])
    parser.add_argument("--action-repr", type=str, default="lie_algebra",
        choices=["lie_algebra", "rotmat", "rotmat_6d", "quat", "quat_pos", "euler"])
    parser.add_argument("--max-angle", type=float, default=0.1 * np.pi)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--n-sampled-goal", type=int, default=4,
        help="HER: virtual transitions per real transition")
    parser.add_argument("--noise-sigma", type=float, default=0.1,
        help="Gaussian action noise std for exploration")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="./runs")
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--device", type=str, default="auto")

    if config_defaults:
        parser.set_defaults(**config_defaults)
    return parser.parse_args()


def main():
    args = parse_args()

    from stable_baselines3 import DDPG, HerReplayBuffer
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor

    exp_name = f"ddpg_her_{args.env}_s-{args.state_repr}_a-{args.action_repr}__{args.seed}__{int(time.time())}"
    log_path = os.path.join(args.log_dir, exp_name)
    os.makedirs(log_path, exist_ok=True)

    # Save args
    with open(os.path.join(log_path, "args.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # Defaults per env type
    if args.env == 'franka' and args.max_episode_steps == 50:
        args.max_episode_steps = 100
    if args.env == 'franka' and args.threshold == 0.1:
        args.threshold = 0.15

    print(f"{'='*60}")
    print(f"  DDPG + HER — Lie Group Orientations")
    print(f"  Environment: {args.env}")
    print(f"  State repr:  {args.state_repr}")
    print(f"  Action repr: {args.action_repr}")
    print(f"  Timesteps:   {args.total_timesteps:,}")
    print(f"  HER goals:   {args.n_sampled_goal} per transition")
    print(f"  Sparse reward (HER-compatible)")
    print(f"  Log dir:     {log_path}")
    print(f"{'='*60}\n")

    # Create environments
    def _make_goal_env(seed):
        if args.env == 'franka':
            from envs.mujoco_orientation_env import make_franka_goal_env
            return make_franka_goal_env(
                state_repr=args.state_repr,
                action_repr=args.action_repr,
                max_angle=args.max_angle,
                threshold=args.threshold,
                max_steps=args.max_episode_steps,
                seed=seed,
            )
        else:
            return OrientationGoalEnv(
                state_repr=args.state_repr,
                action_repr=args.action_repr,
                max_angle=args.max_angle,
                threshold=args.threshold,
                max_steps=args.max_episode_steps,
                seed=seed,
            )

    env = Monitor(_make_goal_env(args.seed), log_path)
    eval_env = _make_goal_env(args.seed + 1000)

    # Action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=args.noise_sigma * np.ones(n_actions)
    )

    # Network architecture
    net_arch = [args.hidden_dim] * args.num_layers

    # Model: DDPG + HER
    model = DDPG(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=args.n_sampled_goal,
            goal_selection_strategy="future",
        ),
        action_noise=action_noise,
        verbose=1,
        buffer_size=args.buffer_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        policy_kwargs=dict(net_arch=net_arch),
        tensorboard_log=log_path,
        seed=args.seed,
        device=args.device,
    )

    # Eval callback — logs success rate
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_path,
        log_path=log_path,
        eval_freq=args.eval_freq,
        n_eval_episodes=50,
        deterministic=True,
    )

    # Train
    print("Training...")
    t0 = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )
    elapsed = time.time() - t0

    # Save
    model.save(os.path.join(log_path, "final_model"))
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Model saved to {log_path}/final_model.zip")
    print(f"Eval results in {log_path}/evaluations.npz")
    print(f"\nTo plot: python plot_her_results.py {log_path}/evaluations.npz")


if __name__ == "__main__":
    main()