"""
orientation_env.py — Direct Orientation Control Environment

Implements Task A from Schuck et al.: Pure orientation control without robot embodiment.
An agent must rotate an initial frame to match a goal frame using incremental rotation actions.

This is the cleanest testbed for comparing orientation representations because:
- No robot dynamics to confuse the results
- No position component — pure orientation
- Fast to train (no physics simulation needed)
- Directly tests if the network can learn rotational control

Environment Spec:
    State:  Current orientation + Goal orientation (concatenated in chosen representation)
    Action: Incremental rotation (in chosen representation), bounded by max_angle
    Reward: Sparse — 0 if within threshold, -1 otherwise (matching the paper)
    Done:   Truncated after max_steps

Supports all 6 representations from the paper for both state and action.
"""

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        # Minimal standalone definitions for testing without gym installed
        class _Space:
            def __init__(self, low, high, shape, dtype):
                self.low = np.full(shape, low, dtype=dtype) if isinstance(low, (int, float)) else low
                self.high = np.full(shape, high, dtype=dtype) if isinstance(high, (int, float)) else high
                self.shape = shape
                self.dtype = dtype
            def sample(self):
                return np.random.uniform(self.low, self.high).astype(self.dtype)
        class spaces:
            Box = _Space
        class gym:
            class Env:
                pass
import numpy as np
from typing import Optional, Tuple, Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.lie_utils import (
    exp_so3_np, log_so3_np, compose_so3_np, geodesic_distance_np,
    sample_uniform_quaternion_np, quat_to_rotmat_np,
    OrientationRepresentation
)


class DirectOrientationControlEnv(gym.Env):
    """
    Direct orientation control: rotate initial frame to goal frame.

    Matches Task A in Schuck et al. (2024):
    - State and goal sampled uniformly in SO(3)
    - Action is a bounded relative rotation
    - Sparse reward: 0 if close enough, -1 otherwise
    - Episode length: 50 steps (configurable)

    Args:
        state_repr:  Representation for observations ('lie_algebra', 'rotmat', 'quat', etc.)
        action_repr: Representation for actions
        max_angle:   Maximum rotation angle per step in radians (paper uses 0.1π ≈ 0.314 rad)
        threshold:   Goal threshold in radians (paper uses different values per experiment)
        max_steps:   Maximum steps per episode
        dense_reward: If True, use dense (negative distance) reward instead of sparse
        seed:        Random seed
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        state_repr: str = 'lie_algebra',
        action_repr: str = 'lie_algebra',
        max_angle: float = 0.1 * np.pi,
        threshold: float = 0.1,
        max_steps: int = 50,
        dense_reward: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.state_repr = state_repr
        self.action_repr = action_repr
        self.max_angle = max_angle
        self.threshold = threshold
        self.max_steps = max_steps
        self.dense_reward = dense_reward

        self.rng = np.random.default_rng(seed)

        # Observation: [state_repr(current_orientation), state_repr(goal_orientation)]
        state_dim = OrientationRepresentation.REPR_DIMS[state_repr]
        self.obs_dim = state_dim * 2  # current + goal

        # Action: bounded in the chosen representation
        action_dim = OrientationRepresentation.REPR_DIMS[action_repr]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # For Lie algebra actions: bound is the max rotation angle per axis
        # The total rotation magnitude is bounded by √3 * max_angle (worst case all axes)
        # but typically the network learns to stay within bounds
        if action_repr == 'lie_algebra':
            action_bound = max_angle
        elif action_repr in ('euler',):
            action_bound = max_angle
        elif action_repr in ('quat', 'quat_pos'):
            action_bound = 1.0  # quaternion components
        elif action_repr in ('rotmat', 'rotmat_6d'):
            action_bound = 1.0  # matrix elements
        else:
            action_bound = max_angle

        self.action_space = spaces.Box(
            low=-action_bound, high=action_bound,
            shape=(action_dim,), dtype=np.float32
        )

        # Internal state (always rotation matrices for canonical computation)
        self.current_R: Optional[np.ndarray] = None
        self.goal_R: Optional[np.ndarray] = None
        self.step_count: int = 0

        # Tracking
        self._initial_distance = 0.0

    def _get_obs(self) -> np.ndarray:
        """Build observation vector from current and goal orientations."""
        s_vec = OrientationRepresentation.to_network_input(
            self.current_R.astype(np.float32), self.state_repr
        )
        g_vec = OrientationRepresentation.to_network_input(
            self.goal_R.astype(np.float32), self.state_repr
        )
        return np.concatenate([s_vec, g_vec]).astype(np.float32)

    def _get_distance(self) -> float:
        """Compute geodesic distance between current and goal orientation."""
        return float(geodesic_distance_np(self.current_R, self.goal_R))

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment with new random initial and goal orientations."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Sample uniform random orientations
        q_init = sample_uniform_quaternion_np(self.rng)
        q_goal = sample_uniform_quaternion_np(self.rng)

        self.current_R = quat_to_rotmat_np(q_init).astype(np.float32)
        self.goal_R = quat_to_rotmat_np(q_goal).astype(np.float32)
        self.step_count = 0
        self._initial_distance = self._get_distance()

        obs = self._get_obs()
        info = {
            'distance': self._get_distance(),
            'initial_distance': self._initial_distance,
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Apply rotation action, compute reward.

        Pipeline (for Lie algebra actions):
            1. action τ_a ∈ ℝ³ (from network, clipped to max_angle)
            2. R_action = Exp(τ_a) ∈ SO(3)
            3. R_new = R_current · R_action  (group composition)

        For other representations:
            1. Convert action to rotation matrix
            2. Same composition
        """
        self.step_count += 1

        # Clip action to bounds
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)

        if self.action_repr == 'lie_algebra':
            # === THE PAPER'S PROPOSED METHOD ===
            # Clip magnitude to max_angle
            angle = np.linalg.norm(action)
            if angle > self.max_angle:
                action = action * (self.max_angle / angle)
            R_action = exp_so3_np(action)
        else:
            # Convert from other representations
            R_action = OrientationRepresentation.action_to_rotation(action, self.action_repr)

        # Compose: s' = s · a
        self.current_R = compose_so3_np(self.current_R, R_action)

        # Ensure we stay exactly on SO(3) (numerical drift correction)
        U, _, Vt = np.linalg.svd(self.current_R)
        self.current_R = (U @ Vt).astype(np.float32)

        # Compute distance and reward
        distance = self._get_distance()

        if self.dense_reward:
            # Dense reward: negative distance (closer to 0 is better)
            reward = -distance
        else:
            # Sparse reward (matching paper): 0 if reached, -1 otherwise
            reward = 0.0 if distance <= self.threshold else -1.0

        # Check termination
        terminated = False  # No terminal condition in paper
        truncated = (self.step_count >= self.max_steps)

        obs = self._get_obs()
        info = {
            'distance': distance,
            'success': distance <= self.threshold,
            'initial_distance': self._initial_distance,
            'distance_reduction': self._initial_distance - distance,
        }

        return obs, reward, terminated, truncated, info


class DirectOrientationControlDenseEnv(DirectOrientationControlEnv):
    """Convenience wrapper with dense reward enabled by default.

    Dense reward works much better with PPO (no HER needed).
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('dense_reward', True)
        kwargs.setdefault('max_steps', 50)
        super().__init__(**kwargs)


# =============================================================================
# Gymnasium Registration
# =============================================================================
def make_orientation_env(
    state_repr: str = 'lie_algebra',
    action_repr: str = 'lie_algebra',
    dense_reward: bool = True,
    **kwargs
) -> DirectOrientationControlEnv:
    """Factory function for creating orientation environments.

    Example:
        env = make_orientation_env(state_repr='lie_algebra', action_repr='lie_algebra')
        env = make_orientation_env(state_repr='quat', action_repr='euler')
    """
    return DirectOrientationControlDenseEnv(
        state_repr=state_repr,
        action_repr=action_repr,
        dense_reward=dense_reward,
        **kwargs
    )


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    print("=== Orientation Environment Self-Test ===\n")

    for state_repr in ['lie_algebra', 'quat', 'euler', 'rotmat']:
        for action_repr in ['lie_algebra', 'euler']:
            env = make_orientation_env(
                state_repr=state_repr,
                action_repr=action_repr,
                seed=42
            )

            obs, info = env.reset()
            print(f"state={state_repr:15s} action={action_repr:15s} | "
                  f"obs_dim={obs.shape[0]:3d} act_dim={env.action_space.shape[0]:2d} | "
                  f"init_dist={info['distance']:.3f} rad ({np.degrees(info['distance']):.1f}°)")

            # Run a few random steps
            total_reward = 0.0
            for _ in range(50):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

            print(f"  → After {env.step_count} steps: "
                  f"dist={info['distance']:.3f} rad, "
                  f"reward={total_reward:.1f}, "
                  f"success={info['success']}")

    print("\n=== Environment test passed ===")
