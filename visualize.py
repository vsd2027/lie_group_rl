"""
visualize.py — Watch a trained policy run in MuJoCo viewer

Usage:
    # DDPG/TD3 + HER model
    python visualize.py --model runs/ddpg_her_*/best_model.zip --env orientation
    python visualize.py --model runs/ddpg_her_*/best_model.zip --env franka

    # Just watch random actions (no trained model)
    python visualize.py --env orientation --random
    python visualize.py --env franka --random
"""

import argparse
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_orientation(args):
    """Visualize pure orientation control — print to terminal since there's no physics to render."""
    from envs.orientation_env import make_orientation_env
    from utils.lie_utils import geodesic_distance_np

    env = make_orientation_env(
        state_repr=args.state_repr,
        action_repr=args.action_repr,
        dense_reward=False,
        seed=args.seed,
    )

    if args.model and not args.random:
        from stable_baselines3 import DDPG, TD3
        # Try TD3 first, fall back to DDPG
        try:
            model = TD3.load(args.model, env=env)
        except:
            model = DDPG.load(args.model, env=env)
        print(f"Loaded model: {args.model}")
    else:
        model = None
        print("Random actions")

    for ep in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        print(f"\n=== Episode {ep+1} | init distance: {info['distance']:.3f} rad ({np.degrees(info['distance']):.1f}°) ===")

        for step in range(100):
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward

            if step % 10 == 0 or info.get('success', False):
                dist = info.get('distance', 0)
                print(f"  step {step:3d} | dist: {dist:.3f} rad ({np.degrees(dist):.1f}°) | reward: {reward:.0f} | {'SUCCESS' if info.get('success') else ''}")

            if term or trunc:
                break

        print(f"  Total reward: {total_reward:.0f} | Final success: {info.get('success', False)}")


def run_franka(args):
    """Visualize Franka arm in MuJoCo viewer."""
    from envs.mujoco_orientation_env import make_franka_orientation_env, make_franka_goal_env

    env = make_franka_orientation_env(
        state_repr=args.state_repr,
        action_repr=args.action_repr,
        render_mode='human',
        seed=args.seed,
    )

    if args.model and not args.random:
        from stable_baselines3 import DDPG, TD3
        # For GoalEnv models, we need the goal env for predict but flat env for render
        goal_env = make_franka_goal_env(
            state_repr=args.state_repr,
            action_repr=args.action_repr,
            seed=args.seed,
        )
        try:
            model = TD3.load(args.model, env=goal_env)
        except:
            model = DDPG.load(args.model, env=goal_env)
        print(f"Loaded model: {args.model}")
        use_goal_obs = True
    else:
        model = None
        use_goal_obs = False
        print("Random actions")

    for ep in range(args.episodes):
        obs, info = env.reset()
        print(f"\n=== Episode {ep+1} | init distance: {info['distance']:.3f} rad ({np.degrees(info['distance']):.1f}°) ===")
        total_reward = 0

        for step in range(200):
            if model:
                # Build goal obs from flat env state
                from utils.lie_utils import OrientationRepresentation
                qpos = env.data.qpos[:env.n_joints].astype(np.float32)
                qvel = env.data.qvel[:env.n_joints].astype(np.float32)
                achieved = OrientationRepresentation.to_network_input(env._get_ee_rotmat(), args.action_repr).astype(np.float32)
                desired = OrientationRepresentation.to_network_input(env.goal_R, args.action_repr).astype(np.float32)
                goal_obs = {
                    'observation': np.concatenate([qpos, qvel]),
                    'achieved_goal': achieved,
                    'desired_goal': desired,
                }
                action, _ = model.predict(goal_obs, deterministic=True)
            else:
                action = env.action_space.sample() * 0.3

            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(0.02)

            if step % 20 == 0:
                dist = info.get('distance', 0)
                print(f"  step {step:3d} | dist: {dist:.3f} rad ({np.degrees(dist):.1f}°) | {'SUCCESS' if info.get('success') else ''}")

            if term or trunc:
                break

        print(f"  Total reward: {total_reward:.1f}")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize trained policy")
    parser.add_argument("--model", type=str, default=None, help="Path to best_model.zip")
    parser.add_argument("--env", type=str, default="orientation", choices=["orientation", "franka"])
    parser.add_argument("--state-repr", type=str, default="lie_algebra")
    parser.add_argument("--action-repr", type=str, default="lie_algebra")
    parser.add_argument("--random", action="store_true", help="Use random actions instead of model")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.env == 'franka':
        run_franka(args)
    else:
        run_orientation(args)
