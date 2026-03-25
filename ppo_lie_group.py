"""
ppo_lie_group.py — PPO with Lie Group Orientations for Robotics

Implementation of Schuck et al. (2024) "Reinforcement Learning with Lie Group Orientations
for Robotics" using PPO instead of DDPG. CleanRL-style single-file implementation.

Core idea:
    - State orientations → Log map → ℝ³ vector → network input
    - Network output → ℝ³ vector → Exp map → rotation matrix → compose with current state
    - This respects the Lie group structure of SO(3) at both input and output

Usage:
    # Compare Lie algebra vs Euler angles vs Quaternions
    python ppo_lie_group.py --state-repr lie_algebra --action-repr lie_algebra
    python ppo_lie_group.py --state-repr euler --action-repr euler
    python ppo_lie_group.py --state-repr quat --action-repr quat

    # Dense vs Sparse reward
    python ppo_lie_group.py --dense-reward
    python ppo_lie_group.py --no-dense-reward  # needs HER, not implemented here

    # Hyperparameter sweep
    python ppo_lie_group.py --total-timesteps 500000 --learning-rate 3e-4 --max-angle 0.314

Based on: CleanRL ppo_continuous_action.py (Huang et al., 2022)
Paper:    arxiv.org/abs/2409.11935
"""

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.orientation_env import make_orientation_env
from utils.lie_utils import OrientationRepresentation


def parse_args():
    # First pass: check for --config before anything else
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML config file")
    pre_args, _ = pre.parse_known_args()

    config_defaults = {}
    if pre_args.config:
        import yaml
        with open(pre_args.config) as f:
            config_defaults = yaml.safe_load(f)
        # Convert underscores to match argparse dest names
        config_defaults = {k.replace('-', '_'): v for k, v in config_defaults.items()}

    parser = argparse.ArgumentParser(description="PPO with Lie Group Orientations")
    parser.add_argument("--config", type=str, default=None,
        help="YAML config file. CLI args override config values.")

    # === Experiment ===
    parser.add_argument("--exp-name", type=str, default=None,
        help="Experiment name (auto-generated if None)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-deterministic", type=bool, default=True)
    parser.add_argument("--cuda", type=bool, default=True,
        help="Use GPU if available")
    parser.add_argument("--device", type=str, default=None,
        help="Force device: 'cpu', 'cuda', or 'mps'. Auto-detects if not set.")
    parser.add_argument("--log-dir", type=str, default="./runs")

    # === Environment ===
    parser.add_argument("--env", type=str, default="orientation",
        choices=["orientation", "franka"],
        help="'orientation' = pure rotation (Task A), 'franka' = MuJoCo arm (Task B)")
    parser.add_argument("--state-repr", type=str, default="lie_algebra",
        choices=["lie_algebra", "rotmat", "rotmat_6d", "quat", "quat_pos", "euler"],
        help="State (observation) orientation representation")
    parser.add_argument("--action-repr", type=str, default="lie_algebra",
        choices=["lie_algebra", "rotmat", "rotmat_6d", "quat", "quat_pos", "euler"],
        help="Action orientation representation (only used for 'orientation' env)")
    parser.add_argument("--max-angle", type=float, default=0.1 * np.pi,
        help="Maximum rotation angle per step (rad). Paper uses 0.1π")
    parser.add_argument("--threshold", type=float, default=0.1,
        help="Goal success threshold (rad)")
    parser.add_argument("--max-episode-steps", type=int, default=50,
        help="Max steps per episode. 50 for Task A, 100 for Task B")
    parser.add_argument("--dense-reward", action="store_true", default=True,
        help="Use dense reward (recommended for PPO without HER)")
    parser.add_argument("--no-dense-reward", action="store_true", default=False)

    # === PPO Hyperparameters ===
    parser.add_argument("--total-timesteps", type=int, default=200_000,
        help="Total training timesteps. Paper uses 200k for Task A")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=4,
        help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=256,
        help="Rollout length per env before update")
    parser.add_argument("--anneal-lr", type=bool, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--norm-adv", type=bool, default=True)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", type=bool, default=True)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)

    # === Network ===
    parser.add_argument("--hidden-dim", type=int, default=256,
        help="Hidden layer dimension")
    parser.add_argument("--num-layers", type=int, default=2,
        help="Number of hidden layers")

    # === Checkpointing ===
    parser.add_argument("--save-every", type=int, default=50,
        help="Save checkpoint every N iterations (0 to disable)")
    parser.add_argument("--resume", type=str, default=None,
        help="Path to checkpoint to resume from (e.g. runs/.../checkpoint_976.pt)")

    # Apply config file defaults (CLI args override these)
    if config_defaults:
        parser.set_defaults(**config_defaults)

    args = parser.parse_args()

    if args.no_dense_reward:
        args.dense_reward = False

    # Defaults per env type
    if args.env == 'franka' and args.max_episode_steps == 50:
        args.max_episode_steps = 100  # franka needs more steps

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.exp_name is None:
        args.exp_name = f"ppo_{args.env}_s-{args.state_repr}_a-{args.action_repr}"

    return args


def _make_env(args, seed: int):
    """Factory: create a single env from args."""
    if args.env == 'franka':
        from envs.mujoco_orientation_env import make_franka_orientation_env
        return make_franka_orientation_env(
            state_repr=args.state_repr,
            max_steps=args.max_episode_steps,
            threshold=args.threshold,
            dense_reward=args.dense_reward,
            seed=seed,
        )
    else:
        return make_orientation_env(
            state_repr=args.state_repr,
            action_repr=args.action_repr,
            max_angle=args.max_angle,
            threshold=args.threshold,
            max_steps=args.max_episode_steps,
            dense_reward=args.dense_reward,
            seed=seed,
        )


# =============================================================================
# Vectorized Environment Wrapper
# =============================================================================

class VecEnv:
    """Simple vectorized wrapper. Works with any gym-like environment."""

    def __init__(self, env_fn, num_envs: int):
        self.envs = [env_fn(i) for i in range(num_envs)]
        self.num_envs = num_envs
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs = np.stack([r[0] for r in results])
        infos = [r[1] for r in results]
        return obs, infos

    def step(self, actions: np.ndarray):
        results = [env.step(actions[i]) for i, env in enumerate(self.envs)]
        obs = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        terminateds = np.array([r[2] for r in results])
        truncateds = np.array([r[3] for r in results])
        infos = [r[4] for r in results]

        # Auto-reset on done
        final_infos = []
        for i, (term, trunc) in enumerate(zip(terminateds, truncateds)):
            info = infos[i]
            if term or trunc:
                info['final_info'] = info.copy()
                new_obs, new_info = self.envs[i].reset()
                obs[i] = new_obs
            final_infos.append(info)

        return obs, rewards, terminateds, truncateds, final_infos


# =============================================================================
# Actor-Critic Network
# =============================================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization (standard for PPO)."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO Actor-Critic with separate networks for policy and value.

    Architecture matches CleanRL's ppo_continuous_action.py with configurable depth.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()

        # Build critic network
        critic_layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            critic_layers.append(layer_init(nn.Linear(in_dim, hidden_dim)))
            critic_layers.append(nn.Tanh())
            in_dim = hidden_dim
        critic_layers.append(layer_init(nn.Linear(hidden_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

        # Build actor network (mean)
        actor_layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            actor_layers.append(layer_init(nn.Linear(in_dim, hidden_dim)))
            actor_layers.append(nn.Tanh())
            in_dim = hidden_dim
        actor_layers.append(layer_init(nn.Linear(hidden_dim, act_dim), std=0.01))
        self.actor_mean = nn.Sequential(*actor_layers)

        # State-independent log standard deviation (standard PPO trick)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ):
        """Get action from policy and compute value.

        Returns:
            action: sampled or provided action
            log_prob: log probability of the action
            entropy: entropy of the policy distribution
            value: state value estimate
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        value = self.critic(x)

        return action, log_prob, entropy, value


# =============================================================================
# Training Loop
# =============================================================================

def train(args):
    """Main PPO training loop."""

    # Setup
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    log_path = os.path.join(args.log_dir, run_name)
    os.makedirs(log_path, exist_ok=True)

    # Save args
    with open(os.path.join(log_path, "args.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
        if args.cuda:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
    print(f"Device: {device}")

    # Create vectorized environments
    envs = VecEnv(
        env_fn=lambda seed: _make_env(args, seed),
        num_envs=args.num_envs,
    )

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    print(f"\n{'='*60}")
    print(f"  Experiment: {args.exp_name}")
    print(f"  Environment: {args.env}")
    print(f"  State repr:  {args.state_repr} (dim={OrientationRepresentation.REPR_DIMS[args.state_repr]})")
    if args.env == 'orientation':
        print(f"  Action repr: {args.action_repr} (dim={OrientationRepresentation.REPR_DIMS[args.action_repr]})")
    else:
        print(f"  Action: delta joint positions (dim={act_dim})")
    print(f"  Obs dim: {obs_dim}, Act dim: {act_dim}")
    print(f"  Dense reward: {args.dense_reward}")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Batch size: {args.batch_size}, Minibatch: {args.minibatch_size}")
    print(f"  Iterations: {args.num_iterations}")
    print(f"{'='*60}\n")

    # Create agent
    agent = Agent(obs_dim, act_dim, args.hidden_dim, args.num_layers).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # === Resume from checkpoint ===
    start_iteration = 1
    global_step = 0
    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt['agent'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iteration = ckpt['iteration'] + 1
        global_step = ckpt['global_step']
        # Use the checkpoint's run directory so CSV appends in the same place
        log_path = ckpt.get('log_path', log_path)
        os.makedirs(log_path, exist_ok=True)
        print(f"  Resuming at iteration {start_iteration}, step {global_step}")
        print(f"  Logging to: {log_path}")

    # Rollout storage
    obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    # Logging
    episode_returns = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_successes = deque(maxlen=100)
    episode_distances = deque(maxlen=100)
    start_time = time.time()

    # CSV log — append if resuming, write header if fresh
    csv_path = os.path.join(log_path, "progress.csv")
    if args.resume and os.path.exists(csv_path):
        print(f"  Appending to existing CSV: {csv_path}")
    else:
        with open(csv_path, "w") as f:
            f.write("iteration,global_step,mean_return,mean_length,success_rate,"
                    "mean_distance,policy_loss,value_loss,entropy,approx_kl,"
                    "sps,elapsed\n")

    # Total iterations including resumed offset
    total_iterations = start_iteration - 1 + args.num_iterations

    # Initial reset
    next_obs_np, _ = envs.reset()
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    # Episode return tracking
    _ep_returns = np.zeros(args.num_envs)
    _ep_lengths = np.zeros(args.num_envs, dtype=int)

    # === Training Loop ===
    for iteration in range(start_iteration, total_iterations + 1):

        # Annealing LR (based on total progress including resumed steps)
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / total_iterations
            lr = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lr

        # === Rollout Phase ===
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            # Get action from policy
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()

            actions_buf[step] = action
            logprobs_buf[step] = logprob

            # Step environment
            action_np = action.cpu().numpy()

            # Clip actions to action space bounds
            action_np = np.clip(
                action_np,
                envs.single_action_space.low,
                envs.single_action_space.high
            )

            next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(action_np)

            rewards_buf[step] = torch.tensor(reward_np, dtype=torch.float32, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(
                np.logical_or(terminated_np, truncated_np),
                dtype=torch.float32, device=device
            )

            # Log completed episodes
            for info in infos:
                if 'final_info' in info:
                    fi = info['final_info']
                    episode_successes.append(fi.get('success', False))
                    episode_distances.append(fi.get('distance', 0.0))

            # Track episode returns manually
            _ep_returns += reward_np
            _ep_lengths += 1
            for i in range(args.num_envs):
                if terminated_np[i] or truncated_np[i]:
                    episode_returns.append(_ep_returns[i])
                    episode_lengths.append(_ep_lengths[i])
                    _ep_returns[i] = 0.0
                    _ep_lengths[i] = 0

        # === GAE Computation ===
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = 0.0

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]

                delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )

            returns = advantages + values_buf

        # === PPO Update Phase ===
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1, act_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # Minibatch indices
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        pg_losses = []
        v_losses = []
        entropy_losses = []
        approx_kls = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Approximate KL for early stopping
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss (clipped)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                approx_kls.append(approx_kl.item())

            # Early stopping on KL divergence
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # === Logging ===
        elapsed = time.time() - start_time
        sps = int(global_step / elapsed)

        mean_return = np.mean(episode_returns) if episode_returns else 0.0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0.0
        success_rate = np.mean(episode_successes) if episode_successes else 0.0
        mean_distance = np.mean(episode_distances) if episode_distances else 0.0

        if iteration % 1 == 0:
            print(
                f"Iter {iteration:4d}/{args.num_iterations} | "
                f"Step {global_step:7d} | "
                f"Return {mean_return:7.2f} | "
                f"Success {success_rate:5.1%} | "
                f"Dist {mean_distance:.3f}rad ({np.degrees(mean_distance):.1f}°) | "
                f"PgLoss {np.mean(pg_losses):.4f} | "
                f"VLoss {np.mean(v_losses):.4f} | "
                f"Entropy {np.mean(entropy_losses):.3f} | "
                f"KL {np.mean(approx_kls):.4f} | "
                f"SPS {sps}"
            )

        # CSV log
        with open(csv_path, "a") as f:
            f.write(
                f"{iteration},{global_step},{mean_return:.4f},{mean_length:.1f},"
                f"{success_rate:.4f},{mean_distance:.4f},"
                f"{np.mean(pg_losses):.6f},{np.mean(v_losses):.6f},"
                f"{np.mean(entropy_losses):.6f},{np.mean(approx_kls):.6f},"
                f"{sps},{elapsed:.1f}\n"
            )

        # === Periodic checkpoint ===
        if args.save_every > 0 and iteration % args.save_every == 0:
            ckpt_path = os.path.join(log_path, f"checkpoint_{iteration}.pt")
            torch.save({
                'agent': agent.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration,
                'global_step': global_step,
                'log_path': log_path,
                'args': vars(args),
            }, ckpt_path)
            print(f"  [Checkpoint saved: {ckpt_path}]")

    # === Save final checkpoint ===
    final_ckpt_path = os.path.join(log_path, f"checkpoint_{iteration}.pt")
    torch.save({
        'agent': agent.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
        'global_step': global_step,
        'log_path': log_path,
        'args': vars(args),
    }, final_ckpt_path)
    # Also save a convenience copy
    torch.save(agent.state_dict(), os.path.join(log_path, "agent.pt"))
    print(f"\nFinal checkpoint: {final_ckpt_path}")
    print(f"CSV log: {csv_path}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"\nTo resume: python ppo_lie_group.py --resume {final_ckpt_path} --total-timesteps <MORE>")

    return log_path


# =============================================================================
# Comparison Runner
# =============================================================================

def run_comparison(args):
    """Run comparison across different representation combinations.

    This is the main experiment from the paper: compare all combinations of
    state and action representations to find the best one.
    """
    import json

    state_reprs = ['lie_algebra', 'quat', 'euler']
    action_reprs = ['lie_algebra', 'quat', 'euler']

    results = {}

    for sr in state_reprs:
        for ar in action_reprs:
            print(f"\n{'#'*60}")
            print(f"  Running: state={sr}, action={ar}")
            print(f"{'#'*60}")

            args.state_repr = sr
            args.action_repr = ar
            args.exp_name = f"ppo_s-{sr}_a-{ar}"

            log_path = train(args)

            # Read final metrics from CSV
            csv_path = os.path.join(log_path, "progress.csv")
            import csv
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last = rows[-1]
                    results[f"{sr}/{ar}"] = {
                        'mean_return': float(last['mean_return']),
                        'success_rate': float(last['success_rate']),
                        'mean_distance': float(last['mean_distance']),
                    }

    # Print summary
    print(f"\n{'='*70}")
    print("  COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"{'State':>15s} {'Action':>15s} {'Return':>10s} {'Success':>10s} {'Distance':>12s}")
    print("-" * 70)

    for key, val in sorted(results.items()):
        sr, ar = key.split('/')
        print(f"{sr:>15s} {ar:>15s} "
              f"{val['mean_return']:>10.2f} "
              f"{val['success_rate']:>9.1%} "
              f"{val['mean_distance']:>10.4f} rad")

    # Save results
    results_path = os.path.join(args.log_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    args = parse_args()

    # Check if running comparison mode
    if "--compare" in sys.argv:
        run_comparison(args)
    else:
        train(args)