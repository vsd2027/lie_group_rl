#!/usr/bin/env python3
"""
plot_her_results.py — Plot DDPG/TD3+HER evaluation results from SB3's evaluations.npz

Usage:
    python plot_her_results.py runs/ddpg_her_*/evaluations.npz
    python plot_her_results.py run1/evaluations.npz run2/evaluations.npz
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_evals(npz_paths):
    colors = ['#534AB7', '#D85A30', '#1D9E75', '#378ADD', '#D4537E', '#639922']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, path in enumerate(npz_paths):
        data = np.load(path)
        timesteps = data['timesteps']
        results = data['results']  # (n_evals, n_episodes) — episode returns

        # Print available keys for debugging
        if i == 0:
            print(f"Keys in npz: {list(data.keys())}")

        mean_return = results.mean(axis=1)
        std_return = results.std(axis=1)

        # Label from directory name
        label = os.path.basename(os.path.dirname(os.path.abspath(path)))
        for tag in ['ddpg_her_', 'td3_her_']:
            label = label.replace(tag, '')
        label = label.split('__')[0]
        c = colors[i % len(colors)]

        # Plot 1: Mean return with std band
        axes[0].plot(timesteps, mean_return, color=c, linewidth=2, label=label)
        axes[0].fill_between(timesteps, mean_return - std_return, mean_return + std_return,
                            color=c, alpha=0.15)

        # Plot 2: Success rate — use SB3's saved successes if available
        if 'successes' in data:
            # SB3 saves this when env info has 'is_success'
            # Shape: (n_evals, n_episodes) boolean
            successes = data['successes']
            success_rate = successes.mean(axis=1)
            axes[1].plot(timesteps, success_rate, color=c, linewidth=2, label=label)
            axes[1].set_title('Success rate (from env)')
        else:
            # Fallback: estimate from returns
            # With sparse reward, return = -(steps before reaching goal)
            # return == 0 means instant success, return == -max_steps means never reached
            # Use return > -max_steps as "reached goal at least once"
            ep_lengths = data.get('ep_lengths', None)
            if ep_lengths is not None:
                max_steps = int(ep_lengths.max())
            else:
                max_steps = int(-results.min())
            success_rate = (results > -max_steps).mean(axis=1)
            axes[1].plot(timesteps, success_rate, color=c, linewidth=2, label=label)
            axes[1].set_title('Success rate (estimated from returns)')

        # Plot 3: Mean steps to goal (lower = faster)
        # With sparse reward: steps_to_goal = -return (if successful) or max_steps (if failed)
        steps_to_goal = -mean_return
        axes[2].plot(timesteps, steps_to_goal, color=c, linewidth=2, label=label)

    axes[0].set_title('Mean evaluation return')
    axes[0].set_xlabel('Timesteps')
    axes[0].set_ylabel('Return')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.15)

    axes[1].set_xlabel('Timesteps')
    axes[1].set_ylabel('Rate')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.15)

    axes[2].set_title('Mean steps to goal')
    axes[2].set_xlabel('Timesteps')
    axes[2].set_ylabel('Steps')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.15)

    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(npz_paths[0]))
    save_path = os.path.join(out_dir, 'her_results.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_her_results.py evaluations.npz [evaluations2.npz ...]")
        sys.exit(1)
    plot_evals(sys.argv[1:])