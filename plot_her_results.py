#!/usr/bin/env python3
"""
plot_her_results.py — Plot DDPG+HER evaluation results from SB3's evaluations.npz

Usage:
    python plot_her_results.py runs/ddpg_her_s-lie_algebra*/evaluations.npz
    python plot_her_results.py run1/evaluations.npz run2/evaluations.npz  # overlay
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_evals(npz_paths):
    colors = ['#534AB7', '#D85A30', '#1D9E75', '#378ADD', '#D4537E', '#639922']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, path in enumerate(npz_paths):
        data = np.load(path)
        timesteps = data['timesteps']
        results = data['results']  # (n_evals, n_episodes)

        mean_return = results.mean(axis=1)
        std_return = results.std(axis=1)

        # Success = episodes where return > -max_steps (got reward 0 at least once)
        # With sparse reward: return = -N means N steps of -1, so 0 means all successes
        # Actually: return of 0 is impossible unless solved instantly.
        # Success = return > -(max_steps), i.e. got at least one 0 reward
        # More precisely: each step is -1 or 0, so higher return = more 0s
        # But SB3 EvalCallback also saves ep_lengths. Let's use mean_return as proxy.
        success_rate = (results == 0.0).any(axis=1).mean(axis=0) if results.ndim > 1 else None

        # Actually: with sparse reward, final return = -(number_of_failed_steps)
        # A fully successful episode = 0 total reward (all steps within threshold)
        # That's unlikely. Better: count episodes where final reward was 0
        # SB3 stores per-episode rewards, so return = sum of per-step rewards
        success_per_eval = (results >= -5).mean(axis=1)  # within 5 steps of optimal

        label = os.path.basename(os.path.dirname(path))
        for tag in ['ddpg_her_']:
            label = label.replace(tag, '')
        label = label.split('__')[0]
        c = colors[i % len(colors)]

        axes[0].plot(timesteps, mean_return, color=c, linewidth=2, label=label)
        axes[0].fill_between(timesteps, mean_return - std_return, mean_return + std_return,
                            color=c, alpha=0.15)
        axes[1].plot(timesteps, success_per_eval, color=c, linewidth=2, label=label)

    axes[0].set_title('Mean evaluation return')
    axes[0].set_xlabel('Timesteps')
    axes[0].set_ylabel('Return')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.15)

    axes[1].set_title('Success rate (return >= -5)')
    axes[1].set_xlabel('Timesteps')
    axes[1].set_ylabel('Rate')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.15)

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
