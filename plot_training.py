#!/usr/bin/env python3
"""
plot_training.py — Save training plots from progress.csv

Usage:
    python plot_training.py runs/ppo_s-lie_algebra_a-lie_algebra__42__*/progress.csv
    python plot_training.py path/to/progress.csv
    python plot_training.py run1/progress.csv run2/progress.csv   # overlay multiple runs

Saves PNG files in the same directory as the first CSV.
"""

import sys
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import numpy as np


def smooth(y, window=20):
    """Simple moving average."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='same')


def plot_single(csv_path: str):
    """Plot a single run's training curves."""
    df = pd.read_csv(csv_path)
    out_dir = os.path.dirname(os.path.abspath(csv_path))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(os.path.basename(out_dir), fontsize=12, y=0.98)

    # 1. Distance to goal
    ax = axes[0, 0]
    ax.plot(df['global_step'], df['mean_distance'], alpha=0.3, color='#534AB7')
    ax.plot(df['global_step'], smooth(df['mean_distance']), color='#534AB7', linewidth=2)
    ax.axhline(y=0.1, color='#E24B4A', linestyle='--', linewidth=1, label='threshold')
    ax.set_ylabel('Geodesic distance (rad)')
    ax.set_xlabel('Steps')
    ax.set_title('Mean distance to goal')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    # Distance in degrees on right axis
    ax2 = ax.twinx()
    ax2.set_ylim(np.degrees(ax.get_ylim()[0]), np.degrees(ax.get_ylim()[1]))
    ax2.set_ylabel('Degrees')

    # 2. Episode return
    ax = axes[0, 1]
    ax.plot(df['global_step'], df['mean_return'], alpha=0.3, color='#1D9E75')
    ax.plot(df['global_step'], smooth(df['mean_return']), color='#1D9E75', linewidth=2)
    ax.set_ylabel('Mean return')
    ax.set_xlabel('Steps')
    ax.set_title('Episode return')
    ax.grid(alpha=0.15)

    # 3. Success rate
    ax = axes[0, 2]
    ax.plot(df['global_step'], df['success_rate'], alpha=0.3, color='#D85A30')
    ax.plot(df['global_step'], smooth(df['success_rate']), color='#D85A30', linewidth=2)
    ax.set_ylabel('Success rate')
    ax.set_xlabel('Steps')
    ax.set_title('Success rate')
    ax.set_ylim(-0.05, max(0.2, df['success_rate'].max() * 1.2))
    ax.grid(alpha=0.15)

    # 4. Policy loss
    ax = axes[1, 0]
    ax.plot(df['global_step'], df['policy_loss'], alpha=0.3, color='#378ADD')
    ax.plot(df['global_step'], smooth(df['policy_loss']), color='#378ADD', linewidth=2)
    ax.set_ylabel('Policy loss')
    ax.set_xlabel('Steps')
    ax.set_title('Policy loss')
    ax.grid(alpha=0.15)

    # 5. Value loss
    ax = axes[1, 1]
    ax.plot(df['global_step'], df['value_loss'], alpha=0.3, color='#639922')
    ax.plot(df['global_step'], smooth(df['value_loss']), color='#639922', linewidth=2)
    ax.set_ylabel('Value loss')
    ax.set_xlabel('Steps')
    ax.set_title('Value loss')
    ax.grid(alpha=0.15)

    # 6. KL divergence + learning rate
    ax = axes[1, 2]
    ax.plot(df['global_step'], df['approx_kl'], alpha=0.3, color='#D85A30')
    ax.plot(df['global_step'], smooth(df['approx_kl']), color='#D85A30', linewidth=2, label='KL')
    ax.set_ylabel('Approx KL', color='#D85A30')
    ax.set_xlabel('Steps')
    ax.set_title('KL divergence & effective LR')
    ax.grid(alpha=0.15)
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'training_curves.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_comparison(csv_paths: list):
    """Overlay multiple runs on the same axes for comparison."""
    out_dir = os.path.dirname(os.path.abspath(csv_paths[0]))

    colors = ['#534AB7', '#D85A30', '#1D9E75', '#378ADD', '#D4537E', '#639922']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        # Extract label from parent directory name
        label = os.path.basename(os.path.dirname(os.path.abspath(path)))
        # Shorten: just keep the repr info
        for tag in ['ppo_', 'ppo_orientation_', 'ppo_franka_']:
            label = label.replace(tag, '')
        label = label.split('__')[0]  # drop seed/timestamp
        c = colors[i % len(colors)]

        axes[0].plot(df['global_step'], smooth(df['mean_distance'], 30), color=c, linewidth=2, label=label)
        axes[1].plot(df['global_step'], smooth(df['mean_return'], 30), color=c, linewidth=2, label=label)
        axes[2].plot(df['global_step'], smooth(df['success_rate'], 30), color=c, linewidth=2, label=label)

    axes[0].axhline(y=0.1, color='#E24B4A', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].set_title('Mean distance to goal')
    axes[0].set_ylabel('Distance (rad)')
    axes[0].set_xlabel('Steps')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.15)

    axes[1].set_title('Episode return')
    axes[1].set_ylabel('Return')
    axes[1].set_xlabel('Steps')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.15)

    axes[2].set_title('Success rate')
    axes[2].set_ylabel('Rate')
    axes[2].set_xlabel('Steps')
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.15)

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'comparison.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_training.py progress.csv [progress2.csv ...]")
        sys.exit(1)

    paths = sys.argv[1:]
    for p in paths:
        if not os.path.exists(p):
            print(f"Not found: {p}")
            sys.exit(1)

    if len(paths) == 1:
        plot_single(paths[0])
    else:
        # Both individual + comparison
        for p in paths:
            plot_single(p)
        plot_comparison(paths)
