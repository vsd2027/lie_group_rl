# Lie Group Orientations for RL — Implementation

Reproduction of **"Reinforcement Learning with Lie Group Orientations for Robotics"**
(Schuck, Brüdigam, Hirche, Schoellig, 2024 — [arXiv:2409.11935](https://arxiv.org/abs/2409.11935))

## What this project does

Tests whether representing orientations via the **Lie algebra (axis-angle via Log/Exp maps)**
improves RL training compared to Euler angles, quaternions, and rotation matrices.

The paper's core claim: **the action representation matters more than the state representation**.
When the network's output goes through `Exp()` to become a valid rotation, training is faster
and more stable than when it goes through gimbal-lock-prone Euler conversion or
double-cover quaternion normalization.

## Algorithm

The paper uses **DDPG + Hindsight Experience Replay (HER)** with sparse reward — NOT PPO.
This matters enormously:

- **DDPG**: off-policy, replay buffer, reuses each transition thousands of times
- **HER**: relabels failed episodes ("pretend the goal was where you ended up"), giving dense
  learning signal from sparse reward
- **PPO**: on-policy, discards data after each update, plateaus at ~0.3 rad on this task
  even with 10M steps and dense reward

We provide both for comparison, but `ddpg_her_lie_group.py` is the correct reproduction.

## Quick start

```bash
# Setup
conda activate lie_group_rl  # or: pip install torch stable-baselines3 gymnasium mujoco scipy matplotlib pandas pyyaml

# === DDPG+HER (paper's method) — should hit 0.9+ success in ~200k steps ===
python ddpg_her_lie_group.py --config configs/her_lie_algebra.yaml
python ddpg_her_lie_group.py --config configs/her_euler.yaml
python ddpg_her_lie_group.py --config configs/her_quat.yaml

# Plot comparison
python plot_her_results.py runs/ddpg_her_s-lie_algebra*/evaluations.npz \
                           runs/ddpg_her_s-euler*/evaluations.npz

# === PPO (for reference / if you want to experiment) ===
python ppo_lie_group.py --config configs/lie_algebra_1M.yaml
python plot_training.py runs/ppo_*/progress.csv

# === MuJoCo Franka arm (Task B — task-space actions through IK) ===
# DDPG/TD3 + HER (recommended)
python ddpg_her_lie_group.py --config configs/her_franka_lie_algebra.yaml
python ddpg_her_lie_group.py --config configs/her_franka_euler.yaml

# Plot Franka comparison
python plot_her_results.py runs/ddpg_her_franka*lie*/evaluations.npz \
                           runs/ddpg_her_franka*euler*/evaluations.npz

# PPO variant (slower, for reference)
python ppo_lie_group.py --config configs/franka_lie_algebra.yaml

# Resume from checkpoint (PPO only)
python ppo_lie_group.py --resume runs/.../checkpoint_500.pt --total-timesteps 1000000

# Force CPU on Mac if MPS gives issues
python ppo_lie_group.py --device cpu

# Watch random actions on the Franka arm (no model needed)
python visualize.py --env franka --random

# Watch a trained model
python visualize.py --env franka --model runs/ddpg_her_franka*/best_model.zip

# Pure orientation just prints distances to terminal
python visualize.py --env orientation --model runs/ddpg_her_orientation*/best_model.zip
```

## Project structure

```
lie_group_rl/
├── ddpg_her_lie_group.py          # DDPG+HER trainer (SB3) — paper's actual algorithm
├── ppo_lie_group.py               # PPO trainer (CleanRL-style) — for comparison
├── plot_her_results.py            # Plot HER evaluation curves
├── plot_training.py               # Plot PPO training CSVs
├── envs/
│   ├── orientation_env.py         # Pure rotation control (Task A)
│   └── mujoco_orientation_env.py  # Franka arm with IK task-space actions (Task B)
├── utils/
│   └── lie_utils.py               # SO(3) Lie group ops (Exp, Log, distance, conversions)
├── configs/                       # YAML configs for all experiments
├── environment.yml                # Conda env (cross-platform)
├── setup.sh                       # Auto-detect platform & install
└── RESEARCH_NOTES.md              # Theory + research survey
```

## What we learned

1. **PPO cannot solve this task efficiently.** With sparse reward it gets no signal. With dense reward
   it plateaus at ~0.3 rad after 10M steps. The paper's DDPG+HER is essential.

2. **The action representation is what matters.** The paper shows state representation differences are
   "less conclusive" — networks can extract features from any input. But the output path through
   Exp (Lie algebra) vs euler_to_rotmat (singularities) vs quaternion normalization (double cover)
   directly affects gradient quality.

3. **The MuJoCo env must use task-space actions.** If actions are joint deltas, the orientation
   representation only appears in observations (which matters less). The correct setup: policy
   outputs orientation deltas, IK controller converts to joint commands.

4. **Observation design matters for PPO.** Providing `Log(R_current)` and `Log(R_goal)` separately
   forces the network to learn a nonlinear relative rotation. Providing the error
   `Log(R_current⁻¹ · R_goal)` directly would make the optimal policy nearly trivial. DDPG+HER
   sidesteps this issue through goal relabeling.

## References

- Schuck et al., "RL with Lie Group Orientations for Robotics", 2024 — [arXiv:2409.11935](https://arxiv.org/abs/2409.11935)
- Sola et al., "A micro Lie theory for state estimation in robotics", 2018 — [arXiv:1812.01537](https://arxiv.org/abs/1812.01537)
- Andrychowicz et al., "Hindsight Experience Replay", NeurIPS 2017
- Zhou et al., "On the continuity of rotation representations in neural networks", CVPR 2019
- Geist et al., "Learning with 3D rotations, a hitchhiker's guide to SO(3)", 2024