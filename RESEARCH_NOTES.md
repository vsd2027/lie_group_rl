# Lie Groups for Robotics Control: Research Notes

## 1. The orientation problem

Positions live in ℝ³ — add, subtract, scale, no surprises. Orientations live on SO(3), a
curved 3D manifold. You cannot flatten it into ℝ³ without discontinuities somewhere.
Neural networks operate in ℝⁿ. This mismatch causes:

- **Euler angles**: gimbal lock at specific configurations. Small physical rotations → huge
  representation jumps. The network sees discontinuous training data.
- **Quaternions**: q and -q represent the same rotation (double cover). The network wastes
  capacity learning this equivalence.
- **Rotation matrices**: 9 parameters for 3 DoF. Network outputs aren't valid SO(3) — need
  SVD projection with weird gradients.

## 2. Lie algebra solution

The **Lie algebra so(3) ≅ ℝ³** is the tangent space to SO(3) at the identity. Its elements are
axis-angle vectors: direction = rotation axis, magnitude = angle. It's a normal vector space.

- **Exp: ℝ³ → SO(3)** — axis-angle to rotation matrix (Rodrigues' formula). Always produces
  valid rotation. Smooth everywhere except at ±π (measure zero).
- **Log: SO(3) → ℝ³** — rotation matrix to axis-angle. The inverse.
- **Composition**: `R_new = R_current · Exp(action)` — stays on manifold, no projection needed.
- **Distance**: `d(R₁, R₂) = ‖Log(R₁⁻¹ · R₂)‖` — the actual rotation angle. Representation-independent.

Network pipeline:
```
R_current → Log → τ ∈ ℝ³ → [network] → Δτ ∈ ℝ³ → Exp → ΔR → R_current · ΔR = R_new
```
The network only sees and produces regular vectors. Log/Exp handle the manifold.

## 3. Paper's findings (Schuck et al., 2024)

Tested 36 combinations (6 state × 6 action representations) across three tasks:

| Task | Setting | Key result |
|------|---------|------------|
| A. Direct orientation control | Pure rotation, no robot | Lie algebra actions converge fastest |
| B. End-effector control | Robot arm, orientation only | Lie algebra actions best, state repr less important |
| C. Pick-and-place | Full manipulation | Lie algebra works on real hardware (47-50% success) |

**Critical finding**: Action representation has a larger effect than state representation.
Networks can extract features from any input, but the output path matters — Exp is clean,
Euler conversion has singularities, quaternion normalization has double cover.

**Algorithm**: DDPG + HER with sparse reward. Not PPO. HER is essential — it relabels failed
episodes as successes with different goals, providing learning signal from sparse reward.

## 4. Why legged locomotion is hard (context for extensions)

- **Underactuated**: torso has 6 DoF, no actuator controls it directly. Control only through
  intermittent ground contact forces.
- **Hybrid dynamics**: flight (ballistic) ↔ stance (constrained) ↔ impact (discontinuous).
  Standard smooth optimization breaks at contact transitions.
- **High-dimensional + redundant**: 12+ joints for 6 DoF of torso control. Infinite solutions.
- **Non-smooth contacts**: friction cones, complementarity constraints, no gradients at transitions.

For legged robots, Lie group theory applies to:
- **Observation**: torso orientation as `Log(R_torso)` instead of quaternion — cleaner input signal.
  Orientation error as `Log(R_target⁻¹ · R_current)` — geometrically meaningful 3D error vector.
- **Reward**: geodesic distance `‖Log(R_target⁻¹ · R_current)‖` instead of quaternion distance.
- **Foot placement targets**: SE(3) Lie algebra se(3) ∈ ℝ⁶ for combined position + orientation.
- **Action space**: only if actions include orientation commands (e.g., desired body orientation
  in hierarchical controllers, or EE targets with task-space control).

## 5. Extensions beyond model-free RL

### MPC
Formulate optimization on SE(3)/SO(3) directly. Lie-algebraic DDP (Alcan et al., 2023)
lifts to the algebra during backward pass, retracts to manifold during forward pass.
SE₂(3) MPC for UAVs achieves superior trajectory tracking.

### MPPI
Sample perturbations in Lie algebra, Exp them to valid rotations — no rejection sampling.
All sampled trajectories have valid orientations by construction.

### Model-based RL / World models
Predict Lie algebra increments instead of next-state orientations. The learned dynamics
stay on the manifold. Compatible with DiffTORI-style differentiable trajectory optimization.

### Equivariant networks
Build SO(3) symmetry into the network architecture itself (Finzi et al., 2020).
More principled but less practical — requires custom network layers instead of just
wrapping standard MLPs with Log/Exp.

## 6. Representation comparison

| Repr | Params | Singularities | Cover | Best for |
|------|--------|---------------|-------|----------|
| Lie algebra (axis-angle) | 3 | At ±π only | Multi | Actions (paper's recommendation) |
| Rotation matrix SO(3) | 9 | None | Single | States (overcomplete but safe) |
| 6D rotation SO(3)₁:₂ | 6 | None | Single | States (continuous, compact) |
| Quaternion S³ | 4 | None | Double | Fast computation, needs normalization |
| Quaternion S³₊ | 4 | At boundary | Single | Breaks Lie group structure |
| Euler angles | 3 | Gimbal lock | Multi | Avoid — worst in every experiment |

## 7. Key references

- Schuck et al., "RL with Lie Group Orientations for Robotics", 2024 — the paper
- Sola et al., "A micro Lie theory for state estimation in robotics", 2018 — the math reference
- Andrychowicz et al., "Hindsight Experience Replay", NeurIPS 2017 — essential for sparse reward
- Zhou et al., "On the continuity of rotation representations in neural networks", CVPR 2019
- Geist et al., "Learning with 3D rotations, a hitchhiker's guide to SO(3)", 2024
- Alcan et al., "Constrained Trajectory Optimization on Matrix Lie Groups via Lie-Algebraic DDP", 2023
- Alhoussani et al., "Geometric RL for robotic manipulation", IEEE Access, 2023
- Alhoussani et al., "RL for orientation on the Lie algebra", SIU 2023