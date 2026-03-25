"""
lie_utils.py — Core SO(3) Lie Group Operations for RL

Implements the mathematical operations from the Schuck et al. paper:
  - Exp: so(3) → SO(3)  (axis-angle vector to rotation matrix)
  - Log: SO(3) → so(3)  (rotation matrix to axis-angle vector)
  - Composition: s' = s · a  (group multiplication)
  - Distance: d(s, g) = ‖Log(s⁻¹ · g)‖  (geodesic distance)
  - Conversions between representations (quaternion, euler, rotation matrix, axis-angle)

All operations are batched and work with PyTorch tensors for GPU compatibility.
NumPy versions are also provided for environment-side computation.

Reference: Sola et al., "A micro Lie theory for state estimation in robotics", 2018
"""

import numpy as np
from typing import Union, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# NumPy versions (for environments / CPU-side computation)
# =============================================================================

def skew_symmetric_np(v: np.ndarray) -> np.ndarray:
    """Create 3x3 skew-symmetric matrix from 3-vector.

    [v]× = [[0, -v3, v2],
             [v3, 0, -v1],
             [-v2, v1, 0]]

    Supports batched input: (..., 3) -> (..., 3, 3)
    """
    if v.ndim == 1:
        v = v[np.newaxis]
        squeeze = True
    else:
        squeeze = False

    batch_shape = v.shape[:-1]
    result = np.zeros(batch_shape + (3, 3), dtype=v.dtype)
    result[..., 0, 1] = -v[..., 2]
    result[..., 0, 2] = v[..., 1]
    result[..., 1, 0] = v[..., 2]
    result[..., 1, 2] = -v[..., 0]
    result[..., 2, 0] = -v[..., 1]
    result[..., 2, 1] = v[..., 0]

    if squeeze:
        result = result.squeeze(0)
    return result


def exp_so3_np(tau: np.ndarray) -> np.ndarray:
    """Exponential map: so(3) → SO(3).

    Converts axis-angle vector τ ∈ ℝ³ to rotation matrix R ∈ SO(3).
    Uses Rodrigues' formula: R = I + sin(θ)/θ · [τ]× + (1-cos(θ))/θ² · [τ]×²

    Args:
        tau: Axis-angle vector(s), shape (..., 3). Direction = axis, magnitude = angle.

    Returns:
        Rotation matrix/matrices, shape (..., 3, 3).
    """
    if tau.ndim == 1:
        tau = tau[np.newaxis]
        squeeze = True
    else:
        squeeze = False

    theta = np.linalg.norm(tau, axis=-1, keepdims=True)  # (..., 1)
    theta_sq = theta ** 2

    # Small angle approximation to avoid division by zero
    # For θ < 1e-6: sin(θ)/θ ≈ 1 - θ²/6, (1-cos(θ))/θ² ≈ 1/2 - θ²/24
    small = (theta.squeeze(-1) < 1e-6)

    # Compute coefficients (safe division)
    safe_theta = np.where(theta_sq.squeeze(-1) < 1e-12, np.ones_like(theta.squeeze(-1)), theta.squeeze(-1))
    safe_theta_sq = np.where(theta_sq.squeeze(-1) < 1e-12, np.ones_like(theta_sq.squeeze(-1)), theta_sq.squeeze(-1))

    a = np.where(
        theta_sq.squeeze(-1) < 1e-12,
        1.0 - theta_sq.squeeze(-1) / 6.0,
        np.sin(safe_theta) / safe_theta
    )
    b = np.where(
        theta_sq.squeeze(-1) < 1e-12,
        0.5 - theta_sq.squeeze(-1) / 24.0,
        (1.0 - np.cos(safe_theta)) / safe_theta_sq
    )

    K = skew_symmetric_np(tau)  # (..., 3, 3)
    K_sq = np.einsum('...ij,...jk->...ik', K, K)

    I = np.eye(3, dtype=tau.dtype)
    R = I + a[..., np.newaxis, np.newaxis] * K + b[..., np.newaxis, np.newaxis] * K_sq

    if squeeze:
        R = R.squeeze(0)
    return R


def log_so3_np(R: np.ndarray) -> np.ndarray:
    """Logarithmic map: SO(3) → so(3).

    Converts rotation matrix R ∈ SO(3) to axis-angle vector τ ∈ ℝ³.

    Args:
        R: Rotation matrix/matrices, shape (..., 3, 3).

    Returns:
        Axis-angle vector(s), shape (..., 3).
    """
    if R.ndim == 2:
        R = R[np.newaxis]
        squeeze = True
    else:
        squeeze = False

    # Compute rotation angle: cos(θ) = (tr(R) - 1) / 2
    trace = np.trace(R, axis1=-2, axis2=-1)  # (...)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # (...)

    # Extract axis from skew-symmetric part: [τ]× = θ/(2·sin(θ)) · (R - R^T)
    skew = R - np.swapaxes(R, -2, -1)  # R - R^T

    # Handle small angles: sin(θ)/θ ≈ 1, so τ ≈ vee(R - R^T) / 2
    # Handle θ ≈ π: need eigenvalue decomposition
    small = (theta < 1e-6)
    near_pi = (theta > np.pi - 1e-6)
    regular = ~small & ~near_pi

    tau = np.zeros(R.shape[:-1], dtype=R.dtype)  # (..., 3)

    # Vee map: extract vector from skew-symmetric matrix
    vee = np.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], axis=-1)  # (..., 3)

    # Regular case
    coeff = np.where(
        regular,
        theta / (2.0 * np.sin(theta + 1e-12)),
        0.5  # small angle default
    )
    tau = coeff[..., np.newaxis] * vee

    # Near-π case: find the column of (R + I) with largest norm
    for i in range(len(theta.flat)):
        idx = np.unravel_index(i, theta.shape)
        if near_pi.flat[i]:
            RpI = R[idx] + np.eye(3)
            col_norms = np.linalg.norm(RpI, axis=0)
            best_col = np.argmax(col_norms)
            v = RpI[:, best_col]
            v = v / np.linalg.norm(v)
            tau[idx] = theta[idx] * v

    if squeeze:
        tau = tau.squeeze(0)
    return tau


def compose_so3_np(s: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Compose two orientations: s' = s · a (matrix multiplication).

    Args:
        s: Current orientation(s), shape (..., 3, 3)
        a: Relative rotation action(s), shape (..., 3, 3)

    Returns:
        New orientation(s), shape (..., 3, 3)
    """
    return np.einsum('...ij,...jk->...ik', s, a)


def geodesic_distance_np(s: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Geodesic distance between two orientations.

    d(s, g) = ‖Log(s⁻¹ · g)‖ = rotation angle between s and g.

    Args:
        s, g: Rotation matrices, shape (..., 3, 3)

    Returns:
        Geodesic distance(s), shape (...)
    """
    # s⁻¹ = s^T for rotation matrices
    s_inv = np.swapaxes(s, -2, -1)
    diff = compose_so3_np(s_inv, g)
    tau = log_so3_np(diff)
    return np.linalg.norm(tau, axis=-1)


def quat_to_rotmat_np(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion to rotation matrix.

    Quaternion convention: q = [w, x, y, z] (scalar-first).

    Args:
        q: Unit quaternion(s), shape (..., 4)

    Returns:
        Rotation matrix/matrices, shape (..., 3, 3)
    """
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = np.zeros(q.shape[:-1] + (3, 3), dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (y*y + z*z)
    R[..., 0, 1] = 2 * (x*y - w*z)
    R[..., 0, 2] = 2 * (x*z + w*y)
    R[..., 1, 0] = 2 * (x*y + w*z)
    R[..., 1, 1] = 1 - 2 * (x*x + z*z)
    R[..., 1, 2] = 2 * (y*z - w*x)
    R[..., 2, 0] = 2 * (x*z - w*y)
    R[..., 2, 1] = 2 * (y*z + w*x)
    R[..., 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def rotmat_to_quat_np(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to unit quaternion (scalar-first, positive w)."""
    if R.ndim == 2:
        R = R[np.newaxis]
        squeeze = True
    else:
        squeeze = False

    batch = R.shape[:-2]
    q = np.zeros(batch + (4,), dtype=R.dtype)

    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    for i in range(np.prod(batch) if batch else 1):
        idx = np.unravel_index(i, batch) if batch else ()
        Ri = R[idx]
        tr = trace[idx] if batch else trace

        if tr > 0:
            s = 2.0 * np.sqrt(tr + 1.0)
            q[idx + (0,)] = 0.25 * s
            q[idx + (1,)] = (Ri[2, 1] - Ri[1, 2]) / s
            q[idx + (2,)] = (Ri[0, 2] - Ri[2, 0]) / s
            q[idx + (3,)] = (Ri[1, 0] - Ri[0, 1]) / s
        elif Ri[0, 0] > Ri[1, 1] and Ri[0, 0] > Ri[2, 2]:
            s = 2.0 * np.sqrt(1.0 + Ri[0, 0] - Ri[1, 1] - Ri[2, 2])
            q[idx + (0,)] = (Ri[2, 1] - Ri[1, 2]) / s
            q[idx + (1,)] = 0.25 * s
            q[idx + (2,)] = (Ri[0, 1] + Ri[1, 0]) / s
            q[idx + (3,)] = (Ri[0, 2] + Ri[2, 0]) / s
        elif Ri[1, 1] > Ri[2, 2]:
            s = 2.0 * np.sqrt(1.0 + Ri[1, 1] - Ri[0, 0] - Ri[2, 2])
            q[idx + (0,)] = (Ri[0, 2] - Ri[2, 0]) / s
            q[idx + (1,)] = (Ri[0, 1] + Ri[1, 0]) / s
            q[idx + (2,)] = 0.25 * s
            q[idx + (3,)] = (Ri[1, 2] + Ri[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + Ri[2, 2] - Ri[0, 0] - Ri[1, 1])
            q[idx + (0,)] = (Ri[1, 0] - Ri[0, 1]) / s
            q[idx + (1,)] = (Ri[0, 2] + Ri[2, 0]) / s
            q[idx + (2,)] = (Ri[1, 2] + Ri[2, 1]) / s
            q[idx + (3,)] = 0.25 * s

    # Enforce positive w (scalar part)
    sign = np.sign(q[..., 0:1])
    sign = np.where(sign == 0, 1.0, sign)
    q = q * sign

    if squeeze:
        q = q.squeeze(0)
    return q


def random_rotation_np(rng: np.random.Generator = None) -> np.ndarray:
    """Sample a uniform random rotation matrix from SO(3).

    Uses the quaternion method: sample uniform quaternion on S³,
    convert to rotation matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample uniform quaternion via Gaussian method
    q = rng.standard_normal(4)
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return quat_to_rotmat_np(q)


def sample_uniform_quaternion_np(rng: np.random.Generator = None) -> np.ndarray:
    """Sample a uniform random unit quaternion."""
    if rng is None:
        rng = np.random.default_rng()
    q = rng.standard_normal(4).astype(np.float32)
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def euler_to_rotmat_np(euler: np.ndarray, order: str = 'xyz') -> np.ndarray:
    """Convert Euler angles to rotation matrix. Intrinsic rotations."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler(order, euler).as_matrix().astype(np.float32)


def rotmat_to_euler_np(R: np.ndarray, order: str = 'xyz') -> np.ndarray:
    """Convert rotation matrix to Euler angles."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_euler(order).astype(np.float32)


# =============================================================================
# PyTorch versions (for network-side computation, GPU-friendly)
# Requires: pip install torch
# =============================================================================

if TORCH_AVAILABLE:

    def skew_symmetric_torch(v: 'torch.Tensor') -> 'torch.Tensor':
        """Batched skew-symmetric matrix from 3-vector. (..., 3) -> (..., 3, 3)"""
        batch_shape = v.shape[:-1]
        zero = torch.zeros(*batch_shape, device=v.device, dtype=v.dtype)
        result = torch.stack([
            zero, -v[..., 2], v[..., 1],
            v[..., 2], zero, -v[..., 0],
            -v[..., 1], v[..., 0], zero
        ], dim=-1).reshape(*batch_shape, 3, 3)
        return result

    def exp_so3_torch(tau: 'torch.Tensor') -> 'torch.Tensor':
        """Batched exponential map: so(3) -> SO(3). (..., 3) -> (..., 3, 3)"""
        theta = torch.norm(tau, dim=-1, keepdim=True).clamp(min=1e-12)
        theta_sq = theta ** 2

        a = torch.where(theta_sq < 1e-12, 1.0 - theta_sq / 6.0, torch.sin(theta) / theta)
        b = torch.where(theta_sq < 1e-12, 0.5 - theta_sq / 24.0, (1.0 - torch.cos(theta)) / theta_sq)

        K = skew_symmetric_torch(tau)
        K_sq = K @ K

        I = torch.eye(3, device=tau.device, dtype=tau.dtype)
        R = I + a.unsqueeze(-1) * K + b.unsqueeze(-1) * K_sq
        return R

    def log_so3_torch(R: 'torch.Tensor') -> 'torch.Tensor':
        """Batched logarithmic map: SO(3) -> so(3). (..., 3, 3) -> (..., 3)"""
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)

        skew = (R - R.transpose(-2, -1)) / 2.0
        vee = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)

        sin_theta = torch.sin(theta).clamp(min=1e-12)
        coeff = torch.where(
            theta.abs() < 1e-6,
            torch.ones_like(theta),
            theta / sin_theta
        )

        return coeff.unsqueeze(-1) * vee

    def geodesic_distance_torch(s: 'torch.Tensor', g: 'torch.Tensor') -> 'torch.Tensor':
        """Batched geodesic distance. (..., 3, 3) x (..., 3, 3) -> (...)"""
        diff = s.transpose(-2, -1) @ g
        tau = log_so3_torch(diff)
        return torch.norm(tau, dim=-1)


# =============================================================================
# Representation Conversion Utilities
# =============================================================================

class OrientationRepresentation:
    """Handles conversion between different orientation representations for RL.

    Supported representations:
        'lie_algebra' : ℝ³ axis-angle (Log of rotation matrix) — THE PAPER'S PROPOSAL
        'rotmat'      : ℝ⁹ flattened rotation matrix (SO(3))
        'rotmat_6d'   : ℝ⁶ first two columns of rotation matrix (SO(3)_{1:2})
        'quat'        : ℝ⁴ unit quaternion (S³)
        'quat_pos'    : ℝ⁴ positive-real-part quaternion (S³₊)
        'euler'       : ℝ³ Euler angles (xyz intrinsic)
    """

    REPR_DIMS = {
        'lie_algebra': 3,
        'rotmat': 9,
        'rotmat_6d': 6,
        'quat': 4,
        'quat_pos': 4,
        'euler': 3,
    }

    @staticmethod
    def to_network_input(R: np.ndarray, repr_type: str) -> np.ndarray:
        """Convert rotation matrix to network input representation.

        Args:
            R: Rotation matrix, shape (3, 3) or (B, 3, 3)
            repr_type: One of the supported representation types

        Returns:
            Vector representation, shape (dim,) or (B, dim)
        """
        single = (R.ndim == 2)
        if single:
            R = R[np.newaxis]

        if repr_type == 'lie_algebra':
            result = log_so3_np(R)
        elif repr_type == 'rotmat':
            result = R.reshape(-1, 9)
        elif repr_type == 'rotmat_6d':
            result = R[..., :, :2].reshape(-1, 6)
        elif repr_type == 'quat':
            result = rotmat_to_quat_np(R)
        elif repr_type == 'quat_pos':
            q = rotmat_to_quat_np(R)
            # Ensure positive real part
            sign = np.sign(q[..., 0:1])
            sign = np.where(sign == 0, 1.0, sign)
            result = q * sign
        elif repr_type == 'euler':
            result = np.array([rotmat_to_euler_np(R[i]) for i in range(len(R))])
        else:
            raise ValueError(f"Unknown representation: {repr_type}")

        return result.squeeze(0) if single else result

    @staticmethod
    def action_to_rotation(action_vec: np.ndarray, repr_type: str) -> np.ndarray:
        """Convert network output (action vector) to a relative rotation matrix.

        For Lie algebra: action IS axis-angle, directly use Exp.
        For others: interpret as the respective representation and convert.

        Args:
            action_vec: Action vector from network, shape (dim,) or (B, dim)
            repr_type: Representation type

        Returns:
            Relative rotation matrix, shape (3, 3) or (B, 3, 3)
        """
        single = (action_vec.ndim == 1)
        if single:
            action_vec = action_vec[np.newaxis]

        if repr_type == 'lie_algebra':
            # Direct: axis-angle → rotation via Exp
            result = exp_so3_np(action_vec)
        elif repr_type == 'rotmat':
            # Interpret as flattened rotation matrix, project to SO(3) via SVD
            M = action_vec.reshape(-1, 3, 3)
            result = np.array([_project_to_so3(M[i]) for i in range(len(M))])
        elif repr_type == 'rotmat_6d':
            # Reconstruct from first two columns via Gram-Schmidt
            result = np.array([_reconstruct_from_6d(action_vec[i]) for i in range(len(action_vec))])
        elif repr_type in ('quat', 'quat_pos'):
            # Normalize to unit quaternion, convert to rotmat
            q = action_vec / (np.linalg.norm(action_vec, axis=-1, keepdims=True) + 1e-8)
            if repr_type == 'quat_pos':
                sign = np.sign(q[..., 0:1])
                sign = np.where(sign == 0, 1.0, sign)
                q = q * sign
            result = quat_to_rotmat_np(q)
        elif repr_type == 'euler':
            result = np.array([euler_to_rotmat_np(action_vec[i]) for i in range(len(action_vec))])
        else:
            raise ValueError(f"Unknown representation: {repr_type}")

        return result.squeeze(0) if single else result


def _project_to_so3(M: np.ndarray) -> np.ndarray:
    """Project a 3×3 matrix to nearest rotation matrix via SVD."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    # Ensure det(R) = 1
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _reconstruct_from_6d(v6: np.ndarray) -> np.ndarray:
    """Reconstruct rotation matrix from 6D representation (first 2 columns)."""
    a1 = v6[:3]
    a2 = v6[3:6]
    # Gram-Schmidt
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])


# =============================================================================
# Quick self-test
# =============================================================================
if __name__ == "__main__":
    print("=== Lie Group Utils Self-Test ===\n")

    rng = np.random.default_rng(42)

    # Test Exp/Log roundtrip
    tau_original = rng.standard_normal(3).astype(np.float32) * 0.5
    R = exp_so3_np(tau_original)
    tau_recovered = log_so3_np(R)
    print(f"Exp/Log roundtrip error: {np.linalg.norm(tau_original - tau_recovered):.2e}")

    # Test that Exp produces valid SO(3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), "R·R^T != I"
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6), "det(R) != 1"
    print(f"R is valid SO(3): ✓  (det={np.linalg.det(R):.6f})")

    # Test composition
    R1 = exp_so3_np(rng.standard_normal(3).astype(np.float32) * 0.5)
    R2 = exp_so3_np(rng.standard_normal(3).astype(np.float32) * 0.5)
    R12 = compose_so3_np(R1, R2)
    assert np.allclose(R12 @ R12.T, np.eye(3), atol=1e-6), "Composition not in SO(3)"
    print(f"Composition stays on SO(3): ✓")

    # Test distance
    d = geodesic_distance_np(R1, R1)
    assert np.isclose(d, 0.0, atol=1e-6), "Self-distance != 0"
    d12 = geodesic_distance_np(R1, R2)
    d21 = geodesic_distance_np(R2, R1)
    assert np.isclose(d12, d21, atol=1e-6), "Distance not symmetric"
    print(f"Distance properties: ✓  (d(R1,R2)={d12:.4f} rad = {np.degrees(d12):.1f}°)")

    # Test PyTorch versions match NumPy
    if TORCH_AVAILABLE:
        tau_t = torch.from_numpy(tau_original)
        R_t = exp_so3_torch(tau_t)
        tau_t2 = log_so3_torch(R_t)
        print(f"\nPyTorch Exp/Log roundtrip error: {torch.norm(tau_t - tau_t2).item():.2e}")
    else:
        print("\n(PyTorch not available, skipping torch tests)")

    # Test representation conversions
    print("\n--- Representation Dimensions ---")
    for name, dim in OrientationRepresentation.REPR_DIMS.items():
        R_test = exp_so3_np(rng.standard_normal(3).astype(np.float32) * 0.5)
        vec = OrientationRepresentation.to_network_input(R_test, name)
        print(f"  {name:15s}: dim={dim}, actual shape={vec.shape}")

    print("\n=== All tests passed ===")
