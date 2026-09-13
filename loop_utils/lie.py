# STAC patch — Lie-group helpers for the keyframe pose graph (claude_stac.txt
# §4.3): explicit, tested so3/se3 exponential and logarithm maps, numpy AND
# torch (the solver differentiates through the torch versions by autograd on
# the perturbation vectors). Rodrigues with the small-angle series near zero,
# float64 throughout.
#
# Conventions: rotation vector ω (rad·axis); ξ = (ω, ν) ∈ ℝ⁶; T = [R t; 0 1];
# Exp(ξ) = [exp(ω) V(ω)ν; 0 1] with V the left Jacobian of SO(3); left
# perturbation T ← Exp(δ)·T.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import numpy as np

_SMALL = 1e-6


# ── numpy ─────────────────────────────────────────────────────────────────

def hat(w):
    w = np.asarray(w, np.float64).reshape(3)
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def so3_exp(w):
    w = np.asarray(w, np.float64).reshape(3)
    th = float(np.linalg.norm(w))
    K = hat(w)
    if th < _SMALL:
        # series: sin θ/θ ≈ 1 − θ²/6, (1 − cos θ)/θ² ≈ 1/2 − θ²/24
        a = 1.0 - th * th / 6.0
        b = 0.5 - th * th / 24.0
    else:
        a = np.sin(th) / th
        b = (1.0 - np.cos(th)) / (th * th)
    return np.eye(3) + a * K + b * (K @ K)


def so3_log(R):
    R = np.asarray(R, np.float64)
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(tr))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < _SMALL:
        return 0.5 * (1.0 + th * th / 6.0) * v
    if np.pi - th < 1e-4:
        # near π: axis from the symmetric part (R + I)/2 = a aᵀ
        A = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k] / np.sqrt(max(A[k, k], 1e-18))
        axis = axis / (np.linalg.norm(axis) + 1e-18)
        # sign from v (may be ~0 exactly at π — any sign is a valid log there)
        if float(axis @ v) < 0:
            axis = -axis
        return th * axis
    return (th / (2.0 * np.sin(th))) * v


def _left_jacobian(w):
    w = np.asarray(w, np.float64).reshape(3)
    th = float(np.linalg.norm(w))
    K = hat(w)
    if th < _SMALL:
        return np.eye(3) + 0.5 * K + (1.0 / 6.0) * (K @ K)
    return (np.eye(3) + (1.0 - np.cos(th)) / (th * th) * K
            + (th - np.sin(th)) / (th ** 3) * (K @ K))


def _left_jacobian_inv(w):
    w = np.asarray(w, np.float64).reshape(3)
    th = float(np.linalg.norm(w))
    K = hat(w)
    if th < _SMALL:
        return np.eye(3) - 0.5 * K + (1.0 / 12.0) * (K @ K)
    half = th / 2.0
    cot = 1.0 / np.tan(half)
    return np.eye(3) - 0.5 * K + (1.0 / (th * th)) * (1.0 - half * cot) * (K @ K)


def se3_exp(xi):
    xi = np.asarray(xi, np.float64).reshape(6)
    w, v = xi[:3], xi[3:]
    T = np.eye(4)
    T[:3, :3] = so3_exp(w)
    T[:3, 3] = _left_jacobian(w) @ v
    return T


def se3_log(T):
    T = np.asarray(T, np.float64)
    w = so3_log(T[:3, :3])
    v = _left_jacobian_inv(w) @ T[:3, 3]
    return np.concatenate([w, v])


def se3_inv(T):
    T = np.asarray(T, np.float64)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


# ── torch (differentiable; same conventions) ──────────────────────────────

def t_hat(w):
    import torch
    z = torch.zeros((), dtype=w.dtype, device=w.device)
    return torch.stack([torch.stack([z, -w[2], w[1]]),
                        torch.stack([w[2], z, -w[0]]),
                        torch.stack([-w[1], w[0], z])])


def t_so3_exp(w):
    import torch
    th = torch.linalg.norm(w)
    K = t_hat(w)
    th2 = th * th
    # branchless small-angle series keeps autograd finite at θ = 0
    small = th < _SMALL
    th_safe = torch.where(small, torch.ones_like(th), th)
    a = torch.where(small, 1.0 - th2 / 6.0, torch.sin(th_safe) / th_safe)
    b = torch.where(small, 0.5 - th2 / 24.0, (1.0 - torch.cos(th_safe)) / (th_safe * th_safe))
    I = torch.eye(3, dtype=w.dtype, device=w.device)
    return I + a * K + b * (K @ K)


def t_left_jacobian(w):
    import torch
    th = torch.linalg.norm(w)
    K = t_hat(w)
    th2 = th * th
    small = th < _SMALL
    th_safe = torch.where(small, torch.ones_like(th), th)
    b = torch.where(small, 0.5 - th2 / 24.0, (1.0 - torch.cos(th_safe)) / (th_safe * th_safe))
    c = torch.where(small, 1.0 / 6.0 - th2 / 120.0,
                    (th_safe - torch.sin(th_safe)) / (th_safe ** 3))
    I = torch.eye(3, dtype=w.dtype, device=w.device)
    return I + b * K + c * (K @ K)


def t_se3_exp(xi):
    import torch
    w, v = xi[:3], xi[3:]
    R = t_so3_exp(w)
    t = t_left_jacobian(w) @ v
    T = torch.eye(4, dtype=xi.dtype, device=xi.device)
    T = torch.cat([torch.cat([R, t[:, None]], dim=1), T[3:4]], dim=0)
    return T


def t_so3_log(R):
    """Differentiable SO(3) log (away from π; the solver's residuals are
    small-angle relative rotations)."""
    import torch
    tr = torch.clamp((R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0, -1.0 + 1e-12, 1.0 - 1e-12)
    th = torch.arccos(tr)
    v = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    small = th < _SMALL
    th_safe = torch.where(small, torch.ones_like(th), th)
    k = torch.where(small, 0.5 * (1.0 + th * th / 6.0), th_safe / (2.0 * torch.sin(th_safe)))
    return k * v


def t_se3_log(T):
    import torch
    w = t_so3_log(T[:3, :3])
    th = torch.linalg.norm(w)
    K = t_hat(w)
    th2 = th * th
    small = th < _SMALL
    th_safe = torch.where(small, torch.ones_like(th), th)
    half = th_safe / 2.0
    coef = torch.where(small, 1.0 / 12.0 + th2 / 720.0,
                       (1.0 - half * torch.cos(half) / torch.sin(half)) / (th_safe * th_safe))
    I = torch.eye(3, dtype=T.dtype, device=T.device)
    Jinv = I - 0.5 * K + coef * (K @ K)
    return torch.cat([w, Jinv @ T[:3, 3]])


def t_se3_inv(T):
    import torch
    R = T[:3, :3]
    t = T[:3, 3]
    top = torch.cat([R.T, (-R.T @ t)[:, None]], dim=1)
    return torch.cat([top, T[3:4]], dim=0)
