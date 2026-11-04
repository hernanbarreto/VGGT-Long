"""STAC (USER 2026-11-04, "hacé 1 y 2"): REVISIT closures found by GEOMETRY, beside SALAD's.

SALAD retrieves by appearance; a place walked twice and seen from other sides (pccr: frames
~800–1000 and ~1400–1725, the second pass 9–12 cm off the first) is never retrieved, and nothing
joins the two passes. Here the provisional chain (the metric-locked chunks joined by their seams,
vggt_long._StacChainView) proposes them: keyframes i, j whose chunks are not neighbours (they share
no frame — a seam already ties neighbours) and that see each other's surfaces — the spatial gate's
own reading, ``visible_fraction`` (the frustum widened by the drift budget of the walk between them,
loops.spatial), in BOTH directions at least ``min_visible_frac``, the gate's own bar. Per pair of
chunks the best few are returned, best first; the caller runs the spatial gate on them (its
occlusion test with each frame's own depth included) and spends a bridge on the first it accepts.
The bridge then MEASURES the closure like any SALAD one (loop_bridges), and the chunk graph's
consistency test and judge decide whether it enters."""

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


def quick_visibility(points: Sequence[np.ndarray], poses: np.ndarray, Ks: np.ndarray, hw,
                     cfg: dict, walk_cum: np.ndarray, budget: Callable[[float], Tuple[float, float]]
                     ) -> np.ndarray:
    """F[i, j] = the share of keyframe i's sampled points inside keyframe j's frustum, widened by
    the drift budget of the walk between them (no occlusion test — the gate runs it on the few
    pairs proposed). Vectorised over j for every i."""
    n = len(points)
    H, W = int(hw[0]), int(hw[1])
    zmin, zmax = float(cfg["min_depth_m"]), float(cfg["max_depth_m"])
    marg0 = float(cfg["frustum_margin_px"])
    R = poses[:, :3, :3]
    t = poses[:, :3, 3]
    fx, fy, cx, cy = Ks[:, 0, 0], Ks[:, 1, 1], Ks[:, 0, 2], Ks[:, 1, 2]
    f = 0.5 * (fx + fy)
    F = np.zeros((n, n))
    for i in range(n):
        p = np.asarray(points[i], np.float64)
        if len(p) == 0:
            continue
        L = np.abs(walk_cum - walk_cum[i])
        dl, th = np.array([budget(float(x)) for x in L]).T            # (n,), (n,)
        pc = np.einsum("nij,mi->nmj", R, p) - np.einsum("nij,ni->nj", R, t)[:, None, :]
        z = pc[..., 2]
        zs = np.maximum(z, zmin if zmin > 0 else 1e-6)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx[:, None] * pc[..., 0] / z + cx[:, None]
            v = fy[:, None] * pc[..., 1] / z + cy[:, None]
        margin = marg0 + f[:, None] * dl[:, None] / zs + f[:, None] * np.tan(np.radians(th))[:, None]
        ok = ((z >= zmin - dl[:, None]) & (z <= zmax + dl[:, None]) & (u >= -margin)
              & (u < W + margin) & (v >= -margin) & (v < H + margin))
        F[i] = ok.mean(axis=1)
    return F


def propose(F: np.ndarray, owner: Sequence[int], walk_cum: np.ndarray, min_walk_m: float,
            min_visible_frac: float, per_pair: int = 3) -> Dict[Tuple[int, int], List[Tuple[float, int, int]]]:
    """{(chunk a, chunk b): [(score, i, j), …]} — keyframe pairs of chunks that share no frame
    (|a − b| ≥ 2), walked ≥ ``min_walk_m`` apart (rule 0, the one definition of a visit), whose
    mutual visibility min(F[i,j], F[j,i]) ≥ ``min_visible_frac``; the ``per_pair`` best of each
    chunk pair, best first (ties → the lower i, then j). Deterministic."""
    owner = np.asarray(owner)
    n = len(owner)
    S = np.minimum(F, F.T)
    out: Dict[Tuple[int, int], List[Tuple[float, int, int]]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = int(owner[i]), int(owner[j])
            if a < 0 or b < 0 or abs(a - b) < 2:
                continue
            if abs(float(walk_cum[j] - walk_cum[i])) < float(min_walk_m):
                continue
            s = float(S[i, j])
            if s < float(min_visible_frac):
                continue
            out.setdefault((min(a, b), max(a, b)), []).append((s, i, j))
    for k in out:
        out[k] = sorted(out[k], key=lambda x: (-x[0], x[1], x[2]))[:int(per_pair)]
    return dict(sorted(out.items()))
