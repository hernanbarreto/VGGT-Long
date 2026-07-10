# STAC patch — per-chunk METRIC LOCK for the chunked Omega pipeline.
#
# Omega chunks are up-to-scale, each with its OWN arbitrary scale. The old windowed
# pipeline let the Sim(3) overlap alignment negotiate the relative scales — chunks
# disagreed by ±18-50% and the chained scale error produced double surfaces ("onion")
# and metric drift. This module locks EVERY chunk to metric BEFORE alignment, using
# isolated DA3 metric depth on a few anchor frames inside the chunk:
#
#     s_chunk = median over anchors of median(da3_depth / omega_depth)
#
# With all chunks metric, the overlap alignment runs as SE(3) (config
# Model.using_sim3: false) — scale is no longer a degree of freedom — and loop-closure
# constraints come out with relative scale ≈ 1 by construction.
#
# Depth convention: omega depth here is the chunk's per-frame 'depth' (camera-forward
# units, same scale as world_points/poses). The ratio is computed on the near-field
# band (lowest-quartile omega depth) where monocular metric depth is most reliable —
# the same policy as server/reconstruction/scale_align.py, which remains the global
# verifier after alignment.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import re

import cv2
import numpy as np

_FRAME_NUM = re.compile(r"(\d+)")


def real_frame_number(image_path):
    """Numeric stem of a processed frame's filename (the REAL video frame index)."""
    m = _FRAME_NUM.search(os.path.basename(str(image_path)))
    return int(m.group(1)) if m else -1


def anchor_ratio(omega_depth, da3_depth, conf=None, near_frac=0.25, min_px=50):
    """median(da3/omega) over pixels valid in BOTH maps, restricted to the near band
    (omega depth <= its `near_frac` quantile). Returns None when starved."""
    om = np.asarray(omega_depth, np.float32).squeeze()
    da = np.asarray(da3_depth, np.float32).squeeze()
    if om.ndim != 2 or da.ndim != 2 or om.size == 0 or da.size == 0:
        return None
    if da.shape != om.shape:
        da = cv2.resize(da, (om.shape[1], om.shape[0]), interpolation=cv2.INTER_LINEAR)
    m = np.isfinite(om) & np.isfinite(da) & (om > 1e-6) & (da > 1e-6)
    if conf is not None:
        cf = np.asarray(conf, np.float32).squeeze()
        if cf.shape == om.shape:
            m &= cf > 1e-5          # sky is masked by zeroing conf — exclude it
    if m.sum() < min_px:
        return None
    near = om <= np.quantile(om[m], near_frac)
    mm = m & near
    if mm.sum() < min_px:           # near band too thin → all valid pixels
        mm = m
    return float(np.median(da[mm] / om[mm]))


def chunk_scale(chunk_data, frame_numbers, anchor_dir, near_frac=0.25):
    """Metric scale for one chunk from the DA3 anchors that fall inside it.

    chunk_data: the chunk's prediction dict ('depth', 'world_points_conf', ...).
    frame_numbers: REAL frame number of each local frame (len == S).
    anchor_dir: dir holding frame_<num>.npz (keys: depth [, conf]) from isolated DA3.

    Returns (s, n_anchors, ratios) — s is None when no anchor lands in the chunk.
    """
    depth = np.asarray(chunk_data["depth"])
    conf3 = chunk_data.get("world_points_conf")
    ratios = []
    for local, num in enumerate(frame_numbers):
        npz_path = os.path.join(str(anchor_dir), f"frame_{int(num)}.npz")
        if not os.path.exists(npz_path):
            continue
        z = np.load(npz_path)
        if "depth" not in z:
            continue
        conf = None
        if conf3 is not None:
            c = np.asarray(conf3)
            conf = c[local] if local < c.shape[0] else None
        r = anchor_ratio(depth[local], z["depth"], conf=conf, near_frac=near_frac)
        if r is not None and np.isfinite(r) and r > 0:
            ratios.append(r)
    if not ratios:
        return None, 0, []
    return float(np.median(ratios)), len(ratios), ratios


def apply_scale(chunk_data, s):
    """Scale a chunk's whole metric IN PLACE: world_points, per-frame depth and the
    camera translations. Rotations and intrinsics are scale-invariant."""
    s = float(s)
    if chunk_data.get("world_points") is not None:
        chunk_data["world_points"] = np.asarray(chunk_data["world_points"]) * s
    if chunk_data.get("depth") is not None:
        chunk_data["depth"] = np.asarray(chunk_data["depth"]) * s
    ext = chunk_data.get("extrinsic")
    if ext is not None:
        ext = np.asarray(ext).copy()
        ext[..., :3, 3] *= s
        chunk_data["extrinsic"] = ext
    return chunk_data


def seam_relative_scale(depth_a, depth_b, min_px=1000):
    """Relative scale s_B/s_A that makes chunk B's depth agree with chunk A's on
    the SAME frame (same pixels): if raw d_B = r * d_A, then s_B/s_A = 1/r.
    Median over valid pixels; None when starved."""
    da = np.asarray(depth_a, np.float32).squeeze()
    db = np.asarray(depth_b, np.float32).squeeze()
    if da.shape != db.shape:
        return None
    m = np.isfinite(da) & np.isfinite(db) & (da > 1e-6) & (db > 1e-6)
    if m.sum() < min_px:
        return None
    return float(1.0 / np.median(db[m] / da[m]))


def solve_scale_graph(s_da3, n_anchors, seam_rel, n_chunks,
                      sigma_seam=0.003, sigma_anchor=0.08):
    """Optimal per-chunk metric scales from BOTH sensors, weighted least squares
    in log-scale space.

    The overlap frames measure the RELATIVE scale between neighbours to ~0.1-1%
    (same pixels, a ratio); the DA3 anchors measure each chunk's ABSOLUTE scale
    with ±8-15% monocular noise. Locking chunks to their own noisy DA3 median
    left neighbours disagreeing 5-27% — decimetres-to-metres of seam decoupling
    an SE(3) glue can never fix (measured, test4). Fusing both:

        minimize  Σ_seams   [(x_{k+1} - x_k) - log r_k]² / σ_seam²
                + Σ_anchors [x_k - log s_k^DA3]²         / (σ_anchor/√n_k)²

    x_k = log s_k. Linear, tiny (one variable per chunk). Seams make the scales
    CONSISTENT; anchors pin the global metre without dragging neighbours apart.

    s_da3: {k: s} absolute estimates; n_anchors: {k: count}; seam_rel: {k: r}
    with r = s_{k+1}/s_k (from seam_relative_scale). Returns np.ndarray scales.
    """
    rows, rhs, w = [], [], []
    for k, r in seam_rel.items():
        if r is None or not np.isfinite(r) or r <= 0:
            continue
        row = np.zeros(n_chunks)
        row[k], row[k + 1] = -1.0, 1.0
        rows.append(row); rhs.append(np.log(r)); w.append(1.0 / sigma_seam)
    for k, s in s_da3.items():
        if s is None or not np.isfinite(s) or s <= 0:
            continue
        row = np.zeros(n_chunks)
        row[k] = 1.0
        sig = sigma_anchor / max(np.sqrt(float(n_anchors.get(k, 1))), 1.0)
        rows.append(row); rhs.append(np.log(s)); w.append(1.0 / sig)
    if not rows:
        raise ValueError("scale graph has no constraints")
    A = np.asarray(rows) * np.asarray(w)[:, None]
    b = np.asarray(rhs) * np.asarray(w)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.exp(x)


def robust_rigid(src, dst, iters=8, sample=200000, seed=0):
    """Rigid fit dst ≈ R·src + t from EXACT correspondences (same pixel, same
    frame, two chunks), IRLS with a Cauchy weight on the residuals so the
    chunk-internal disagreement (non-rigid noise + far-field junk) does not
    drag the fit. Returns (R, t, median_residual_m, n_used) or None."""
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    m = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    src, dst = src[m], dst[m]
    if len(src) < 1000:
        return None
    if len(src) > sample:
        idx = np.random.default_rng(seed).choice(len(src), sample, replace=False)
        src, dst = src[idx], dst[idx]
    w = np.ones(len(src))
    R, t = np.eye(3), np.zeros(3)
    for _ in range(int(iters)):
        ws = w.sum()
        cs_ = (src * w[:, None]).sum(0) / ws
        cd_ = (dst * w[:, None]).sum(0) / ws
        H = ((src - cs_) * w[:, None]).T @ (dst - cd_)
        U, _, Vt = np.linalg.svd(H)
        D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ D @ U.T
        t = cd_ - R @ cs_
        r = np.linalg.norm(dst - (src @ R.T + t), axis=1)
        c = max(3.0 * 1.4826 * np.median(np.abs(r - np.median(r))), 1e-4)
        w = 1.0 / (1.0 + (r / c) ** 2)
    r = np.linalg.norm(dst - (src @ R.T + t), axis=1)
    return R, t, float(np.median(r)), int(len(src))


def frame_owner(chunk_indices, n_frames):
    """owner[g] = index of the chunk that WRITES frame g's points to the cloud.
    Every frame is written by exactly ONE chunk (the one whose centre is nearest)
    — overlap frames used to be written by BOTH chunks, putting two displaced
    copies of the same pixels into the cloud (the mechanical half of the
    duplicated-objects problem)."""
    owner = np.full(int(n_frames), -1, np.int32)
    centers = [(s0 + e0) / 2.0 for s0, e0 in chunk_indices]
    for g in range(int(n_frames)):
        best, bd = -1, None
        for k, (s0, e0) in enumerate(chunk_indices):
            if s0 <= g < e0:
                d = abs(g - centers[k])
                if bd is None or d < bd:
                    best, bd = k, d
        owner[g] = best
    return owner
