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


def chunk_anchor_ratios(chunk_data, frame_numbers, anchor_dir, near_frac=0.25):
    """Per-anchor (local_index, ratio) pairs for one chunk — the positioned form
    of the anchor evidence, so a scale that DRIFTS along the chunk is visible
    instead of being collapsed into one median.

    chunk_data: the chunk's prediction dict ('depth', 'world_points_conf', ...).
    frame_numbers: REAL frame number of each local frame (len == S).
    anchor_dir: dir holding frame_<num>.npz (keys: depth [, conf]) from isolated DA3.
    """
    depth = np.asarray(chunk_data["depth"])
    conf3 = chunk_data.get("world_points_conf")
    locals_, ratios = [], []
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
            locals_.append(int(local))
            ratios.append(r)
    return locals_, ratios


def chunk_scale(chunk_data, frame_numbers, anchor_dir, near_frac=0.25):
    """Metric scale for one chunk from the DA3 anchors that fall inside it.

    Returns (s, n_anchors, ratios) — s is None when no anchor lands in the chunk.
    """
    _, ratios = chunk_anchor_ratios(chunk_data, frame_numbers, anchor_dir,
                                    near_frac=near_frac)
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


def solve_scale_drift(anchors, seam_obs, n_chunks, prior=None,
                      sigma_anchor=0.08, sigma_seam=0.01,
                      sigma_drift=0.12, sigma_prior=0.5):
    """Per-chunk LINEAR scale drift, weighted least squares in log-scale space.

    A single scalar per chunk cannot represent a chunk whose internal scale
    DRIFTS (measured on test4: DA3 anchor ratios spread 10.9-16.1 inside one
    chunk — 48% — while adjacent locked scales jumped 18-29%; the leftover
    warp is the z-drift on tall structures). Model: log s_k(u) is linear in
    the chunk position u∈[0,1], two unknowns per chunk (x_k0 at the start,
    x_k1 at the end):

        anchors  : (1-u)·x_k0 + u·x_k1 = log r          / sigma_anchor
        seams    : x_{k+1}(u') - x_k(u) = log r_seam    / sigma_seam
        drift    : x_k1 - x_k0 = 0                      / sigma_drift  (prior)
        prior    : x_k(0.5) = log s_prior[k]            / sigma_prior  (weak)

    anchors:  {k: [(u, ratio), ...]} positioned DA3 evidence per chunk.
    seam_obs: {k: [(u_k, u_k1, r), ...]} per-shared-frame relative scale
              between chunk k (at u_k) and k+1 (at u_k1) — the same-pixel
              seam sensor, now positioned inside BOTH chunks.
    prior:    {k: s} optional (e.g. the constant scale-graph solution) — keeps
              data-starved chunks near the proven constant answer.

    Returns (s0, s1) arrays of per-chunk start/end scales.
    """
    rows, rhs, w = [], [], []

    def _row(coeffs, value, sigma):
        row = np.zeros(2 * n_chunks)
        for j, c in coeffs:
            row[j] += c
        rows.append(row); rhs.append(value); w.append(1.0 / sigma)

    for k, obs in (anchors or {}).items():
        for u, r in obs:
            if r is None or not np.isfinite(r) or r <= 0:
                continue
            u = min(max(float(u), 0.0), 1.0)
            _row([(2 * k, 1.0 - u), (2 * k + 1, u)], np.log(r), sigma_anchor)
    for k, obs in (seam_obs or {}).items():
        if k + 1 >= n_chunks:
            continue
        for u_k, u_k1, r in obs:
            if r is None or not np.isfinite(r) or r <= 0:
                continue
            u_k = min(max(float(u_k), 0.0), 1.0)
            u_k1 = min(max(float(u_k1), 0.0), 1.0)
            _row([(2 * k, -(1.0 - u_k)), (2 * k + 1, -u_k),
                  (2 * (k + 1), 1.0 - u_k1), (2 * (k + 1) + 1, u_k1)],
                 np.log(r), sigma_seam)
    for k in range(n_chunks):
        _row([(2 * k, -1.0), (2 * k + 1, 1.0)], 0.0, sigma_drift)
        s_p = (prior or {}).get(k)
        if s_p is not None and np.isfinite(s_p) and s_p > 0:
            _row([(2 * k, 0.5), (2 * k + 1, 0.5)], np.log(s_p), sigma_prior)
    if not rows:
        raise ValueError("scale drift graph has no constraints")
    A = np.asarray(rows) * np.asarray(w)[:, None]
    b = np.asarray(rhs) * np.asarray(w)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.exp(x[0::2]), np.exp(x[1::2])


def scale_drift_gate(anchors, s_const, seam_obs, n_chunks,
                     min_holdout=5, improve=0.75, max_drift_log=None, **solve_kw):
    """Self-validation for the drift model — it must EARN the right to touch
    the geometry (same discipline as the depth graph).

    The judge is the PRECISE sensor: the per-frame seam ratios (same pixels,
    0.3-1% noise — the drift signature is their variation ALONG the overlap;
    DA3 anchors at 8-15% noise cannot discriminate a model this fine). Every
    3rd seam observation is HELD OUT; both the drift model and a CONSTANT
    reference are fitted on the SAME remaining data — the constant one is
    the identical solver with the drift DOF pinned, so the comparison leaks
    nothing and favours nobody:

      apply ⟺ held-out |log error| median improves by ≥ (1-improve)
              (default 25% — measured: real ±10-20% drift improves ~9x while
              pure seam noise buys at most ~17% by overfitting the 2 extra
              DOF/chunk, so the cut separates them with an order of magnitude
              of margin)
              AND every chunk's |log(s1/s0)| ≤ max_drift_log (default log 1.6).

    Returns (verdict_bool, info dict). The caller refits on ALL data when
    the verdict is True."""
    if max_drift_log is None:
        max_drift_log = float(np.log(1.6))
    fit_seams, held = {}, []
    for k, obs in (seam_obs or {}).items():
        keep = []
        for i, (u_k, u_k1, r) in enumerate(obs):
            if len(obs) >= 3 and i % 3 == 2:
                held.append((k, u_k, u_k1, r))
            else:
                keep.append((u_k, u_k1, r))
        fit_seams[k] = keep
    if len(held) < int(min_holdout):
        return False, {"reason": f"held-out too thin ({len(held)} seam obs "
                                 f"< {min_holdout})",
                       "n_holdout": len(held)}
    s0, s1 = solve_scale_drift(anchors, fit_seams, n_chunks,
                               prior=s_const, **solve_kw)
    kw_const = dict(solve_kw, sigma_drift=1e-6)      # same solver, drift pinned
    c0, c1 = solve_scale_drift(anchors, fit_seams, n_chunks,
                               prior=s_const, **kw_const)

    def _pred(a0, a1, k, u):
        return (1.0 - u) * np.log(a0[k]) + u * np.log(a1[k])

    err_d, err_c = [], []
    for k, u_k, u_k1, r in held:
        err_d.append(abs(np.log(r) - (_pred(s0, s1, k + 1, u_k1)
                                      - _pred(s0, s1, k, u_k))))
        err_c.append(abs(np.log(r) - (_pred(c0, c1, k + 1, u_k1)
                                      - _pred(c0, c1, k, u_k))))
    med_d, med_c = float(np.median(err_d)), float(np.median(err_c))
    drift_mag = float(np.max(np.abs(np.log(s1 / s0))))
    ok = med_d <= float(improve) * med_c and drift_mag <= max_drift_log
    return ok, {"n_holdout": len(held), "holdout_const": med_c,
                "holdout_drift": med_d, "max_drift_log": drift_mag,
                "bound_log": max_drift_log}


def apply_scale_drift(chunk_data, s_frames):
    """Scale a chunk with a PER-FRAME factor, in place. Depth scales about each
    frame's own camera; the trajectory is re-integrated step by step so camera
    spacing follows the local scale. With a constant factor this reduces EXACTLY
    to apply_scale. Extrinsics here are c2w (center = translation column)."""
    s = np.asarray(s_frames, np.float64).reshape(-1)
    ext0 = np.asarray(chunk_data["extrinsic"])
    ext = ext0.astype(np.float64).copy()
    S = ext.shape[0]
    if len(s) != S:
        raise ValueError(f"s_frames has {len(s)} entries for {S} frames")
    c_old = ext[:, :3, 3].copy()
    c_new = np.empty_like(c_old)
    c_new[0] = s[0] * c_old[0]
    for i in range(1, S):
        c_new[i] = c_new[i - 1] + np.sqrt(s[i - 1] * s[i]) * (c_old[i] - c_old[i - 1])
    ext[:, :3, 3] = c_new
    chunk_data["extrinsic"] = ext.astype(ext0.dtype)

    if chunk_data.get("depth") is not None:
        d0 = np.asarray(chunk_data["depth"])
        d = d0.astype(np.float64) * s.reshape((S,) + (1,) * (d0.ndim - 1))
        chunk_data["depth"] = d.astype(d0.dtype)

    if chunk_data.get("world_points") is not None:
        wp0 = np.asarray(chunk_data["world_points"])
        lead = wp0.ndim == 5
        w = (wp0[0] if lead else wp0).astype(np.float64)
        for i in range(S):
            w[i] = c_new[i] + s[i] * (w[i] - c_old[i])
        chunk_data["world_points"] = (w[None] if lead else w).astype(wp0.dtype)
    return chunk_data


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


def rigid_fraction(R, t, alpha):
    """Fractional rigid transform as a 4x4: the rotation angle is scaled on its own
    axis (Rodrigues) and the translation linearly. Smooth in alpha, EXACTLY identity
    at alpha=0 and exactly (R, t) at alpha=1. Seam agreement never depends on the
    interpolation path (elastic_corrections composes the dst side as B @ T^-1), so
    the path only needs smoothness and exact endpoints."""
    a = float(alpha)
    rvec, _ = cv2.Rodrigues(np.asarray(R, np.float64))
    Ra, _ = cv2.Rodrigues(rvec * a)
    M = np.eye(4)
    M[:3, :3] = Ra
    M[:3, 3] = a * np.asarray(t, np.float64).reshape(3)
    return M


def rigid_mat(R, t):
    """(R, t) as a 4x4 homogeneous matrix."""
    M = np.eye(4)
    M[:3, :3] = np.asarray(R, np.float64)
    M[:3, 3] = np.asarray(t, np.float64).reshape(3)
    return M


def elastic_corrections(chunk_indices, k, seam_fits, sick=None, suspect=None):
    """Per-frame ELASTIC seam corrections for chunk k: [S, 4, 4] world-space rigid
    moves, one per local frame (identity where nothing constrains the frame).

    The anchoring directive this stage exists for: the same pixel of a frame present
    in two chunks MUST land at the same 3D position. Per shared frame g of seam j
    (between chunks j and j+1), seam_fits[j][g] = (R, t) is the rigid residual T_g
    mapping chunk j+1's copy of the frame onto chunk j's copy (robust_rigid on the
    exact pixel-to-pixel correspondences, AFTER the global alignment). Both copies
    are moved onto ONE consensus pose:

        chunk j+1 (src side of the fit):  B_g = rigid_fraction(T_g, alpha_g)
        chunk j   (dst side of the fit):  A_g = B_g @ T_g^-1

    A_g @ T_g == B_g by construction, so the two corrected copies COINCIDE exactly
    — the only residual left is the per-frame fit residual (intra-frame non-rigid
    disagreement, the physical floor).

    alpha_g is the weight of chunk j's opinion, linear in the frame's position
    inside the overlap: 1 at the overlap start (chunk j's centre — its copy stays
    put, A = I) down to 0 at the overlap end (chunk j+1's centre — ITS copy stays
    put, B = I). Each chunk's correction field is therefore identity at its own
    centre and grows toward its edges: interiors are never torn, edges bend onto
    the consensus, and the fused cloud is continuous through the frame-ownership
    switch in the middle of every overlap. (With the pipeline's 50% overlap each
    frame belongs to at most two chunks, so the two seams of a chunk touch
    disjoint frame ranges.)

    A shared frame whose fit starved inherits the nearest fitted frame of the same
    seam (the seam is already rigid-glued, so per-frame residuals are small and
    smooth); a seam with no fits at all contributes identity.

    `sick`: chunk indices flagged by the health gate (flag_sick_chunks). A healthy
    chunk must NEVER bend toward a sick neighbour's garbage, so on a mixed seam
    alpha is pinned to keep the consensus entirely at the healthy side's copy (the
    sick side adopts it fully — harmless, its frames don't write points anyway).

    `suspect`: the SOFT tier (flag_suspect_chunks). On a seam where exactly one
    side is suspect the consensus is BIASED toward the trusted side (alpha warped
    quadratically) instead of pinned — endpoints stay exact (1→0), so interiors
    are never torn and the field stays continuous; the shaky chunk just gets
    less say in where the shared pixels land.
    """
    start, end = chunk_indices[k]
    S = end - start
    corr = np.tile(np.eye(4), (S, 1, 1))
    sick = frozenset(sick or ())
    suspect = frozenset(suspect or ()) - sick

    def _fit_for(j, g, shared):
        d = seam_fits.get(j) or {}
        if g in d:
            return d[g]
        fitted = [gg for gg in shared if gg in d]
        if not fitted:
            return None
        return d[min(fitted, key=lambda gg: abs(gg - g))]

    def _alpha(j, i, L):
        # weight of chunk j's (dst-side) opinion; pinned on mixed-health seams
        if (j in sick) != (j + 1 in sick):
            return 0.0 if j in sick else 1.0
        a = 1.0 - (i / (L - 1.0)) if L > 1 else 0.5
        # soft tier: bias toward the trusted side, endpoints kept exact
        if (j in suspect) != (j + 1 in suspect):
            a = a * a if j in suspect else 1.0 - (1.0 - a) ** 2
        return a

    # left seam (j = k-1): this chunk is the SRC side of the fit -> B_g
    if k > 0:
        _, e_prev = chunk_indices[k - 1]
        shared = list(range(start, min(e_prev, end)))
        L = len(shared)
        for i, g in enumerate(shared):
            f = _fit_for(k - 1, g, shared)
            if f is None:
                continue
            corr[g - start] = rigid_fraction(f[0], f[1], _alpha(k - 1, i, L))
    # right seam (j = k): this chunk is the DST side -> A_g = B_g @ T_g^-1
    if k < len(chunk_indices) - 1:
        s_next, _ = chunk_indices[k + 1]
        shared = list(range(max(s_next, start), end))
        L = len(shared)
        for i, g in enumerate(shared):
            f = _fit_for(k, g, shared)
            if f is None:
                continue
            T = rigid_mat(f[0], f[1])
            corr[g - start] = rigid_fraction(f[0], f[1], _alpha(k, i, L)) @ np.linalg.inv(T)
    return corr


def depth_pair_samples(wp_src, conf_src, wp_dst, conf_dst, w2c_dst, K_dst,
                       max_samples=8000, seed=0):
    """EXACT-surface depth samples between two frames: project src's valid points
    into dst's camera, read dst's OWN depth at the hit pixel. Returns (z_src, z_dst)
    — the depth src's geometry implies in dst's frame vs the depth dst itself
    predicts for that surface — or None when starved. Purely geometric: the same
    association the seam work uses, generalized to any nearby frame pair."""
    H, W = wp_dst.shape[:2]
    p = np.asarray(wp_src, np.float64).reshape(-1, 3)
    c = np.asarray(conf_src, np.float32).reshape(-1)
    idx = np.flatnonzero(c > 1e-5)
    if len(idx) < 500:
        return None
    if len(idx) > max_samples:
        idx = np.random.default_rng(seed).choice(idx, max_samples, replace=False)
    p = p[idx]
    w2c = np.asarray(w2c_dst, np.float64)
    X = p @ w2c[:3, :3].T + w2c[:3, 3]
    z = X[:, 2]
    m = z > 0.3
    if m.sum() < 300:
        return None
    fx, fy, cx, cy = float(K_dst[0, 0]), float(K_dst[1, 1]), float(K_dst[0, 2]), float(K_dst[1, 2])
    u = np.round(X[m, 0] / z[m] * fx + cx).astype(int)
    v = np.round(X[m, 1] / z[m] * fy + cy).astype(int)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if inb.sum() < 300:
        return None
    u, v, z_src = u[inb], v[inb], z[m][inb]
    q = np.asarray(wp_dst, np.float64)[v, u]
    cq = np.asarray(conf_dst, np.float32).reshape(H, W)[v, u]
    good = cq > 1e-5
    if good.sum() < 300:
        return None
    Xq = q[good] @ w2c[:3, :3].T + w2c[:3, 3]
    return z_src[good], Xq[:, 2]


def pair_depth_relation(z_src, z_dst, iters=6):
    """Robust affine relation z_dst ~= alpha*z_src + beta between two frames'
    depths of the SAME surfaces (IRLS, Cauchy weights — occlusion mismatches land
    in the tail and get killed). Returns (alpha, beta, median_rel_mismatch, n)
    or None. rel mismatch = |z_src - z_dst| / z_dst BEFORE the fit (diagnostic)."""
    zs = np.asarray(z_src, np.float64)
    zd = np.asarray(z_dst, np.float64)
    m = np.isfinite(zs) & np.isfinite(zd) & (zs > 1e-3) & (zd > 1e-3)
    zs, zd = zs[m], zd[m]
    if len(zs) < 300:
        return None
    before = float(np.median(np.abs(zs - zd) / zd))
    w = np.ones(len(zs))
    a, b = float(np.median(zd / zs)), 0.0
    for _ in range(int(iters)):
        sw = w.sum()
        mx = (zs * w).sum() / sw
        my = (zd * w).sum() / sw
        vx = (w * (zs - mx) ** 2).sum()
        if vx < 1e-12:
            return None
        a = (w * (zs - mx) * (zd - my)).sum() / vx
        b = my - a * mx
        r = zd - (a * zs + b)
        cch = max(3.0 * 1.4826 * np.median(np.abs(r - np.median(r))), 1e-4)
        w = 1.0 / (1.0 + (r / cch) ** 2)
    if not (0.8 < a < 1.25):      # a pair this broken is occlusion/garbage, not a
        return None               # depth-field measurement (measured pairs: <=5%)
    return float(a), float(b), before, int(len(zs))


def solve_depth_graph(measurements, n_frames, sick_frames=(), scale_only=False):
    """Per-frame depth corrections z' = a_f*z + b_f from pairwise affine relations,
    the frame-level analogue of solve_scale_graph. For a pair (f, g) with measured
    z_g = alpha*z_f + beta, corrected consistency (a_f z + b_f == a_g(alpha z +
    beta) + b_g for all z) splits into two LINEAR systems solved in sequence:

        log a_f - log a_g = log alpha        (scale graph, gauge: mean log a = 0)
        b_f - b_g         = a_g * beta       (offset graph, gauge: mean b = 0)

    The gauges preserve the session's global metre (set by the metric lock +
    scale_align) — the graph only REDISTRIBUTES depth so every frame agrees on
    every shared surface. Sick frames get identity (a=1, b=0) and their
    measurements must not be fed in. Returns (a[n], b[n]).

    ``scale_only``: skip the offset system entirely (b stays 0) — the fallback
    rung of the model ladder. After the metric lock + scale drift the residual
    inter-frame depth disagreement is mostly MULTIPLICATIVE (leftover scale
    error); the free offset is where an unbounded low-frequency warp hides
    (measured on test4: b ran to ±112 cm while the scale stayed near 1)."""
    meas = [(f, g, al, be) for f, g, al, be in measurements
            if f not in sick_frames and g not in sick_frames
            and np.isfinite(al) and al > 0 and np.isfinite(be)]
    a = np.ones(n_frames)
    b = np.zeros(n_frames)
    if not meas:
        return a, b
    frames = sorted({f for f, g, _, _ in meas} | {g for _, g, _, _ in meas})
    col = {f: i for i, f in enumerate(frames)}
    n = len(frames)
    rows, rhs = [], []
    for f, g, al, _ in meas:
        r = np.zeros(n)
        r[col[f]], r[col[g]] = 1.0, -1.0
        rows.append(r)
        rhs.append(np.log(al))
    gauge = np.full(n, float(len(rows)) / n)      # strong: pins the mean exactly
    rows.append(gauge)
    rhs.append(0.0)
    x, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    for f, i in col.items():
        a[f] = float(np.exp(x[i]))
    if scale_only:
        return a, b
    rows, rhs = [], []
    for f, g, _, be in meas:
        r = np.zeros(n)
        r[col[f]], r[col[g]] = 1.0, -1.0
        rows.append(r)
        rhs.append(a[g] * be)
    rows.append(gauge)
    rhs.append(0.0)
    y, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    for f, i in col.items():
        b[f] = float(y[i])
    return a, b


def depth_graph_verdict(a, b, meas, held, zref=5.0, improve=0.8, bound=5.0):
    """Self-validation shared by every rung of the depth-graph model ladder.
    Judged ONLY on held-out pairs (never fitted): the corrected disagreement at
    ``zref`` must improve by ≥(1-improve), and the corrections must stay within
    ``bound``× the pairwise signal (an order of magnitude beyond what the pairs
    show is noise integration along the chain, not signal). Returns a dict with
    bounded / improves / med_before / med_after / sig_a / sig_b."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    rb = [abs(al * zref + be - zref) / zref for _, _, al, be in held]
    ra = [abs((a[f] * zref + b[f]) - (a[g] * (al * zref + be) + b[g])) / zref
          for f, g, al, be in held]
    med_b, med_a = float(np.median(rb)), float(np.median(ra))
    las = np.abs(np.log([al for _, _, al, _ in meas]))
    bes = np.abs([be for _, _, _, be in meas])
    sig_a = max(float(np.median(las)), 1e-4)
    sig_b = max(float(np.median(bes)), 1e-3)
    bounded = (float(np.percentile(np.abs(np.log(a)), 99)) <= bound * sig_a
               and float(np.percentile(np.abs(b), 99)) <= bound * sig_b)
    improves = med_a <= improve * med_b
    return {"bounded": bounded, "improves": improves,
            "med_before": med_b, "med_after": med_a,
            "sig_a": sig_a, "sig_b": sig_b}


def apply_depth_correction(world_points, depth, cam_center, a, b):
    """Move ONE frame's points along their camera rays so its depth becomes
    a*z + b: p' = c + (p - c) * (a*z + b)/z (angles fixed, pixels unchanged —
    cameras do NOT move). Returns (world_points', depth'). Pixels with no valid
    depth are left untouched."""
    wp = np.asarray(world_points)
    d = np.asarray(depth, np.float32).reshape(wp.shape[0], wp.shape[1])
    t = np.ones_like(d, np.float64)
    m = np.isfinite(d) & (d > 1e-6)
    t[m] = (float(a) * d[m].astype(np.float64) + float(b)) / d[m]
    c = np.asarray(cam_center, np.float64).reshape(3)
    wp2 = (c + (wp.astype(np.float64) - c) * t[..., None]).astype(wp.dtype)
    d2 = d.copy()
    d2[m] = (float(a) * d[m] + float(b)).astype(d.dtype)
    return wp2, d2


def blend_two_copies(wp1, cf1, wp2, cf2, d1=None, d2=None):
    """TWO-COPY CONSENSUS for one frame: every overlap frame is predicted by BOTH
    of its chunks; ownership used to discard one copy. The two copies are two
    independent measurements of the same depth field (elastic-aligned) — their
    per-pixel mean cuts the field noise ~sqrt(2) and, because neighbouring frames'
    blends share sources, it halves the field jump at the ownership switch
    (measured on test4: cross-owner pair disagreement 1.51% -> 1.01%).

    Returns (wp, cf[, d]): mean where BOTH copies are valid, the valid copy where
    only one is, conf = elementwise max over the union. IDEMPOTENT once applied
    (both copies set to the same blend -> re-blending is a no-op)."""
    v1 = np.asarray(cf1, np.float32) > 1e-5
    v2 = np.asarray(cf2, np.float32) > 1e-5
    both = v1 & v2
    wp = np.where(v1[..., None], np.asarray(wp1), np.asarray(wp2)).astype(np.float64)
    wp[both] = 0.5 * (np.asarray(wp1)[both].astype(np.float64)
                      + np.asarray(wp2)[both].astype(np.float64))
    cf = (np.maximum(np.asarray(cf1, np.float32), np.asarray(cf2, np.float32))
          * (v1 | v2)).astype(np.float32)
    wp = wp.astype(np.asarray(wp1).dtype)
    if d1 is None or d2 is None:
        return wp, cf
    d = np.where(v1, np.asarray(d1), np.asarray(d2)).astype(np.float64)
    d[both] = 0.5 * (np.asarray(d1)[both].astype(np.float64)
                     + np.asarray(d2)[both].astype(np.float64))
    return wp, cf, d.astype(np.asarray(d1).dtype)


def classify_far_points(pts, conf_ok, cam_g, depth_g, conf_g, w2c_g, K_g,
                        cap, floor_m, rate, k_sigma=3.0):
    """One frame-pair step of the CONTRADICTION test for far-observed points.

    A far observation may only be dropped when a NEARBY frame looked at the same
    spot and saw something else (a displaced duplicate); if nobody saw it from
    close, it is UNIQUE coverage and must stay (measured on test4: a blanket
    distance cap destroyed 87% real coverage — narrow FOV, side/elevated
    structures never get a near pass while in frame).

    pts: [n,3] the far points to test. cam_g/depth_g/conf_g/w2c_g/K_g: the
    candidate near frame. A point is TESTED only if it lies within `cap` of
    frame g's camera (g could have observed it within the error budget).
    Projecting it into g: |z_projected - z_g| <= k_sigma*max(floor, rate*z_g)
    -> AGREE (corroborated, keep); beyond -> CONTRADICTED (displaced duplicate
    or free-space violation). Returns (agree, contra) boolean masks over pts;
    untested points are False in both (unseen -> caller keeps them)."""
    n = len(pts)
    agree = np.zeros(n, bool)
    contra = np.zeros(n, bool)
    sel = conf_ok & (np.linalg.norm(pts - np.asarray(cam_g).reshape(3), axis=1) <= cap)
    if not sel.any():
        return agree, contra
    idx = np.flatnonzero(sel)
    w2c = np.asarray(w2c_g, np.float64)
    X = pts[idx] @ w2c[:3, :3].T + w2c[:3, 3]
    z = X[:, 2]
    m = z > 0.3
    if not m.any():
        return agree, contra
    H, W = depth_g.shape[:2]
    fx, fy, cx, cy = float(K_g[0, 0]), float(K_g[1, 1]), float(K_g[0, 2]), float(K_g[1, 2])
    u = np.round(X[m, 0] / z[m] * fx + cx).astype(int)
    v = np.round(X[m, 1] / z[m] * fy + cy).astype(int)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not inb.any():
        return agree, contra
    ii = idx[m][inb]
    zg = np.asarray(depth_g)[v[inb], u[inb]].astype(np.float64)
    cg = np.asarray(conf_g)[v[inb], u[inb]]
    ok = (cg > 1e-5) & (zg > 0.3)
    if not ok.any():
        return agree, contra
    ii = ii[ok]
    diff = np.abs(z[m][inb][ok] - zg[ok])
    tol = float(k_sigma) * np.maximum(float(floor_m), float(rate) * zg[ok])
    agree[ii[diff <= tol]] = True
    contra[ii[diff > tol]] = True
    return agree, contra


def chunk_tri_angle(depth, conf, extrinsic):
    """Per-chunk triangulation angle (rad): median keyframe camera step over the
    median valid depth — the amount of PARALLAX the chunk actually holds. Depth
    error of any multi-view geometry scales as pixel_error / tri_angle, so a chunk
    an order of magnitude below its session's walking pace carries no usable 3D
    information (rotation-only / far-field — measured on test4: body 0.08-0.11 rad,
    rotten tail 0.005/0.0008). SCALE-INVARIANT: steps and depth share the chunk's
    scale, so it can be computed before or after the metric lock. None if starved."""
    ext = np.asarray(extrinsic, np.float64)
    if ext.ndim != 3 or ext.shape[0] < 2:
        return None
    steps = np.linalg.norm(np.diff(ext[:, :3, 3], axis=0), axis=1)
    d = np.asarray(depth, np.float32).reshape(-1)
    c = np.asarray(conf, np.float32).reshape(-1)
    m = np.isfinite(d) & (d > 1e-6) & (c > 1e-5)
    if not m.any() or not np.isfinite(steps).all():
        return None
    med_depth = float(np.median(d[m]))
    if med_depth <= 0:
        return None
    return float(np.median(steps)) / med_depth


def flag_sick_chunks(tri_angle, anchor_iqr, fx_median=None,
                     parallax_floor_ratio=10.0, anchor_z_cut=3.5, zoom_z_cut=3.5):
    """Health gate over a session's own chunks. Returns {k: [reasons]} for the
    chunks whose numbers say their geometry cannot be trusted. Two independent
    signals, thresholds derived from the SESSION itself (no external truth):

    1. Parallax: tri_angle[k] < median(tri_angle) / parallax_floor_ratio — an
       order of magnitude below the session's own walking pace. Physical basis:
       relative depth error ~ pixel_error / tri_angle; the measured gap on test4
       is 18-107x (body 0.081-0.112 rad vs rotten 0.0051/0.00084), so the decade
       cut sits far from both sides.
    2. DA3 anchor incoherence: robust z-score (Iglewicz-Hoaglin, MAD-based,
       standard 3.5 cut) of the chunk's anchor-ratio IQR/median across chunks —
       DA3 and Omega disagreeing WITHIN one chunk means its internal scale is not
       a single number (test4 chunk 10: 0.374 vs session median 0.100, z=4.4).

    Sick chunks stay in the alignment + scale graph (with 50% overlap they are the
    only bridge between their neighbours) — the caller must only stop them from
    WRITING points."""
    sick = {}
    tri = {k: v for k, v in (tri_angle or {}).items() if v is not None and np.isfinite(v)}
    if len(tri) >= 2:
        med = float(np.median(list(tri.values())))
        for k, v in tri.items():
            if med > 0 and v < med / float(parallax_floor_ratio):
                sick.setdefault(k, []).append(
                    f"triangulation angle {v:.5f} rad is {med / max(v, 1e-12):.0f}x below "
                    f"the session median {med:.4f} — rotation-only/far-field, no 3D information")
    aiq = {k: v for k, v in (anchor_iqr or {}).items() if v is not None and np.isfinite(v)}
    if len(aiq) >= 3:
        vals = np.array(list(aiq.values()), np.float64)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        if mad > 0:
            for k, v in aiq.items():
                z = (v - med) / (1.4826 * mad)
                if z > float(anchor_z_cut):
                    sick.setdefault(k, []).append(
                        f"DA3 anchors disagree within the chunk: ratio IQR/median {v:.3f} "
                        f"(session median {med:.3f}, robust z={z:.1f}) — internal scale "
                        f"is not one number")
    # 3. Optical ZOOM: the per-frame focal the model itself estimates. A zoom
    #    segment magnifies without adding baseline — no parallax, no 3D — and it
    #    also breaks DA3's metric depth (assumed focal). Robust z (same 3.5 cut)
    #    of the chunk-median fx across chunks: the body is stable to ~2% while a
    #    real zoom is +25..140% (measured on test4: fx 550 body vs 704-1334 tail).
    fxm = {k: v for k, v in (fx_median or {}).items() if v is not None and np.isfinite(v)}
    if len(fxm) >= 3:
        vals = np.array(list(fxm.values()), np.float64)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        if mad > 0:
            for k, v in fxm.items():
                z = abs(v - med) / (1.4826 * mad)
                if z > float(zoom_z_cut):
                    sick.setdefault(k, []).append(
                        f"optical ZOOM: chunk-median focal {v:.0f} vs session {med:.0f} "
                        f"(robust z={z:.1f}) — magnification adds no baseline: no "
                        f"parallax, no 3D information, and DA3 metric depth breaks")
    return sick


def flag_suspect_chunks(anchor_iqr, sick=None, spread_cut=0.30):
    """SOFT health tier below `sick`: chunks whose DA3 anchor spread says their
    internal scale is shaky (IQR/median > spread_cut — test4 chunk 3 sat at
    0.12 raw but its full anchor RANGE spanned 48%) yet not bad enough for the
    sick gate's robust-z cut. A suspect chunk keeps writing points; it just
    argues MORE QUIETLY where opinions are weighed (elastic seam consensus,
    fine_register unit confidence). Returns {k: reason}."""
    sick = frozenset(sick or ())
    out = {}
    for k, v in (anchor_iqr or {}).items():
        if k in sick or v is None or not np.isfinite(v):
            continue
        if v > float(spread_cut):
            out[k] = (f"DA3 anchor spread IQR/median {v:.3f} > {spread_cut:.2f} — "
                      f"internal scale is shaky (opinion down-weighted, "
                      f"points still written)")
    return out


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


# ── INTRA-CHUNK per-frame consensus (bounded fields, anchored boundaries) ──
# History (test4, 2026-07-10): a GLOBAL per-frame pose graph with free
# boundaries hallucinated wavelengths longer than its pair span (163 cm bend,
# chimney 7→130 cm) — short-span evidence cannot certify long corrections.
# This design inverts it: each chunk is solved ALONE with its endpoints
# CLAMPED to zero (the seam consensus the elastic already established), so no
# correction longer than one chunk can exist, and per-chunk self-gates keep
# the worst case at identity for that chunk only.


def surface_pair_correspondences(wp_src, conf_src, wp_dst, conf_dst, w2c_dst, K_dst,
                                 max_samples=8000, seed=0):
    """EXACT-surface 3D correspondences between two frames: project src's valid
    points into dst's camera and pair them with dst's OWN 3D point at the hit
    pixel. Returns (p_src[n,3], q_dst[n,3]) or None when starved — the rigid
    analogue of depth_pair_samples (same association, full 3D instead of z)."""
    H, W = wp_dst.shape[:2]
    p = np.asarray(wp_src, np.float64).reshape(-1, 3)
    c = np.asarray(conf_src, np.float32).reshape(-1)
    idx = np.flatnonzero(c > 1e-5)
    if len(idx) < 500:
        return None
    if len(idx) > max_samples:
        idx = np.random.default_rng(seed).choice(idx, max_samples, replace=False)
    p = p[idx]
    w2c = np.asarray(w2c_dst, np.float64)
    X = p @ w2c[:3, :3].T + w2c[:3, 3]
    z = X[:, 2]
    m = z > 0.3
    if m.sum() < 300:
        return None
    fx, fy, cx, cy = float(K_dst[0, 0]), float(K_dst[1, 1]), float(K_dst[0, 2]), float(K_dst[1, 2])
    u = np.round(X[m, 0] / z[m] * fx + cx).astype(int)
    v = np.round(X[m, 1] / z[m] * fy + cy).astype(int)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if inb.sum() < 300:
        return None
    u, v, p_in = u[inb], v[inb], p[m][inb]
    q = np.asarray(wp_dst, np.float64)[v, u]
    cq = np.asarray(conf_dst, np.float32).reshape(H, W)[v, u]
    good = cq > 1e-5
    if good.sum() < 300:
        return None
    return p_in[good], q[good]


def se3_matrices(xi):
    """(N,6) se(3) vectors (rotvec, t) → (N,4,4) rigid matrices."""
    xi = np.asarray(xi, np.float64).reshape(-1, 6)
    out = np.tile(np.eye(4), (len(xi), 1, 1))
    for i, x in enumerate(xi):
        R, _ = cv2.Rodrigues(x[:3].copy())
        out[i, :3, :3] = R
        out[i, :3, 3] = x[3:]
    return out


def filter_pair_fits(fits, min_points=800, res_factor=3.0):
    """Pair-fit hygiene: drop occlusion-contaminated fits — residual beyond
    ``res_factor``× the SESSION's own median residual, or too few inliers —
    and weight the survivors by inverse residual (session-derived, nothing
    absolute). fits: [(f, g, R, t, res_m, n_used)]. Returns
    [(f, g, tau[6], w)] ready for the solver."""
    good = [(f, g, R, t, res, n) for f, g, R, t, res, n in fits
            if n >= int(min_points) and np.isfinite(res)]
    if not good:
        return []
    med = max(float(np.median([res for *_, res, _n in good])), 1e-4)
    out = []
    for f, g, R, t, res, n in good:
        if res > res_factor * med:
            continue
        rvec, _ = cv2.Rodrigues(np.asarray(R, np.float64))
        tau = np.concatenate([rvec.ravel(), np.asarray(t, np.float64).reshape(3)])
        out.append((f, g, tau, med / max(res, 0.25 * med)))
    return out


def solve_chunk_field(pair_taus, S, smooth_w=0.5, prior_w=0.3):
    """Per-frame rigid corrections for ONE chunk, endpoints CLAMPED to zero.

    Constraints (local frame indices, 0..S-1):

        xi_f - xi_g = tau_fg     (within-chunk exact-surface pair residuals)
        xi_{f+1} - xi_f = 0      (smoothness, weight smooth_w)
        xi_f = 0                 (zero-prior, weight prior_w — damping)
        xi_0 = xi_{S-1} = 0      (HARD: solved-out, not weighted — no field
                                  longer than the chunk can exist)

    Six independent scalar problems, one lstsq. Returns xi (S,6); zeros when
    starved."""
    xi = np.zeros((S, 6))
    taus = [(int(f), int(g), np.asarray(t, np.float64).reshape(6), float(w))
            for f, g, t, w in (pair_taus or [])
            if 0 <= f < S and 0 <= g < S and np.all(np.isfinite(t))]
    if not taus or S < 3:
        return xi
    free = list(range(1, S - 1))               # endpoints clamped out
    col = {f: i for i, f in enumerate(free)}
    n = len(free)
    rows, rhs = [], []
    for f, g, t, w in taus:
        r = np.zeros(n)
        if f in col:
            r[col[f]] += w
        if g in col:
            r[col[g]] -= w
        if not r.any():
            continue
        rows.append(r)
        rhs.append(w * t)
    for i in range(S - 1):                     # smoothness incl. clamped ends
        r = np.zeros(n)
        if i in col:
            r[col[i]] -= smooth_w
        if i + 1 in col:
            r[col[i + 1]] += smooth_w
        if r.any():
            rows.append(r)
            rhs.append(np.zeros(6))
    for i in free:                             # zero-prior damping
        r = np.zeros(n)
        r[col[i]] = prior_w
        rows.append(r)
        rhs.append(np.zeros(6))
    if not rows:
        return xi
    x, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    for f, i in col.items():
        xi[f] = x[i]
    return xi


def blend_chunk_fields(chunk_indices, fields, n_frames):
    """Per-GLOBAL-frame field from the per-chunk solutions: where two chunks
    share a frame their fields are blended with the elastic's triangular
    weights (1 at a chunk's own centre → 0 at its edges). Every chunk field is
    zero at its endpoints, so the blend is continuous and both copies of every
    shared frame receive ONE identical correction — the seam consensus is
    preserved exactly. Returns xi (n_frames, 6)."""
    num = np.zeros((n_frames, 6))
    den = np.zeros(n_frames)
    for k, (start, end) in enumerate(chunk_indices):
        S = end - start
        fld = np.asarray(fields.get(k)) if fields.get(k) is not None else None
        if fld is None or S < 2:
            continue
        w = 1.0 - np.abs(np.linspace(-1.0, 1.0, S))
        w = np.maximum(w, 1e-6)
        for local in range(S):
            g = start + local
            num[g] += w[local] * fld[local]
            den[g] += w[local]
    out = np.zeros((n_frames, 6))
    m = den > 0
    out[m] = num[m] / den[m, None]
    return out


def chunk_field_verdict(xi, pair_taus, held, improve=0.8, bound=5.0):
    """Per-chunk self-gate (the family discipline): held-out within-chunk pairs
    judge, corrections bounded by 5× the P90 pair magnitude — a smooth bump's
    pointwise correction legitimately exceeds the MEDIAN pairwise difference,
    while p99 proved inflatable by occlusion outliers (run 4); p90 sits
    between, and filter_pair_fits has already capped the tail.
    held: [(f, g, p[n,3], q[n,3])] with LOCAL indices. Returns dict
    bounded/improves/med_before/med_after."""
    xi = np.asarray(xi, np.float64)
    X = se3_matrices(xi)
    before, after = [], []
    for f, g, p, q in held:
        p = np.asarray(p, np.float64)
        q = np.asarray(q, np.float64)
        before.append(float(np.median(np.linalg.norm(p - q, axis=1))))
        p2 = p @ X[f][:3, :3].T + X[f][:3, 3]
        q2 = q @ X[g][:3, :3].T + X[g][:3, 3]
        after.append(float(np.median(np.linalg.norm(p2 - q2, axis=1))))
    med_b, med_a = float(np.median(before)), float(np.median(after))
    t_pair = [float(np.linalg.norm(np.asarray(t, np.float64).reshape(6)[3:]))
              for _, _, t, _ in pair_taus]
    r_pair = [float(np.linalg.norm(np.asarray(t, np.float64).reshape(6)[:3]))
              for _, _, t, _ in pair_taus]
    t_sig = max(float(np.percentile(t_pair, 90)), 1e-3)
    r_sig = max(float(np.percentile(r_pair, 90)), 1e-4)
    bounded = (float(np.max(np.linalg.norm(xi[:, 3:], axis=1))) <= bound * t_sig
               and float(np.max(np.linalg.norm(xi[:, :3], axis=1))) <= bound * r_sig)
    improves = med_a <= improve * med_b
    return {"bounded": bounded, "improves": improves,
            "med_before": med_b, "med_after": med_a,
            "t_sig": t_sig, "r_sig": r_sig}
