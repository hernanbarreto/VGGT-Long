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


def elastic_corrections(chunk_indices, k, seam_fits, sick=None):
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
    """
    start, end = chunk_indices[k]
    S = end - start
    corr = np.tile(np.eye(4), (S, 1, 1))
    sick = frozenset(sick or ())

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
        return 1.0 - (i / (L - 1.0)) if L > 1 else 0.5

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


def solve_depth_graph(measurements, n_frames, sick_frames=()):
    """Per-frame depth corrections z' = a_f*z + b_f from pairwise affine relations,
    the frame-level analogue of solve_scale_graph. For a pair (f, g) with measured
    z_g = alpha*z_f + beta, corrected consistency (a_f z + b_f == a_g(alpha z +
    beta) + b_g for all z) splits into two LINEAR systems solved in sequence:

        log a_f - log a_g = log alpha        (scale graph, gauge: mean log a = 0)
        b_f - b_g         = a_g * beta       (offset graph, gauge: mean b = 0)

    The gauges preserve the session's global metre (set by the metric lock +
    scale_align) — the graph only REDISTRIBUTES depth so every frame agrees on
    every shared surface. Sick frames get identity (a=1, b=0) and their
    measurements must not be fed in. Returns (a[n], b[n])."""
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
