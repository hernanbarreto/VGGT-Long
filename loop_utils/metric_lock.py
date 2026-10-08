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


def seam_scale_error(depth_a, depth_b, min_px=1000):
    """The RESOLUTION of seam_relative_scale on one shared frame (log units): the frame's
    valid pixels split in two halves by their stable key (loop_utils.stable_sample — a pixel
    entering or leaving does not reshuffle the others), the log of each half's ratio; two
    independent half-estimates differ by twice the error of the whole, so the error is
    |log r_A - log r_B| / 2. None when either half is under ``min_px`` / 2 pixels. The
    measured error of a seam observation in the user's rule (plan point 18)."""
    from loop_utils.stable_sample import pixel_keys, stable_half
    da = np.asarray(depth_a, np.float32).squeeze()
    db = np.asarray(depth_b, np.float32).squeeze()
    if da.shape != db.shape:
        return None
    m = (np.isfinite(da) & np.isfinite(db) & (da > 1e-6) & (db > 1e-6)).reshape(-1)
    idx = np.flatnonzero(m)
    if len(idx) < min_px:
        return None
    in_a = stable_half(pixel_keys(0, idx))
    ra, rb = da.reshape(-1)[idx], db.reshape(-1)[idx]
    h = []
    for sel in (in_a, ~in_a):
        if int(sel.sum()) < min_px // 2:
            return None
        h.append(float(np.log(np.median(rb[sel] / ra[sel]))))
    return float(abs(h[0] - h[1]) / 2.0)


def fx_frame_error(fx):
    """The measured ERROR of a chunk's focal (px): the spread of its per-frame focals —
    what one frame's focal estimate scatters by inside the chunk (robust: 1.4826 x the MAD
    about their median, the normal-consistency constant of the MAD — mathematics, not a
    decision). It is the instrument's error on fx, the analogue of a bridge's σ in the pose
    graph's rule (point 1): a chunk whose focal departs from the session by less than a few
    times what a single frame scatters by has not zoomed. None with fewer than two frames
    (one focal says nothing about its own scatter). The error of the zoom rule (plan point
    18)."""
    fx = np.asarray(fx, np.float64).ravel()
    if fx.size < 2:
        return None
    if not np.all(np.isfinite(fx)):
        raise ValueError("fx_frame_error: non-finite focal values")
    return float(1.4826 * np.median(np.abs(fx - np.median(fx))))


def inference_focal_error(fx_frames, chunk_indices):
    """Omega's error on a chunk's focal BETWEEN inferences (px), MEASURED on the frames two
    chunks share: the same frame — the same lens — gets a focal from each chunk's inference, so
    the offset between the two copies is pure estimation error, never a zoom. Per seam the median
    offset of its shared frames; the error of ONE inference = RMS of those offsets / √2 (each
    offset holds two inferences' errors). pccr 2026-10-07: offsets +1.4, +11.8, −1.9, +21.8 px →
    8.8 px, while each chunk's own frame scatter (the old error) read 2–5 px, so Omega's noise
    passed for a zoom on 4 of 5 chunks and the scale verification failed by 18.6 %. None with no
    shared frame."""
    offs = []
    ranges = [(int(a), int(b)) for a, b in chunk_indices]
    for k in range(len(ranges) - 1):
        if k not in fx_frames or (k + 1) not in fx_frames:
            continue
        a, b = ranges[k + 1][0], ranges[k][1]
        if b <= a:
            continue
        fk = np.asarray(fx_frames[k], np.float64).ravel()[a - ranges[k][0]: b - ranges[k][0]]
        fk1 = np.asarray(fx_frames[k + 1], np.float64).ravel()[0: b - a]
        n = min(len(fk), len(fk1))
        if n:
            offs.append(float(np.median(fk[:n] - fk1[:n])))
    if not offs:
        return None
    return float(np.sqrt(np.mean(np.square(offs)) / 2.0))


def zoom_anchor_test(fx_chunk, fx_rest, *, error_factor, confidence, n_boot=2000, seed=0,
                     inference_error_px=None):
    """USER 2026-10-07 (plan point 18): a chunk's DA3 anchors are excluded as ZOOMED only when
    its focal differs from the rest of the session SIGNIFICANTLY — the whole ``confidence``
    bootstrap interval of median(fx of the chunk's frames) - median(fx of every other chunk's
    frames) on one side of zero (both samples resampled, seeded) — AND by at least
    ``error_factor`` (the user's 2) x the chunk's own measured fx error (fx_frame_error: the
    scatter of its per-frame focals). No robust-z cut
    (the 3.5 was invented; its MAD came from five numbers). Returns a JSON-able dict: the
    verdict ``zoomed`` and every margin."""
    a = np.asarray(fx_chunk, np.float64).ravel()
    b = np.asarray(fx_rest, np.float64).ravel()
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("zoom_anchor_test: non-finite focal values")
    conf = float(confidence)
    fac = float(error_factor)
    if not (0.0 < conf < 1.0):
        raise ValueError(f"zoom_anchor_test: confidence {confidence!r} must lie in (0, 1)")
    if not (np.isfinite(fac) and fac > 0.0):
        raise ValueError(f"zoom_anchor_test: error_factor {error_factor!r} must be > 0")
    out = {"rule": "USER 2026-10-07: significant AND |median diff| >= factor x the chunk's fx error",
           "n_chunk_frames": int(a.size), "n_rest_frames": int(b.size), "confidence": conf,
           "error_factor": fac, "n_boot": int(n_boot), "seed": int(seed),
           "error_source": "the scatter of the chunk's per-frame focals (1.4826 x MAD)"}
    err = fx_frame_error(a)
    if b.size < 2 or err is None:
        out.update({"zoomed": False, "fx_error_px": err,
                    "reason": "too few frames to measure the chunk's focal, its error or the "
                              "session's — no exclusion"})
        return out
    # the chunk's focal is ONE inference: its error is the larger of its frames' scatter and
    # Omega's measured error between inferences (inference_focal_error) — USER 2026-10-07
    out["fx_frame_scatter_px"] = float(err)
    out["inference_error_px"] = (float(inference_error_px) if inference_error_px is not None else None)
    if inference_error_px is not None and np.isfinite(inference_error_px):
        err = max(float(err), float(inference_error_px))
        out["error_source"] = ("max(the scatter of the chunk's per-frame focals, Omega's focal error "
                               "between inferences measured on the shared frames)")
    diff = float(np.median(a) - np.median(b))
    rng = np.random.default_rng(int(seed))
    bd = np.empty(int(n_boot), np.float64)
    for q in range(int(n_boot)):
        bd[q] = (np.median(a[rng.integers(0, a.size, a.size)])
                 - np.median(b[rng.integers(0, b.size, b.size)]))
    alpha = (1.0 - conf) / 2.0
    lo = float(np.percentile(bd, 100.0 * alpha))
    hi = float(np.percentile(bd, 100.0 * (1.0 - alpha)))
    significant = bool(lo > 0.0 or hi < 0.0)
    required = fac * float(err)
    beyond = bool(abs(diff) >= required)
    out.update({"fx_chunk_median": float(np.median(a)), "fx_rest_median": float(np.median(b)),
                "diff_px": diff, "ci_low": lo, "ci_high": hi, "significant": significant,
                "fx_error_px": float(err), "required_px": float(required),
                "error_margin_px": float(abs(diff) - required), "beyond_error": beyond,
                "zoomed": bool(significant and beyond)})
    return out


def solve_scale_graph(s_da3, n_anchors, seam_rel, n_chunks,
                      sigma_seam=0.003, sigma_anchor=0.08,
                      loop_rel=None, absolute=None, skip_loop=None):
    """Optimal per-chunk metric scales from EVERY sensor, weighted least squares
    in log-scale space — a GRAPH, no longer a chain (claude_stac.txt §5).

    The overlap frames measure the RELATIVE scale between neighbours to ~0.1-1%
    (same pixels, a ratio); the DA3 anchors measure each chunk's ABSOLUTE scale
    with ±8-15% monocular noise. Locking chunks to their own noisy DA3 median
    left neighbours disagreeing 5-27% — decimetres-to-metres of seam decoupling
    an SE(3) glue can never fix (measured, test4). Fusing all of them:

        minimize  Σ_seams   [(x_{k+1} - x_k) - log r_k]²      / σ_seam²
                + Σ_anchors [x_k - log s_k^DA3]²              / (σ_anchor/√n_k)²
                + Σ_loops   [(x_j - x_i) - log r_ij]²         / σ_ij²      (§5.1)
                + Σ_abs     [x_k - log s_k^src]²              / σ_src²     (§5.2)

    x_k = log s_k. Linear, tiny (one variable per chunk). Seams make neighbours
    CONSISTENT; loop rows CLOSE the walk (without them the scale random-walks
    along the chain, P4); absolute rows (VIO, regulated dimensions, a user
    measurement) pin the metre with their own σ — the most precise source wins,
    DA3 becomes the cross-check.

    s_da3: {k: s} absolute estimates; n_anchors: {k: count}; seam_rel: {k: r}
    with r = s_{k+1}/s_k (from seam_relative_scale).
    loop_rel: {(i, j): (log_r, sigma)} with log_r = log(s_j/s_i) measured by an
    exact loop bridge (loop_bridges.loop_scale_row); skip_loop: a key of
    loop_rel to leave out (scale-break diagnosis, §5.3).
    absolute: iterable of (k, log_s, sigma[, source]) extra absolute rows.
    Returns np.ndarray scales (len n_chunks).
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
    for key, val in (loop_rel or {}).items():
        if skip_loop is not None and key == skip_loop:
            continue
        i, j = int(key[0]), int(key[1])
        log_r, sig = float(val[0]), float(val[1])
        if i == j or not np.isfinite(log_r) or not (sig > 0):
            continue
        row = np.zeros(n_chunks)
        row[i], row[j] = -1.0, 1.0
        rows.append(row); rhs.append(log_r); w.append(1.0 / sig)
    for rec in (absolute or ()):
        k, log_s, sig = int(rec[0]), float(rec[1]), float(rec[2])
        if not np.isfinite(log_s) or not (sig > 0):
            continue
        row = np.zeros(n_chunks)
        row[k] = 1.0
        rows.append(row); rhs.append(log_s); w.append(1.0 / sig)
    if not rows:
        raise ValueError("scale graph has no constraints")
    A = np.asarray(rows) * np.asarray(w)[:, None]
    b = np.asarray(rhs) * np.asarray(w)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.exp(x)


def seam_residuals(scales, seam_rel):
    """Per-seam |log| residual of a scale solution against the measured seam
    ratios: {k: |(x_{k+1} - x_k) - log r_k|}."""
    x = np.log(np.asarray(scales, np.float64))
    out = {}
    for k, r in seam_rel.items():
        if r is None or not np.isfinite(r) or r <= 0 or k + 1 >= len(x):
            continue
        out[int(k)] = float(abs((x[k + 1] - x[k]) - np.log(r)))
    return out


def scale_break_diagnosis(s_da3, n_anchors, seam_rel, n_chunks, loop_rel,
                          break_key, sigma_seam, sigma_anchor, absolute=None,
                          localisation_gap=2.0):
    """§5.3: a loop whose measured relative scale contradicts the chain
    (scale_break) is NOT discarded — the graph is solved WITH and WITHOUT its
    row, and the seam where the scale jumped is searched for.

    Least squares spreads one bad seam over every seam the loop spans, so the
    residual pattern alone cannot name it; and a single cycle (seams + one
    loop row) cannot localise a jump by itself — dropping ANY seam of the
    cycle makes the rest consistent. The witnesses that break the tie are the
    ABSOLUTE rows (DA3 anchors, VIO, regulated dims): LEAVE-ONE-SEAM-OUT solves
    the graph WITH the loop row and without each spanned seam in turn; the
    seam whose absence lets every remaining measurement — seams, loop AND
    absolute rows — agree best is the suspect. The verdict is honest about
    its power: ``localised`` is False when the second-best candidate's cost is
    within ``localisation_gap`` (Δχ²) of the best, and the ranking is reported
    so the kit visual can show the ambiguity instead of a false certainty."""
    i, j = int(break_key[0]), int(break_key[1])
    lo, hi = min(i, j), max(i, j)
    with_row = solve_scale_graph(s_da3, n_anchors, seam_rel, n_chunks,
                                 sigma_seam=sigma_seam, sigma_anchor=sigma_anchor,
                                 loop_rel=loop_rel, absolute=absolute)
    without = solve_scale_graph(s_da3, n_anchors, seam_rel, n_chunks,
                                sigma_seam=sigma_seam, sigma_anchor=sigma_anchor,
                                loop_rel=loop_rel, absolute=absolute,
                                skip_loop=break_key)
    r_with = seam_residuals(with_row, seam_rel)
    r_without = seam_residuals(without, seam_rel)
    # Least squares spreads one bad seam over every seam the loop spans, so
    # the residual pattern alone cannot name it. LEAVE-ONE-SEAM-OUT: drop each
    # spanned seam in turn and solve WITH the loop row; the seam whose absence
    # lets every remaining measurement agree (minimum weighted cost of the
    # rest) is the one that contradicted the loop — the scale jump.
    loop_log, loop_sig = float(loop_rel[break_key][0]), float(loop_rel[break_key][1])
    cands = [k for k in seam_rel if lo <= k < hi and seam_rel[k] is not None
             and np.isfinite(seam_rel[k]) and seam_rel[k] > 0]
    if not cands:
        return {"loop": [i, j], "suspect_seam": None, "growth_log": None,
                "jump_pct": None, "note": "the loop spans no measured seam"}
    ranking = []
    for k in cands:
        seams_k = {kk: v for kk, v in seam_rel.items() if kk != k}
        x = solve_scale_graph(s_da3, n_anchors, seams_k, n_chunks,
                              sigma_seam=sigma_seam, sigma_anchor=sigma_anchor,
                              loop_rel=loop_rel, absolute=absolute)
        lx = np.log(x)
        cost = 0.0
        for kk, v in seams_k.items():
            if v is None or not np.isfinite(v) or v <= 0:
                continue
            cost += ((lx[kk + 1] - lx[kk]) - np.log(v)) ** 2 / sigma_seam ** 2
        cost += ((lx[j] - lx[i]) - loop_log) ** 2 / loop_sig ** 2
        for kk, s in s_da3.items():
            if s is None or not np.isfinite(s) or s <= 0:
                continue
            sig = sigma_anchor / max(np.sqrt(float(n_anchors.get(kk, 1))), 1.0)
            cost += (lx[kk] - np.log(s)) ** 2 / sig ** 2
        for rec in (absolute or ()):
            kk, log_s, sig = int(rec[0]), float(rec[1]), float(rec[2])
            if np.isfinite(log_s) and sig > 0:
                cost += (lx[kk] - log_s) ** 2 / sig ** 2
        jump = float(abs((lx[k + 1] - lx[k]) - np.log(seam_rel[k])))
        ranking.append((float(cost), int(k), jump))
    ranking.sort()
    best_cost, best_k, best_jump = ranking[0]
    localised = len(ranking) == 1 or (ranking[1][0] - best_cost) >= float(localisation_gap)
    return {"loop": [i, j], "suspect_seam": int(best_k),
            "localised": bool(localised),
            "growth_log": float(r_with.get(best_k, 0.0) - r_without.get(best_k, 0.0)),
            "jump_pct": float((np.exp(best_jump) - 1.0) * 100.0),
            "leave_one_out_cost": best_cost,
            "ranking": [{"seam": k_, "cost": c_, "jump_pct": float((np.exp(j_) - 1.0) * 100.0)}
                        for c_, k_, j_ in ranking],
            "localisation_gap": float(localisation_gap),
            "residual_with_row_log": r_with, "residual_without_row_log": r_without}


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


def scale_drift_gate(anchors, s_const, seam_obs, n_chunks, *, seam_frames, seam_err,
                     error_factor, confidence, max_drift_log=None, **solve_kw):
    """Self-validation for the drift model — it must EARN the right to touch
    the geometry (same discipline as the depth graph).

    The judge is the PRECISE sensor: the per-frame seam ratios (same pixels,
    0.3-1% noise — the drift signature is their variation ALONG the overlap;
    DA3 anchors at 8-15% noise cannot discriminate a model this fine). Every
    shared frame whose GLOBAL index is 2 mod 3 is HELD OUT (a stable choice: a
    frame whose ratio starves does not shift which others are held out — plan
    point 11); both the drift model and a CONSTANT reference are fitted on the
    SAME remaining data — the constant one is the identical solver with the
    drift DOF pinned, so the comparison leaks nothing and favours nobody.

    THE USER'S RULE (2026-10-07, plan point 18 — "fuera el 0,75"): the drift
    applies only when metric_lock.decide_change says so on the held-out
    |log| errors, constant (before) vs drift (after): significant at
    ``confidence``, at least min_judge_closures(confidence) held-out frames,
    and a median improvement >= ``error_factor`` x the largest measured error
    of those held-out ratios (``seam_err``: seam_scale_error per frame) — AND
    every chunk's |log(s1/s0)| <= max_drift_log (default log 1.6, the bound).

    ``seam_frames``: {k: [global frame of each observation]}, ``seam_err``:
    {k: [its measured error or None]} — parallel to ``seam_obs``. Returns
    (verdict_bool, info dict). The caller refits on ALL data when True."""
    if max_drift_log is None:
        max_drift_log = float(np.log(1.6))
    fit_seams, held = {}, []
    for k, obs in (seam_obs or {}).items():
        frames = list((seam_frames or {}).get(k) or [])
        errs = list((seam_err or {}).get(k) or [])
        if len(frames) != len(obs) or len(errs) != len(obs):
            raise ValueError(f"scale_drift_gate: seam {k} has {len(obs)} observation(s), "
                             f"{len(frames)} frame(s), {len(errs)} error(s)")
        keep = []
        for (u_k, u_k1, r), g, e in zip(obs, frames, errs):
            if int(g) % 3 == 2:
                held.append((k, u_k, u_k1, r, e))
            else:
                keep.append((u_k, u_k1, r))
        fit_seams[k] = keep
    # a held-out frame whose ratio has no measurable error cannot testify (counted)
    n_no_err = sum(1 for h in held if h[4] is None or not np.isfinite(h[4]))
    held = [h for h in held if h[4] is not None and np.isfinite(h[4])]
    info = {"n_holdout": len(held), "n_holdout_without_error": int(n_no_err),
            "bound_log": max_drift_log}
    if not held:
        info.update({"reason": "no held-out seam observation with a measured error",
                     "decision": None})
        return False, info
    s0, s1 = solve_scale_drift(anchors, fit_seams, n_chunks,
                               prior=s_const, **solve_kw)
    kw_const = dict(solve_kw, sigma_drift=1e-6)      # same solver, drift pinned
    c0, c1 = solve_scale_drift(anchors, fit_seams, n_chunks,
                               prior=s_const, **kw_const)

    def _pred(a0, a1, k, u):
        return (1.0 - u) * np.log(a0[k]) + u * np.log(a1[k])

    err_d, err_c = [], []
    for k, u_k, u_k1, r, _e in held:
        err_d.append(abs(np.log(r) - (_pred(s0, s1, k + 1, u_k1)
                                      - _pred(s0, s1, k, u_k))))
        err_c.append(abs(np.log(r) - (_pred(c0, c1, k + 1, u_k1)
                                      - _pred(c0, c1, k, u_k))))
    med_d, med_c = float(np.median(err_d)), float(np.median(err_c))
    drift_mag = float(np.max(np.abs(np.log(s1 / s0))))
    error = float(max(h[4] for h in held))
    decision = decide_change(err_c, err_d, error=error, error_factor=error_factor,
                             confidence=confidence)
    decision["error_source"] = "the largest measured error of the held-out seam ratios"
    bounded = drift_mag <= max_drift_log
    ok = bool(decision["improves"]) and bounded
    info.update({"holdout_const": med_c, "holdout_drift": med_d,
                 "max_drift_log": drift_mag, "bounded": bool(bounded),
                 "decision": decision,
                 "reason": decision["reason"] + ("" if bounded else
                                                 f"; drift {drift_mag:.3f} beyond the bound "
                                                 f"{max_drift_log:.3f}")})
    return ok, info


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


def robust_rigid(src, dst, iters=8, sample=200000, seed=0, keys=None):
    """Rigid fit dst ≈ R·src + t from EXACT correspondences (same pixel, same
    frame, two chunks), IRLS with a Cauchy weight on the residuals so the
    chunk-internal disagreement (non-rigid noise + far-field junk) does not
    drag the fit. Returns (R, t, median_residual_m, n_used) or None.

    Past ``sample`` correspondences the fit uses a subsample chosen by a STABLE
    KEY of each correspondence (plan point 11, loop_utils.stable_sample):
    ``keys`` — one uint64 per row, the pixel identity (stable_sample.pixel_keys)
    when the caller has it — else the bits of the row itself; ``seed`` salts
    the choice. One correspondence entering or leaving no longer re-draws the
    others."""
    from loop_utils.stable_sample import row_keys, stable_pick
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    m = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    if keys is not None:
        keys = np.asarray(keys).ravel()
        if len(keys) != len(m):
            raise ValueError(f"robust_rigid: {len(keys)} keys for {len(m)} correspondences")
        keys = keys[m]
    src, dst = src[m], dst[m]
    if len(src) < 1000:
        return None
    if len(src) > sample:
        idx = stable_pick(row_keys(src, dst) if keys is None else keys, sample, salt=seed)
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


def backfill_mask(conf_owner, thr_owner):
    """OWNERSHIP BACKFILL: pixels of a shared frame that the OWNER chunk will NOT
    write — below its frozen write threshold, or invalid (sky/masked). The
    non-owner copy may write exactly these pixels: the overlap redundancy that
    frame ownership discarded pays for HOLES instead of re-creating duplicates
    (a pixel is never written twice: owner-writes and backfill are complements
    by construction). ``thr_owner=None`` means the owner writes nothing at all
    (it owns no frame rows) — everything may be backfilled."""
    c = np.asarray(conf_owner, np.float32).reshape(-1)
    if thr_owner is None:
        return np.ones(c.shape, bool)
    return ~((c >= max(float(thr_owner), 0.0)) & (c > 1e-5))


def demote_disproportionate_fits(seam_fits, report, log=print):
    """Scale down the fits that move the points FURTHER than the disagreement
    they remove. No absolute ceiling — the allowance is measured per frame.

    USER 2026-09-16: *"podría haber una corrección de más de 30 cm y ser
    perfectamente correcta"*. True: what made the old `elastic_max_t_m` cap
    necessary was never the size. It was disproportion — the pathology its own
    comment recorded, "seam 7->8 fitted up to 1.59 m to close an 8 cm gap": a
    degenerate pair on a flat cost surface, where the solver wanders far and
    explains nothing. A correction of 50 cm that removes 48 cm of disagreement
    is right, and the cap used to shrink it for no reason.

    The two copies of a shared frame are the SAME frame, so a fit that puts one
    onto the other moves the points by about how far apart they were. The
    allowance for each frame is therefore its OWN measurement:

        allowed = before_m + residual_m          (the gap, plus this seam's
                                                  own non-rigid noise floor)

    A fit whose median point motion exceeds it is scaled as a whole twist down
    to that allowance — direction preserved, authority bounded by what the data
    shows. Frames whose report lacks the measurement (an older
    elastic_seams.json) are left exactly as they are.

    Returns (fits, n_demoted, stats).
    """
    seams = (report or {}).get("seams") or {}
    out, n_dem = {}, 0
    worst = {"ratio": 0.0, "seam": None}
    allowances = []
    for j, d in seam_fits.items():
        rep_j = seams.get(str(j)) or seams.get(j) or {}
        sd = {}
        for g, (R_, t_) in d.items():
            entry = rep_j.get(str(g)) or rep_j.get(g) or {}
            motion = entry.get("motion_m")
            before = entry.get("before_m")
            resid = entry.get("residual_m")
            if motion is None or before is None or resid is None:
                sd[g] = (R_, t_)
                continue
            allowed = float(before) + float(resid)
            allowances.append(allowed)
            motion = float(motion)
            if allowed > 0 and motion > allowed:
                ratio = motion / allowed
                M = rigid_fraction(np.asarray(R_, np.float64),
                                   np.asarray(t_, np.float64), allowed / motion)
                sd[g] = (M[:3, :3], M[:3, 3])
                n_dem += 1
                if ratio > worst["ratio"]:
                    worst = {"ratio": float(ratio), "seam": int(j)}
            else:
                sd[g] = (R_, t_)
        out[j] = sd
    stats = {"worst_ratio": worst["ratio"], "worst_seam": worst["seam"],
             "allowance_max_m": float(max(allowances)) if allowances else None,
             "allowance_median_m": float(np.median(allowances)) if allowances else None}
    return out, n_dem, stats


def compose_frame_fields(prev, new):
    """Compose a new per-frame correction field ON TOP of whatever a previous
    stage already put there: result[k][i] = new[k][i] @ prev[k][i].

    Why this exists (bug found on pccr 2026-09-16, the first run where a loop
    actually closed): the per-frame field `_stac_elastic_corr` is not the
    elastic stage's private variable — it is where EVERY stage that moves
    frames accumulates its correction, and it is what the camera poses are
    built from (`_stac_aligned_pose`, `save_camera_poses`). The pose graph
    composes its closure into it and moves the POINTS in the npy directly; the
    elastic stage then rebuilt the field from scratch and the closure vanished
    from the camera side while staying in the points.

    The damage was silent and large: points and cameras drifted apart by the
    size of the closure (29 cm median, 38 cm max on that run), so `intra_chunk`
    — which compares one against the other — measured a disagreement that did
    not exist (held-out 2-5 cm historically → 8-53 cm), "repaired" it with
    corrections up to 129 cm, and the warped depth failed the global scale
    verification by 18%. It never showed before because every previous run
    rejected its loop edges: the closure was identity, and overwriting identity
    loses nothing.
    """
    prev = prev or {}
    out = {}
    for k, cur in new.items():
        cur = np.asarray(cur, np.float64)
        before = prev.get(k)
        out[k] = cur if before is None else (cur @ np.asarray(before, np.float64))
    for k, before in prev.items():          # a chunk the new field does not mention
        out.setdefault(k, np.asarray(before, np.float64))
    return out


def smooth_seam_fits(seam_fits, window=5, max_t=None):
    """Tame the per-frame elastic seam fits BEFORE they become corrections.

    Two measured pathologies of the raw fits (test4 2026-07-11): (1) adjacent
    shared frames get INDEPENDENT rigid fits with no along-trajectory coherence,
    so a continuous surface receives uncorrelated moves frame to frame; (2) a
    degenerate frame pair can fit a huge transform to close a small disagreement
    (seam 7->8: |t| median 63 cm, max 159 cm, to close an 8 cm gap).

    (1) moving-average smoothing over the fitted frames of each seam, in
        rotation-vector + translation space (seam residuals are small-angle, so
        averaging Rodrigues vectors is exact enough and dependency-free);
    (2) per-frame translation cap: a smoothed fit whose |t| still exceeds
        ``max_t`` is scaled DOWN as a whole twist (rigid_fraction), preserving
        the direction of the correction but bounding its authority — beyond the
        cap it is an upstream pose problem this stage must not fake away.

    Both sides of a seam consume the SAME smoothed fit, so the two-copy
    coincidence property of elastic_corrections is preserved exactly.

    Returns (new_fits, n_capped)."""
    if window <= 1 and not max_t:
        return seam_fits, 0
    out, n_capped = {}, 0
    half = max(int(window), 1) // 2
    for j, d in seam_fits.items():
        gs = sorted(d.keys())
        rvecs, ts = {}, {}
        for g in gs:
            R_, t_ = d[g]
            rv, _ = cv2.Rodrigues(np.asarray(R_, np.float64))
            rvecs[g] = rv.reshape(3)
            ts[g] = np.asarray(t_, np.float64).reshape(3)
        sd = {}
        for i, g in enumerate(gs):
            nb = gs[max(0, i - half):i + half + 1]
            rv = np.mean([rvecs[gg] for gg in nb], axis=0)
            t = np.mean([ts[gg] for gg in nb], axis=0)
            R_, _ = cv2.Rodrigues(rv)
            tn = float(np.linalg.norm(t))
            if max_t and tn > float(max_t):
                M = rigid_fraction(R_, t, float(max_t) / tn)
                R_, t = M[:3, :3], M[:3, 3]
                n_capped += 1
            sd[g] = (R_, t)
        out[j] = sd
    return out, n_capped


def chunk_trust(anchor_iqr):
    """Per-chunk trust in (0, 1] from the MEASURED anchor spread — continuous,
    with no cut-off anywhere.

    USER 2026-09-16, on `[health] chunk 1 SUSPECT: IQR/median 0.311 > 0.30`:
    a chunk does not become unreliable at 0.30 and stay perfect at 0.299. What
    the number says is RELATIVE — this chunk's anchors agree worse than the
    session's other chunks — so the reference is the session itself (the median
    spread) and the result is a weight, not a label:

        trust_k = 1 / (1 + spread_k / median_spread)

    All chunks equally spread → every trust is 0.5 → no chunk is favoured. A
    chunk twice as spread as its peers → 1/3 against 1/2: less say, still heard.
    A chunk with no measurement keeps the median trust (it is not evidence).
    """
    vals = {int(k): float(v) for k, v in (anchor_iqr or {}).items()
            if v is not None and np.isfinite(v) and v >= 0}
    if not vals:
        return {}
    ref = float(np.median(list(vals.values()))) or 1e-6
    trust = {k: 1.0 / (1.0 + (v / ref)) for k, v in vals.items()}
    med_trust = float(np.median(list(trust.values())))
    for k in (anchor_iqr or {}):
        trust.setdefault(int(k), med_trust)
    return trust


def elastic_corrections(chunk_indices, k, seam_fits, suspect=None, trust=None):
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
    switch in the middle of every overlap. (With the pipeline's 50% overlap — and
    on any explicit layout, by validate_chunk_ranges rule 5 — each frame belongs
    to at most two chunks, so the two seams of a chunk touch disjoint frame
    ranges.)

    A shared frame whose fit starved inherits the nearest fitted frame of the same
    seam (the seam is already rigid-glued, so per-frame residuals are small and
    smooth); a seam with no fits at all contributes identity.

    `trust` (preferred): per-chunk weight from `chunk_trust`, measured from the
    anchor spread. The consensus is biased CONTINUOUSLY toward the chunk whose
    anchors agree better — alpha is warped by the exponent p = trust_b/trust_a,
    which is exactly 1 (no warp) when the two sides are equally trustworthy and
    grows smoothly as they diverge. Endpoints stay exact (1→0), so interiors are
    never torn and the field stays continuous.

    `suspect` (legacy): the binary SOFT tier. Used only when no `trust` is
    given — it warped alpha quadratically the moment a chunk crossed a fixed
    spread cut (USER 2026-09-16: a chunk is not sound at 0.299 and shaky at
    0.311).
    """
    start, end = chunk_indices[k]
    S = end - start
    corr = np.tile(np.eye(4), (S, 1, 1))
    suspect = frozenset(suspect or ())

    def _fit_for(j, g, shared):
        d = seam_fits.get(j) or {}
        if g in d:
            return d[g]
        fitted = [gg for gg in shared if gg in d]
        if not fitted:
            return None
        return d[min(fitted, key=lambda gg: abs(gg - g))]

    def _alpha(j, i, L):
        # weight of chunk j's (dst-side) opinion, linear inside the overlap
        a = 1.0 - (i / (L - 1.0)) if L > 1 else 0.5
        if trust:
            ta, tb = float(trust.get(j, 0.0)), float(trust.get(j + 1, 0.0))
            if ta > 0 and tb > 0:
                p = tb / ta                      # 1 when equally trusted
                if p != 1.0:
                    a = a ** p if p > 1 else 1.0 - (1.0 - a) ** (1.0 / p)
            return a
        # legacy binary tier (no measured trust available)
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
    association the seam work uses, generalized to any nearby frame pair.
    Past ``max_samples`` the source pixels are chosen by a stable per-pixel key
    in the namespace ``seed`` (the source frame's key — plan point 11)."""
    from loop_utils.stable_sample import pixel_keys, stable_pick
    H, W = wp_dst.shape[:2]
    p = np.asarray(wp_src, np.float64).reshape(-1, 3)
    c = np.asarray(conf_src, np.float32).reshape(-1)
    idx = np.flatnonzero(c > 1e-5)
    if len(idx) < 500:
        return None
    if len(idx) > max_samples:
        # stable per-pixel choice (plan point 11): ``seed`` is the frame key of the
        # source frame (the fork passes its global keyframe index)
        idx = idx[stable_pick(pixel_keys(int(seed), idx), max_samples)]
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


def pair_relation_error(z_src, z_dst, zref, iters=6):
    """The RESOLUTION of one pair's depth relation at ``zref`` (relative units, the unit of
    the depth graph's held-out judge): the pair's samples split in two halves by a stable key
    of each sample (its own values — loop_utils.stable_sample.row_keys), the affine relation
    fitted on each half (pair_depth_relation), and |pred_A(zref) - pred_B(zref)| / (2 zref) —
    two independent half-estimates differ by twice the error of the whole. None when a half
    does not fit. The measured error of a held-out pair in the user's rule (plan point 19)."""
    from loop_utils.stable_sample import row_keys, stable_half
    zs = np.asarray(z_src, np.float64).ravel()
    zd = np.asarray(z_dst, np.float64).ravel()
    if zs.size != zd.size or zs.size == 0:
        return None
    in_a = stable_half(row_keys(zs, zd))
    preds = []
    for sel in (in_a, ~in_a):
        rel = pair_depth_relation(zs[sel], zd[sel], iters=iters)
        if rel is None:
            return None
        preds.append(rel[0] * float(zref) + rel[1])
    return float(abs(preds[0] - preds[1]) / (2.0 * float(zref)))


def solve_depth_graph(measurements, n_frames, scale_only=False, weights=None, weights_b=None):
    """Per-frame depth corrections z' = a_f*z + b_f from pairwise affine relations,
    the frame-level analogue of solve_scale_graph. For a pair (f, g) with measured
    z_g = alpha*z_f + beta, corrected consistency (a_f z + b_f == a_g(alpha z +
    beta) + b_g for all z) splits into two LINEAR systems solved in sequence:

        log a_f - log a_g = log alpha        (scale graph, gauge: mean log a = 0)
        b_f - b_g         = a_g * beta       (offset graph, gauge: mean b = 0)

    The gauges preserve the session's global metre (set by the metric lock +
    scale_align) — the graph only REDISTRIBUTES depth so every frame agrees on
    every shared surface. Returns (a[n], b[n]).

    ``scale_only``: skip the offset system entirely (b stays 0) — the fallback
    rung of the model ladder. After the metric lock + scale drift the residual
    inter-frame depth disagreement is mostly MULTIPLICATIVE (leftover scale
    error); the free offset is where an unbounded low-frequency warp hides
    (measured on test4: b ran to ±112 cm while the scale stayed near 1).

    ``weights``: optional per-measurement weights (1/σ, same length as
    ``measurements``) applied to both systems — the projection sensor's rows
    and the correspondence rows of claude_stac.txt §6.4 (triangulated
    tracks) carry very different precisions; unweighted, 200 biased pair
    rows outvoted 40 exact track rows and dragged clean frames (measured,
    F3 depth test). None = every row weighs 1 (the F1/F2 behaviour).
    ``weights_b``: weights of the OFFSET system (its residuals are metres,
    the scale system's are log units — one 1/σ does not fit both); defaults
    to ``weights``."""
    idx = [q for q, (f, g, al, be) in enumerate(measurements)
           if np.isfinite(al) and al > 0 and np.isfinite(be)]
    meas = [tuple(measurements[q])[:4] for q in idx]
    wts = (np.ones(len(meas)) if weights is None
           else np.asarray([float(weights[q]) for q in idx], np.float64))
    wts_b = (wts if weights_b is None
             else np.asarray([float(weights_b[q]) for q in idx], np.float64))
    a = np.ones(n_frames)
    b = np.zeros(n_frames)
    if not meas:
        return a, b
    frames = sorted({f for f, g, _, _ in meas} | {g for _, g, _, _ in meas})
    col = {f: i for i, f in enumerate(frames)}
    n = len(frames)
    rows, rhs = [], []
    for (f, g, al, _), wq in zip(meas, wts):
        r = np.zeros(n)
        r[col[f]], r[col[g]] = wq, -wq
        rows.append(r)
        rhs.append(wq * np.log(al))
    gauge = np.full(n, float(np.mean(wts)) * float(len(rows)) / n)   # strong: pins the mean exactly
    rows.append(gauge)
    rhs.append(0.0)
    x, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    for f, i in col.items():
        a[f] = float(np.exp(x[i]))
    if scale_only:
        return a, b
    rows, rhs = [], []
    for (f, g, _, be), wq in zip(meas, wts_b):
        r = np.zeros(n)
        r[col[f]], r[col[g]] = wq, -wq
        rows.append(r)
        rhs.append(wq * a[g] * be)
    rows.append(np.full(n, float(np.mean(wts_b)) * float(len(rows)) / n))
    rhs.append(0.0)
    y, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    for f, i in col.items():
        b[f] = float(y[i])
    return a, b


def heldout_change(before, after, confidence=0.95, n_boot=2000, seed=0):
    """Did the correction change the held-out disagreement by more than the
    sample's OWN noise?

    `before`/`after` are the PAIRED per-held-out-pair disagreements (same pair,
    same order; any unit, as long as both share it). The statistic is the median
    of d = before - after (positive = the correction helped) and its sampling
    noise is MEASURED by bootstrapping over the pairs, so the bar adapts to how
    many pairs there are and how much they disagree — instead of being a
    constant someone chose.

    improves / worsens = the whole confidence interval of median(d) sits on one
    side of zero. NEITHER means the change is inside the noise: the held-out
    sample cannot tell, which is not the same as "no change".

    USER 2026-09-23 ("me parece bien"), after pccr showed the two halves of the
    same evidence judged by two invented round numbers pointing opposite ways:
    0.8 rejected a measured 12 % improvement while 0.005 m accepted a measured
    10 % degradation.
    """
    b = np.asarray(before, np.float64).ravel()
    a = np.asarray(after, np.float64).ravel()
    if b.size == 0 or b.size != a.size:
        return {"improves": False, "worsens": False, "n": int(b.size),
                "median_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    d = b - a
    n = int(d.size)
    rng = np.random.default_rng(int(seed))
    meds = np.median(d[rng.integers(0, n, size=(int(n_boot), n))], axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    lo = float(np.percentile(meds, 100.0 * alpha))
    hi = float(np.percentile(meds, 100.0 * (1.0 - alpha)))
    return {"improves": bool(lo > 0.0), "worsens": bool(hi < 0.0), "n": n,
            "median_delta": float(np.median(d)), "ci_low": lo, "ci_high": hi}


# ── THE USER'S RULE for every "apply this correction / model?" decision ─────────
# USER 2026-10-07 (docs/plan_determinismo.md, point 1, binding for points 1, 2, 18, 19, 30,
# 46, 48): a correction is applied only when THREE things hold at once —
#   (a) it is SIGNIFICANT: the whole bootstrap confidence interval of the paired change lies on
#       the improving side (fixed seed: the same sample gives the same interval);
#   (b) enough JUDGES testify: n >= min_judge_closures(confidence) (5 at 0.95 — the fewest
#       independent judges that can all improve by chance with probability below 1 - confidence);
#   (c) it is LARGER THAN THE ERROR: the median paired improvement >= error_factor x the measured
#       error of what is being judged (the factor 2 is the user's, read from
#       correction_graph.graph.improvement_error_factor — never a literal here).
# heldout_change() above stays for the reports that only describe a change.

# memory bound of one bootstrap block (elements of the resample matrix): changes neither the draws
# (numpy's Generator.integers with int64 consumes the stream value by value, so blocks of rows
# draw exactly what one big matrix would) nor any number the rule returns
_BOOT_BLOCK_ELEMS = 1 << 22


def _pooled_median(vals_sorted, weights_sorted):
    """np.median of the multiset where ``vals_sorted[i]`` appears ``weights_sorted[i]`` times —
    the same middle element(s) and the same (a + b) / 2 as np.median on the expanded array,
    without expanding it."""
    cw = np.cumsum(weights_sorted)
    m = int(cw[-1])
    i_lo = int(np.searchsorted(cw, (m - 1) // 2, side="right"))
    i_hi = int(np.searchsorted(cw, m // 2, side="right"))
    if i_lo == i_hi:
        return float(vals_sorted[i_lo])
    return float((vals_sorted[i_lo] + vals_sorted[i_hi]) / 2.0)


def decide_change(before, after, *, error, error_factor, confidence, min_judges=None,
                  clusters=None, n_boot=2000, seed=0):
    """THE USER'S RULE (2026-10-07): does a correction improve what it is judged on — significantly,
    with enough judges, and by more than ``error_factor`` x the measured ``error``?

    ``before`` / ``after``: the PAIRED disagreements of the judges at the state without and with the
    correction (same judge, same order, lower = better, one unit shared with ``error``). Every value
    must be finite — observations that are not valid in both states are the caller's to discard
    AND COUNT (plan point 60) before calling.
    ``error``: the measured error of the thing judged, in the same unit (point 1: the largest bridge
    sigma; 18: the chunk's measured fx error; 46: the solver error of point 59; ...). Finite, >= 0.
    ``error_factor``: correction_graph.graph.improvement_error_factor (the user's 2).
    ``confidence``: the declared confidence of the interval (heldout_confidence, 0.95).
    ``min_judges``: None = min_judge_closures(confidence) (5 at 0.95).
    ``clusters``: one label per observation (e.g. the keyframe of each F5 observation, point 46):
    the bootstrap resamples whole CLUSTERS with replacement and the judges are the clusters; None =
    every observation is its own judge (and the resample is bit-identical to heldout_change's).
    The statistic is the median of d = before - after (positive = improves), over the pooled
    observations of the resample; its interval is the bootstrap percentile interval (seeded).

    Returns a JSON-able dict: ``improves`` (= the verdict: a and b and c), ``significant`` (a),
    ``enough_judges`` (b), ``beyond_error`` (c), ``worsens`` (the interval entirely on the worsening
    side — for reports), every margin (``ci_margin`` = ci_low, ``judges_margin`` = n_judges -
    min_judges, ``error_margin`` = median_delta - required_delta), ``failed`` (the conditions that
    did not hold, in the order judges / significance / error) and ``reason`` (one line for the log).
    """
    b = np.asarray(before, np.float64).ravel()
    a = np.asarray(after, np.float64).ravel()
    if b.size != a.size:
        raise ValueError(f"decide_change: {b.size} 'before' values for {a.size} 'after' values — "
                         f"the comparison is paired, judge by judge")
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(a))):
        raise ValueError("decide_change: non-finite disagreements — discard the observations that "
                         "are not valid in both states (and count them) before judging")
    err = float(error)
    fac = float(error_factor)
    conf = float(confidence)
    if not (np.isfinite(err) and err >= 0.0):
        raise ValueError(f"decide_change: error {error!r} must be a finite measured value >= 0")
    if not (np.isfinite(fac) and fac > 0.0):
        raise ValueError(f"decide_change: error_factor {error_factor!r} must be finite and > 0 "
                         f"(correction_graph.graph.improvement_error_factor)")
    if not (0.0 < conf < 1.0):
        raise ValueError(f"decide_change: confidence {confidence!r} must lie in (0, 1)")
    if min_judges is None:
        from loop_utils.loop_judge import min_judge_closures
        min_judges = min_judge_closures(conf)
    min_judges = int(min_judges)
    if min_judges < 1:
        raise ValueError(f"decide_change: min_judges {min_judges} must be >= 1")
    n_boot = int(n_boot)
    if n_boot < 1:
        raise ValueError(f"decide_change: n_boot {n_boot} must be >= 1")

    d = b - a
    n_obs = int(d.size)
    if clusters is None:
        inv, n_judges = None, n_obs
    else:
        cl = np.asarray(clusters).ravel()
        if cl.size != n_obs:
            raise ValueError(f"decide_change: {cl.size} cluster labels for {n_obs} observations")
        _labels, inv = np.unique(cl, return_inverse=True)
        n_judges = int(_labels.size)
    required = fac * err
    out = {"rule": "USER 2026-10-07: significant AND >= min_judges AND median >= factor x error",
           "statistic": "median of paired (before - after); positive = improves",
           "n_obs": n_obs, "n_judges": int(n_judges), "min_judges": min_judges,
           "clustered": clusters is not None, "confidence": conf, "n_boot": n_boot,
           "seed": int(seed), "error": err, "error_factor": fac, "required_delta": float(required)}

    if n_obs == 0:
        out.update({"median_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0})
    else:
        med = float(np.median(d))
        rng = np.random.default_rng(int(seed))
        meds = np.empty(n_boot, np.float64)
        if inv is None:
            rows = max(1, _BOOT_BLOCK_ELEMS // n_obs)
            for s in range(0, n_boot, rows):
                r = min(rows, n_boot - s)
                meds[s:s + r] = np.median(d[rng.integers(0, n_obs, size=(r, n_obs))], axis=1)
        else:
            order = np.argsort(d, kind="stable")
            d_sorted = d[order]
            inv_sorted = inv[order]
            k = int(n_judges)
            rows = max(1, _BOOT_BLOCK_ELEMS // k)
            for s in range(0, n_boot, rows):
                r = min(rows, n_boot - s)
                draw = rng.integers(0, k, size=(r, k))
                for q in range(r):
                    counts = np.bincount(draw[q], minlength=k)
                    meds[s + q] = _pooled_median(d_sorted, counts[inv_sorted])
        alpha = (1.0 - conf) / 2.0
        lo = float(np.percentile(meds, 100.0 * alpha))
        hi = float(np.percentile(meds, 100.0 * (1.0 - alpha)))
        out.update({"median_delta": med, "ci_low": lo, "ci_high": hi})

    enough = bool(n_judges >= min_judges)
    significant = bool(n_obs > 0 and out["ci_low"] > 0.0)
    beyond = bool(n_obs > 0 and out["median_delta"] >= required)
    failed = []
    if not enough:
        failed.append("judges")
    if not significant:
        failed.append("significance")
    if not beyond:
        failed.append("error")
    pct = 100.0 * conf
    reason = (f"median improvement {out['median_delta']:.6g} vs {fac:g} x error {err:.6g} = "
              f"{required:.6g} (margin {out['median_delta'] - required:+.6g}); {pct:g} % CI "
              f"[{out['ci_low']:.6g}, {out['ci_high']:.6g}]; {n_judges} judge(s) for {min_judges} "
              f"required")
    out.update({"improves": bool(enough and significant and beyond), "significant": significant,
                "enough_judges": enough, "beyond_error": beyond,
                "worsens": bool(n_obs > 0 and out["ci_high"] < 0.0),
                "ci_margin": float(out["ci_low"]), "judges_margin": int(n_judges - min_judges),
                "error_margin": float(out["median_delta"] - required), "failed": failed,
                "reason": ("IMPROVES — " if not failed else
                           "does NOT pass (" + ", ".join(failed) + ") — ") + reason})
    return out


def depth_graph_verdict(a, b, meas, held, zref=5.0, confidence=0.95, bound=5.0,
                        n_boot=2000, seed=0, *, held_err=None, error_factor=None):
    """Self-validation shared by every rung of the depth-graph model ladder.
    Judged ONLY on held-out pairs (never fitted), by THE USER'S RULE (2026-10-07,
    plan point 19 — metric_lock.decide_change): the paired change of the
    held-out disagreement at ``zref`` must be significant at ``confidence``,
    come from at least min_judge_closures(confidence) pairs, and its median must
    be >= ``error_factor`` x the largest measured error of those pairs
    (``held_err``: pair_relation_error per held-out pair; a pair without one
    cannot testify and is counted out). AND the corrections must stay within
    ``bound``x the pairwise signal (an order of magnitude beyond what the pairs
    show is noise integration along the chain, not signal). Returns a dict with
    bounded / improves / med_before / med_after / sig_a / sig_b / decision.

    Without ``held_err`` and ``error_factor`` it MEASURES only — bounded, the
    held-out medians, sig_a / sig_b, no 'improves' and no decision: the server's
    witness depth stage (reconstruction/witness/depth_tracks) reads those and
    judges with its own held-out test. Giving one of the two without the other
    is refused."""
    if (held_err is None) != (error_factor is None):
        raise ValueError("depth_graph_verdict: held_err and error_factor go together (the "
                         "rule needs both), or neither (measurement only)")
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    rb_all = [abs(al * zref + be - zref) / zref for _, _, al, be in held]
    ra_all = [abs((a[f] * zref + b[f]) - (a[g] * (al * zref + be) + b[g])) / zref
              for f, g, al, be in held]
    if held_err is None:
        keep = list(range(len(held)))
    else:
        if len(held_err) != len(held):
            raise ValueError(f"depth_graph_verdict: {len(held_err)} errors for {len(held)} pairs")
        keep = [q for q, e in enumerate(held_err) if e is not None and np.isfinite(e)]
    rb = [rb_all[q] for q in keep]
    ra = [ra_all[q] for q in keep]
    med_b = float(np.median(rb)) if rb else float("nan")
    med_a = float(np.median(ra)) if ra else float("nan")
    las = np.abs(np.log([al for _, _, al, _ in meas]))
    bes = np.abs([be for _, _, _, be in meas])
    sig_a = max(float(np.median(las)), 1e-4)
    sig_b = max(float(np.median(bes)), 1e-3)
    bounded = (float(np.percentile(np.abs(np.log(a)), 99)) <= bound * sig_a
               and float(np.percentile(np.abs(b), 99)) <= bound * sig_b)
    out = {"bounded": bool(bounded), "med_before": med_b, "med_after": med_a,
           "sig_a": sig_a, "sig_b": sig_b}
    if held_err is None:
        return out
    error = float(max(held_err[q] for q in keep)) if keep else 0.0
    decision = decide_change(rb, ra, error=error, error_factor=error_factor,
                             confidence=confidence, n_boot=n_boot, seed=seed)
    decision["error_source"] = "the largest measured error of the held-out pairs (split-half)"
    decision["n_pairs_without_error"] = int(len(held) - len(keep))
    out.update({"improves": bool(decision["improves"]), "decision": decision})
    return out


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


def flag_sick_chunks(tri_angle, anchor_iqr, zoom=None,
                     parallax_floor_ratio=10.0, anchor_z_cut=3.5):
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
    #    also breaks DA3's metric depth (assumed focal). The label is the zoom
    #    rule's own verdict (``zoom``: {chunk: zoom_anchor_test(...)} — USER
    #    2026-10-07, plan point 18: significant AND >= factor x the chunk's fx
    #    error), never a second criterion: the robust z > 3.5 is gone.
    for k, zt in sorted((zoom or {}).items(), key=lambda kv: int(kv[0])):
        if zt and zt.get("zoomed"):
            sick.setdefault(int(k), []).append(
                f"optical ZOOM: chunk-median focal {zt['fx_chunk_median']:.0f} vs session "
                f"{zt['fx_rest_median']:.0f} (CI [{zt['ci_low']:+.1f}, {zt['ci_high']:+.1f}] px, "
                f"|Δ| beyond {zt['error_factor']:g} x its error by {zt['error_margin_px']:.1f} px)"
                f" — magnification adds no baseline: no parallax, no 3D information, and "
                f"DA3 metric depth breaks")
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
    Every frame is written by exactly ONE chunk — overlap frames used to be written by
    BOTH chunks, putting two displaced copies of the same pixels into the cloud (the
    mechanical half of the duplicated-objects problem). The block two chunks share is
    split at its MIDPOINT: the first half is written by the earlier chunk, the second by
    the later one — each chunk writes half of what they share (USER 2026-10-07, "50 % de
    quién"). On the uniform layouts this rule was validated on (equal chunks, 50 % overlap)
    the midpoint IS the old nearest-centre rule; on the co-visibility planner's unequal
    chunks the nearest-centre rule handed the WHOLE shared block to the smaller chunk
    (pccr 2026-08-31: chunk 0 wrote 63/63 frames, chunk 1 only 50/139)."""
    owner = np.full(int(n_frames), -1, np.int32)
    ranges = [(int(s0), int(e0)) for s0, e0 in chunk_indices]
    for k, (s0, e0) in enumerate(ranges):
        owner[s0:e0] = k                                  # provisional: the later chunk
    for k in range(len(ranges) - 1):
        a, b = ranges[k + 1][0], ranges[k][1]            # the block chunks k and k+1 share
        if b > a:
            # first half → k, second half → k+1; the frame AT the midpoint goes to the earlier
            # chunk — exactly the nearest-centre rule's tie on a uniform layout, so the validated
            # 60/30 sessions keep their ownership bit for bit
            owner[a:a + (b - a) // 2 + 1] = k
    return owner


# ── CHUNK LAYOUT: the vendor's uniform ranges or an EXPLICIT list ──
# USER 2026-10-06: the chunked Omega path runs the co-visibility planner's
# layout (server/reconstruction/chunk_covis.py — variable-size chunks, each seam
# its own overlap) handed over as Model.chunk_ranges. Every stage of the fork
# reads the real per-chunk (start, end) of chunk_indices and the real per-seam
# overlap of seam_overlap; the uniform construction below stays the
# layout when no explicit list is given (bit-identical to the vendor's).

# Fewest frames two consecutive chunks may share. NOT a tuning knob: it is the
# floor of the seam's own fallback fit — when the exact seam starves, the seam is
# glued by weighted_align_point_maps (sim3utils.py), which wants
# align_min_inlier_frames (default 8) mutually consistent overlap frames and only
# warns below it — so a seam narrower than this could not be measured the way
# every other seam is. The co-visibility planner's seams are whole blocks of
# >= MIN_CHUNK_FRAMES // 2 = 12 frames (server/reconstruction/chunk_covis.py).
MIN_SEAM_FRAMES = 8


class ChunkRangesError(ValueError):
    """A Model.chunk_ranges the fork cannot run. The message names the problem."""


def uniform_chunk_ranges(n_frames, chunk_size, overlap):
    """The vendor's uniform layout, EXACTLY as VGGT-Long's process_long_sequence
    built it: step = chunk_size - overlap, ceil((N - overlap) / step) chunks, the
    last one clipped to N; one chunk [(0, N)] when N <= chunk_size."""
    n_frames, chunk_size, overlap = int(n_frames), int(chunk_size), int(overlap)
    if overlap >= chunk_size:
        raise ValueError(f"[SETTING ERROR] Overlap ({overlap}) must be less than chunk "
                         f"size ({chunk_size})")
    if n_frames <= chunk_size:
        return [(0, n_frames)]
    step = chunk_size - overlap
    num_chunks = (n_frames - overlap + step - 1) // step
    return [(i * step, min(i * step + chunk_size, n_frames)) for i in range(num_chunks)]


def _as_index(v):
    """An integer keyframe index (int or numpy integer — never bool, never float)."""
    if isinstance(v, (bool, np.bool_)):
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    return None


def validate_chunk_ranges(ranges, n_frames=None, min_seam=MIN_SEAM_FRAMES):
    """Explicit chunk layout → list of (start, end) int tuples, or ChunkRangesError
    naming what is wrong. Ranges are [start, end) keyframe indices AFTER any
    keyframe filter and frame stride (the fork's img_list).

    Rules (each one an assumption some fork stage is built on):
      1. a non-empty list of [start, end] integer pairs, start >= 0, start < end;
      2. the first chunk starts at 0 and the last ends at n_frames (a frame no
         chunk covers has no pose: save_camera_poses fails on it) — skipped when
         n_frames is None (the early structural check, before the frame list
         exists);
      3. sorted: starts AND ends strictly increasing (the seams pair chunk k with
         k+1; find_chunk_index bisects on the starts);
      4. consecutive chunks share >= min_seam frames (MIN_SEAM_FRAMES): every seam
         is measured on its shared frames (exact seam, metric-lock seam ratio,
         elastic fits, two-copy uncertainty);
      5. no frame in three chunks (end of k <= start of k+2): the elastic
         corrections, the two-copy blend, the backfill and the uncertainty
         pair chunk k with k-1 and k+1 only, on disjoint frame ranges;
      6. frame ownership (frame_owner) steps 0 or +1 from chunk 0 to the last
         chunk, so every chunk writes at least one frame and the pose graph's
         ownership-crossing link carries THAT seam's residual. Rules 1-5 imply it
         (no layout passing them failed it: every 3-chunk layout up to 63 frames,
         200 000 random 3-6-chunk ones); it is checked anyway because the pose
         graph and the writers stand on it."""
    if isinstance(ranges, np.ndarray):
        ranges = ranges.tolist()
    if not isinstance(ranges, (list, tuple)) or len(ranges) == 0:
        raise ChunkRangesError(f"Model.chunk_ranges must be a non-empty list of [start, end] "
                               f"pairs, got {ranges!r}")
    out = []
    for k, pair in enumerate(ranges):
        if isinstance(pair, np.ndarray):
            pair = pair.tolist()
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ChunkRangesError(f"Model.chunk_ranges: chunk {k} is {pair!r}, not a "
                                   f"[start, end] pair")
        s, e = _as_index(pair[0]), _as_index(pair[1])
        if s is None or e is None:
            raise ChunkRangesError(f"Model.chunk_ranges: chunk {k} is {list(pair)!r} — start "
                                   f"and end must be integer keyframe indices")
        if s < 0 or e <= s:
            raise ChunkRangesError(f"Model.chunk_ranges: chunk {k} [{s}, {e}) is empty or "
                                   f"negative")
        out.append((s, e))
    if out[0][0] != 0:
        raise ChunkRangesError(f"Model.chunk_ranges: the first chunk starts at {out[0][0]}, "
                               f"it must start at 0 (frames 0..{out[0][0] - 1} would have no "
                               f"chunk)")
    n = out[-1][1] if n_frames is None else int(n_frames)
    if n_frames is not None and out[-1][1] != n:
        raise ChunkRangesError(f"Model.chunk_ranges: the last chunk ends at {out[-1][1]} but "
                               f"this run has {n} frames after the keyframe filter and the "
                               f"frame stride — the ranges were planned for another frame "
                               f"list")
    for k in range(len(out) - 1):
        (s0, e0), (s1, e1) = out[k], out[k + 1]
        if not (s1 > s0 and e1 > e0):
            raise ChunkRangesError(f"Model.chunk_ranges: not sorted — chunk {k + 1} [{s1}, "
                                   f"{e1}) does not start AND end after chunk {k} [{s0}, {e0})")
        ov = e0 - s1
        if ov < int(min_seam):
            what = (f"a gap of {-ov} frame(s) [{e0}, {s1})" if ov < 0
                    else f"{ov} shared frame(s) [{s1}, {e0})")
            raise ChunkRangesError(f"Model.chunk_ranges: seam {k}->{k + 1} has {what}, at "
                                   f"least {int(min_seam)} shared frames are needed "
                                   f"(MIN_SEAM_FRAMES)")
    for k in range(len(out) - 2):
        e0, s2 = out[k][1], out[k + 2][0]
        if e0 > s2:
            raise ChunkRangesError(f"Model.chunk_ranges: frames [{s2}, {e0}) are in three "
                                   f"chunks ({k}, {k + 1}, {k + 2}) — a frame may be in two "
                                   f"chunks at most (end of chunk {k} <= start of chunk "
                                   f"{k + 2})")
    owner = frame_owner(out, n)
    if owner[0] != 0 or owner[-1] != len(out) - 1:
        raise ChunkRangesError(f"Model.chunk_ranges: frame ownership runs from chunk "
                               f"{int(owner[0])} to {int(owner[-1])}, it must run from 0 to "
                               f"{len(out) - 1}")
    steps = np.diff(owner.astype(np.int64))
    bad = np.flatnonzero((steps < 0) | (steps > 1))
    if bad.size:
        g = int(bad[0]) + 1
        raise ChunkRangesError(f"Model.chunk_ranges: frame ownership jumps from chunk "
                               f"{int(owner[g - 1])} to chunk {int(owner[g])} at frame {g} — "
                               f"chunk(s) in between own no frame (chunk centres too close "
                               f"for their lengths)")
    return out


def seam_overlap(chunk_indices, k):
    """Frames shared by chunks k and k+1 = end of chunk k - start of chunk k+1: the
    vendor's `overlap` on a uniform layout (every seam), this seam's own count on
    an explicit one. Chunk k's last seam_overlap frames are chunk k+1's first ones —
    the seam fits slice [-ov:] / [:ov] with it."""
    return int(chunk_indices[k][1]) - int(chunk_indices[k + 1][0])


# ── INTRA-CHUNK per-frame consensus (bounded fields, anchored boundaries) ──
# History (test4, 2026-07-10): a GLOBAL per-frame pose graph with free
# boundaries hallucinated wavelengths longer than its pair span (163 cm bend,
# chimney 7→130 cm) — short-span evidence cannot certify long corrections.
# This design inverts it: each chunk is solved ALONE with its endpoints
# CLAMPED to zero (the seam consensus the elastic already established), so no
# correction longer than one chunk can exist, and per-chunk self-gates keep
# the worst case at identity for that chunk only.


def surface_pair_correspondences(wp_src, conf_src, wp_dst, conf_dst, w2c_dst, K_dst,
                                 max_samples=8000, seed=0, conf_min_norm=0.0,
                                 return_keys=False):
    """EXACT-surface 3D correspondences between two frames: project src's valid
    points into dst's camera and pair them with dst's OWN 3D point at the hit
    pixel. Returns (p_src[n,3], q_dst[n,3]) or None when starved — the rigid
    analogue of depth_pair_samples (same association, full 3D instead of z).

    `conf_min_norm` (USER 2026-09-23) is a QUALITY floor on top of the sky mask,
    as a min-max fraction of this frame's own valid confidences — the same
    arithmetic the viewer slider and the CloudCompy gate use, so one number means
    one thing everywhere. It matters more here than in the cloud: these pairs are
    what the intra-chunk field and the pose graph FIT on *and* what their held-out
    pairs JUDGE with, so an unconfident point biases the correction and corrupts
    its own examiner. `conf > 1e-5` alone is the SKY MASK, not a quality gate.
    0.0 keeps the historical behaviour.

    Past ``max_samples`` the source pixels are chosen by their STABLE KEY (plan point
    11): pixel (``seed``, flat index) — the fork passes the source frame's global
    keyframe index as ``seed``. ``return_keys``: also return each pair's key (uint64),
    for a stable sub-selection downstream."""
    from loop_utils.stable_sample import pixel_keys, stable_pick
    H, W = wp_dst.shape[:2]
    p = np.asarray(wp_src, np.float64).reshape(-1, 3)
    c = np.asarray(conf_src, np.float32).reshape(-1)
    idx = np.flatnonzero(c > 1e-5)
    if float(conf_min_norm) > 0.0 and idx.size:
        v = c[idx]
        lo, hi = float(v.min()), float(v.max())
        if hi > lo:
            keep = v >= lo + float(conf_min_norm) * (hi - lo)
            # never starve the fit: a frame whose confidences are nearly uniform
            # would lose everything to a floor that means nothing there
            if keep.sum() >= 500:
                idx = idx[keep]
    if len(idx) < 500:
        return None
    keys = pixel_keys(int(seed), idx)
    if len(idx) > max_samples:
        # stable per-pixel choice (plan point 11): ``seed`` is the frame key of the
        # source frame (the fork passes its global keyframe index)
        sel = stable_pick(keys, max_samples)
        idx, keys = idx[sel], keys[sel]
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
    u, v, p_in, k_in = u[inb], v[inb], p[m][inb], keys[m][inb]
    q = np.asarray(wp_dst, np.float64)[v, u]
    cq = np.asarray(conf_dst, np.float32).reshape(H, W)[v, u]
    good = cq > 1e-5
    if good.sum() < 300:
        return None
    if return_keys:
        return p_in[good], q[good], k_in[good]
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


def chunk_field_verdict(xi, pair_taus, held, confidence=0.95, bound=5.0,
                        n_boot=2000, seed=0):
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
    chg = heldout_change(before, after, confidence=confidence,
                         n_boot=n_boot, seed=seed)
    improves = bool(chg["improves"])
    return {"bounded": bounded, "improves": improves,
            "med_before": med_b, "med_after": med_a,
            "t_sig": t_sig, "r_sig": r_sig, "heldout": chg}
