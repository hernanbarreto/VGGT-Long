# STAC patch — EXACT loop bridges (claude_stac.txt §4.1 / §4.2 / §4.9).
#
# A loop bridge is one extra Omega pass over two frame windows: `loop_chunk_size`
# keyframes around frame i (inside chunk a) and around frame j (inside chunk b).
# Those windows are the SAME frames — pixel for pixel — as the corresponding
# slices of chunk a and chunk b, so the bridge↔chunk relation is measured on
# millions of EXACT correspondences (the seam discipline), not on a coarse
# point-map fit. Two fits give the a↔b relation:
#
#     chunk_a ≈ s_a·R_a·bridge + t_a         chunk_b ≈ s_b·R_b·bridge + t_b
#     a→b Sim3 = compose(fit_b, fit_a⁻¹):    s_ab = s_b/s_a, R_ab = R_b·R_aᵀ,
#                                            t_ab = t_b − s_ab·R_ab·t_a
#
# `s_ab` is the RELATIVE SCALE between the two chunks as observed through one
# coherent prediction — it is kept as a MEASUREMENT for the scale graph (a loop
# row, metric_lock.solve_scale_graph) and never negotiated in the poses. The
# pose edge that enters the graph is the SE(3) part, re-measured with a rigid
# fit once the scale graph has closed.
#
# Bridge frames that are NOT keyframes (§4.9) may be added to densify the
# revisit; they take part in the Omega pass only — they are never part of the
# chain, add no drift, and are excluded from the exact correspondences (which
# by definition need a frame present in a chunk).
#
# Every decision threshold comes from the config (Model.loops.*); a missing key
# fails naming it. Nothing here is silent: a starved fit is recorded as a
# low-confidence edge with the configured sigma, never dropped quietly.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import numpy as np

from loop_utils.metric_lock import robust_rigid


# ── config access (fail-fast, named) ──────────────────────────────────

def cfg_req(section: dict, key: str, path: str):
    """Mandatory config key: a missing one aborts naming `Model.<path>.<key>`."""
    if section is None or key not in section:
        raise RuntimeError(f"config key Model.{path}.{key} is missing — every "
                           f"loop/scale decision threshold must be declared "
                           f"(server/config.yaml, section loops:/scale:)")
    return section[key]


# ── robust similarity fit on exact correspondences ─────────────────────

def robust_sim3(src, dst, iters=8, sample=200000, seed=0, min_points=1000, keys=None):
    """dst ≈ s·R·src + t from EXACT correspondences, weighted Umeyama with
    IRLS Cauchy weights (the Sim3 sibling of metric_lock.robust_rigid).
    Returns (s, R, t, median_residual_m, n_used) or None when starved.
    Past ``sample`` the subsample is chosen by a STABLE KEY per correspondence
    (``keys``, else the row's own bits; ``seed`` salts it — plan point 11)."""
    from loop_utils.stable_sample import row_keys, stable_pick
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    m = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    if keys is not None:
        keys = np.asarray(keys).ravel()
        if len(keys) != len(m):
            raise ValueError(f"robust_sim3: {len(keys)} keys for {len(m)} correspondences")
        keys = keys[m]
    src, dst = src[m], dst[m]
    if len(src) < int(min_points):
        return None
    if len(src) > int(sample):
        idx = stable_pick(row_keys(src, dst) if keys is None else keys, int(sample), salt=seed)
        src, dst = src[idx], dst[idx]
    w = np.ones(len(src))
    s, R, t = 1.0, np.eye(3), np.zeros(3)
    for _ in range(int(iters)):
        ws = w.sum()
        cs = (src * w[:, None]).sum(0) / ws
        cd = (dst * w[:, None]).sum(0) / ws
        X = src - cs
        Y = dst - cd
        H = ((X * w[:, None]).T @ Y) / ws
        U, D, Vt = np.linalg.svd(H)
        S = np.eye(3)
        S[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ S @ U.T
        var_src = (w * (X ** 2).sum(1)).sum() / ws
        if var_src <= 1e-18:
            return None
        s = float(np.trace(np.diag(D) @ S) / var_src)
        if not np.isfinite(s) or s <= 0:
            return None
        t = cd - s * (R @ cs)
        r = np.linalg.norm(dst - (s * (src @ R.T) + t), axis=1)
        c = max(3.0 * 1.4826 * np.median(np.abs(r - np.median(r))), 1e-4)
        w = 1.0 / (1.0 + (r / c) ** 2)
    r = np.linalg.norm(dst - (s * (src @ R.T) + t), axis=1)
    return float(s), R, t, float(np.median(r)), int(len(src))


def compose_ab(fit_a, fit_b):
    """(s_ab, R_ab, t_ab) mapping chunk-a coordinates onto chunk-b coordinates
    from the two bridge fits (same algebra as sim3utils.compute_sim3_ab)."""
    s_a, R_a, t_a = fit_a
    s_b, R_b, t_b = fit_b
    s_ab = float(s_b) / float(s_a)
    R_ab = np.asarray(R_b) @ np.asarray(R_a).T
    t_ab = np.asarray(t_b) - s_ab * (R_ab @ np.asarray(t_a))
    return s_ab, R_ab, t_ab


# ── exact correspondences bridge ↔ chunk ───────────────────────────────

def _wp(data):
    wp = np.asarray(data["world_points"])
    return wp[0] if wp.ndim == 5 else wp


def _conf(data, wp):
    return np.asarray(data["world_points_conf"]).reshape(wp.shape[:3])


def exact_correspondences(bridge, bridge_locals: Sequence[int],
                          chunk, chunk_locals: Sequence[int],
                          per_frame_cap: int, seed: int = 0,
                          frame_keys: Optional[Sequence[int]] = None,
                          return_keys: bool = False):
    """Pixel-to-pixel pairs (p_bridge, q_chunk) over frames present in BOTH
    predictions. bridge_locals[k] and chunk_locals[k] index the SAME global
    frame in each prediction. Pixels valid (conf > 1e-5) in both are used,
    capped per frame so every frame has the same voice.

    The cap keeps the pixels whose STABLE KEY ranks first (plan point 11):
    pixel (frame_keys[k], flat index) — the global keyframe index when the
    caller passes it, else the chunk-local index — salted by ``seed``. A pixel
    that enters or leaves the valid set no longer re-draws the frame's sample.
    ``return_keys``: also return those keys (one per pair), for the robust fits
    and the split-half test downstream."""
    from loop_utils.stable_sample import pixel_keys, stable_pick
    wb, wc = _wp(bridge), _wp(chunk)
    cb, cc = _conf(bridge, wb), _conf(chunk, wc)
    if wb.shape[1:3] != wc.shape[1:3]:
        raise RuntimeError(f"bridge grid {wb.shape[1:3]} != chunk grid "
                           f"{wc.shape[1:3]} — bridge and chunk must share the "
                           f"prediction resolution for exact correspondences")
    fkeys = list(chunk_locals) if frame_keys is None else list(frame_keys)
    if len(fkeys) != len(chunk_locals):
        raise ValueError(f"exact_correspondences: {len(fkeys)} frame keys for "
                         f"{len(chunk_locals)} frames")
    P, Q, KEYS = [], [], []
    for lb, lc, fk in zip(bridge_locals, chunk_locals, fkeys):
        ok = (cb[lb] > 1e-5) & (cc[lc] > 1e-5)
        idx = np.flatnonzero(ok.reshape(-1))
        if len(idx) == 0:
            continue
        keys = pixel_keys(int(fk), idx)
        if len(idx) > int(per_frame_cap):
            sel = stable_pick(keys, int(per_frame_cap), salt=seed)
            idx, keys = idx[sel], keys[sel]
        P.append(wb[lb].reshape(-1, 3)[idx])
        Q.append(wc[lc].reshape(-1, 3)[idx])
        KEYS.append(keys)
    if not P:
        out = (np.zeros((0, 3)), np.zeros((0, 3)))
        return out + (np.zeros(0, np.uint64),) if return_keys else out
    out = (np.concatenate(P).astype(np.float64), np.concatenate(Q).astype(np.float64))
    return out + (np.concatenate(KEYS),) if return_keys else out


# ── bridge layout (which bridge frame is which chunk frame) ────────────

def bridge_layout(item, chunk_indices, n_extra_a: int = 0, n_extra_b: int = 0) -> dict:
    """Local index bookkeeping for one bridge prediction whose image list is
    [window_a keyframes, extra_a frames, window_b keyframes, extra_b frames].

    item = (chunk_a, (a0, a1), chunk_b, (b0, b1)) in GLOBAL keyframe indices."""
    ka, (a0, a1), kb, (b0, b1) = item[0], item[1], item[2], item[3]
    na, nb = a1 - a0, b1 - b0
    off_b = na + n_extra_a
    return {
        "chunk_a": int(ka), "chunk_b": int(kb),
        "range_a": [int(a0), int(a1)], "range_b": [int(b0), int(b1)],
        "bridge_a": list(range(0, na)),
        "bridge_b": list(range(off_b, off_b + nb)),
        "chunk_a_local": [int(g - chunk_indices[ka][0]) for g in range(a0, a1)],
        "chunk_b_local": [int(g - chunk_indices[kb][0]) for g in range(b0, b1)],
        "n_extra_a": int(n_extra_a), "n_extra_b": int(n_extra_b),
        "n_frames": int(na + n_extra_a + nb + n_extra_b),
    }


def _apply(fit, src):
    """s·R·src + t for a (s, R, t) fit."""
    s_, R_, t_ = fit
    return float(s_) * (np.asarray(src, np.float64) @ np.asarray(R_).T) + np.asarray(t_)


def split_half_residual(p, q, rigid: bool, sample: int, min_pts: int,
                        seed: int = 0, keys=None) -> Optional[dict]:
    """Fit on half the correspondences, MEASURE on the other half.

    This is what tells a real association from a lucky one, and it replaces
    judging a fit by the SIZE of its residual (USER 2026-09-16: *"podría haber
    una corrección de más de 30 cm y ser perfectamente correcta"*). A fit that
    describes the geometry predicts correspondences it never saw just as well:
    held-out ≈ fit. One that latched onto the wrong structure does not.

    Returns {fit_residual_m, holdout_residual_m, ratio, n_fit, n_held} or None
    when either half starves (the caller keeps the full-sample verdict then).
    """
    from loop_utils.stable_sample import row_keys, stable_half
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    n = len(p)
    if n < 2 * int(min_pts):
        return None
    # the halves are decided per correspondence by its STABLE KEY (plan point 11):
    # a pair entering or leaving no longer reshuffles the split of all the others.
    # The salt differs from the subsample's so the half does not follow the pick.
    keys = row_keys(p, q) if keys is None else np.asarray(keys).ravel()
    if len(keys) != n:
        raise ValueError(f"split_half_residual: {len(keys)} keys for {n} correspondences")
    in_a = stable_half(keys, salt=int(seed) + 9176)
    a, b = np.flatnonzero(in_a), np.flatnonzero(~in_a)
    if min(len(a), len(b)) < int(min_pts):
        return None
    if rigid:
        f = robust_rigid(p[a], q[a], sample=sample, seed=seed, keys=keys[a])
        fit = None if f is None else (1.0, f[0], f[1])
        res_fit = None if f is None else float(f[2])
    else:
        f = robust_sim3(p[a], q[a], sample=sample, seed=seed, min_points=min_pts,
                        keys=keys[a])
        fit = None if f is None else (f[0], f[1], f[2])
        res_fit = None if f is None else float(f[3])
    if fit is None:
        return None
    held = np.linalg.norm(q[b] - _apply(fit, p[b]), axis=1)
    held = held[np.isfinite(held)]
    if not len(held):
        return None
    res_held = float(np.median(held))
    return {"fit_residual_m": res_fit, "holdout_residual_m": res_held,
            "ratio": float(res_held / max(res_fit, 1e-9)),
            "n_fit": int(len(a)), "n_held": int(len(held))}


# ── the measurement ─────────────────────────────────────────────────────

def _side_range_m(q: np.ndarray, chunk, chunk_locals: Sequence[int]):
    """Median distance of the chunk-side correspondences ``q`` to the cameras
    of the chunk frames they came from (their centroid — the frames of one
    window sit within a few tens of centimetres of each other). ``extrinsic``
    is (S,4,4) c2w in the STAC npy; a (S,3,4) w2c is inverted. None when the
    chunk carries no extrinsics or the side has no correspondences."""
    if q is None or len(q) == 0 or "extrinsic" not in chunk:
        return None
    ext = np.asarray(chunk["extrinsic"], np.float64)
    ext = ext[0] if ext.ndim == 4 else ext
    idx = [int(c) for c in chunk_locals if 0 <= int(c) < len(ext)]
    if not idx:
        return None
    if ext.shape[1:] == (4, 4):
        C = ext[idx, :3, 3]
    else:                                   # (3,4) w2c: C = -Rᵀ t
        R = ext[idx, :3, :3]; t = ext[idx, :3, 3]
        C = -np.einsum("nji,nj->ni", R, t)
    c0 = np.median(C, axis=0)
    return float(np.median(np.linalg.norm(np.asarray(q, np.float64) - c0, axis=1)))


def measure_bridge(bridge, layout: dict, chunk_a, chunk_b, loops_cfg: dict,
                   rigid: bool, seed: int = 0) -> dict:
    """Fit the bridge against both chunks on exact correspondences.

    rigid=False → Sim3 fits (scale MEASURED: s_ab is the residual relative
    scale between the chunks, a scale-graph row). rigid=True → SE(3) fits on
    scale-closed chunks (the pose edge). A side whose exact fit starves falls
    back to NOTHING here — the caller decides (vendor coarse fit, recorded as
    low confidence) — so the verdict is never hidden inside this function."""
    cap = int(cfg_req(loops_cfg, "corr_per_frame", "loops"))
    sample = int(cfg_req(loops_cfg, "fit_sample", "loops"))
    min_pts = int(cfg_req(loops_cfg, "min_correspondences", "loops"))
    out = {"rigid": bool(rigid), "sides": {}}
    fits = {}
    for side, chunk, bl, cl, rg in (("a", chunk_a, layout["bridge_a"], layout["chunk_a_local"],
                                     layout.get("range_a")),
                                    ("b", chunk_b, layout["bridge_b"], layout["chunk_b_local"],
                                     layout.get("range_b"))):
        # pixels keyed by their GLOBAL keyframe (plan point 11) when the layout says it
        fk = list(range(int(rg[0]), int(rg[1]))) if rg is not None else None
        p, q, keys = exact_correspondences(bridge, bl, chunk, cl, cap, seed=seed,
                                           frame_keys=fk, return_keys=True)
        rec = {"n_corr": int(len(p)), "starved": False}
        # the LEVER ARM of this side: how far the surfaces that constrained the
        # fit sit from the cameras that saw them. A translation error bar σ on
        # those surfaces is a rotation error bar σ/range on the pose — the
        # same measurement read in angle (USER 2026-09-25: no constant σ_rot)
        rng_m = _side_range_m(q, chunk, cl)
        if rng_m is not None:
            rec["range_m"] = float(rng_m)
        fit = None
        if len(p) >= min_pts:
            if rigid:
                f = robust_rigid(p, q, sample=sample, seed=seed, keys=keys)
                if f is not None:
                    R_, t_, res, n = f
                    fit = (1.0, R_, t_)
                    rec.update({"s": 1.0, "residual_m": float(res), "n_fit": int(n)})
            else:
                f = robust_sim3(p, q, sample=sample, seed=seed, min_points=min_pts, keys=keys)
                if f is not None:
                    s_, R_, t_, res, n = f
                    fit = (s_, R_, t_)
                    rec.update({"s": float(s_), "residual_m": float(res), "n_fit": int(n)})
        if fit is None:
            rec["starved"] = True
        else:
            rec["R"] = np.asarray(fit[1]).tolist()
            rec["t"] = np.asarray(fit[2]).tolist()
            # does the fit describe the geometry, or only the points it saw?
            sh = split_half_residual(p, q, rigid, sample, min_pts, seed=seed, keys=keys)
            if sh is not None:
                rec["split_half"] = sh
        fits[side] = fit
        out["sides"][side] = rec
    if fits["a"] is not None and fits["b"] is not None:
        s_ab, R_ab, t_ab = compose_ab(fits["a"], fits["b"])
        sh_a = out["sides"]["a"].get("split_half")
        sh_b = out["sides"]["b"].get("split_half")
        ranges = [out["sides"][s_]["range_m"] for s_ in ("a", "b")
                  if out["sides"][s_].get("range_m") is not None]
        out.update({"ok": True, "s_ab": float(s_ab), "R_ab": R_ab.tolist(),
                    "t_ab": t_ab.tolist(),
                    # the SHORTER lever arm decides: it is the side whose
                    # translation error bar turns into the larger angle
                    "range_m": (float(min(ranges)) if ranges else None),
                    "residual_m": float(max(out["sides"]["a"]["residual_m"],
                                            out["sides"]["b"]["residual_m"])),
                    "n_corr": int(min(out["sides"]["a"]["n_corr"],
                                      out["sides"]["b"]["n_corr"]))})
        if sh_a is not None and sh_b is not None:
            # the worse side decides: an edge is as good as its weaker half
            out["holdout_residual_m"] = float(max(sh_a["holdout_residual_m"],
                                                  sh_b["holdout_residual_m"]))
            out["holdout_ratio"] = float(max(sh_a["ratio"], sh_b["ratio"]))
    else:
        out["ok"] = False
    return out


def scale_row_sigma(sigma_loop: float, log_r_measured: float) -> float:
    """σ (log units) of a bridge's scale row in the scale graph (USER 2026-10-07, plan point
    17 — no threshold, no factor): the row's base σ with its MEASURED scale disagreement
    |log r| added in quadrature — it grows continuously with the disagreement. The old rule
    multiplied σ by scale_break_sigma_factor once |log r| crossed scale_tol_log."""
    s, d = float(sigma_loop), float(log_r_measured)
    if not (np.isfinite(s) and s > 0.0 and np.isfinite(d)):
        raise ValueError(f"scale_row_sigma: σ {sigma_loop!r} / log r {log_r_measured!r}")
    return float(np.hypot(s, d))


def loop_scale_row(s_ab: float) -> float:
    """log r for the scale graph, r = s_B/s_A in metric_lock's convention
    (s_k = factor that makes chunk k metric). The bridge sees chunk b's units
    as s_ab = u_b/u_a times chunk a's, so the factors relate as s_B/s_A =
    1/s_ab."""
    return float(-np.log(float(s_ab)))


# ── verification (§4.2, steps 1–4; step 0 is the spatial gate, upstream) ──

def verify_loop(meas: dict, loops_cfg: dict, semantic: Optional[dict] = None,
                spatial: Optional[dict] = None,
                reference_m: Optional[float] = None,
                scale_disagreement_log: Optional[float] = None) -> dict:
    """Verdict for one measured bridge: σ is MEASURED, magnitude never vetoes.

    USER 2026-09-16, after pccr closed nothing: *"no debes rechazar correcciones
    por umbrales arbitrarios"*. The old rule dropped any bridge whose residual
    exceeded `max_residual_m` (0.10), so the only two edges that could close a
    44 m walk vanished at 17 cm and the pose graph fell back to IDENTITY. A big
    residual is not evidence of a wrong measurement — it is a measurement with
    a wide error bar, and a pose graph already knows what to do with that.

    What decides now, all of it measured:
    1. the fit must EXIST (≥ min_correspondences on both sides) — starvation is
       the only geometric rejection left;
    2. σ = the SPLIT-HALF held-out residual: fit on half the correspondences,
       measure on the other half. A fit that describes the geometry predicts
       what it never saw (held-out ≈ fit); one that latched onto the wrong
       structure does not, and pays for it in σ — automatically, with no
       threshold to invent;
    3. the residual is compared against what THIS session achieves where the
       geometry is known to be the same (`reference_m`, the median exact-seam
       residual of its own chain; `max_residual_m` is only the fallback when
       the caller has nothing measured yet). Worse than the session's own
       overlaps → σ is inflated by the measured ratio and the edge is declared
       `weak_evidence` in the report. It still enters the graph, weighted.
    4. scale (USER 2026-10-07, plan point 17 — no threshold, no factor): the
       bridge's MEASURED scale disagreement d (log units — ``scale_disagreement_log``
       when the caller measured it, else log s_ab of this measurement) displaces the
       surfaces it was fitted on by |d| x their lever arm (``range_m``); that error
       is added IN QUADRATURE to σ, so σ grows continuously with the disagreement.
       The 'scale_break' step (σ x factor beyond |log s_ab| > tol) is gone;
    5. semantic: when BOTH frames carry SAM3 instances, ≥ min_shared_structural_labels
       structural labels in common (movable labels neither help nor hurt).
    A spatial verdict 'ambiguous' inflates σ by ambiguous_sigma_factor; 'reject'
    never reaches this function (no bridge is spent on it)."""
    fallback_ref = float(cfg_req(loops_cfg, "max_residual_m", "loops"))
    min_corr = int(cfg_req(loops_cfg, "min_correspondences", "loops"))
    amb_factor = float(cfg_req(loops_cfg, "ambiguous_sigma_factor", "loops"))
    min_shared = int(cfg_req(loops_cfg, "min_shared_structural_labels", "loops"))
    movable = set(str(x).lower() for x in cfg_req(loops_cfg, "movable_labels", "loops"))

    v = {"status": "rejected", "reasons": [], "checks": {}}
    if not meas.get("ok"):
        v["reasons"].append("exact fit starved on at least one side")
        v["checks"]["geometric"] = False
        return v
    # 1. starvation is the only geometric rejection: a fit that does not exist
    ref = float(reference_m) if (reference_m is not None
                                 and np.isfinite(reference_m)
                                 and reference_m > 0) else fallback_ref
    enough = meas["n_corr"] >= min_corr
    v["checks"]["geometric"] = {"residual_m": meas["residual_m"],
                                "reference_m": ref,
                                "reference_source": ("session_seams"
                                                     if reference_m else "config_fallback"),
                                "holdout_residual_m": meas.get("holdout_residual_m"),
                                "holdout_ratio": meas.get("holdout_ratio"),
                                "n_corr": meas["n_corr"], "min_correspondences": min_corr,
                                "passed": bool(enough)}
    if not enough:
        v["reasons"].append(f"starved: {meas['n_corr']} exact correspondence(s) < "
                            f"{min_corr} — no fit to trust or distrust")
        return v
    log_s = float(np.log(meas["s_ab"]))
    d_log = log_s if scale_disagreement_log is None else float(scale_disagreement_log)
    if not np.isfinite(d_log):
        raise ValueError(f"verify_loop: scale disagreement {d_log!r} is not finite")
    lever = meas.get("range_m")
    if d_log != 0.0 and not (lever is not None and np.isfinite(lever) and lever > 0):
        raise ValueError("verify_loop: the bridge has a scale disagreement but no lever arm "
                         "(range_m) to read it in metres")
    sigma_scale = abs(d_log) * float(lever) if d_log != 0.0 else 0.0
    v["checks"]["scale"] = {"log_s_ab": log_s, "scale_disagreement_log": d_log,
                            "disagreement_source": ("measured by the caller"
                                                    if scale_disagreement_log is not None
                                                    else "log s_ab of this measurement"),
                            "lever_arm_m": (None if lever is None else float(lever)),
                            "sigma_scale_m": float(sigma_scale),
                            "rule": "added in quadrature to σ (USER 2026-10-07, no threshold)"}
    if semantic is not None and semantic.get("a") is not None and semantic.get("b") is not None:
        la = {str(x).lower() for x in semantic["a"]} - movable
        lb = {str(x).lower() for x in semantic["b"]} - movable
        shared = sorted(la & lb)
        sem_ok = len(shared) >= min_shared
        v["checks"]["semantic"] = {"shared_structural": shared, "min": min_shared,
                                   "passed": bool(sem_ok)}
        if not sem_ok:
            v["reasons"].append(f"semantic: {len(shared)} shared structural label(s) < {min_shared}")
            return v
    else:
        v["checks"]["semantic"] = {"passed": None, "note": "no instances on both frames"}
    # 2. σ IS the held-out error — what the fit failed to predict, in metres.
    #    No extra widening: the error bar already says how much to believe it,
    #    and inflating it again for being large would count the same fact twice.
    sigma = float(meas.get("holdout_residual_m") or meas["residual_m"])
    # 3. …and it is DECLARED against what this session achieves where the
    #    geometry is known to be the same. A ratio, not a verdict.
    worse = sigma / max(ref, 1e-9)
    v["checks"]["evidence"] = {"sigma_m": sigma, "reference_m": ref,
                               "ratio_vs_session": float(worse)}
    if worse > 1.0:
        v["evidence"] = "weak"
        v["reasons"].append(
            f"weak evidence: held-out {sigma * 100:.1f} cm vs this session's own "
            f"overlap agreement {ref * 100:.1f} cm (×{worse:.1f}) — edge kept, "
            f"the graph weighs it by σ")
    if spatial is not None and spatial.get("verdict") == "ambiguous":
        sigma *= amb_factor
        v["checks"]["spatial"] = {"verdict": "ambiguous", "sigma_factor": amb_factor}
    v["checks"]["scale"]["sigma_before_m"] = float(sigma)
    sigma = float(np.hypot(sigma, sigma_scale))
    if sigma_scale > 0.0:
        v["reasons"].append(f"scale disagreement {abs(d_log) * 100:.2f}% x lever arm "
                            f"{float(lever):.2f} m = {sigma_scale * 100:.1f} cm, added in "
                            f"quadrature to σ")
    v["status"] = "accepted"
    v["sigma_m"] = sigma
    return v


# ── persistence ─────────────────────────────────────────────────────────

def save_loop_report(path: str, edges: List[dict], extra: Optional[dict] = None) -> None:
    rep = {"version": 1, "edges": edges}
    if extra:
        rep.update(extra)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rep, f, indent=1)
    os.replace(tmp, path)


# ── the loop candidate files (plan point 8, audit omega-08) ─────────────
#
# loop_closures.txt holds SALAD's candidates: POSITIONS in the keyframe list it was computed
# on. The map worker keeps it across new chunk plans, and the fork used to reuse it whenever
# it existed — whatever keyframes, weights, parameters or revisit reference it came from. Now
# it carries ONE stamp line (``# stac-stamp: {json}`` — every reader skips '#' lines): the
# repro.stamp of the keyframes (names, order, bytes), the SALAD weights, the detector's code,
# its parameters and the geometric revisit reference its bar is calibrated on, plus the bar it
# applied (``salad_threshold``, recorded). It is reused only on a matching stamp; otherwise it
# is deleted (with its calibration) and SALAD runs again. The SERVER never rewrites it.
#
# The post-hoc candidates (the server's instance / manual loops) live in their OWN file,
# loop_closures_posthoc.txt, stamped with the SALAD stamp they were merged against and the
# digest of their own lines: when SALAD is re-run (another keyframe list, another anything)
# they are stale — deleted by the fork, ignored by every reader.

LOOP_STAMP_PREFIX = "# stac-stamp: "
LOOP_CLOSURES_NAME = "loop_closures.txt"
POSTHOC_NAME = "loop_closures_posthoc.txt"
SALAD_CALIBRATION_NAME = "salad_calibration.json"


def posthoc_path(loop_txt: str) -> str:
    """The post-hoc candidate file next to ``loop_txt``."""
    return os.path.join(os.path.dirname(os.path.abspath(loop_txt)), POSTHOC_NAME)


def _parse_candidates(path: str) -> List[dict]:
    """Candidate lines `i, j, sim[, source]` of a file (`#` lines are comments; source
    defaults to 'salad', the vendor detector)."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            rec = {"i": int(float(parts[0])), "j": int(float(parts[1])),
                   "sim": float(parts[2]) if len(parts) >= 3 and parts[2] else None,
                   "source": parts[3] if len(parts) >= 4 and parts[3] else "salad"}
            out.append(rec)
    return out


def read_loop_stamp(path: str) -> Optional[dict]:
    """The stamp line of a candidate file (None: no file, no stamp, or unreadable)."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            if line.startswith(LOOP_STAMP_PREFIX):
                try:
                    doc = json.loads(line[len(LOOP_STAMP_PREFIX):])
                except ValueError:
                    return None
                return doc if isinstance(doc, dict) else None
    return None


def _stamp_line(stamp: dict) -> str:
    return LOOP_STAMP_PREFIX + json.dumps(stamp, sort_keys=True, separators=(",", ":"))


def write_loop_stamp(path: str, stamp: dict) -> None:
    """(Re)write the stamp line of ``path`` (atomically; every other line kept)."""
    with open(path) as f:
        lines = [ln for ln in f.read().splitlines() if not ln.startswith(LOOP_STAMP_PREFIX)]
    lines.insert(0, _stamp_line(stamp))
    _write_lines(path, lines)


def _write_lines(path: str, lines: List[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


def format_similarity(sim) -> str:
    """A candidate's similarity as text that reads back as the SAME float (repr): a resumed
    run must hold the very candidates a fresh run held (plan points 8 and 45)."""
    return "" if sim is None else repr(float(sim))


def _candidate_line(c: dict) -> str:
    return (f"{int(c['i'])}, {int(c['j'])}, {format_similarity(c.get('sim'))}, "
            f"{c.get('source', 'salad')}")


def _candidates_sha256(cands: Sequence[dict]) -> str:
    import hashlib
    return hashlib.sha256("\n".join(_candidate_line(c) for c in cands).encode()).hexdigest()


def salad_loop_stamp(img_list: Sequence[str], config: dict) -> dict:
    """repro.stamp of what SALAD's candidates are made of: the keyframes (names, order and
    bytes — the candidates are POSITIONS in this list), the SALAD weights, the detector's code
    (LoopModels/, its VPR model and DINOv2 backbone included), its parameters (Loop.SALAD, the
    bridge window that widens the odometry band, the frame stride) and the geometric revisit
    reference its bar is calibrated on."""
    from pathlib import Path
    from loop_utils.stac_repro import FORK_DIR, repro
    sal = dict((config.get("Loop") or {}).get("SALAD") or {})
    inputs = {f"frames/{os.path.basename(p)}": p for p in img_list}
    ref = sal.pop("revisit_reference", None)
    if ref and os.path.exists(ref):
        inputs["revisit_reference"] = ref
    sal["revisit_reference_used"] = bool(ref and os.path.exists(ref))
    w = (config.get("Weights") or {}).get("SALAD")
    if not (w and os.path.isfile(str(w))):
        raise RuntimeError(f"Weights.SALAD {w!r} is not a local file — the loop candidates "
                           f"cannot be stamped (plan point 8)")
    inputs["weights/SALAD"] = str(w)
    code = sorted(p for p in (Path(FORK_DIR) / "LoopModels").rglob("*.py")
                  if "__pycache__" not in p.parts)
    model = config.get("Model") or {}
    return repro().stamp(inputs=inputs, code=code,
                         config={"salad": sal,
                                 "loop_chunk_size": model.get("loop_chunk_size"),
                                 "frame_stride": model.get("frame_stride", 1),
                                 "frame_list": [os.path.basename(p) for p in img_list]})


def _salad_digest(loop_txt: str) -> Optional[str]:
    st = read_loop_stamp(loop_txt)
    return None if st is None else st.get("sha256")


def posthoc_state(loop_txt: str):
    """(valid, reason, candidates) of the post-hoc file next to ``loop_txt``: valid when its
    stamp names the SALAD stamp ``loop_txt`` carries NOW and the digest of its own lines."""
    pp = posthoc_path(loop_txt)
    if not os.path.exists(pp):
        return False, "absent", []
    st = read_loop_stamp(pp)
    cands = _parse_candidates(pp)
    if st is None:
        return False, "it carries no stamp", cands
    salad = _salad_digest(loop_txt)
    if st.get("salad_stamp") != salad:
        return False, (f"merged against SALAD stamp {str(st.get('salad_stamp'))[:12]}…, the "
                       f"candidates on disk are {str(salad)[:12]}…"), cands
    if st.get("candidates_sha256") != _candidates_sha256(cands):
        return False, "its lines do not match their stamp (edited)", cands
    return True, "matches the SALAD stamp", cands


def load_loop_candidates(path: str, include_posthoc: bool = True) -> List[dict]:
    """loop_closures.txt → [{i, j, sim, source}] — SALAD's candidates, then (``include_posthoc``)
    the post-hoc ones of loop_closures_posthoc.txt when that file's stamp matches (a stale one
    is ignored here, deleted by the fork)."""
    out = _parse_candidates(path)
    if include_posthoc:
        ok, _why, post = posthoc_state(path)
        if ok:
            out += post
    return out


def write_loop_candidates(path: str, cands: List[dict], header: Optional[str] = None) -> None:
    """Persist candidates for the fork's next run. SALAD's (source 'salad') are the fork's own:
    when ``path`` exists they must be exactly the ones it holds — the SERVER never rewrites the
    stamped SALAD file. Every other source goes to loop_closures_posthoc.txt, stamped with the
    SALAD stamp of ``path`` and the digest of its own lines (no post-hoc candidate: the file is
    removed). A ``path`` that does not exist yet is written unstamped (the fork re-runs SALAD
    over an unstamped file)."""
    salad = [c for c in cands if c.get("source", "salad") == "salad"]
    post = [c for c in cands if c.get("source", "salad") != "salad"]
    if os.path.exists(path):
        have = [_candidate_line(c) for c in _parse_candidates(path)
                if c.get("source", "salad") == "salad"]
        want = [_candidate_line(c) for c in salad]
        if sorted(have) != sorted(want):
            raise ValueError(f"{path}: the SALAD candidates are the fork's (stamped, plan point "
                             f"8) — {len(want)} given vs {len(have)} on disk; only post-hoc "
                             f"candidates are written by the server")
    elif salad:
        lines = ["# Loop candidates (index1, index2, similarity, source)"]
        if header:
            lines.append(f"# {header}")
        lines += ["", "# Loop pairs:"] + [_candidate_line(c) for c in salad]
        _write_lines(path, lines)
    pp = posthoc_path(path)
    if not post:
        if os.path.exists(pp):
            os.remove(pp)
        return
    stamp = {"salad_stamp": _salad_digest(path), "candidates_sha256": _candidates_sha256(post),
             "n": len(post)}
    lines = [_stamp_line(stamp), "# Post-hoc loop candidates (index1, index2, similarity, source)"]
    if header:
        lines.append(f"# {header}")
    lines += ["", "# Loop pairs:"] + [_candidate_line(c) for c in post]
    _write_lines(pp, lines)


def reconcile_loop_files(loop_txt: str, now: dict, log=print) -> dict:
    """Hold loop_closures.txt against this run's SALAD stamp ``now``: a file of ANOTHER stamp
    (or none) is deleted with its calibration and the post-hoc file — SALAD runs again. A
    post-hoc file that does not match the SALAD file left standing is deleted too. Returns
    {'salad': 'reused' | 'deleted' | 'absent', 'posthoc': 'kept' | 'deleted' | 'absent',
    'diffs': [...]} (call :func:`reconcile_posthoc` again once a fresh SALAD file is stamped)."""
    from loop_utils.stac_repro import repro
    out = {"salad": "absent", "posthoc": "absent", "diffs": []}
    d = os.path.dirname(os.path.abspath(loop_txt))
    if os.path.exists(loop_txt):
        diffs = repro().check_stamp(read_loop_stamp(loop_txt), now)
        if diffs:
            out.update({"salad": "deleted", "diffs": diffs})
            log(f"[STAC loops] {os.path.basename(loop_txt)} does not match this run "
                f"({'; '.join(diffs[:6])}{' …' if len(diffs) > 6 else ''}) — deleted with "
                f"{SALAD_CALIBRATION_NAME} and {POSTHOC_NAME}; SALAD runs again (plan point 8)")
            for fn in (os.path.basename(loop_txt), SALAD_CALIBRATION_NAME, POSTHOC_NAME):
                p = os.path.join(d, fn)
                if os.path.exists(p):
                    os.remove(p)
        else:
            out["salad"] = "reused"
    out["posthoc"] = reconcile_posthoc(loop_txt, log=log)
    return out


def reconcile_posthoc(loop_txt: str, log=print) -> str:
    """Delete loop_closures_posthoc.txt unless it matches the SALAD file on disk."""
    pp = posthoc_path(loop_txt)
    if not os.path.exists(pp):
        return "absent"
    ok, why, cands = posthoc_state(loop_txt)
    if ok:
        return "kept"
    os.remove(pp)
    log(f"[STAC loops] {POSTHOC_NAME}: {len(cands)} post-hoc candidate(s) deleted — {why} "
        f"(plan point 8)")
    return "deleted"
