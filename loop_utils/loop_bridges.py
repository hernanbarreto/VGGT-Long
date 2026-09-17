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

def robust_sim3(src, dst, iters=8, sample=200000, seed=0, min_points=1000):
    """dst ≈ s·R·src + t from EXACT correspondences, weighted Umeyama with
    IRLS Cauchy weights (the Sim3 sibling of metric_lock.robust_rigid).
    Returns (s, R, t, median_residual_m, n_used) or None when starved."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    m = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    src, dst = src[m], dst[m]
    if len(src) < int(min_points):
        return None
    if len(src) > int(sample):
        idx = np.random.default_rng(seed).choice(len(src), int(sample), replace=False)
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
                          per_frame_cap: int, seed: int = 0):
    """Pixel-to-pixel pairs (p_bridge, q_chunk) over frames present in BOTH
    predictions. bridge_locals[k] and chunk_locals[k] index the SAME global
    frame in each prediction. Pixels valid (conf > 1e-5) in both are used,
    capped per frame so every frame has the same voice."""
    wb, wc = _wp(bridge), _wp(chunk)
    cb, cc = _conf(bridge, wb), _conf(chunk, wc)
    if wb.shape[1:3] != wc.shape[1:3]:
        raise RuntimeError(f"bridge grid {wb.shape[1:3]} != chunk grid "
                           f"{wc.shape[1:3]} — bridge and chunk must share the "
                           f"prediction resolution for exact correspondences")
    rng = np.random.default_rng(seed)
    P, Q = [], []
    for lb, lc in zip(bridge_locals, chunk_locals):
        ok = (cb[lb] > 1e-5) & (cc[lc] > 1e-5)
        idx = np.flatnonzero(ok.reshape(-1))
        if len(idx) == 0:
            continue
        if len(idx) > int(per_frame_cap):
            idx = rng.choice(idx, int(per_frame_cap), replace=False)
        P.append(wb[lb].reshape(-1, 3)[idx])
        Q.append(wc[lc].reshape(-1, 3)[idx])
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return (np.concatenate(P).astype(np.float64),
            np.concatenate(Q).astype(np.float64))


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
                        seed: int = 0) -> Optional[dict]:
    """Fit on half the correspondences, MEASURE on the other half.

    This is what tells a real association from a lucky one, and it replaces
    judging a fit by the SIZE of its residual (USER 2026-09-16: *"podría haber
    una corrección de más de 30 cm y ser perfectamente correcta"*). A fit that
    describes the geometry predicts correspondences it never saw just as well:
    held-out ≈ fit. One that latched onto the wrong structure does not.

    Returns {fit_residual_m, holdout_residual_m, ratio, n_fit, n_held} or None
    when either half starves (the caller keeps the full-sample verdict then).
    """
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    n = len(p)
    if n < 2 * int(min_pts):
        return None
    idx = np.random.default_rng(int(seed) + 9176).permutation(n)
    a, b = idx[: n // 2], idx[n // 2:]
    if rigid:
        f = robust_rigid(p[a], q[a], sample=sample, seed=seed)
        fit = None if f is None else (1.0, f[0], f[1])
        res_fit = None if f is None else float(f[2])
    else:
        f = robust_sim3(p[a], q[a], sample=sample, seed=seed, min_points=min_pts)
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
    for side, chunk, bl, cl in (("a", chunk_a, layout["bridge_a"], layout["chunk_a_local"]),
                                ("b", chunk_b, layout["bridge_b"], layout["chunk_b_local"])):
        p, q = exact_correspondences(bridge, bl, chunk, cl, cap, seed=seed)
        rec = {"n_corr": int(len(p)), "starved": False}
        fit = None
        if len(p) >= min_pts:
            if rigid:
                f = robust_rigid(p, q, sample=sample, seed=seed)
                if f is not None:
                    R_, t_, res, n = f
                    fit = (1.0, R_, t_)
                    rec.update({"s": 1.0, "residual_m": float(res), "n_fit": int(n)})
            else:
                f = robust_sim3(p, q, sample=sample, seed=seed, min_points=min_pts)
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
            sh = split_half_residual(p, q, rigid, sample, min_pts, seed=seed)
            if sh is not None:
                rec["split_half"] = sh
        fits[side] = fit
        out["sides"][side] = rec
    if fits["a"] is not None and fits["b"] is not None:
        s_ab, R_ab, t_ab = compose_ab(fits["a"], fits["b"])
        sh_a = out["sides"]["a"].get("split_half")
        sh_b = out["sides"]["b"].get("split_half")
        out.update({"ok": True, "s_ab": float(s_ab), "R_ab": R_ab.tolist(),
                    "t_ab": t_ab.tolist(),
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


def loop_scale_row(s_ab: float) -> float:
    """log r for the scale graph, r = s_B/s_A in metric_lock's convention
    (s_k = factor that makes chunk k metric). The bridge sees chunk b's units
    as s_ab = u_b/u_a times chunk a's, so the factors relate as s_B/s_A =
    1/s_ab."""
    return float(-np.log(float(s_ab)))


# ── verification (§4.2, steps 1–4; step 0 is the spatial gate, upstream) ──

def verify_loop(meas: dict, loops_cfg: dict, semantic: Optional[dict] = None,
                spatial: Optional[dict] = None,
                reference_m: Optional[float] = None) -> dict:
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
    4. scale: |log s_ab| ≤ scale_tol_log → row + edge; beyond → scale_break;
    5. semantic: when BOTH frames carry SAM3 instances, ≥ min_shared_structural_labels
       structural labels in common (movable labels neither help nor hurt).
    A spatial verdict 'ambiguous' inflates σ by ambiguous_sigma_factor; 'reject'
    never reaches this function (no bridge is spent on it)."""
    fallback_ref = float(cfg_req(loops_cfg, "max_residual_m", "loops"))
    min_corr = int(cfg_req(loops_cfg, "min_correspondences", "loops"))
    tol_log = float(cfg_req(loops_cfg, "scale_tol_log", "loops"))
    sb_factor = float(cfg_req(loops_cfg, "scale_break_sigma_factor", "loops"))
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
    scale_ok = abs(log_s) <= tol_log
    v["checks"]["scale"] = {"log_s_ab": log_s, "scale_tol_log": tol_log, "passed": bool(scale_ok)}
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
    if scale_ok:
        v["status"] = "accepted"
    else:
        v["status"] = "scale_break"
        sigma *= sb_factor
        v["reasons"].append(f"scale break: |log s_ab| {abs(log_s):.4f} > {tol_log:.4f} "
                            f"— edge kept with σ×{sb_factor:g}, seams flagged suspect")
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


def load_loop_candidates(path: str) -> List[dict]:
    """loop_closures.txt → [{i, j, sim, source}]. Lines: `i, j, sim[, source]`;
    `#` lines are comments. Source defaults to 'salad' (the vendor detector)."""
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


def write_loop_candidates(path: str, cands: List[dict], header: Optional[str] = None) -> None:
    lines = ["# Loop candidates (index1, index2, similarity, source)"]
    if header:
        lines.append(f"# {header}")
    lines.append("")
    lines.append("# Loop pairs:")
    for c in cands:
        sim = "" if c.get("sim") is None else f"{float(c['sim']):.4f}"
        lines.append(f"{int(c['i'])}, {int(c['j'])}, {sim}, {c.get('source', 'salad')}")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
