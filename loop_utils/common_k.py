"""ONE calibration for every Ω inference of the session — claude_stac_slam2.txt, phase 2 (idea 1 of
VGGT-SLAM 2.0, MIT-SPARK arXiv 2601.19887: two submaps are stitched only when the two inferences
of the shared frame agree in pose AND calibration).

``Model.seams.focal_model: common_K`` re-unprojects every chunk (right after the metric lock: the
zoom detector keeps reading Ω's own focals; before the bridges are planned and the seams fitted)
and every bridge (after its lock, before the loops are measured) with ONE session K: the depth per
pixel does not change, the rays do, c2w does not; ``world_points`` and ``intrinsic`` of the npy
change together; the provenance (frame, pixel) is invariant. The K comes from
``common_K_source``: ``session_median`` (the median fx, fy, cx, cy of Ω over the frames by owner,
zoom chunks excluded) or ``intake_probe`` (the K of intake.focal, handed over by the map worker on
the Ω grid as ``Model.seams.intake_probe_K``; absent for the session → declared not run).

THE JUDGEMENT (the user's rule, metric_lock.decide_change, with the run's own graph parameters):
the correction is applied only when (1) the held-out residual of the seams improves by the rule —
the same exact correspondences exact_seam_align fits, split by stable key into a fit half and a
held-out half, the judges being the shared frames, pooled over every seam — AND (2) the surface
pairs held out INSIDE the chunks (metric_lock.surface_pair_correspondences at _stac_holdout_pairs'
offsets) do not worsen by the MIRROR of the rule. Otherwise: identity, declared. Everything lands in
``focal_model.json`` (stamped, resume-safe: the same stamp decides exactly as before). A chunk the
metric lock declared ZOOMED keeps its own K, declared.

Every statistic is computed on the CPU in float64 from the arrays; the re-unprojection itself is
float64 and stored in the arrays' own dtype.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from loop_utils.metric_lock import frame_owner

MARK_KEY = "_stac_common_K"
REPORT_NAME = "focal_model.json"
MAD_TO_SIGMA = 1.4826


class CommonKError(RuntimeError):
    """The common K cannot be formed or judged — with the exact reason."""


# ── geometry ──────────────────────────────────────────────────────────────────────────────

def _wp(d: dict) -> np.ndarray:
    wp = np.asarray(d["world_points"])
    return wp[0] if wp.ndim == 5 else wp


def _ext(d: dict) -> np.ndarray:
    ext = np.asarray(d["extrinsic"], np.float64)
    return ext[0] if ext.ndim == 4 else ext


def _K(d: dict) -> np.ndarray:
    K = np.asarray(d["intrinsic"], np.float64)
    return K[0] if K.ndim == 4 else K


def _depth(d: dict) -> np.ndarray:
    dep = np.asarray(d["depth"])
    return dep[0] if dep.ndim == 4 else dep


def unproject_chunk(d: dict, K: Optional[np.ndarray] = None, frames: Optional[Sequence[int]] = None
                    ) -> np.ndarray:
    """World points of a chunk (or bridge) from its depth and c2w with ``K`` (one 3x3 for every
    frame; None = each frame's own K): the adapter's unprojection — x = (u − cx)·d/fx,
    y = (v − cy)·d/fy, z = d; world = cam·Rᵀ + t — in float64. ``frames``: local indices to
    compute (default all)."""
    dep = _depth(d).astype(np.float64)
    ext = _ext(d)
    Ks = _K(d)
    S, H, W = dep.shape
    vv, uu = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    idx = range(S) if frames is None else [int(i) for i in frames]
    out = np.zeros((len(idx), H, W, 3), np.float64)
    for j, i in enumerate(idx):
        Ki = Ks[i] if K is None else np.asarray(K, np.float64)
        fx, fy, cx, cy = Ki[0, 0], Ki[1, 1], Ki[0, 2], Ki[1, 2]
        z = dep[i]
        cam = np.stack([(uu - cx) * z / fx, (vv - cy) * z / fy, z], -1)
        R, t = ext[i, :3, :3], ext[i, :3, 3]
        out[j] = cam @ R.T + t
    return out


def reunproject_in_place(d: dict, K: np.ndarray, frames: Optional[Sequence[int]] = None, *,
                         source: str, mark: bool = True) -> dict:
    """``d``'s world_points recomputed with ``K`` for ``frames`` (local indices; None = all) in the
    array's own dtype, its intrinsic rows set to ``K`` (own dtype), the mark recorded. The depth,
    the extrinsics, the confidences and the images are untouched — the provenance (frame, pixel)
    is invariant by construction (every pixel keeps its slot)."""
    K = np.asarray(K, np.float64)
    wp0 = np.asarray(d["world_points"])
    lead = wp0.ndim == 5
    wp = wp0[0] if lead else wp0
    S = wp.shape[0]
    idx = list(range(S)) if frames is None else [int(i) for i in frames]
    new = unproject_chunk(d, K, frames=idx)
    wp = wp.copy()
    for j, i in enumerate(idx):
        wp[i] = new[j].astype(wp.dtype)
    d["world_points"] = wp[None] if lead else wp
    K0 = np.asarray(d["intrinsic"])
    leadK = K0.ndim == 4
    Ks = (K0[0] if leadK else K0).copy()
    for i in idx:
        Ks[i] = K.astype(Ks.dtype)
    d["intrinsic"] = Ks[None] if leadK else Ks
    if mark:
        d[MARK_KEY] = {"applied": True, "K": K.tolist(), "source": str(source),
                       "frames": (None if frames is None else [int(i) for i in idx])}
    return d


def session_median_K(chunk_Ks: Dict[int, np.ndarray], layout: Sequence[Tuple[int, int]], n_frames: int,
                     exclude: Sequence[int] = ()) -> Dict[str, Any]:
    """ONE session K: the median fx, fy, cx, cy of Ω over the frames by OWNER (frame_owner — every
    frame counted once, from the chunk that writes it), the chunks in ``exclude`` (zoomed) left out."""
    owner = frame_owner(layout, n_frames)
    ex = {int(k) for k in exclude}
    rows = []
    for k, (a, b) in enumerate(layout):
        if k in ex or k not in chunk_Ks:
            continue
        Ks = np.asarray(chunk_Ks[k], np.float64)
        Ks = Ks[0] if Ks.ndim == 4 else Ks
        for local, g in enumerate(range(a, b)):
            if owner[g] == k:
                Ki = Ks[local]
                rows.append([Ki[0, 0], Ki[1, 1], Ki[0, 2], Ki[1, 2]])
    rows = np.asarray(rows, np.float64)
    if rows.size == 0:
        raise CommonKError("no owned frame of a non-zoom chunk carries intrinsics — the session median K "
                           "cannot be formed")
    med = np.median(rows, axis=0)
    K = np.array([[med[0], 0.0, med[2]], [0.0, med[1], med[3]], [0.0, 0.0, 1.0]])
    spread = MAD_TO_SIGMA * np.median(np.abs(rows - med), axis=0)
    return {"K": K, "fx": float(med[0]), "fy": float(med[1]), "cx": float(med[2]), "cy": float(med[3]),
            "n_frames": int(len(rows)), "chunks_excluded": sorted(ex),
            "spread_sigma_px": {n: float(v) for n, v in zip(("fx", "fy", "cx", "cy"), spread)}}


# ── the judgement ─────────────────────────────────────────────────────────────────────────

def judge_common_K(layout: Sequence[Tuple[int, int]], load_chunk: Callable[[int], dict], K: np.ndarray, *,
                   params: Dict[str, Any], zoom_chunks: Sequence[int], judge_samples: int,
                   n_frames: int, log: Callable = print) -> Dict[str, Any]:
    """Before / after of every seam's held-out residual and of every chunk's held-out pairs under
    ``K`` (zoom chunks keep their own K on both sides), judged pooled by the user's rule and its
    mirror. Chunks are loaded two at a time through ``load_chunk``; nothing is written."""
    from loop_utils import seam_radial as SR
    from loop_utils.stable_sample import stable_pick
    zoom = {int(k) for k in zoom_chunks}
    owner = frame_owner(layout, n_frames)
    factor = float(params["improvement_error_factor"])
    conf = float(params["heldout_confidence"])
    seams_rec: List[Dict[str, Any]] = []
    before_all: List[np.ndarray] = []
    after_all: List[np.ndarray] = []
    cl_all: List[np.ndarray] = []
    pairs_before: Dict[Tuple[int, int], Dict[str, Any]] = {}
    pairs_after: Dict[Tuple[int, int], Dict[str, Any]] = {}
    pairs_by_chunk: Dict[int, Tuple[dict, dict]] = {}
    n_chunks = len(layout)
    cache: Dict[int, dict] = {}
    after_wp: Dict[int, np.ndarray] = {}

    def _chunk(k):
        if k not in cache:
            cache[k] = load_chunk(k)
        return cache[k]

    def _after_points(k):
        """The chunk's world points under the common K — its own when zoomed."""
        if k not in after_wp:
            d = _chunk(k)
            after_wp[k] = (_wp(d).astype(np.float64) if k in zoom else unproject_chunk(d, K))
        return after_wp[k]

    for k in range(n_chunks):
        d = _chunk(k)
        pb = SR.chunk_holdout_pairs(d, layout[k], owner, params)
        if k in zoom:
            pa = pb                                   # its own K on both states: unchanged
        else:
            pa = SR.chunk_holdout_pairs(d, layout[k], owner, params, wp_override=_after_points(k), K_override=K)
        pairs_before.update(pb)
        pairs_after.update(pa)
        pairs_by_chunk[k] = (pb, pa)
        if k >= 1:
            d1, d2 = _chunk(k - 1), d
            corr = SR.seam_correspondences(layout, k - 1, d1, d2)
            rec = {"seam": k - 1, "chunks": [k - 1, k], "shared_frames": corr["ov"], "n_valid": corr["n_valid"],
                   "zoom_side": [j for j in (k - 1, k) if j in zoom]}
            if corr["n_valid"] < 2000:
                rec["starved"] = True
                seams_rec.append(rec)
            elif k - 1 in zoom and k in zoom:
                rec["skipped"] = "both chunks zoomed: nothing changes on this seam"
                seams_rec.append(rec)
            else:
                fit_idx, held_idx = SR.split_fit_heldout(corr["keys"])
                ov = corr["ov"]
                P1 = SR._points_at(_wp(d1)[-ov:], corr["flat"])
                P2 = SR._points_at(_wp(d2)[:ov], corr["flat"])
                P1c = SR._points_at(_after_points(k - 1)[-ov:], corr["flat"])
                P2c = SR._points_at(_after_points(k)[:ov], corr["flat"])
                fb = SR.fit_seam(P1, P2, corr["keys"], fit_idx, held_idx)
                fa = SR.fit_seam(P1c, P2c, corr["keys"], fit_idx, held_idx)
                if fb.get("starved") or fa.get("starved"):
                    rec["starved"] = True
                else:
                    nb = np.linalg.norm(fb["held_residual"], axis=1)
                    na = np.linalg.norm(fa["held_residual"], axis=1)
                    judge_idx = held_idx
                    if judge_samples and len(held_idx) > int(judge_samples):
                        judge_idx = held_idx[stable_pick(corr["keys"][held_idx], int(judge_samples),
                                                         salt=SR.SALT_JUDGE)]
                    pos = np.searchsorted(held_idx, judge_idx)
                    frames_g = int(layout[k][0]) + corr["frame_local"][judge_idx]   # global shared frame
                    before_all.append(nb[pos]); after_all.append(na[pos]); cl_all.append(frames_g)
                    rec.update({"n_fit": int(len(fit_idx)), "n_held": int(len(held_idx)), "n_judged": int(len(judge_idx)),
                                "held_median_before_m": float(np.median(nb)), "held_median_after_m": float(np.median(na)),
                                "fit_median_before_m": fb["fit_median_m"], "fit_median_after_m": fa["fit_median_m"]})
                seams_rec.append(rec)
            cache.pop(k - 1, None)
            after_wp.pop(k - 1, None)
    report: Dict[str, Any] = {"seams": seams_rec}
    if before_all:
        b = np.concatenate(before_all); a = np.concatenate(after_all); cl = np.concatenate(cl_all)
        err = SR.judge_error_per_judge(b, cl)
        rule = SR.rule_verdict(b, a, cl, error_m=err["error_m"], factor=factor, confidence=conf)
        rule["error_source"] = ("median over the shared frames (the judges) of each frame's own standard error "
                                "of its median held-out residual (before)")
        rule["n_judges_frames"] = err["n_judges"]
        report["seams_rule"] = rule
    else:
        report["seams_rule"] = {"verdict": None, "reason": "no seam could be judged (starved or both sides zoomed)"}
    # the pairs are judged CHUNK BY CHUNK (the K is applied chunk by chunk, and a pooled median
    # over the session hides one chunk whose own geometry the common K breaks — a zoom the
    # detector did not flag); the pooled numbers are reported beside them
    pooled = SR.pairs_before_after(pairs_before, pairs_after, factor=factor, confidence=conf)
    pooled.pop("pairs", None)
    by_chunk = {}
    for k, (pb, pa) in sorted(pairs_by_chunk.items()):
        rec_k = SR.pairs_before_after(pb, pa, factor=factor, confidence=conf)
        rec_k.pop("pairs", None)
        rec_k["zoomed"] = k in zoom
        by_chunk[str(k)] = rec_k
    worsening_chunks = [int(k) for k, v in by_chunk.items() if v.get("worsens_by_rule")]
    pooled["by_chunk"] = by_chunk
    pooled["worsening_chunks"] = worsening_chunks
    pooled["worsens_by_rule"] = bool(worsening_chunks)
    pooled["rule"] = ("a chunk's held-out pairs worsen significantly, with enough judges, beyond their own "
                      "error (the mirror of decide_change), judged chunk by chunk; pooled numbers for the report")
    if worsening_chunks:
        pooled["reason"] = "chunk(s) " + ", ".join(str(k) for k in worsening_chunks) + ": " + "; ".join(
            f"{k}: {by_chunk[str(k)].get('reason', '')}" for k in worsening_chunks)
    report["pairs_rule"] = pooled
    improves = bool(report["seams_rule"].get("improves_by_rule", False))
    worsens = bool(report["pairs_rule"].get("worsens_by_rule", False))
    pairs_judged = int(report["pairs_rule"].get("n_pairs_judged", 0) or 0)
    report["applied"] = bool(improves and not worsens)
    if improves and not worsens:
        report["reason"] = ("the seams' held-out residual improves by the rule and the chunks' held-out pairs "
                            "do not worsen by its mirror" + ("" if pairs_judged else " (no pair could be judged)"))
    elif not improves:
        report["reason"] = "the seams' held-out residual does not improve by the rule: " + str(
            report["seams_rule"].get("reason", ""))
    else:
        report["reason"] = "the chunks' held-out pairs worsen by the mirror of the rule: " + str(
            report["pairs_rule"].get("reason", ""))
    return report


def judge_and_decide(layout: Sequence[Tuple[int, int]], load_chunk: Callable[[int], dict], *, seams_cfg: Dict[str, Any],
                     params: Dict[str, Any], zoom_chunks: Sequence[int], n_frames: int,
                     log: Callable = print) -> Dict[str, Any]:
    """The whole decision of ``_stac_common_K_chunks``: the K from the configured source, the
    judgement, the verdict. Returns the focal_model.json document (without the stamp)."""
    zoom = sorted({int(k) for k in zoom_chunks})
    layout = [(int(a), int(b)) for a, b in layout]
    doc: Dict[str, Any] = {"version": 1, "focal_model": str(seams_cfg["focal_model"]),
                           "source": str(seams_cfg["common_K_source"]),
                           "judge_samples": int(seams_cfg["judge_samples"]),
                           "chunk_indices": [list(x) for x in layout], "chunks_excluded_zoom": zoom,
                           "params": {k: params[k] for k in ("heldout_confidence", "improvement_error_factor",
                                                             "holdout_offsets", "holdout_stride", "holdout_samples",
                                                             "pose_fit_conf_min_norm")}}
    if doc["source"] == "session_median":
        Ks = {}
        for k in range(len(layout)):
            if k in zoom:
                continue
            Ks[k] = _K(load_chunk(k))
        sm = session_median_K(Ks, layout, n_frames, exclude=zoom)
        K = sm["K"]
        doc["K_source_detail"] = {k_: v for k_, v in sm.items() if k_ != "K"}
    else:
        probe = seams_cfg.get("intake_probe_K")
        if probe is None:
            doc.update({"run": False, "applied": False, "K": None,
                        "reason": "intake probe K not available for this session (the map worker found no "
                                  "intake/focal_probe.json) — the variant is recorded as not run"})
            log(f"[common-K] {doc['reason']}")
            return doc
        K = np.asarray(probe, np.float64).reshape(3, 3)
        doc["K_source_detail"] = seams_cfg.get("intake_probe") or {}
    doc["run"] = True
    doc["K"] = K.tolist()
    log(f"[common-K] candidate K from {doc['source']}: fx {K[0, 0]:.2f} fy {K[1, 1]:.2f} cx {K[0, 2]:.2f} "
        f"cy {K[1, 2]:.2f}" + (f"; zoom chunks keep their own K: {zoom}" if zoom else ""))
    rep = judge_common_K(layout, load_chunk, K, params=params, zoom_chunks=zoom,
                         judge_samples=int(seams_cfg["judge_samples"]), n_frames=n_frames, log=log)
    doc.update(rep)
    sr = rep["seams_rule"]
    pr = rep["pairs_rule"]
    log(f"[common-K] seams held-out median {sr.get('median_before_m')} → {sr.get('median_after_m')} m "
        f"({sr.get('n_judges_frames', 0)} judge frames; improves by the rule: {sr.get('improves_by_rule')}); "
        f"intra-chunk pairs {pr.get('median_before_m')} → {pr.get('median_after_m')} m "
        f"({pr.get('n_pairs_judged', 0)} pairs; worsen by the mirror: {pr.get('worsens_by_rule')}) → "
        f"{'APPLIED' if doc['applied'] else 'IDENTITY'}: {doc['reason']}")
    return doc


def bridge_frames_to_reunproject(layout_rec: Dict[str, Any], zoom_chunks: Sequence[int]) -> List[int]:
    """The bridge positions whose parent chunk is not zoomed: window a (+ its extra frames) belong
    to chunk_a, window b (+ its extras) to chunk_b."""
    zoom = {int(k) for k in zoom_chunks}
    na = len(layout_rec["bridge_a"])
    nb = len(layout_rec["bridge_b"])
    n_ex_a = int(layout_rec.get("n_extra_a", 0))
    n_ex_b = int(layout_rec.get("n_extra_b", 0))
    out: List[int] = []
    if int(layout_rec["chunk_a"]) not in zoom:
        out += list(range(0, na + n_ex_a))
    if int(layout_rec["chunk_b"]) not in zoom:
        off = na + n_ex_a
        out += list(range(off, off + nb + n_ex_b))
    return out
