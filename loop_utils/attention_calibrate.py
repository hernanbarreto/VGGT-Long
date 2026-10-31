"""Offline calibration of the attention loop verifier — claude_stac_slam2.txt, phase 3.

    python -m loop_utils.attention_calibrate --run <maplong_run> [--labels <json>] [--params <json>]
                                             [--out <dir>] [--no-salad-grid] [--max-pairs N]

GPU: one Ω pass of TWO frames per labelled pair, capturing every GLOBAL inter-frame block at once
(loop_utils.attention_gate), the model loaded from the run's own fork config. Never beside another
GPU process (repro.require_exclusive_gpu).

LABELS of a finished run — or declared impossible, then only the manual ones (``--labels``, the
same schema {"positives": [[i, j], …], "negatives": [[i, j], …]} in keyframe indices):
- positives: the candidates whose bridge the verification ACCEPTED (loop_edges.json, stage
  verification, status accepted) AND that survived the cycle consistency of the keyframe pose
  graph (pose_graph.json → cycle_consistency.groups[*].kept);
- negatives: the bridges the verification REJECTED, plus keyframe pairs outside the non-local
  band (the detector's own band: Loop.SALAD.min_gap / min_gap_frac / one bridge window) whose
  frusta do not overlap under the FINAL poses (camera_poses.txt, intrinsic.txt, the spatial
  gate's depth range), a stable pick of ``n_negatives`` of them.
Per global block: the AUC (rank statistic, ties half) with a bootstrap interval (positives and
negatives resampled independently, seeded), the precision–recall curve, n positives / negatives.
THE CHOICE, by the declared rule (calibration.select max_auc_lower_ci): the layer with the highest
lower bound of its AUC interval, and the lowest threshold reaching calibration.target_precision on
the labels; with fewer than calibration.min_labels positives or negatives nothing is chosen —
declared. For every SALAD similarity of calibration.salad_grid: how many candidates the detector's
own rule proposes at that bar (its descriptors recomputed on the card) and how many of them pass
the chosen layer and threshold.

Outputs in <out> (default <run>/attention_calibration/): attention_calibration.json (the choice,
every layer's numbers, the labels and their sources, the grid), per_layer.csv, roc_pr.png. All
statistics in float64 on the CPU from the captured tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from loop_utils import attention_gate as AG
from loop_utils.stable_sample import pixel_keys, stable_pick

LOG_TAG = "[attention-calib]"
PARAM_KEYS = ("select", "target_precision", "min_labels", "n_negatives", "n_boot", "confidence", "salad_grid")
SELECT_RULES = ("max_auc_lower_ci",)


class CalibrationError(RuntimeError):
    """The calibration cannot run as asked — with the exact reason."""


# ── the run ───────────────────────────────────────────────────────────────────────────────

def read_run(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise CalibrationError(f"run directory is missing: {run_dir}")
    cfg_path = run_dir / "vggt_omega_config.yaml"
    if not cfg_path.is_file():
        raise CalibrationError(f"vggt_omega_config.yaml is missing: {cfg_path}")
    import yaml
    with open(cfg_path) as f:
        config = yaml.safe_load(f) or {}
    fl = run_dir / "frame_list.json"
    if not fl.is_file():
        raise CalibrationError(f"frame_list.json is missing: {fl} — the run's keyframe list is unknown")
    frames = json.loads(fl.read_text())
    if not isinstance(frames, list) or not frames:
        raise CalibrationError(f"{fl} holds no keyframe list")
    session = run_dir.resolve().parent.parent
    frames_dir = session / "frames"
    paths = [str(frames_dir / os.path.basename(str(p))) for p in frames]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise CalibrationError(f"{len(missing)} keyframe image(s) of frame_list.json are not under {frames_dir} "
                               f"(first: {missing[0]})")
    stamp = None
    sp = run_dir / "fork_stamp.json"
    if sp.is_file():
        try:
            stamp = json.loads(sp.read_text()).get("sha256")
        except ValueError:
            stamp = None
    layout = None
    cs = run_dir / "chunk_sim3.json"
    if cs.is_file():
        try:
            layout = [(int(a), int(b)) for a, b in json.loads(cs.read_text()).get("chunk_indices") or []]
        except (ValueError, TypeError):
            layout = None
    return {"run_dir": str(run_dir), "session_dir": str(session), "config": config, "frames": [str(x) for x in frames],
            "image_paths": paths, "n_frames": len(paths), "fork_stamp": stamp, "chunk_indices": layout}


def calibration_params(config: dict, params_path: Optional[Path]) -> Dict[str, Any]:
    """The calibration's parameters: the run's Model.loops.attention_gate (calibration block +
    top_fraction) when the run carried the section, else the JSON at ``params_path`` with the same
    keys; a missing key is refused by name — never a default."""
    gate = ((config.get("Model") or {}).get("loops") or {}).get("attention_gate")
    src = None
    doc: Dict[str, Any] = {}
    if isinstance(gate, dict) and isinstance(gate.get("calibration"), dict):
        doc = dict(gate["calibration"])
        doc["top_fraction"] = gate.get("top_fraction")
        src = "the run's Model.loops.attention_gate"
    if params_path is not None:
        p = Path(params_path)
        if not p.is_file():
            raise CalibrationError(f"--params {p} does not exist")
        ext = json.loads(p.read_text())
        if not isinstance(ext, dict):
            raise CalibrationError(f"--params {p} must hold a JSON object")
        doc.update(ext)
        src = f"{src} + {p}" if src else str(p)
    missing = [k for k in PARAM_KEYS + ("top_fraction",) if doc.get(k) is None]
    if missing:
        raise CalibrationError(f"calibration parameter(s) missing: {missing} — the run carries no "
                               f"Model.loops.attention_gate section; pass --params <json> with "
                               f"{list(PARAM_KEYS) + ['top_fraction']}")
    if doc["select"] not in SELECT_RULES:
        raise CalibrationError(f"calibration.select must be one of {SELECT_RULES}, got {doc['select']!r}")
    out = {"select": str(doc["select"]), "target_precision": float(doc["target_precision"]),
           "min_labels": int(doc["min_labels"]), "n_negatives": int(doc["n_negatives"]),
           "n_boot": int(doc["n_boot"]), "confidence": float(doc["confidence"]),
           "salad_grid": [float(x) for x in doc["salad_grid"]], "top_fraction": float(doc["top_fraction"]),
           "source": src}
    if not (0.0 < out["target_precision"] <= 1.0) or not (0.5 < out["confidence"] < 1.0) or out["min_labels"] < 1:
        raise CalibrationError("calibration parameters out of range (target_precision in (0,1], confidence in "
                               "(0.5,1), min_labels >= 1)")
    return out


# ── labels ────────────────────────────────────────────────────────────────────────────────

def _unordered(i: int, j: int) -> Tuple[int, int]:
    return (int(min(i, j)), int(max(i, j)))


def run_labels(run_dir: Path) -> Dict[str, Any]:
    """Positives and negatives of a finished run from loop_edges.json and pose_graph.json, or the
    declaration that they cannot be read unambiguously. Returns {'positives', 'negatives_rejected',
    'available': bool, 'reason', 'detail'} — pairs as unordered (i, j) keyframe indices."""
    run_dir = Path(run_dir)
    le = run_dir / "loop_edges.json"
    pg = run_dir / "pose_graph.json"
    out: Dict[str, Any] = {"positives": [], "negatives_rejected": [], "available": False, "reason": None,
                           "detail": {}}
    if not le.is_file():
        out["reason"] = f"{le.name} is missing — the run's bridges and their verdicts are unknown"
        return out
    try:
        edges = json.loads(le.read_text()).get("edges") or []
    except ValueError as e:
        out["reason"] = f"{le.name} is not readable JSON ({e})"
        return out
    ver = [e for e in edges if e.get("stage") == "verification"]
    accepted, rejected = {}, []
    for e in ver:
        c = e.get("candidate") or {}
        if c.get("i") is None or c.get("j") is None:
            continue
        pair = _unordered(c["i"], c["j"])
        if e.get("status") == "accepted":
            accepted[int(e.get("bridge", -1))] = pair
        elif e.get("status") == "rejected":
            rejected.append(pair)
    out["negatives_rejected"] = sorted(set(rejected))
    out["detail"]["verification_edges"] = len(ver)
    out["detail"]["accepted_bridges"] = sorted(accepted)
    if not pg.is_file():
        out["reason"] = (f"{pg.name} is missing — which accepted closures survived the cycle consistency "
                         f"cannot be known; positives undefined")
        return out
    try:
        pgd = json.loads(pg.read_text())
    except ValueError as e:
        out["reason"] = f"{pg.name} is not readable JSON ({e})"
        return out
    cyc = pgd.get("cycle_consistency")
    if not isinstance(cyc, dict) or not isinstance(cyc.get("groups"), list):
        out["reason"] = (f"{pg.name} carries no cycle_consistency record (verdict {pgd.get('verdict')!r}: "
                         f"{pgd.get('reason', '')}) — positives undefined")
        return out
    kept = set()
    for g in cyc["groups"]:
        for b in g.get("kept") or []:
            kept.add(int(b))
    positives = sorted({accepted[b] for b in accepted if b in kept})
    out["positives"] = positives
    out["available"] = True
    out["detail"]["cycle_kept_bridges"] = sorted(kept)
    out["detail"]["pose_graph_verdict"] = pgd.get("verdict")
    return out


def read_final_cameras(run_dir: Path, n_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """The run's final per-keyframe c2w (camera_poses.txt, 16 values per row) and K rows
    (intrinsic.txt: fx fy cx cy) on the Ω grid."""
    run_dir = Path(run_dir)
    pp, ip = run_dir / "camera_poses.txt", run_dir / "intrinsic.txt"
    if not pp.is_file() or not ip.is_file():
        raise CalibrationError(f"{pp.name} / {ip.name} missing under {run_dir} — the final poses are needed "
                               f"for the frustum-disjoint negatives")
    poses = np.loadtxt(str(pp), dtype=np.float64, ndmin=2)
    if poses.shape != (n_frames, 16):
        raise CalibrationError(f"{pp} holds {poses.shape}, expected ({n_frames}, 16)")
    Ks = np.loadtxt(str(ip), dtype=np.float64, ndmin=2)
    if Ks.shape[0] != n_frames or Ks.shape[1] < 4:
        raise CalibrationError(f"{ip} holds {Ks.shape}, expected ({n_frames}, >=4)")
    return poses.reshape(n_frames, 4, 4), Ks[:, :4]


def frustum_disjoint_pairs(poses: np.ndarray, K_rows: np.ndarray, grid_hw: Tuple[int, int], *, min_gap: int,
                           depth_range: Tuple[float, float], n_pick: int, grid_pts: int = 3, n_depths: int = 3
                           ) -> Dict[str, Any]:
    """Non-local keyframe pairs (|i − j| >= min_gap) whose frusta do NOT overlap under the final
    poses: a lattice of ``grid_pts``² pixels at ``n_depths`` depths in [min, max] of each frame,
    carried to the world and projected into the other frame — a pair is disjoint when no sample of
    either lands inside the other's image in front of it within the depth range. A stable pick of
    ``n_pick`` of them by pair key."""
    N = poses.shape[0]
    H, W = int(grid_hw[0]), int(grid_hw[1])
    zmin, zmax = float(depth_range[0]), float(depth_range[1])
    us = np.linspace(0.5, W - 0.5, int(grid_pts))
    vs = np.linspace(0.5, H - 0.5, int(grid_pts))
    zs = np.linspace(zmin, zmax, int(n_depths))
    uu, vv, zz = np.meshgrid(us, vs, zs, indexing="ij")
    pts_world = np.zeros((N, uu.size, 3))
    for i in range(N):
        fx, fy, cx, cy = K_rows[i]
        cam = np.stack([(uu.ravel() - cx) * zz.ravel() / fx, (vv.ravel() - cy) * zz.ravel() / fy, zz.ravel()], 1)
        pts_world[i] = cam @ poses[i, :3, :3].T + poses[i, :3, 3]
    w2c = np.linalg.inv(poses)
    visible = np.zeros((N, N), bool)                    # visible[i, j]: some sample of i seen by j
    for j in range(N):
        fx, fy, cx, cy = K_rows[j]
        P = pts_world.reshape(-1, 3) @ w2c[j, :3, :3].T + w2c[j, :3, 3]
        z = P[:, 2]
        ok = (z > zmin) & (z < zmax)
        u = np.where(ok, P[:, 0] / np.where(ok, z, 1.0) * fx + cx, -1.0)
        v = np.where(ok, P[:, 1] / np.where(ok, z, 1.0) * fy + cy, -1.0)
        inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        visible[:, j] = inb.reshape(N, -1).any(axis=1)
    overlap = visible | visible.T
    idx = np.arange(N)
    nonlocal_ = np.abs(idx[:, None] - idx[None, :]) >= int(min_gap)
    disjoint = nonlocal_ & ~overlap & (idx[:, None] < idx[None, :])
    ii, jj = np.nonzero(disjoint)
    keys = pixel_keys(ii.astype(np.int64), jj.astype(np.int64)) if ii.size else np.zeros(0, np.uint64)
    pick = stable_pick(keys, int(n_pick)) if ii.size else np.zeros(0, np.int64)
    pairs = [(int(ii[k]), int(jj[k])) for k in pick]
    return {"pairs": pairs, "n_disjoint": int(ii.size), "n_nonlocal": int((nonlocal_ & (idx[:, None] < idx[None, :])).sum()),
            "min_gap": int(min_gap), "depth_range_m": [zmin, zmax], "samples_per_frame": int(uu.size)}


def manual_labels(path: Optional[Path]) -> Dict[str, List[Tuple[int, int]]]:
    if path is None:
        return {"positives": [], "negatives": []}
    p = Path(path)
    if not p.is_file():
        raise CalibrationError(f"--labels {p} does not exist")
    doc = json.loads(p.read_text())
    if not isinstance(doc, dict):
        raise CalibrationError(f"--labels {p} must hold {{'positives': [[i, j], …], 'negatives': [[i, j], …]}}")
    out = {}
    for key in ("positives", "negatives"):
        lst = doc.get(key) or []
        pairs = []
        for x in lst:
            if not (isinstance(x, (list, tuple)) and len(x) == 2):
                raise CalibrationError(f"--labels {p}: {key} entries must be [i, j] pairs")
            pairs.append(_unordered(int(x[0]), int(x[1])))
        out[key] = sorted(set(pairs))
    return out


# ── statistics ────────────────────────────────────────────────────────────────────────────

def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) + ½ P(tie) — the rank statistic, exact."""
    pos = np.asarray(pos, np.float64)
    neg = np.asarray(neg, np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def auc_with_ci(pos: np.ndarray, neg: np.ndarray, *, n_boot: int, confidence: float, seed: int = 0) -> Dict[str, Any]:
    pos = np.asarray(pos, np.float64)
    neg = np.asarray(neg, np.float64)
    a = auc(pos, neg)
    rec = {"auc": a, "n_pos": int(pos.size), "n_neg": int(neg.size), "n_boot": int(n_boot), "confidence": float(confidence)}
    if pos.size == 0 or neg.size == 0:
        rec.update({"ci_low": None, "ci_high": None})
        return rec
    rng = np.random.default_rng(int(seed))
    bs = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        bs[b] = auc(pos[rng.integers(0, pos.size, pos.size)], neg[rng.integers(0, neg.size, neg.size)])
    alpha = (1.0 - float(confidence)) / 2.0
    rec["ci_low"] = float(np.percentile(bs, 100.0 * alpha))
    rec["ci_high"] = float(np.percentile(bs, 100.0 * (1.0 - alpha)))
    return rec


def pr_curve(pos: np.ndarray, neg: np.ndarray) -> Dict[str, List[float]]:
    """Precision and recall at every distinct score taken as the threshold (score >= t passes)."""
    pos = np.asarray(pos, np.float64)
    neg = np.asarray(neg, np.float64)
    ts = np.unique(np.concatenate([pos, neg]))[::-1]
    prec, rec = [], []
    for t in ts:
        tp = int((pos >= t).sum())
        fp = int((neg >= t).sum())
        prec.append(tp / max(tp + fp, 1))
        rec.append(tp / max(pos.size, 1))
    return {"thresholds": [float(t) for t in ts], "precision": [float(p) for p in prec], "recall": [float(r) for r in rec]}


def threshold_for_precision(pos: np.ndarray, neg: np.ndarray, target: float) -> Optional[Dict[str, Any]]:
    """The LOWEST threshold (among the distinct scores) whose precision on the labels reaches
    ``target``; None when none does."""
    curve = pr_curve(pos, neg)
    best = None
    for t, p, r in zip(curve["thresholds"], curve["precision"], curve["recall"]):
        if p >= float(target):
            best = {"threshold": float(t), "precision": float(p), "recall": float(r)}
    return best


def choose(per_layer: Dict[int, Dict[str, Any]], *, select: str, target_precision: float, min_labels: int
           ) -> Dict[str, Any]:
    """The declared rule over the per-layer records ({layer: {'auc': {...}, 'pos': [...], 'neg': [...]}})."""
    if select not in SELECT_RULES:
        raise CalibrationError(f"select must be one of {SELECT_RULES}")
    eligible = {L: r for L, r in per_layer.items()
                if r["auc"]["n_pos"] >= int(min_labels) and r["auc"]["n_neg"] >= int(min_labels)
                and r["auc"].get("ci_low") is not None}
    if not eligible:
        n_pos = max((r["auc"]["n_pos"] for r in per_layer.values()), default=0)
        n_neg = max((r["auc"]["n_neg"] for r in per_layer.values()), default=0)
        return {"chosen": False, "reason": f"insufficient labels: {n_pos} positive(s), {n_neg} negative(s), "
                                           f"{int(min_labels)} of each needed", "rule": select}
    # ties on the lower bound (several layers separating the labels perfectly) are broken by a
    # DECLARED order: the AUC point estimate, then the separation of the medians, then the lower index
    def _sep(rec):
        p_, q_ = np.asarray(rec["pos"], np.float64), np.asarray(rec["neg"], np.float64)
        return float(np.median(p_) - np.median(q_)) if p_.size and q_.size else -np.inf
    L = min(eligible, key=lambda k: (-eligible[k]["auc"]["ci_low"], -eligible[k]["auc"]["auc"], -_sep(eligible[k]), k))
    r = eligible[L]
    th = threshold_for_precision(np.asarray(r["pos"]), np.asarray(r["neg"]), target_precision)
    if th is None:
        return {"chosen": False, "layer": int(L), "reason": f"layer {L} (AUC lower bound {r['auc']['ci_low']:.3f}) "
                                                             f"reaches no threshold with precision >= {target_precision:g} "
                                                             f"on the labels", "rule": select}
    return {"chosen": True, "layer": int(L), "threshold": th["threshold"], "precision": th["precision"],
            "recall": th["recall"], "auc": r["auc"], "n_pos": r["auc"]["n_pos"], "n_neg": r["auc"]["n_neg"],
            "median_separation": _sep(r),
            "rule": f"{select}: the layer with the highest lower bound of its AUC interval (ties: the AUC, then the "
                    f"separation of the medians, then the lower index), the lowest threshold reaching precision "
                    f"{target_precision:g} on the labels"}


# ── the scorer (GPU) ──────────────────────────────────────────────────────────────────────

def make_omega_scorer(adapter, image_paths: Sequence[str], layers: Sequence[int], top_fraction: float,
                      log: Callable = print) -> Callable[[int, int], Dict[int, float]]:
    """scorer(i, j) → {layer: min(score(A,B), score(B,A))} from ONE two-frame Ω pass capturing every
    requested global layer; cached per unordered pair."""
    cache: Dict[Tuple[int, int], Dict[int, float]] = {}
    agg = AG.aggregator_of(adapter.model)

    def scorer(i: int, j: int) -> Dict[int, float]:
        key = _unordered(i, j)
        if key not in cache:
            cap = AG.omega_pair_pass(adapter, [image_paths[key[0]], image_paths[key[1]]], layers)
            sc = AG.scores_from_captures(agg, cap, top_fraction=top_fraction)
            cache[key] = {int(L): float(v["score_min"]) for L, v in sc.items()}
            if len(cache) % 10 == 0:
                log(f"{LOG_TAG} {len(cache)} pair(s) scored")
        return cache[key]
    scorer.cache = cache                                              # type: ignore[attr-defined]
    return scorer


def salad_similarities(config: dict, image_paths: Sequence[str], frames_dir: str, log: Callable = print) -> np.ndarray:
    """The SALAD detector's cosine similarity over the run's keyframes (its descriptors recomputed
    on the card, the image list pinned to the run's)."""
    from LoopModels.LoopModel import LoopDetector
    det = LoopDetector(image_dir=frames_dir, output=None, config=config)
    det.image_paths = [Path(p) for p in image_paths]
    D = det.extract_descriptors().float()
    D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (D @ D.T).numpy().astype(np.float64)


def detector_min_gap(config: dict, n: int) -> int:
    sal = (config.get("Loop") or {}).get("SALAD") or {}
    model = config.get("Model") or {}
    return max(int(sal.get("min_gap", 0)), int(math.ceil(float(sal.get("min_gap_frac", 0.0)) * n)),
               int(model.get("loop_chunk_size", 0) or 0))


def salad_grid_counts(sims: np.ndarray, config: dict, grid: Sequence[float], scorer: Callable[[int, int], Dict[int, float]],
                      layer: Optional[int], threshold: Optional[float], *, max_pairs: Optional[int],
                      log: Callable = print) -> List[Dict[str, Any]]:
    """For each SALAD bar of the grid: the candidates the detector's own rule proposes and, with the
    chosen layer and threshold, how many pass the gate (the pairs scored once, cached)."""
    from LoopModels.LoopModel import nonlocal_similarities, select_candidates
    sal = (config.get("Loop") or {}).get("SALAD") or {}
    n = sims.shape[0]
    nl = nonlocal_similarities(sims, detector_min_gap(config, n))
    out = []
    for bar in grid:
        cands = select_candidates(nl, float(bar), int(sal.get("top_k", 5)), bool(sal.get("use_nms", True)),
                                  int(sal.get("nms_threshold", 0)))
        rec = {"salad_similarity": float(bar), "n_candidates": len(cands), "n_pass": None, "n_scored": 0,
               "n_not_scored": 0}
        if layer is not None and threshold is not None:
            n_pass, n_scored, n_skip = 0, 0, 0
            for a, b, _sim in cands:
                if max_pairs is not None and n_scored >= int(max_pairs):
                    n_skip += 1
                    continue
                sc = scorer(int(a), int(b))
                n_scored += 1
                if sc.get(int(layer), float("nan")) >= float(threshold):
                    n_pass += 1
            rec.update({"n_pass": n_pass, "n_scored": n_scored, "n_not_scored": n_skip})
        out.append(rec)
        log(f"{LOG_TAG} SALAD bar {bar:g}: {rec['n_candidates']} candidate(s)"
            + (f", {rec['n_pass']} pass the gate ({rec['n_scored']} scored)" if rec["n_pass"] is not None else ""))
    return out


# ── the whole calibration ─────────────────────────────────────────────────────────────────

def calibrate(run: Dict[str, Any], params: Dict[str, Any], scorer: Callable[[int, int], Dict[int, float]],
              layers: Sequence[int], *, labels: Dict[str, Any], manual: Dict[str, List[Tuple[int, int]]],
              negatives_frustum: Dict[str, Any], log: Callable = print) -> Dict[str, Any]:
    """Score every labelled pair, the per-layer statistics, the choice. Pure given a scorer."""
    pos = set(manual["positives"])
    neg = set(manual["negatives"])
    sources = {"positives": {"manual": len(pos)}, "negatives": {"manual": len(neg)}}
    if labels.get("available"):
        pos |= set(labels["positives"])
        neg |= set(labels["negatives_rejected"])
        sources["positives"]["run_accepted_cycle_consistent"] = len(labels["positives"])
        sources["negatives"]["run_rejected_bridges"] = len(labels["negatives_rejected"])
    neg |= set(negatives_frustum.get("pairs") or [])
    sources["negatives"]["frustum_disjoint_nonlocal"] = len(negatives_frustum.get("pairs") or [])
    both = pos & neg
    if both:
        log(f"{LOG_TAG} {len(both)} pair(s) labelled both ways — dropped from both sides: {sorted(both)[:5]}")
        pos -= both
        neg -= both
    pos_l, neg_l = sorted(pos), sorted(neg)
    scores: Dict[Tuple[int, int], Dict[int, float]] = {}
    for pair in pos_l + neg_l:
        scores[pair] = scorer(pair[0], pair[1])
    per_layer: Dict[int, Dict[str, Any]] = {}
    for L in layers:
        p = np.array([scores[x][L] for x in pos_l], np.float64)
        q = np.array([scores[x][L] for x in neg_l], np.float64)
        per_layer[int(L)] = {"pos": p.tolist(), "neg": q.tolist(),
                             "auc": auc_with_ci(p, q, n_boot=params["n_boot"], confidence=params["confidence"]),
                             "pr": pr_curve(p, q)}
    choice = choose(per_layer, select=params["select"], target_precision=params["target_precision"],
                    min_labels=params["min_labels"])
    return {"labels": {"positives": [list(x) for x in pos_l], "negatives": [list(x) for x in neg_l],
                       "sources": sources, "run_labels": {k: v for k, v in labels.items() if k not in ("positives", "negatives_rejected")},
                       "frustum_negatives": {k: v for k, v in negatives_frustum.items() if k != "pairs"}},
            "layers": {str(L): per_layer[L] for L in per_layer}, "choice": choice,
            "scores": {f"{a}-{b}": {str(L): v for L, v in sc.items()} for (a, b), sc in sorted(scores.items())}}


def write_outputs(out_dir: Path, doc: Dict[str, Any], png: bool = True) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / AG.CALIBRATION_NAME
    jp.write_text(json.dumps(doc, indent=1, default=_jsonable))
    cp = out_dir / "per_layer.csv"
    with open(cp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "auc", "ci_low", "ci_high", "n_pos", "n_neg", "chosen"])
        for L, r in sorted(doc["layers"].items(), key=lambda kv: int(kv[0])):
            a = r["auc"]
            w.writerow([L, a["auc"], a["ci_low"], a["ci_high"], a["n_pos"], a["n_neg"],
                        int(doc["choice"].get("chosen", False) and doc["choice"].get("layer") == int(L))])
    outs = {"json": str(jp), "csv": str(cp)}
    if png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            for L, r in sorted(doc["layers"].items(), key=lambda kv: int(kv[0])):
                pr = r["pr"]
                axes[0].plot(pr["recall"], pr["precision"], marker=".", label=f"L{L} AUC {r['auc']['auc']:.2f}")
                pos, neg = np.asarray(r["pos"]), np.asarray(r["neg"])
                ts = np.unique(np.concatenate([pos, neg]))[::-1]
                tpr = [(pos >= t).mean() if pos.size else 0 for t in ts]
                fpr = [(neg >= t).mean() if neg.size else 0 for t in ts]
                axes[1].plot(fpr, tpr, marker=".", label=f"L{L}")
            axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision"); axes[0].set_title("precision–recall per global layer")
            axes[1].set_xlabel("false positive rate"); axes[1].set_ylabel("true positive rate"); axes[1].set_title("ROC per global layer")
            for ax in axes:
                ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
            fig.tight_layout()
            pp = out_dir / "roc_pr.png"
            fig.savefig(pp, dpi=120)
            plt.close(fig)
            outs["png"] = str(pp)
        except Exception as e:  # noqa: BLE001 — the figure is a convenience; declared
            outs["png_error"] = str(e)
    return outs


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON-able: {type(o)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m loop_utils.attention_calibrate",
                                 description="Offline calibration of the attention loop verifier on a finished Ω run "
                                             "(claude_stac_slam2.txt phase 3). GPU: one two-frame pass per pair.")
    ap.add_argument("--run", required=True, help="the maplong_run directory of a finished run")
    ap.add_argument("--labels", default=None, help="manual labels JSON {positives: [[i,j]..], negatives: [[i,j]..]}")
    ap.add_argument("--params", default=None, help="calibration parameters JSON (when the run has no attention_gate section)")
    ap.add_argument("--out", default=None, help="output directory (default <run>/attention_calibration)")
    ap.add_argument("--no-salad-grid", action="store_true", help="skip the SALAD similarity grid (no descriptor pass)")
    ap.add_argument("--max-pairs", type=int, default=None, help="BOUND: pairs scored per SALAD bar of the grid")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args(argv)
    try:
        run = read_run(Path(args.run))
        params = calibration_params(run["config"], None if args.params is None else Path(args.params))
    except CalibrationError as e:
        print(f"{LOG_TAG} REFUSED: {e}", file=sys.stderr)
        return 2
    out_dir = Path(args.out) if args.out else Path(args.run) / "attention_calibration"
    log = print
    # the card: exclusive, deterministic (the fork's own regime), BEFORE torch touches CUDA
    from loop_utils.stac_repro import repro
    R = repro()
    R.require_exclusive_gpu(log=log)
    R.enable_deterministic_torch(42)
    import torch  # noqa: F401
    from base_models.vggtomega_adapter import VGGTOmegaAdapter
    adapter = VGGTOmegaAdapter(run["config"])
    adapter.load()
    agg = AG.aggregator_of(adapter.model)
    layers = AG.global_layers(agg)
    log(f"{LOG_TAG} model loaded: {len(layers)} global layer(s) {layers}, register layers {AG.register_layers(agg)}")
    n = run["n_frames"]
    labels = run_labels(Path(args.run))
    log(f"{LOG_TAG} labels from the run: " + ("available — " if labels["available"] else "NOT available — ")
        + str(labels["reason"] or f"{len(labels['positives'])} positive(s), {len(labels['negatives_rejected'])} rejected bridge(s)"))
    manual = manual_labels(None if args.labels is None else Path(args.labels))
    model_cfg = run["config"].get("Model") or {}
    spatial = (model_cfg.get("loops") or {}).get("spatial") or {}
    try:
        poses, Ks = read_final_cameras(Path(args.run), n)
        grid_hw = (int(round(Ks[0, 3] * 2)), int(round(Ks[0, 2] * 2)))    # the Ω grid from the principal point
        probe = AG.omega_pair_pass(adapter, run["image_paths"][:2], layers[:1])
        n_tok = int(next(iter(probe.values())).shape[1] // 2)
        fr_neg = frustum_disjoint_pairs(poses, Ks, grid_hw, min_gap=detector_min_gap(run["config"], n),
                                        depth_range=(float(spatial.get("min_depth_m", 0.3)), float(spatial.get("max_depth_m", 15.0))),
                                        n_pick=params["n_negatives"])
        fr_neg["grid_hw"] = list(grid_hw)
        fr_neg["tokens_per_frame"] = n_tok
    except CalibrationError as e:
        fr_neg = {"pairs": [], "reason": str(e)}
        log(f"{LOG_TAG} frustum-disjoint negatives not drawn: {e}")
    scorer = make_omega_scorer(adapter, run["image_paths"], layers, params["top_fraction"], log=log)
    doc = calibrate(run, params, scorer, layers, labels=labels, manual=manual, negatives_frustum=fr_neg, log=log)
    ch = doc["choice"]
    log(f"{LOG_TAG} choice: " + (f"layer {ch['layer']}, threshold {ch['threshold']:.4f} (precision {ch['precision']:.2f}, "
                                 f"recall {ch['recall']:.2f}, {ch['n_pos']} pos / {ch['n_neg']} neg)" if ch.get("chosen")
                                 else f"NOT chosen — {ch['reason']}"))
    if not args.no_salad_grid:
        sims = salad_similarities(run["config"], run["image_paths"], str(Path(run["session_dir"]) / "frames"), log=log)
        doc["salad_grid"] = salad_grid_counts(sims, run["config"], params["salad_grid"], scorer,
                                              ch.get("layer") if ch.get("chosen") else None,
                                              ch.get("threshold") if ch.get("chosen") else None,
                                              max_pairs=args.max_pairs, log=log)
    doc.update({"version": 1, "tool": "loop_utils.attention_calibrate", "run_dir": run["run_dir"],
                "fork_stamp": run["fork_stamp"], "chunk_indices": run["chunk_indices"], "params": params,
                "global_layers": layers, "register_layers": AG.register_layers(agg),
                "n_pairs_scored": len(getattr(scorer, "cache", {}))})
    outs = write_outputs(out_dir, doc, png=not args.no_png)
    log(f"{LOG_TAG} → {outs['json']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
