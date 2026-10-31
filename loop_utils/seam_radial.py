"""Offline seam instrument — claude_stac_slam2.txt, phase 1 (no GPU, no model, no behaviour change).

    python -m loop_utils.seam_radial --run <maplong_run> [--out <dir>] [options]

VGGT-SLAM 2.0 (MIT-SPARK, arXiv 2601.19887) stitches two submaps only when the two inferences of
the shared frame agree in pose AND calibration, modelling the intrinsics difference K_i⁻¹·K_j
instead of assuming it zero. In STAC the same frame receives focals differing by up to 21.8 px
between two Ω inferences (pccr: metric_lock.inference_focal_error 8.8 px RMS), every chunk is
unprojected with its OWN K and the seam is rigid (exact_seam_align → robust_rigid, s = 1). This
instrument MEASURES, on a finished run, what that calibration difference does to the seams:

- per seam k→k+1, on the SAME exact correspondences exact_seam_align used (chunk k's last `ov`
  frames against chunk k+1's first `ov`, every valid pixel), split by a stable key into a FIT half
  and a HELD-OUT half: robust_rigid on the fit half, the held-out residual by bins of normalised
  radius r = ‖(u − cx, v − cy)‖ / f and by bins of depth, separated into its LATERAL (X, Y in the
  shared frame's camera, chunk k's copy) and AXIAL (Z) components;
- the offset of fx, fy, cx, cy between the two inferences on the shared frames (median, spread);
- two COUNTERFACTUALS on in-memory copies (the npys are never written):
    common_K     — every frame of both chunks re-unprojected with ONE session K (its depth per
                   pixel unchanged, its rays changed, c2w unchanged): session_median (the median
                   fx, fy, cx, cy over the frames by owner) and intake_probe (the K of
                   intake/focal_probe.json, when the session has it);
    affine_seam  — the literal model of 2.0: A = K_k⁻¹·K_{k+1} from the median intrinsics of the
                   shared frames in each inference, expressed in the camera of the middle shared
                   frame and applied to the WHOLE copy of chunk k+1 before the rigid fit. Measured
                   only: an affine warp of a chunk has no Euclidean camera with one K per frame,
                   which is what F0–F5 need;
  each judged by metric_lock.decide_change with the run's own correction_graph.graph parameters
  (heldout_confidence, improvement_error_factor), the judges being the shared frames; and the
  surface pairs held out INSIDE each chunk (metric_lock.surface_pair_correspondences at the
  offsets of _stac_holdout_pairs) before / after, judged by the MIRROR of the rule (a worsening
  significant, with enough judges and beyond the error). A verdict is a calculation here, never a
  recommendation.

THE STATE OF THE NPYS. The chunk npys are mutated in place by the metric lock (apply_scale /
apply_scale_drift) and by the scale close (δ_k, stamped `_stac_loop_scale_applied`); the exact
seams read them AFTER both. This tool proves, from the marks and stamps, that every chunk npy
is in exactly that state (fork_stamp.json ↔ `_stac_fork_stamp`, metric_lock.json's lock record,
scale_graph.json's δ ↔ the npy's mark, chunk_sim3.json's layout ↔ `_stac_range`) and refuses by
name when it cannot — it never supposes. The npys survive the run only with
Model.keep_unaligned_chunks (server: certify.keep_unaligned_chunks).

Outputs (in <out>, default <run>/seam_radial/): seam_radial.json, seam_<k>.csv per seam (the
bin tables of every state), seam_<k>.png per seam (held-out residual against radius, lateral and
axial, every state). Every statistic is computed on the CPU in float64 from the arrays on disk;
every subsample is a stable pick (loop_utils.stable_sample).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from loop_utils.metric_lock import (decide_change, frame_owner, robust_rigid, seam_overlap,
                                    surface_pair_correspondences)
from loop_utils.stable_sample import block_pixel_keys, stable_pick

STAMP_KEY = "_stac_fork_stamp"
RANGE_KEY = "_stac_range"
LOOP_SCALE_KEY = "_stac_loop_scale_applied"
CONF_FLOOR = 1e-5                      # the sky mask of every fork fit (not a quality gate)
DEFAULT_RADIUS_BINS = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.4)
DEFAULT_DEPTH_BINS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
SALT_FIT = 11                          # the fit / held-out split (salted so it never follows a pick)
SALT_JUDGE = 29                        # the held-out subsample that enters the rule
STATES = ("before", "common_K_session_median", "common_K_intake_probe", "affine_seam")
MAD_TO_SIGMA = 1.4826


class SeamRadialError(RuntimeError):
    """The run cannot be measured without supposing — the message names the file / field."""


# ── the run and the proof of its state ────────────────────────────────────────────────────

def _load_json(path: Path, what: str) -> dict:
    if not path.is_file():
        raise SeamRadialError(f"{what} is missing: {path}")
    try:
        with open(path) as f:
            doc = json.load(f)
    except ValueError as e:
        raise SeamRadialError(f"{what} is not readable JSON: {path} ({e})") from e
    if not isinstance(doc, dict):
        raise SeamRadialError(f"{what} is not a JSON object: {path}")
    return doc


def load_chunk(path: Path) -> dict:
    """One chunk npy (a pickled dict) as the fork saved it."""
    if not path.is_file():
        raise SeamRadialError(f"chunk npy is missing: {path}")
    d = np.load(path, allow_pickle=True)
    if d.dtype == object and d.shape == ():
        d = d.item()
    if not isinstance(d, dict):
        raise SeamRadialError(f"{path} does not hold the fork's chunk dict")
    return d


def _layout_of(config: dict, sim3: dict) -> List[Tuple[int, int]]:
    ci_sim3 = sim3.get("chunk_indices")
    if not isinstance(ci_sim3, list) or not ci_sim3:
        raise SeamRadialError("chunk_sim3.json carries no chunk_indices — the seams' layout is unknown")
    layout = [(int(a), int(b)) for a, b in ci_sim3]
    ranges = (config.get("Model") or {}).get("chunk_ranges")
    if ranges is not None:
        explicit = [(int(a), int(b)) for a, b in ranges]
        if explicit != layout:
            raise SeamRadialError(f"vggt_omega_config.yaml Model.chunk_ranges {explicit} is not "
                                  f"chunk_sim3.json's layout {layout}")
    return layout


def verify_state(run_dir: Path) -> Dict[str, Any]:
    """Prove that every unaligned chunk npy is in the state the exact seams read, or refuse by
    name. Returns the record of what was verified (the layout, the stamp, the lock, the δ)."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise SeamRadialError(f"run directory is missing: {run_dir}")
    cfg_path = run_dir / "vggt_omega_config.yaml"
    if not cfg_path.is_file():
        raise SeamRadialError(f"vggt_omega_config.yaml is missing: {cfg_path}")
    import yaml
    with open(cfg_path) as f:
        config = yaml.safe_load(f) or {}
    stamp_doc = _load_json(run_dir / "fork_stamp.json", "fork_stamp.json (the run stamp)")
    stamp = stamp_doc.get("sha256")
    if not stamp:
        raise SeamRadialError(f"{run_dir / 'fork_stamp.json'} carries no sha256 — the npys cannot be "
                              f"tied to a run")
    stamp = str(stamp)
    sim3 = _load_json(run_dir / "chunk_sim3.json", "chunk_sim3.json (the seams' record)")
    layout = _layout_of(config, sim3)
    if len(layout) < 2:
        raise SeamRadialError(f"the run has {len(layout)} chunk(s) — no seam to measure")
    lock = _load_json(run_dir / "metric_lock.json", "metric_lock.json (the lock record)")
    if str(lock.get("fork_stamp")) != stamp:
        raise SeamRadialError(f"metric_lock.json carries fork stamp {lock.get('fork_stamp')!r}, the run's "
                              f"is {stamp[:12]}… — the lock on disk is another run's")
    if lock.get("chunk_indices") is not None:
        if [(int(a), int(b)) for a, b in lock["chunk_indices"]] != layout:
            raise SeamRadialError(f"metric_lock.json was written for the layout {lock['chunk_indices']}, "
                                  f"the seams' layout is {layout}")
    sg_path = run_dir / "scale_graph.json"
    delta = None
    if sg_path.is_file():
        sg = _load_json(sg_path, "scale_graph.json")
        if str(sg.get("fork_stamp")) != stamp:
            raise SeamRadialError(f"scale_graph.json carries fork stamp {sg.get('fork_stamp')!r}, the "
                                  f"run's is {stamp[:12]}… — the scale close on disk is another run's")
        da = sg.get("delta_applied")
        if not isinstance(da, dict):
            raise SeamRadialError("scale_graph.json carries no delta_applied — whether the chunks "
                                  "were re-scaled by the loop rows cannot be proven")
        delta = {int(k): float(v) for k, v in da.items()}
    unaligned = run_dir / "_tmp_results_unaligned"
    chunks_rec = []
    for k, (a, b) in enumerate(layout):
        p = unaligned / f"chunk_{k}.npy"
        if not p.is_file():
            raise SeamRadialError(f"{p} is missing — the run did not keep its unaligned chunks "
                                  f"(Model.keep_unaligned_chunks / certify.keep_unaligned_chunks)")
        d = load_chunk(p)
        st = d.get(STAMP_KEY)
        if st is None:
            raise SeamRadialError(f"{p.name} carries no fork stamp — it cannot be tied to this run")
        if str(st) != stamp:
            raise SeamRadialError(f"{p.name} carries fork stamp {str(st)[:12]}…, the run's is "
                                  f"{stamp[:12]}… — it was written by another run")
        rng = d.get(RANGE_KEY)
        if rng is not None and [int(x) for x in rng] != [a, b]:
            raise SeamRadialError(f"{p.name} is frames {[int(x) for x in rng]}, the seams' layout has "
                                  f"chunk {k} at {[a, b]}")
        dep = np.asarray(d["depth"])
        n_frames = int(dep.shape[0] if dep.ndim == 3 else dep.shape[1])
        if n_frames != b - a:
            raise SeamRadialError(f"{p.name} holds {n_frames} frames, chunk {k} of the layout has {b - a}")
        rec = lock.get("chunks", {}).get(str(k))
        if not isinstance(rec, dict) or (rec.get("s") is None and rec.get("s_applied") is None):
            raise SeamRadialError(f"metric_lock.json does not record chunk {k} as locked — the npy's "
                                  f"scale state cannot be proven")
        mark = d.get(LOOP_SCALE_KEY)
        if delta is None and mark is not None:
            raise SeamRadialError(f"{p.name} carries a loop-scale mark ({mark}) but the run has no "
                                  f"scale_graph.json — the npy's state is not the one the seams read")
        if delta is not None:
            if k not in delta:
                raise SeamRadialError(f"scale_graph.json's delta_applied has no entry for chunk {k}")
            if mark is None:
                raise SeamRadialError(f"{p.name} carries no loop-scale mark while scale_graph.json applied "
                                      f"δ={delta[k]!r} to chunk {k} — the close did not reach the npy on disk")
            if float(mark) != delta[k]:
                raise SeamRadialError(f"{p.name} carries loop-scale mark {float(mark)!r}, scale_graph.json "
                                      f"applied δ={delta[k]!r} to chunk {k}")
        chunks_rec.append({"chunk": k, "range": [a, b], "frames": n_frames,
                           "lock": {kk: rec.get(kk) for kk in ("s", "s_graph", "s_applied", "s0_s1")
                                    if kk in rec},
                           "loop_scale_mark": (None if mark is None else float(mark))})
        del d
    model = config.get("Model") or {}
    drift = (lock.get("drift") or {}).get("verdict")
    return {"run_dir": str(run_dir), "fork_stamp": stamp, "chunk_indices": [list(x) for x in layout],
            "exact_seam_align": bool(model.get("exact_seam_align", False)),
            "frame_ownership": bool(model.get("frame_ownership", False)),
            "scale_drift_verdict": drift, "delta_applied": (None if delta is None else
                                                            {str(k): v for k, v in sorted(delta.items())}),
            "chunks": chunks_rec,
            "proof": "fork_stamp.json ↔ every chunk's _stac_fork_stamp; metric_lock.json (same stamp) "
                     "records every chunk locked; scale_graph.json (same stamp) δ ↔ every chunk's "
                     "_stac_loop_scale_applied (or neither exists); chunk_sim3.json's layout ↔ "
                     "Model.chunk_ranges / _stac_range / frame counts — the state the exact seams read"}


def run_params(config: dict) -> Dict[str, Any]:
    """The run's own decision parameters the rule and the pairs read (correction_graph.graph via
    Model.graph; the pose fits' confidence floor). Missing keys fail by name — never a default."""
    model = config.get("Model") or {}
    g = model.get("graph")
    if not isinstance(g, dict):
        raise SeamRadialError("vggt_omega_config.yaml has no Model.graph — the rule's confidence and "
                              "factor are unknown")
    out = {}
    for key in ("heldout_confidence", "improvement_error_factor", "holdout_offsets", "holdout_stride",
                "holdout_samples"):
        if key not in g:
            raise SeamRadialError(f"vggt_omega_config.yaml Model.graph.{key} is missing")
        out[key] = g[key]
    out["heldout_confidence"] = float(out["heldout_confidence"])
    out["improvement_error_factor"] = float(out["improvement_error_factor"])
    out["holdout_offsets"] = [int(x) for x in out["holdout_offsets"]]
    out["holdout_stride"] = int(out["holdout_stride"])
    out["holdout_samples"] = int(out["holdout_samples"])
    out["pose_fit_conf_min_norm"] = float(model.get("pose_fit_conf_min_norm", 0.0) or 0.0)
    return out


# ── geometry helpers ──────────────────────────────────────────────────────────────────────

def _wp(d: dict) -> np.ndarray:
    wp = np.asarray(d["world_points"])
    return wp[0] if wp.ndim == 5 else wp


def _conf(d: dict, shape) -> np.ndarray:
    return np.asarray(d["world_points_conf"]).reshape(shape)


def _ext(d: dict) -> np.ndarray:
    ext = np.asarray(d["extrinsic"], np.float64)
    return ext[0] if ext.ndim == 4 else ext


def _K(d: dict) -> np.ndarray:
    K = np.asarray(d["intrinsic"], np.float64)
    return K[0] if K.ndim == 4 else K


def _depth(d: dict) -> np.ndarray:
    dep = np.asarray(d["depth"])
    return dep[0] if dep.ndim == 4 else dep


def _c2w4(ext_i: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :4] = np.asarray(ext_i, np.float64)[:3, :4]
    return M


from loop_utils.common_k import session_median_K, unproject_chunk   # noqa: E402  (one implementation)


def intake_probe_K(run_dir: Path, grid_hw: Tuple[int, int]) -> Dict[str, Any]:
    """The K intake.focal measured (<session>/intake/focal_probe.json, native pixels) carried to
    the Ω grid by proportional scaling; {'K': ..} or {'K': None, 'reason': ..} when the session
    has none or the grid is not a pure scaling of the native frame."""
    session = Path(run_dir).resolve().parent.parent
    path = session / "intake" / "focal_probe.json"
    if not path.is_file():
        return {"K": None, "reason": f"{path} does not exist (intake.focal did not run for this session)"}
    try:
        doc = json.loads(path.read_text())
    except ValueError as e:
        return {"K": None, "reason": f"{path} is not readable JSON ({e})"}
    K = doc.get("K")
    nat = doc.get("native_wh")
    if K is None or nat is None:
        return {"K": None, "reason": f"{path} carries no K / native_wh"}
    K = np.asarray(K, np.float64)
    Wn, Hn = int(nat[0]), int(nat[1])
    H, W = int(grid_hw[0]), int(grid_hw[1])
    sx, sy = W / Wn, H / Hn
    if abs(sx - sy) > 1e-3 * max(sx, sy):
        return {"K": None, "reason": f"the Ω grid {W}x{H} is not a pure scaling of the native frame "
                                     f"{Wn}x{Hn} (sx {sx:.5f} ≠ sy {sy:.5f}: padding / cropping unknown offline)"}
    Kg = K.copy()
    Kg[0, 0] *= sx
    Kg[1, 1] *= sy
    Kg[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    Kg[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return {"K": Kg, "source": str(path.relative_to(session)), "native_wh": [Wn, Hn],
            "grid_wh": [W, H], "scale": [sx, sy], "mapping": "proportional (pixel centres)"}


# ── the seam's exact correspondences ──────────────────────────────────────────────────────

def seam_correspondences(layout: Sequence[Tuple[int, int]], k: int, d1: dict, d2: dict) -> Dict[str, Any]:
    """The correspondences exact_seam_align fits on: chunk k's last ov frames against chunk k+1's
    first ov, pixel for pixel, valid where both confidences clear the sky mask (and the chunks'
    masks when they have one). Returns the flat indices, keys, block-local frame, pixel (u, v)."""
    ov = seam_overlap(layout, k)
    wp1, wp2 = _wp(d1), _wp(d2)
    S1 = wp1.shape[0]
    pm1, pm2 = wp1[-ov:], wp2[:ov]
    c1 = _conf(d1, wp1.shape[:3])[-ov:].reshape(-1)
    c2 = _conf(d2, wp2.shape[:3])[:ov].reshape(-1)
    ok = (c1 > CONF_FLOOR) & (c2 > CONF_FLOOR)
    if d1.get("mask") is not None and d2.get("mask") is not None:
        m1 = np.asarray(d1["mask"])[-ov:]
        m2 = np.asarray(d2["mask"])[:ov]
        ok &= (m1.squeeze() & m2.squeeze()).reshape(-1).astype(bool)
    H, W = pm1.shape[1], pm1.shape[2]
    flat = np.flatnonzero(ok)
    keys = block_pixel_keys(int(layout[k + 1][0]), flat, H * W)
    frame_local = flat // (H * W)              # block-local: 0 = first shared frame
    within = flat % (H * W)
    v = within // W
    u = within % W
    return {"ov": ov, "H": H, "W": W, "flat": flat, "keys": keys, "frame_local": frame_local,
            "u": u, "v": v, "first_local_in_1": S1 - ov,       # chunk k's local index of shared frame 0
            "n_total": int(ok.size), "n_valid": int(flat.size)}


def split_fit_heldout(keys: np.ndarray, salt: int = SALT_FIT) -> Tuple[np.ndarray, np.ndarray]:
    """Half of the correspondences for the fit (stable pick by key), the rest held out."""
    n = len(keys)
    fit = stable_pick(keys, n // 2, salt=salt)
    held_mask = np.ones(n, bool)
    held_mask[fit] = False
    return fit, np.flatnonzero(held_mask)


def _points_at(wp_block: np.ndarray, flat: np.ndarray) -> np.ndarray:
    return np.asarray(wp_block, np.float64).reshape(-1, 3)[flat]


def fit_seam(P1: np.ndarray, P2: np.ndarray, keys: np.ndarray, fit_idx: np.ndarray, held_idx: np.ndarray
             ) -> Dict[str, Any]:
    """robust_rigid on the fit half (chunk k+1 → chunk k, as exact_seam_align), the residual
    VECTORS on the held-out half (chunk k's world), the fit's own median and the split-half
    disagreement of the transform at the held points (the fit on the held half against the fit
    on the fit half)."""
    fit = robust_rigid(P2[fit_idx], P1[fit_idx], keys=keys[fit_idx])
    if fit is None:
        return {"starved": True}
    R, t, res_fit, n_used = fit
    res = (P2[held_idx] @ R.T + t) - P1[held_idx]
    out = {"starved": False, "R": R, "t": t, "fit_median_m": float(res_fit), "n_fit_used": int(n_used),
           "held_residual": res}
    other = robust_rigid(P2[held_idx], P1[held_idx], keys=keys[held_idx])
    if other is not None:
        Rb, tb = other[0], other[1]
        disp = np.linalg.norm((P2[held_idx] @ R.T + t) - (P2[held_idx] @ Rb.T + tb), axis=1)
        out["fit_uncertainty_m"] = float(np.median(disp))
    else:
        out["fit_uncertainty_m"] = None
    return out


def camera_components(res_world: np.ndarray, frame_local: np.ndarray, d1: dict, first_local_in_1: int
                      ) -> Dict[str, np.ndarray]:
    """Residual vectors (chunk k's world) in the camera of each correspondence's shared frame,
    chunk k's copy: lateral (x, y) and axial (z)."""
    ext = _ext(d1)
    out = np.empty_like(res_world)
    for fl in np.unique(frame_local):
        sel = frame_local == fl
        R = ext[first_local_in_1 + int(fl), :3, :3]
        out[sel] = res_world[sel] @ R        # Rᵀ · r, row form
    lateral = np.hypot(out[:, 0], out[:, 1])
    return {"x": out[:, 0], "y": out[:, 1], "axial": out[:, 2], "lateral": lateral}


def normalized_radius(u: np.ndarray, v: np.ndarray, frame_local: np.ndarray, Ks_block: np.ndarray
                      ) -> np.ndarray:
    """r = ‖(u − cx, v − cy)‖ / f with f = (fx + fy) / 2 of the shared frame's K (chunk k's copy)."""
    K = Ks_block[frame_local]
    f = 0.5 * (K[:, 0, 0] + K[:, 1, 1])
    return np.hypot(u - K[:, 0, 2], v - K[:, 1, 2]) / f


BIN_STATS = ("lateral_median_m", "lateral_rms_m", "lateral_p90_m", "axial_abs_median_m",
             "axial_signed_median_m", "axial_rms_m", "axial_abs_p90_m",
             "lateral_over_depth_median", "axial_abs_over_depth_median")


def _bin_stats(lat: np.ndarray, ax: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    """The statistics of one bin: metres, and the residual divided by the point's depth (the
    ANGULAR residual — a focal error moves a point by ≈ z·r·Δf/f, so lateral/z grows with r
    whatever the depth distribution of the scene)."""
    zok = z > 0.1
    return {"lateral_median_m": float(np.median(lat)),
            "lateral_rms_m": float(np.sqrt(np.mean(lat ** 2))),
            "lateral_p90_m": float(np.percentile(lat, 90)),
            "axial_abs_median_m": float(np.median(np.abs(ax))),
            "axial_signed_median_m": float(np.median(ax)),
            "axial_rms_m": float(np.sqrt(np.mean(ax ** 2))),
            "axial_abs_p90_m": float(np.percentile(np.abs(ax), 90)),
            "lateral_over_depth_median": (float(np.median(lat[zok] / z[zok])) if zok.any() else None),
            "axial_abs_over_depth_median": (float(np.median(np.abs(ax[zok]) / z[zok])) if zok.any() else None)}


def bin_table(lateral: np.ndarray, axial: np.ndarray, coord: np.ndarray, edges: Sequence[float],
              kind: str, depth: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    rows = []
    e = [float(x) for x in edges]
    z = coord if depth is None else depth
    for lo, hi in zip(e[:-1], e[1:]):
        sel = (coord >= lo) & (coord < hi)
        n = int(sel.sum())
        row = {"kind": kind, "lo": lo, "hi": hi, "n": n}
        if n:
            row.update(_bin_stats(lateral[sel], axial[sel], z[sel]))
        else:
            row.update({k_: None for k_ in BIN_STATS})
        rows.append(row)
    return rows


def radius_depth_cells(lateral: np.ndarray, axial: np.ndarray, r: np.ndarray, z: np.ndarray,
                       r_edges: Sequence[float], z_edges: Sequence[float]) -> List[Dict[str, Any]]:
    """The joint table radius × depth (one row per non-empty cell): the radius trend at one depth,
    free of the scene's depth distribution (a corridor seen along its axis puts the FAR points at
    the centre of the image)."""
    cells = []
    re_ = [float(x) for x in r_edges]
    ze = [float(x) for x in z_edges]
    for rlo, rhi in zip(re_[:-1], re_[1:]):
        for zlo, zhi in zip(ze[:-1], ze[1:]):
            sel = (r >= rlo) & (r < rhi) & (z >= zlo) & (z < zhi)
            n = int(sel.sum())
            if n:
                cells.append({"r_lo": rlo, "r_hi": rhi, "z_lo": zlo, "z_hi": zhi, "n": n,
                              **_bin_stats(lateral[sel], axial[sel], z[sel])})
    return cells


def intrinsics_offsets(d1: dict, d2: dict, layout: Sequence[Tuple[int, int]], k: int) -> Dict[str, Any]:
    """fx, fy, cx, cy of the shared frames in both inferences: K_{k+1} − K_k per frame, the median
    and the spread (1.4826·MAD) per parameter."""
    ov = seam_overlap(layout, k)
    K1, K2 = _K(d1)[-ov:], _K(d2)[:ov]
    names = ("fx", "fy", "cx", "cy")
    pick = lambda K: np.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], 1)     # noqa: E731
    a, b = pick(K1), pick(K2)
    diff = b - a
    med = np.median(diff, axis=0)
    mad = MAD_TO_SIGMA * np.median(np.abs(diff - med), axis=0)
    return {"n_shared_frames": int(ov),
            "chunk_k_median": {n: float(v) for n, v in zip(names, np.median(a, 0))},
            "chunk_k1_median": {n: float(v) for n, v in zip(names, np.median(b, 0))},
            "offset_median_px": {n: float(v) for n, v in zip(names, med)},
            "offset_spread_sigma_px": {n: float(v) for n, v in zip(names, mad)},
            "per_frame_offset_px": {n: [float(x) for x in diff[:, i]] for i, n in enumerate(names)},
            "shared_frames_global": [int(g) for g in range(layout[k + 1][0], layout[k][1])]}


def affine_seam_matrix(d1: dict, d2: dict, layout: Sequence[Tuple[int, int]], k: int) -> Dict[str, Any]:
    """A = K_k⁻¹·K_{k+1} from the median intrinsics of the shared frames in each inference, and the
    middle shared frame (block index ov // 2) whose camera, in chunk k+1's copy, expresses it."""
    off = intrinsics_offsets(d1, d2, layout, k)
    def _K3(m):
        return np.array([[m["fx"], 0.0, m["cx"]], [0.0, m["fy"], m["cy"]], [0.0, 0.0, 1.0]])
    Kk, Kk1 = _K3(off["chunk_k_median"]), _K3(off["chunk_k1_median"])
    A = np.linalg.inv(Kk) @ Kk1
    mid = int(off["n_shared_frames"] // 2)
    return {"A": A, "K_k": Kk, "K_k1": Kk1, "middle_shared_block_index": mid,
            "middle_shared_frame_global": int(layout[k + 1][0] + mid)}


def warp_chunk_affine(d2: dict, A: np.ndarray, mid_local: int) -> np.ndarray:
    """The whole copy of chunk k+1 under X' = c2w_m · A · w2c_m · X (A on the 3x3 linear part,
    expressed in the middle shared frame's camera)."""
    wp = _wp(d2).astype(np.float64)
    c2w = _c2w4(_ext(d2)[mid_local])
    w2c = np.linalg.inv(c2w)
    M = np.eye(4)
    M[:3, :3] = np.asarray(A, np.float64)
    T = c2w @ M @ w2c
    S, H, W, _ = wp.shape
    flat = wp.reshape(-1, 3)
    out = flat @ T[:3, :3].T + T[:3, 3]
    return out.reshape(S, H, W, 3)


# ── the rule, and its mirror ──────────────────────────────────────────────────────────────

def judge_error_per_judge(values: np.ndarray, clusters: np.ndarray) -> Dict[str, Any]:
    """The measured error of what is judged: each judge's statistic is the median of its
    observations; its standard error 1.4826·MAD/√n; the error of the rule = the median over the
    judges (pooled). Returns {'error_m', 'n_judges', 'per_judge'}."""
    errs, per = [], {}
    for c in np.unique(clusters):
        v = values[clusters == c]
        if v.size == 0:
            continue
        med = np.median(v)
        se = MAD_TO_SIGMA * np.median(np.abs(v - med)) / np.sqrt(v.size)
        errs.append(se)
        per[str(int(c))] = {"n": int(v.size), "median_m": float(med), "se_m": float(se)}
    if not errs:
        return {"error_m": None, "n_judges": 0, "per_judge": {}}
    return {"error_m": float(np.median(errs)), "n_judges": len(errs), "per_judge": per}


def rule_verdict(before: np.ndarray, after: np.ndarray, clusters: np.ndarray, *, error_m: float,
                 factor: float, confidence: float, mirror: bool = False) -> Dict[str, Any]:
    """decide_change on the paired per-observation disagreements, the judges being ``clusters``
    (the shared frames; the intra-chunk pairs). ``mirror``: the same rule applied to the WORSENING
    (before and after swapped) — 'worsens_by_rule' is then its 'improves'."""
    b = np.asarray(before, np.float64)
    a = np.asarray(after, np.float64)
    ok = np.isfinite(b) & np.isfinite(a)
    rec = {"n_observations": int(ok.sum()), "n_discarded_nonfinite": int((~ok).sum()),
           "error_m": float(error_m), "error_factor": float(factor), "confidence": float(confidence),
           "median_before_m": (float(np.median(b[ok])) if ok.any() else None),
           "median_after_m": (float(np.median(a[ok])) if ok.any() else None)}
    if not ok.any():
        rec["verdict"] = None
        rec["reason"] = "no observation valid in both states"
        return rec
    x, y = (a[ok], b[ok]) if mirror else (b[ok], a[ok])
    d = decide_change(x, y, error=error_m, error_factor=factor, confidence=confidence,
                      clusters=np.asarray(clusters)[ok])
    rec["decide_change"] = {k_: (v if not isinstance(v, np.generic) else v.item()) for k_, v in d.items()}
    if mirror:
        rec["worsens_by_rule"] = bool(d["improves"])
        rec["reason"] = ("worsens significantly, with enough judges, beyond the error" if d["improves"]
                         else "no worsening by the rule: " + str(d.get("reason", "")))
    else:
        rec["improves_by_rule"] = bool(d["improves"])
        rec["reason"] = str(d.get("reason", ""))
    return rec


# ── intra-chunk held-out surface pairs ────────────────────────────────────────────────────

def chunk_holdout_pairs(d: dict, chunk_range: Tuple[int, int], owner: np.ndarray, params: Dict[str, Any],
                        wp_override: Optional[np.ndarray] = None, K_override: Optional[np.ndarray] = None
                        ) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """The surface pairs held out inside ONE chunk (frames f, f + d both owned by it, d in
    holdout_offsets, every holdout_stride-th f): {(f, g): {'disagreement_m': median ‖p − q‖,
    'se_m', 'n'}} under the chunk's own world points and K, or under ``wp_override`` /
    ``K_override`` (the common-K counterfactual: the points AND the projection K change together)."""
    a, b = int(chunk_range[0]), int(chunk_range[1])
    wp = _wp(d) if wp_override is None else np.asarray(wp_override)
    cf = _conf(d, wp.shape[:3])
    ext = _ext(d)
    Ks = _K(d)
    N = len(owner)
    out = {}
    for f in range(0, N, max(int(params["holdout_stride"]), 1)):
        for off in params["holdout_offsets"]:
            g = f + int(off)
            if not (a <= f < b and a <= g < b) or owner[f] != owner[g]:
                continue
            if owner[f] != [k for k in range(1)][0] and False:      # (owner equality already checked)
                continue
            lf, lg = f - a, g - a
            Kg = Ks[lg] if K_override is None else np.asarray(K_override, np.float64)
            w2c = np.linalg.inv(_c2w4(ext[lg]))
            pq = surface_pair_correspondences(wp[lf].astype(np.float32), cf[lf].astype(np.float32),
                                              wp[lg].astype(np.float32), cf[lg].astype(np.float32),
                                              w2c, Kg, max_samples=int(params["holdout_samples"]), seed=f,
                                              conf_min_norm=float(params["pose_fit_conf_min_norm"]))
            if pq is None:
                continue
            dist = np.linalg.norm(np.asarray(pq[0], np.float64) - np.asarray(pq[1], np.float64), axis=1)
            med = float(np.median(dist))
            out[(f, g)] = {"disagreement_m": med, "n": int(dist.size),
                           "se_m": float(MAD_TO_SIGMA * np.median(np.abs(dist - med)) / np.sqrt(dist.size))}
    return out


def pairs_before_after(before: Dict[Tuple[int, int], Dict[str, Any]],
                       after: Dict[Tuple[int, int], Dict[str, Any]], *, factor: float, confidence: float
                       ) -> Dict[str, Any]:
    """The intra-chunk pairs valid in both states, judged by the MIRROR of the rule (does the
    change WORSEN them significantly, with enough judges, beyond their own measured error?)."""
    common = sorted(set(before) & set(after))
    rec = {"n_pairs_before": len(before), "n_pairs_after": len(after), "n_pairs_judged": len(common),
           "n_discarded_one_state_only": len(set(before) ^ set(after))}
    if not common:
        rec["verdict"] = None
        rec["reason"] = "no held-out pair valid in both states"
        return rec
    b = np.array([before[p]["disagreement_m"] for p in common])
    a = np.array([after[p]["disagreement_m"] for p in common])
    se = np.array([before[p]["se_m"] for p in common])
    err = float(np.median(se))
    rec.update({"median_before_m": float(np.median(b)), "median_after_m": float(np.median(a)),
                "error_m": err, "error_source": "median over the pairs of each pair's own standard error "
                                                "(1.4826·MAD/√n of its correspondence distances, before)"})
    rec.update(rule_verdict(b, a, np.arange(len(common)), error_m=err, factor=factor,
                            confidence=confidence, mirror=True))
    rec["pairs"] = [{"f": int(p[0]), "g": int(p[1]), "before_m": float(before[p]["disagreement_m"]),
                     "after_m": float(after[p]["disagreement_m"])} for p in common]
    return rec


# ── one seam, every state ─────────────────────────────────────────────────────────────────

def _state_record(P1: np.ndarray, P2: np.ndarray, corr: Dict[str, Any], d1: dict, fit_idx: np.ndarray,
                  held_idx: np.ndarray, radius_bins: Sequence[float], depth_bins: Sequence[float],
                  r_norm: np.ndarray, z_cam: np.ndarray) -> Dict[str, Any]:
    fit = fit_seam(P1, P2, corr["keys"], fit_idx, held_idx)
    if fit["starved"]:
        return {"starved": True, "reason": "robust_rigid starved on the fit half (fewer than 1000 "
                                           "finite correspondences)"}
    comp = camera_components(fit["held_residual"], corr["frame_local"][held_idx], d1, corr["first_local_in_1"])
    rec = {"starved": False, "n_fit": int(len(fit_idx)), "n_fit_used": fit["n_fit_used"],
           "n_held": int(len(held_idx)), "fit_median_m": fit["fit_median_m"],
           "fit_uncertainty_m": fit["fit_uncertainty_m"],
           "held_norm_median_m": float(np.median(np.linalg.norm(fit["held_residual"], axis=1))),
           "held_lateral_median_m": float(np.median(comp["lateral"])),
           "held_axial_abs_median_m": float(np.median(np.abs(comp["axial"]))),
           "held_axial_signed_median_m": float(np.median(comp["axial"])),
           "R": np.asarray(fit["R"]).tolist(), "t": np.asarray(fit["t"]).tolist(),
           "radius_bins": bin_table(comp["lateral"], comp["axial"], r_norm[held_idx], radius_bins, "radius",
                                    depth=z_cam[held_idx]),
           "depth_bins": bin_table(comp["lateral"], comp["axial"], z_cam[held_idx], depth_bins, "depth"),
           "radius_depth_cells": radius_depth_cells(comp["lateral"], comp["axial"], r_norm[held_idx],
                                                    z_cam[held_idx], radius_bins, depth_bins)}
    rec["_held_norm"] = np.linalg.norm(fit["held_residual"], axis=1)       # for the rule (stripped later)
    return rec


def measure_seam(layout: Sequence[Tuple[int, int]], k: int, d1: dict, d2: dict, *, params: Dict[str, Any],
                 owner: np.ndarray, radius_bins: Sequence[float], depth_bins: Sequence[float],
                 common_Ks: Dict[str, Optional[np.ndarray]], judge_samples: int) -> Dict[str, Any]:
    """Everything the instrument says about seam k → k+1, in every state."""
    corr = seam_correspondences(layout, k, d1, d2)
    rec = {"seam": k, "chunks": [k, k + 1], "shared_frames": corr["ov"], "grid_hw": [corr["H"], corr["W"]],
           "n_pixels": corr["n_total"], "n_valid": corr["n_valid"],
           "intrinsics": intrinsics_offsets(d1, d2, layout, k)}
    if corr["n_valid"] < 2000:
        rec["starved"] = True
        rec["reason"] = f"{corr['n_valid']} valid correspondences — too few to split and fit"
        return rec
    fit_idx, held_idx = split_fit_heldout(corr["keys"])
    wp1, wp2 = _wp(d1), _wp(d2)
    P1 = _points_at(wp1[-corr["ov"]:], corr["flat"])
    P2 = _points_at(wp2[:corr["ov"]], corr["flat"])
    Ks1_block = _K(d1)[-corr["ov"]:]
    r_norm = normalized_radius(corr["u"].astype(np.float64), corr["v"].astype(np.float64),
                               corr["frame_local"], Ks1_block)
    dep1 = _depth(d1)[-corr["ov"]:].astype(np.float64)
    z_cam = dep1.reshape(-1)[corr["flat"]]
    states: Dict[str, Dict[str, Any]] = {}
    states["before"] = _state_record(P1, P2, corr, d1, fit_idx, held_idx, radius_bins, depth_bins, r_norm, z_cam)
    # the judged held-out subset (stable): the judges are the shared frames
    judge_idx = held_idx
    if judge_samples and len(held_idx) > int(judge_samples):
        judge_idx = held_idx[stable_pick(corr["keys"][held_idx], int(judge_samples), salt=SALT_JUDGE)]
    judge_pos = np.searchsorted(held_idx, judge_idx)         # positions inside the held arrays
    judge_frames = corr["frame_local"][judge_idx]
    before_norm = None if states["before"].get("starved") else states["before"]["_held_norm"][judge_pos]
    counter: Dict[str, Any] = {}
    # common K counterfactuals (both chunks re-unprojected: the correspondences keep their pixels)
    for name, K in common_Ks.items():
        key = f"common_K_{name}"
        if K is None:
            counter[key] = {"run": False}
            continue
        wp1c = unproject_chunk(d1, K, frames=range(wp1.shape[0] - corr["ov"], wp1.shape[0]))
        wp2c = unproject_chunk(d2, K, frames=range(0, corr["ov"]))
        P1c = _points_at(wp1c, corr["flat"])
        P2c = _points_at(wp2c, corr["flat"])
        st = _state_record(P1c, P2c, corr, d1, fit_idx, held_idx, radius_bins, depth_bins, r_norm, z_cam)
        st["K"] = np.asarray(K).tolist()
        st["run"] = True
        if before_norm is not None and not st.get("starved"):
            err = judge_error_per_judge(before_norm, judge_frames)
            st["rule"] = rule_verdict(before_norm, st["_held_norm"][judge_pos], judge_frames,
                                      error_m=err["error_m"], factor=params["improvement_error_factor"],
                                      confidence=params["heldout_confidence"])
            st["rule"]["error_source"] = ("median over the shared frames (the judges) of each frame's own "
                                          "standard error of its median held-out residual (before)")
            st["rule"]["n_judges_frames"] = err["n_judges"]
        # the pairs held out INSIDE each chunk, before / after (the points and the projection K change together)
        pairs = {}
        for kk, d, wpc in ((k, d1, None), (k + 1, d2, None)):
            before_p = chunk_holdout_pairs(d, layout[kk], owner, params)
            wp_full = unproject_chunk(d, K)
            after_p = chunk_holdout_pairs(d, layout[kk], owner, params, wp_override=wp_full, K_override=K)
            pairs[str(kk)] = pairs_before_after(before_p, after_p, factor=params["improvement_error_factor"],
                                                confidence=params["heldout_confidence"])
        st["intra_chunk_pairs"] = pairs
        counter[key] = st
    # affine seam (the literal 2.0 model) — measured only
    aff = affine_seam_matrix(d1, d2, layout, k)
    wp2a = warp_chunk_affine(d2, aff["A"], aff["middle_shared_block_index"])
    P2a = _points_at(wp2a[:corr["ov"]], corr["flat"])
    st = _state_record(P1, P2a, corr, d1, fit_idx, held_idx, radius_bins, depth_bins, r_norm, z_cam)
    st.update({"run": True, "A": aff["A"].tolist(), "K_k": aff["K_k"].tolist(), "K_k1": aff["K_k1"].tolist(),
               "middle_shared_frame_global": aff["middle_shared_frame_global"],
               "note": "measured only: an affine warp of the whole chunk has no Euclidean camera with one K "
                       "per frame (F0–F5 need one); the intra-chunk pairs after it use the warped points "
                       "with the median K of chunk k and each frame's own camera — a declared approximation"})
    if before_norm is not None and not st.get("starved"):
        err = judge_error_per_judge(before_norm, judge_frames)
        st["rule"] = rule_verdict(before_norm, st["_held_norm"][judge_pos], judge_frames,
                                  error_m=err["error_m"], factor=params["improvement_error_factor"],
                                  confidence=params["heldout_confidence"])
        st["rule"]["error_source"] = ("median over the shared frames (the judges) of each frame's own "
                                      "standard error of its median held-out residual (before)")
        st["rule"]["n_judges_frames"] = err["n_judges"]
    before_p = chunk_holdout_pairs(d2, layout[k + 1], owner, params)
    after_p = chunk_holdout_pairs(d2, layout[k + 1], owner, params, wp_override=wp2a, K_override=aff["K_k"])
    st["intra_chunk_pairs"] = {str(k + 1): pairs_before_after(before_p, after_p,
                                                              factor=params["improvement_error_factor"],
                                                              confidence=params["heldout_confidence"])}
    counter["affine_seam"] = st
    for s in [states["before"]] + list(counter.values()):
        s.pop("_held_norm", None)
    rec["states"] = {"before": states["before"], **counter}
    rec["split"] = {"rule": "stable_pick of half the valid correspondences by their pixel key (salt "
                            f"{SALT_FIT}) fits; the rest is held out; the judged held-out subset is a stable "
                            f"pick of {judge_samples} (salt {SALT_JUDGE}) when larger",
                    "n_judged": int(len(judge_idx))}
    return rec


# ── outputs ───────────────────────────────────────────────────────────────────────────────

def write_csv(path: Path, seam_rec: Dict[str, Any]) -> None:
    cols = ["state", "kind", "lo", "hi", "n"] + list(BIN_STATS)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for name, st in (seam_rec.get("states") or {}).items():
            if not st.get("run", True) or st.get("starved"):
                continue
            for row in st["radius_bins"] + st["depth_bins"]:
                w.writerow({"state": name, **{c: row.get(c) for c in cols if c != "state"}})


def write_png(path: Path, seam_rec: Dict[str, Any]) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 — declared in the JSON, never fatal
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, st in (seam_rec.get("states") or {}).items():
        if not st.get("run", True) or st.get("starved"):
            continue
        rows = [r for r in st["radius_bins"] if r["n"]]
        if not rows:
            continue
        x = [0.5 * (r["lo"] + r["hi"]) for r in rows]
        axes[0].plot(x, [100 * r["lateral_median_m"] for r in rows], marker="o", label=name)
        axes[1].plot(x, [100 * r["axial_abs_median_m"] for r in rows], marker="o", label=name)
    k = seam_rec["seam"]
    axes[0].set_title(f"seam {k}→{k + 1}: held-out residual, LATERAL (median, cm)")
    axes[1].set_title(f"seam {k}→{k + 1}: held-out residual, AXIAL |z| (median, cm)")
    for ax in axes:
        ax.set_xlabel("normalised radius r = ‖(u−cx, v−cy)‖ / f")
        ax.set_ylabel("cm")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-able: {type(o)}")


def run_instrument(run_dir: Path, out_dir: Optional[Path] = None, *, radius_bins=DEFAULT_RADIUS_BINS,
                   depth_bins=DEFAULT_DEPTH_BINS, judge_samples: int = 20000, png: bool = True,
                   log=print) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir / "seam_radial"
    state = verify_state(run_dir)
    log(f"[seam-radial] state proven: {len(state['chunk_indices'])} chunks, stamp {state['fork_stamp'][:12]}…, "
        f"δ {'applied' if state['delta_applied'] else 'none'}, exact_seam_align {state['exact_seam_align']}")
    import yaml
    with open(run_dir / "vggt_omega_config.yaml") as f:
        config = yaml.safe_load(f) or {}
    params = run_params(config)
    layout = [tuple(x) for x in state["chunk_indices"]]
    n_frames = int(layout[-1][1])
    owner = frame_owner(layout, n_frames)
    unaligned = run_dir / "_tmp_results_unaligned"
    # pass 1: every chunk's intrinsics (the session median by owner) and the grid
    chunk_Ks, grid_hw = {}, None
    for k in range(len(layout)):
        d = load_chunk(unaligned / f"chunk_{k}.npy")
        chunk_Ks[k] = _K(d)
        dep = _depth(d)
        grid_hw = (int(dep.shape[1]), int(dep.shape[2]))
        del d
    sm = session_median_K(chunk_Ks, layout, n_frames)
    probe = intake_probe_K(run_dir, grid_hw)
    common_Ks = {"session_median": sm["K"], "intake_probe": probe["K"]}
    log(f"[seam-radial] session median K: fx {sm['fx']:.2f} fy {sm['fy']:.2f} cx {sm['cx']:.2f} cy {sm['cy']:.2f} "
        f"over {sm['n_frames']} owned frames; intake probe K: "
        + (f"fx {probe['K'][0, 0]:.2f} fy {probe['K'][1, 1]:.2f}" if probe["K"] is not None
           else f"not run — {probe['reason']}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    seams = []
    cache: Dict[int, dict] = {}
    for k in range(len(layout) - 1):
        d1 = cache.pop(k, None) or load_chunk(unaligned / f"chunk_{k}.npy")
        d2 = load_chunk(unaligned / f"chunk_{k + 1}.npy")
        rec = measure_seam(layout, k, d1, d2, params=params, owner=owner, radius_bins=radius_bins,
                           depth_bins=depth_bins, common_Ks=common_Ks, judge_samples=judge_samples)
        if not rec.get("starved"):
            b = rec["states"]["before"]
            log(f"[seam-radial] seam {k}->{k + 1}: {rec['n_valid']:,} correspondences, held-out median "
                f"lateral {b['held_lateral_median_m'] * 100:.2f} cm / axial {b['held_axial_abs_median_m'] * 100:.2f} cm"
                + "".join(f"; {n}: {'not run' if not s.get('run', True) else ('starved' if s.get('starved') else f'{s[chr(104) + chr(101) + chr(108) + chr(100) + chr(95) + chr(110) + chr(111) + chr(114) + chr(109) + chr(95) + chr(109) + chr(101) + chr(100) + chr(105) + chr(97) + chr(110) + chr(95) + chr(109)] * 100:.2f} cm')}"
                          for n, s in rec["states"].items() if n != "before"))
            write_csv(out_dir / f"seam_{k}.csv", rec)
            rec["csv"] = str((out_dir / f"seam_{k}.csv").name)
            if png:
                rec["png"] = (str((out_dir / f"seam_{k}.png").name)
                              if write_png(out_dir / f"seam_{k}.png", rec) else None)
        else:
            log(f"[seam-radial] seam {k}->{k + 1}: starved — {rec['reason']}")
        seams.append(rec)
        cache[k + 1] = d2
        del d1
    doc = {"version": 1, "tool": "loop_utils.seam_radial", "state": state, "params": params,
           "bins": {"radius": [float(x) for x in radius_bins], "depth_m": [float(x) for x in depth_bins]},
           "judge_samples": int(judge_samples),
           "session_median_K": {k_: v for k_, v in sm.items() if k_ != "K"} | {"K": sm["K"].tolist()},
           "intake_probe_K": ({k_: v for k_, v in probe.items() if k_ != "K"}
                              | {"K": (None if probe["K"] is None else probe["K"].tolist())}),
           "seams": seams,
           "outputs": {"json": str(out_dir / "seam_radial.json"), "dir": str(out_dir)}}
    with open(out_dir / "seam_radial.json", "w") as f:
        json.dump(doc, f, indent=1, default=_jsonable)
    log(f"[seam-radial] → {out_dir / 'seam_radial.json'}")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m loop_utils.seam_radial",
                                 description="Offline seam instrument on a finished Ω run (claude_stac_slam2.txt "
                                             "phase 1): held-out seam residual by radius / depth, lateral and "
                                             "axial; intrinsics offsets; common-K and affine-seam counterfactuals.")
    ap.add_argument("--run", required=True, help="the maplong_run directory of a finished run")
    ap.add_argument("--out", default=None, help="output directory (default <run>/seam_radial)")
    ap.add_argument("--radius-bins", type=float, nargs="+", default=list(DEFAULT_RADIUS_BINS),
                    help="edges of the normalised-radius bins")
    ap.add_argument("--depth-bins", type=float, nargs="+", default=list(DEFAULT_DEPTH_BINS),
                    help="edges of the depth bins (m)")
    ap.add_argument("--judge-samples", type=int, default=20000,
                    help="held-out correspondences entering the rule per seam (stable pick)")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args(argv)
    try:
        run_instrument(Path(args.run), None if args.out is None else Path(args.out),
                       radius_bins=args.radius_bins, depth_bins=args.depth_bins,
                       judge_samples=args.judge_samples, png=not args.no_png)
    except SeamRadialError as e:
        print(f"[seam-radial] REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
