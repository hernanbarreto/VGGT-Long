"""The synthetic Ω loop-stage scenario of the VGGT-SLAM 2.0 work (claude_stac_slam2.txt), run
by the fork of ANY commit and compared product by product.

`python tests/slam2_scenario.py <out_dir> [--seams-none] [--gate-off]` runs, on the synthetic
session of server/tests/synth_metric.py (4 chunks, one revisit, DA3 anchors), the stages the
STAC path runs without the model — metric lock → spatial gate → synthetic bridge → bridge lock
→ Sim3 loop rows → closed scale graph → exact seam chain → SE(3) edges → verification — exactly
as tests/test_stac_loop_stage.py drives them, and leaves every product in <out_dir>/maplong_run.
It imports `vggt_long` from the tree it is run in, so the same file run inside a `git archive`
of f576cbf produces THAT commit's products; :func:`compare_products` says whether two such
directories agree bit for bit, ignoring only the fork-stamp fields.

`--seams-none` adds a PRESENT but OFF ``Model.seams`` (focal_model none), `--gate-off` a
present but off ``Model.loops.attention_gate`` — both must leave every product untouched
(rule 5). A tree that predates the sections (f576cbf) ignores the keys."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_FORK = Path(__file__).resolve().parents[1]
_SERVER = _FORK.parents[1] / "server"
for p in (_FORK, _SERVER, _SERVER / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

STAMP_KEYS = ("fork_stamp", "_stac_fork_stamp")

SEAMS_OFF = {"focal_model": "none", "common_K_source": "session_median", "judge_samples": 20000}
GATE_OFF = {"mode": "off", "layer": 4, "min_score": 0.5, "top_fraction": 0.25,
            "salad_similarity_threshold": None,
            "calibration": {"select": "max_auc_lower_ci", "target_precision": 0.95,
                            "min_labels": 5, "n_negatives": 50, "n_boot": 2000,
                            "confidence": 0.95, "salad_grid": [0.55, 0.60, 0.65, 0.70]}}


def make_runner(root: Path, sess, chunks, ci, *, seams=None, gate=None):
    """A VGGT_Long without __init__ (no model) over the on-disk layout the loop stage reads —
    the configuration of tests/test_stac_loop_stage.py, plus the optional slam2 sections."""
    import vggt_long as vl
    from synth_metric import fork_graph_cfg, fork_loops_cfg, fork_scale_cfg, write_anchors
    save_dir = root / "maplong_run"
    for d in ("_tmp_results_unaligned", "_tmp_results_aligned", "_tmp_results_loop", "pcd"):
        (save_dir / d).mkdir(parents=True, exist_ok=True)
    for k, c in enumerate(chunks):
        np.save(save_dir / "_tmp_results_unaligned" / f"chunk_{k}.npy", c)
    anchor_dir = root / "da3_run" / "results_output"
    frames = []
    for a, b in ci:
        span = b - a
        frames += [a + int(round(f * (span - 1))) for f in (0.15, 0.5, 0.85)]
    write_anchors(sess, anchor_dir, sorted(set(frames)), noise_rel=0.05)
    loops = fork_loops_cfg(stac_server_dir=str(_SERVER))
    if gate is not None:
        loops["attention_gate"] = dict(gate)
    model = {"chunk_size": 60, "overlap": 30, "loop_chunk_size": 20,
             "loop_enable": True, "using_sim3": False, "align_method": "numpy",
             "IRLS": {"delta": 0.1, "max_iters": 5, "tol": "1e-9"},
             "Pointcloud_Save": {"use_conf_filter": True, "conf_percentile": 20.0,
                                 "sample_ratio": 1.0},
             "metric_lock": {"enable": True, "anchor_dir": str(anchor_dir),
                             "near_frac": 0.25, "scale_drift": False,
                             "zoom_scale_fix": True,
                             "sigma_seam": 0.003, "sigma_anchor": 0.08},
             "loops": loops, "scale": fork_scale_cfg(), "graph": fork_graph_cfg(),
             "exact_seam_align": True, "frame_ownership": True}
    if seams is not None:
        model["seams"] = dict(seams)
    r = object.__new__(vl.VGGT_Long)
    r.config = {"Model": model}
    r.chunk_size, r.overlap = 60, 30
    r.chunk_indices = list(ci)
    r.img_dir = str(root / "frames")
    r.img_list = [str(root / "frames" / f"{int(n):06d}.jpg") for n in sess.frame_numbers]
    r.output_dir = str(save_dir)
    r.result_unaligned_dir = str(save_dir / "_tmp_results_unaligned")
    r.result_aligned_dir = str(save_dir / "_tmp_results_aligned")
    r.result_loop_dir = str(save_dir / "_tmp_results_loop")
    r.pcd_dir = str(save_dir / "pcd")
    r.loop_enable = True
    r.loop_predict_list = []
    r.loop_sim3_list = []
    r.sim3_list = []
    r.loop_list = []
    r.loop_cands = []
    return r, save_dir


def run_scenario(root: Path, *, seams=None, gate=None) -> Path:
    """Run the loop stage on the synthetic session; returns the maplong_run directory."""
    from synth_metric import make_bridge, make_chunks, make_session
    from loop_utils.loop_bridges import bridge_layout
    from loop_utils.sim3utils import process_loop_list
    root = Path(root)
    sess = make_session(n_kf=150, H=40, W=56)
    chunks, ci, errors = make_chunks(sess, scale_err=[1.0, 1.06, 0.97, 1.04],
                                     yaw_err_deg=[0.0, 1.5, -1.0, 2.0],
                                     t_err=[(0, 0, 0), (0.3, 0, 0.1), (-0.2, 0.05, 0.3),
                                            (0.1, 0, -0.4)])
    r, save_dir = make_runner(root, sess, chunks, ci, seams=seams, gate=gate)
    i, j = 140, 8                                   # the revisit: the last chunk sees chunk 0
    r.loop_list = [(i, j)]
    r.loop_cands = [{"i": i, "j": j, "sim": 0.9, "source": "salad"}]
    paired = [tuple(res) + ((i, j),) for res in process_loop_list(ci, r.loop_list, half_window=10)]
    if hasattr(r, "_stac_slam2_cfg"):               # phase 0 and later: validated by name first
        r._stac_slam2_cfg()
    r._stac_metric_lock()
    planned = r._stac_plan_bridges(paired)
    for item, cand, gate_rec in planned:
        pred = make_bridge(sess, item, s_L=1.25, yaw_L_deg=8.0, t_L=(0.7, 0.1, -0.3))
        r.loop_predict_list.append((item, pred, {"candidate": cand, "gate": gate_rec,
                                                 "layout": bridge_layout(item, ci)}))
    r._stac_ensure_bridge_anchors()
    r._stac_lock_bridges()
    if hasattr(r, "_stac_common_K_bridges"):        # phase 2 and later (no-op with the section absent)
        r._stac_common_K_bridges()
    meas = r._stac_measure_loops(rigid=False)
    r._stac_scale_close(meas)
    seq, seam_report = r._stac_seam_chain("final")
    meas_r = r._stac_measure_loops(rigid=True)
    r._stac_verify_loops(meas_r)
    # products of this scenario beyond the fork's own files: the final seam chain, the locked
    # bridges and the stage's verdicts
    (save_dir / "seam_chain.json").write_text(json.dumps(
        {"seams": [{"s": float(s), "R": np.asarray(R).tolist(), "t": np.asarray(t).tolist()}
                   for s, R, t in seq], "report": seam_report}, indent=1, sort_keys=True))
    for li, (item, pred, meta) in enumerate(r.loop_predict_list):
        np.save(save_dir / f"bridge_{li}.npy", pred)
        (save_dir / f"bridge_{li}_meta.json").write_text(json.dumps(
            {"item": [int(item[0]), [int(x) for x in item[1]], int(item[2]), [int(x) for x in item[3]]],
             "candidate": meta["candidate"], "lock": meta.get("lock")}, indent=1, sort_keys=True,
            default=_jsonable))
    (save_dir / "scenario_summary.json").write_text(json.dumps(
        {"loop_enable_opt": bool(r.loop_enable_opt),
         "loop_sim3_list": [[int(a), int(b), float(s), np.asarray(R).tolist(), np.asarray(t).tolist()]
                            for a, b, (s, R, t) in r.loop_sim3_list],
         "scales_applied": {str(k): float(v) for k, v in sorted(r._stac_scales_applied.items())},
         "n_planned": len(planned), "n_rejected": len(getattr(r, "_stac_loops_rejected", []) or [])},
        indent=1, sort_keys=True))
    return save_dir


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-able: {type(o)}")


def _strip_stamps(doc):
    if isinstance(doc, dict):
        return {k: _strip_stamps(v) for k, v in doc.items() if k not in STAMP_KEYS}
    if isinstance(doc, list):
        return [_strip_stamps(v) for v in doc]
    return doc


def _array_equal_bits(a, b) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _compare_values(path: str, a, b, diffs: list) -> None:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            diffs.append(f"{path}: array on one side only")
        elif not _array_equal_bits(a, b):
            diffs.append(f"{path}: arrays differ (dtype {a.dtype}/{b.dtype}, shape {a.shape}/{b.shape})")
        return
    if isinstance(a, dict) and isinstance(b, dict):
        ka = set(a) - set(STAMP_KEYS)
        kb = set(b) - set(STAMP_KEYS)
        if ka != kb:
            diffs.append(f"{path}: keys differ ({sorted(ka ^ kb)})")
        for k in sorted(ka & kb):
            _compare_values(f"{path}.{k}", a[k], b[k], diffs)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{path}: lengths differ ({len(a)} vs {len(b)})")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _compare_values(f"{path}[{i}]", x, y, diffs)
        return
    if isinstance(a, float) and isinstance(b, float):
        if not (a == b or (np.isnan(a) and np.isnan(b))):
            diffs.append(f"{path}: {a!r} != {b!r}")
        return
    if a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")


def compare_products(dir_a: Path, dir_b: Path) -> list:
    """Every .npy / .json under dir_a against dir_b: [] when identical bit for bit (arrays by
    their bytes, dtype and shape; JSON by value), ignoring only the fork-stamp fields. A file
    on one side only is a difference."""
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    files_a = sorted(p.relative_to(dir_a) for p in dir_a.rglob("*") if p.suffix in (".npy", ".json"))
    files_b = sorted(p.relative_to(dir_b) for p in dir_b.rglob("*") if p.suffix in (".npy", ".json"))
    diffs = []
    for rel in sorted(set(files_a) ^ set(files_b)):
        diffs.append(f"{rel}: on one side only")
    for rel in sorted(set(files_a) & set(files_b)):
        if rel.suffix == ".npy":
            a = np.load(dir_a / rel, allow_pickle=True)
            b = np.load(dir_b / rel, allow_pickle=True)
            if a.dtype == object and a.shape == ():
                a = a.item()
            if b.dtype == object and b.shape == ():
                b = b.item()
        else:
            a = _strip_stamps(json.loads((dir_a / rel).read_text()))
            b = _strip_stamps(json.loads((dir_b / rel).read_text()))
        _compare_values(str(rel), a, b, diffs)
    return diffs


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--seams-none", action="store_true", help="a present but OFF Model.seams")
    ap.add_argument("--gate-off", action="store_true",
                    help="a present but OFF Model.loops.attention_gate")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_dir = run_scenario(out, seams=SEAMS_OFF if args.seams_none else None,
                            gate=GATE_OFF if args.gate_off else None)
    print(f"[slam2-scenario] products in {save_dir} (fork tree {_FORK})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
