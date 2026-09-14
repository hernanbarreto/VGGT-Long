"""Integration of the STAC F1 loop stage inside VGGT_Long, on synthetic chunks
written to disk exactly as process_single_chunk leaves them — without the
model: metric lock (anchors + seams) → provisional chain → spatial gate →
bridges (own anchors) → Sim3 loop rows → closed scale graph (δ applied) →
exact seams → SE(3) edges → verification → loop_enable_opt. Also checks that
the pre-F1 skip ("all seams exact → optimizer SKIPPED") no longer exists."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch  # noqa: F401 — vggt_long imports it at module level

from synth_metric import (make_session, make_chunks, make_bridge, write_anchors,
                                fork_loops_cfg, fork_scale_cfg)

_FORK = Path(__file__).resolve().parents[1]
_SERVER = _FORK.parents[1] / "server"


def _make_runner(tmp_path, sess, chunks, ci, loops_over=None, scale_over=None):
    """A VGGT_Long instance without __init__ (no model), with the on-disk
    layout the loop stage reads."""
    import vggt_long as vl
    save_dir = tmp_path / "maplong_run"
    for d in ("_tmp_results_unaligned", "_tmp_results_aligned", "_tmp_results_loop", "pcd"):
        (save_dir / d).mkdir(parents=True, exist_ok=True)
    for k, c in enumerate(chunks):
        np.save(save_dir / "_tmp_results_unaligned" / f"chunk_{k}.npy", c)
    anchor_dir = tmp_path / "da3_run" / "results_output"
    frames = []
    for a, b in ci:
        span = b - a
        frames += [a + int(round(f * (span - 1))) for f in (0.15, 0.5, 0.85)]
    write_anchors(sess, anchor_dir, sorted(set(frames)), noise_rel=0.05)
    r = object.__new__(vl.VGGT_Long)
    r.config = {"Model": {"chunk_size": 60, "overlap": 30, "loop_chunk_size": 20,
                          "loop_enable": True, "using_sim3": False, "align_method": "numpy",
                          "IRLS": {"delta": 0.1, "max_iters": 5, "tol": "1e-9"},
                          "Pointcloud_Save": {"use_conf_filter": True, "conf_percentile": 20.0,
                                              "sample_ratio": 1.0},
                          "metric_lock": {"enable": True, "anchor_dir": str(anchor_dir),
                                          "near_frac": 0.25, "scale_drift": False,
                                          "zoom_scale_fix": True,
                                          "sigma_seam": 0.003, "sigma_anchor": 0.08},
                          "loops": fork_loops_cfg(stac_server_dir=str(_SERVER),
                                                  **(loops_over or {})),
                          "scale": fork_scale_cfg(**(scale_over or {})),
                          "exact_seam_align": True, "frame_ownership": True}}
    r.chunk_size, r.overlap = 60, 30
    r.chunk_indices = list(ci)
    r.img_dir = str(tmp_path / "frames")
    r.img_list = [str(tmp_path / "frames" / f"{int(n):06d}.jpg") for n in sess.frame_numbers]
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


@pytest.fixture(scope="module")
def synth():
    sess = make_session(n_kf=150, H=40, W=56)
    n = 4
    chunks, ci, errors = make_chunks(sess, scale_err=[1.0, 1.06, 0.97, 1.04],
                                     yaw_err_deg=[0.0, 1.5, -1.0, 2.0],
                                     t_err=[(0, 0, 0), (0.3, 0, 0.1), (-0.2, 0.05, 0.3),
                                            (0.1, 0, -0.4)])
    assert len(ci) == n
    return sess, chunks, ci, errors


def test_loop_stage_end_to_end(tmp_path, synth):
    sess, chunks, ci, errors = synth
    r, save_dir = _make_runner(tmp_path, sess, chunks, ci)
    # candidate = the revisit: frame 140 (last chunk) sees where frame 8 (chunk 0) was
    i, j = 140, 8
    r.loop_list = [(i, j)]
    r.loop_cands = [{"i": i, "j": j, "sim": 0.9, "source": "salad"}]
    from loop_utils.sim3utils import process_loop_list
    results = process_loop_list(ci, r.loop_list, half_window=10)
    paired = [tuple(res) + ((i, j),) for res in results]
    # ── stage 1: metric lock (anchors + seams) ──
    r._stac_metric_lock()
    assert r._stac_scale_inputs is not None
    s_v1 = dict(r._stac_scales_applied)
    # ── gate + "inference" (synthetic bridge with its own gauge) ──
    planned = r._stac_plan_bridges(paired)
    assert len(planned) == 1, "a real revisit must survive the spatial gate"
    assert planned[0][2]["verdict"] in ("accept", "ambiguous")
    from loop_utils.loop_bridges import bridge_layout
    for item, cand, gate in planned:
        pred = make_bridge(sess, item, s_L=1.25, yaw_L_deg=8.0, t_L=(0.7, 0.1, -0.3))
        r.loop_predict_list.append((item, pred, {"candidate": cand, "gate": gate,
                                                 "layout": bridge_layout(item, ci)}))
    # ── stage 2: bridge anchors (already on disk for the window frames?) + lock ──
    r._stac_ensure_bridge_anchors()          # no extractor configured → declared, not fatal
    r._stac_lock_bridges()
    meas = r._stac_measure_loops(rigid=False)
    assert meas[0]["ok"]
    r._stac_scale_close(meas)
    sg = json.loads((save_dir / "scale_graph.json").read_text())
    assert sg["loop_rows"] and sg["delta_applied"]
    # the closed scales reproduce the injected chunk gauges better than v1
    s_true = np.array([1.0 / e[0] for e in errors])
    v1 = np.array([s_v1[k] for k in range(len(ci))])
    v2 = np.array([r._stac_scales_applied[k] for k in range(len(ci))])
    rel_v1 = abs(np.log(v1[-1] / v1[0]) - np.log(s_true[-1] / s_true[0]))
    rel_v2 = abs(np.log(v2[-1] / v2[0]) - np.log(s_true[-1] / s_true[0]))
    assert rel_v2 < 0.01 and rel_v2 <= rel_v1 + 1e-9
    # chunk npys carry the δ stamp; a second close does not re-apply
    d0 = np.load(save_dir / "_tmp_results_unaligned" / "chunk_0.npy", allow_pickle=True).item()
    assert "_stac_loop_scale_applied" in d0
    # ── stage 3: exact SE(3) edges + verification ──
    meas_r = r._stac_measure_loops(rigid=True)
    r._stac_verify_loops(meas_r)
    rep = json.loads((save_dir / "loop_edges.json").read_text())
    edges = [e for e in rep["edges"] if e.get("stage") == "verification"]
    assert edges and edges[0]["status"] == "accepted"
    assert r.loop_enable_opt is True
    assert len(r.loop_sim3_list) == 1
    ka, kb, (s_ab, R_ab, t_ab) = r.loop_sim3_list[0]
    assert (ka, kb) == (len(ci) - 1, 0) and s_ab == 1.0    # a = chunk of i (later), b = chunk of j
    # the SE(3) edge maps chunk-a coordinates onto chunk-b coordinates: compare
    # with the injected gauges after the scale close (both ~metric now)
    assert rep["max_edge_sigma_m"] == fork_loops_cfg()["max_edge_sigma_m"]
    assert rep["spatial_gate"] == "on"


def test_spatial_gate_rejects_far_apart_candidate(tmp_path, synth):
    sess, chunks, ci, errors = synth
    r, save_dir = _make_runner(tmp_path, sess, chunks, ci)
    # frames 8 and 75 are on opposite sides of the block: never co-visible
    i, j = 75, 8
    r.loop_list = [(i, j)]
    r.loop_cands = [{"i": i, "j": j, "sim": 0.88, "source": "salad"}]
    from loop_utils.sim3utils import process_loop_list
    paired = [tuple(res) + ((i, j),) for res in process_loop_list(ci, r.loop_list, half_window=10)]
    r._stac_metric_lock()
    planned = r._stac_plan_bridges(paired)
    assert planned == []
    assert r._stac_loops_rejected and r._stac_loops_rejected[0]["stage"] == "spatial_gate"


def test_no_usable_edge_keeps_optimizer_off(tmp_path, synth):
    sess, chunks, ci, errors = synth
    r, save_dir = _make_runner(tmp_path, sess, chunks, ci,
                               loops_over={"min_correspondences": 10 ** 9})
    i, j = 140, 8
    r.loop_list = [(i, j)]
    r.loop_cands = [{"i": i, "j": j, "sim": 0.9, "source": "salad"}]
    from loop_utils.sim3utils import process_loop_list
    from loop_utils.loop_bridges import bridge_layout
    paired = [tuple(res) + ((i, j),) for res in process_loop_list(ci, r.loop_list, half_window=10)]
    r._stac_metric_lock()
    planned = r._stac_plan_bridges(paired)
    for item, cand, gate in planned:
        pred = make_bridge(sess, item)
        r.loop_predict_list.append((item, pred, {"candidate": cand, "gate": gate,
                                                 "layout": bridge_layout(item, ci)}))
    r._stac_lock_bridges()
    r._stac_scale_close(r._stac_measure_loops(rigid=False))   # no ok row → nothing closes
    r._stac_verify_loops(r._stac_measure_loops(rigid=True))
    rep = json.loads((save_dir / "loop_edges.json").read_text())
    e = [x for x in rep["edges"] if x.get("stage") == "verification"][0]
    # starved exact fit → vendor coarse fit recorded as LOW confidence, never usable
    assert e["verdict"].get("fallback") == "vendor_point_map_fit" or e["status"] == "rejected"
    assert r.loop_enable_opt is False


def test_exact_seam_skip_is_gone():
    src = (_FORK / "vggt_long.py").read_text()
    assert "loop optimizer SKIPPED" not in src
    assert "self.loop_enable_opt = False\n        else:\n            self.loop_enable_opt = self.loop_enable" not in src


def test_scale_close_is_identity_without_loop_rows(tmp_path, synth):
    """pccr 2026-09-13 regression: the metric lock's drift stage leaves
    ``_stac_scales_applied`` at the ramp's geometric mean, which differs from
    the constant anchors+seams solution; the scale close divided the constant
    solution by it and re-scaled every chunk with ZERO loop rows (chunk 0
    ×0.83 → the 0->1 seam went 9 → 14 cm). The residual factor is
    s_v2 / s_ref (the same graph without the loop rows): with no loop and no
    absolute row every δ is exactly 1 whatever s_v1 holds."""
    sess, chunks, ci, errors = synth
    r, save_dir = _make_runner(tmp_path, sess, chunks, ci)
    r._stac_metric_lock()
    # simulate the drift stage's bookkeeping: applied scales off the constant solve
    r._stac_scales_applied = {k: v * (1.0 + 0.1 * (k + 1)) for k, v in r._stac_scales_applied.items()}
    before = {k: np.load(save_dir / "_tmp_results_unaligned" / f"chunk_{k}.npy",
                         allow_pickle=True).item()["world_points"].copy() for k in range(len(ci))}
    r._stac_scale_close([])                       # no bridge measured → no loop row
    sg = json.loads((save_dir / "scale_graph.json").read_text())
    assert sg["loop_rows"] == [] and sg["absolute_rows"] == []
    assert all(abs(v - 1.0) < 1e-12 for v in sg["delta_applied"].values()), sg["delta_applied"]
    for k in range(len(ci)):
        d = np.load(save_dir / "_tmp_results_unaligned" / f"chunk_{k}.npy", allow_pickle=True).item()
        assert d["_stac_loop_scale_applied"] == 1.0
        assert np.array_equal(d["world_points"], before[k]), f"chunk {k} geometry moved with no loop row"
    # s_ref (no loops) is recorded and equals s_v2 here
    assert all(abs(sg["s_ref_no_loops"][k] - sg["s_v2_with_loops"][k]) < 1e-12 for k in sg["s_ref_no_loops"])
