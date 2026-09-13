"""claude_stac.txt §12.3 / §12.7c on the fork: the keyframe SE(3) graph stage
inside VGGT_Long on synthetic chunks — intra-chunk drift injected along the
walk (what per-chunk rigid seams cannot remove), one verified exact bridge at
the revisit; the graph closes the loop within tolerance, held-out surface
pairs do not degrade, the §4.7 veto removes a false loop demanding more than
the drift budget, contradictory-only loops leave identity, resume replays."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch  # noqa: F401

from synth_metric import (make_session, make_chunks, make_bridge, write_anchors,
                          fork_loops_cfg, fork_scale_cfg, fork_graph_cfg,
                          fork_authority_cfg, drift_field, yaw_R)
from test_stac_loop_stage import _make_runner as _base_runner

_FORK = Path(__file__).resolve().parents[1]


def _runner(tmp_path, sess, chunks, ci, graph_over=None, auth_over=None):
    r, save_dir = _base_runner(tmp_path, sess, chunks, ci)
    r.config["Model"]["graph"] = fork_graph_cfg(**(graph_over or {}))
    r.config["Model"]["authority"] = fork_authority_cfg(**(auth_over or {}))
    r.config["Model"]["certify"] = {"ensemble_offset_frames": 0}
    return r, save_dir


def _align_and_write(r, save_dir, ci):
    """What process_long_sequence does between the seams and the post-alignment
    stages: exact seams → accumulated chain → aligned npys (world_points moved,
    extrinsic left chunk-local, as the vendor apply loop leaves them)."""
    from loop_utils.sim3utils import accumulate_sim3_transforms, apply_sim3_direct
    seq, rep = r._stac_seam_chain("final")
    r._stac_drop_chunk_cache()
    r._stac_seam_residuals = {int(k): float(v["residual_m"]) for k, v in rep.items()
                              if v.get("exact")}
    r.sim3_list = accumulate_sim3_transforms(seq)
    for k in range(len(ci)):
        d = np.load(save_dir / "_tmp_results_unaligned" / f"chunk_{k}.npy", allow_pickle=True).item()
        if k > 0:
            s, R, t = r.sim3_list[k - 1]
            d["world_points"] = apply_sim3_direct(d["world_points"], s, R, t)
        np.save(save_dir / "_tmp_results_aligned" / f"chunk_{k}.npy", d)


def _run_loop_stage(r, sess, ci, cand, bridge_kw=None):
    from loop_utils.sim3utils import process_loop_list
    from loop_utils.loop_bridges import bridge_layout
    i, j = cand
    r.loop_list = [(i, j)]
    r.loop_cands = [{"i": i, "j": j, "sim": 0.9, "source": "salad"}]
    paired = [tuple(res) + ((i, j),) for res in process_loop_list(ci, r.loop_list, half_window=10)]
    r._stac_metric_lock()
    planned = r._stac_plan_bridges(paired)
    assert planned, "the revisit must pass the spatial gate"
    for item, c, gate in planned:
        pred = make_bridge(sess, item, **(bridge_kw or {}))
        r.loop_predict_list.append((item, pred, {"candidate": c, "gate": gate,
                                                 "layout": bridge_layout(item, ci)}))
    r._stac_lock_bridges()
    r._stac_scale_close(r._stac_measure_loops(rigid=False))
    r._stac_verify_loops(r._stac_measure_loops(rigid=True))


def _chain_pose(r, g, ci):
    """Per-frame c2w as the cloud sees it after the stage (owner copy)."""
    from loop_utils.metric_lock import frame_owner
    owner = frame_owner(ci, len(r.img_list))
    k = int(owner[g])
    d = np.load(Path(r.result_aligned_dir) / f"chunk_{k}.npy", allow_pickle=True).item()
    ext = np.asarray(d["extrinsic"])
    return r._stac_aligned_pose(k, g - ci[k][0], ext[g - ci[k][0]])


@pytest.fixture(scope="module")
def synth_drift():
    sess = make_session(n_kf=150, H=40, W=56)
    # 3.6 mm per keyframe (≈8 mm per metre walked, inside the configured drift
    # budget of 13 mm/m) accumulate to ~0.55 m over the walk — far beyond the
    # seams' reach, inside the pose-graph authority. (A drift ABOVE the budget
    # is, by §4.7, indistinguishable from a false loop and gets vetoed.)
    D = drift_field(sess.n_kf, yaw_deg_per_kf=0.0, t_per_kf=(0.003, 0.0, 0.002))
    chunks, ci, errors = make_chunks(sess, scale_err=[1.0, 1.03, 0.98, 1.02],
                                     yaw_err_deg=[0.0, 1.0, -0.5, 1.5],
                                     t_err=[(0, 0, 0), (0.2, 0, 0.1), (-0.1, 0.05, 0.2),
                                            (0.1, 0, -0.3)], drift=D)
    return sess, chunks, ci, errors, D


def test_graph_closes_the_injected_drift(tmp_path, synth_drift):
    sess, chunks, ci, errors, D = synth_drift
    r, save_dir = _runner(tmp_path, sess, chunks, ci)
    _run_loop_stage(r, sess, ci, (140, 8), bridge_kw={"s_L": 1.2, "yaw_L_deg": 5.0,
                                                       "t_L": (0.5, 0.1, -0.2)})
    assert r._stac_loop_edges_kf, "a verified bridge must yield a keyframe edge"
    _align_and_write(r, save_dir, ci)
    # BEFORE: the loop endpoints disagree with the ground truth by the drift
    e = r._stac_loop_edges_kf[0]
    gi, gj = int(e["i"]), int(e["j"])
    Ti0, Tj0 = _chain_pose(r, gi, ci), _chain_pose(r, gj, ci)
    Z_gt = np.linalg.inv(sess.poses[gi]) @ sess.poses[gj]
    Z_before = np.linalg.inv(Ti0) @ Tj0
    err_before = float(np.linalg.norm((np.linalg.inv(Z_gt) @ Z_before)[:3, 3]))
    assert err_before > 0.2, err_before
    r._stac_uncertainty()
    assert Path(save_dir / "uncertainty.json").exists()
    r._stac_pose_graph()
    rep = json.loads((save_dir / "pose_graph.json").read_text())
    assert rep["verdict"] == "APPLY", rep.get("gates")
    assert rep["gates"]["loop_gain"]["passed"] and rep["gates"]["holdout_surface_pairs"]["passed"]
    Ti1, Tj1 = _chain_pose(r, gi, ci), _chain_pose(r, gj, ci)
    Z_after = np.linalg.inv(Ti1) @ Tj1
    err_after = float(np.linalg.norm((np.linalg.inv(Z_gt) @ Z_after)[:3, 3]))
    assert err_after < 0.05, (err_before, err_after)
    # authority.json carries the stage's fraction
    auth = json.loads((save_dir / "authority.json").read_text())
    assert "pose_graph" in auth and 0.0 < auth["pose_graph"]["fraction_used"] < 1.0
    # the npys are stamped; a second call replays the persisted verdict (resume)
    d0 = np.load(save_dir / "_tmp_results_aligned" / "chunk_0.npy", allow_pickle=True).item()
    assert d0.get("_stac_pose_graph_applied")
    r._stac_pose_graph()
    Ti2 = _chain_pose(r, gi, ci)
    assert np.allclose(Ti2, Ti1)


def test_false_loop_is_vetoed_by_authority(tmp_path, synth_drift):
    sess, chunks, ci, errors, D = synth_drift
    r, save_dir = _runner(tmp_path, sess, chunks, ci)
    _run_loop_stage(r, sess, ci, (140, 8))
    _align_and_write(r, save_dir, ci)
    # a LIAR bypassing the gate: an edge that demands 8 m at the revisit
    honest = dict(r._stac_loop_edges_kf[0])
    Z = np.asarray(honest["Z"], np.float64).copy()
    Z[:3, 3] += np.array([8.0, 0.0, 0.0])
    liar = dict(honest, Z=Z.tolist(), bridge=99)
    r._stac_loop_edges_kf = [honest, liar]
    r._stac_uncertainty()
    r._stac_pose_graph()
    rep = json.loads((save_dir / "pose_graph.json").read_text())
    assert rep["vetoed"] and rep["vetoed"][0]["bridge"] == 99
    assert rep["n_loop_edges_active"] == 1
    assert rep["verdict"] == "APPLY"


def test_contradictory_only_loops_keep_identity(tmp_path, synth_drift):
    sess, chunks, ci, errors, D = synth_drift
    r, save_dir = _runner(tmp_path, sess, chunks, ci, auth_over={"pose_graph_max_m": 0.05})
    _run_loop_stage(r, sess, ci, (140, 8))
    _align_and_write(r, save_dir, ci)
    r._stac_uncertainty()
    r._stac_pose_graph()
    rep = json.loads((save_dir / "pose_graph.json").read_text())
    # closing 1.2 m of drift needs more than the 5 cm authority → IDENTITY
    assert rep["verdict"] == "IDENTITY"
    assert rep["authority"]["exceeded"] is True
    d0 = np.load(save_dir / "_tmp_results_aligned" / "chunk_0.npy", allow_pickle=True).item()
    assert not d0.get("_stac_pose_graph_applied")


def test_no_loop_edge_means_identity(tmp_path, synth_drift):
    sess, chunks, ci, errors, D = synth_drift
    r, save_dir = _runner(tmp_path, sess, chunks, ci)
    r._stac_metric_lock()
    _align_and_write(r, save_dir, ci)
    r._stac_loop_edges_kf = []
    r._stac_uncertainty()
    r._stac_pose_graph()
    rep = json.loads((save_dir / "pose_graph.json").read_text())
    assert rep["verdict"] == "IDENTITY" and rep["n_loop_edges"] == 0


def test_vendor_optimizer_bypassed_in_stac_path():
    src = (_FORK / "vggt_long.py").read_text()
    assert "if self.loop_enable_opt and not _stac_kf_graph:" in src
    assert "self._stac_pose_graph()" in src
