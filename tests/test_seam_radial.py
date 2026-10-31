"""claude_stac_slam2.txt, phase 1 — the offline seam instrument (loop_utils.seam_radial) on
synthetic chunks of known camera, depth and poses. No GPU, no model.

(a) two inferences with the same K → the held-out residual is flat in r and the intrinsics
    offset is exactly 0; (b) a focal 3 % larger in chunk k+1 → the LATERAL residual grows with r,
    the AXIAL does not, the offset is recovered and both counterfactuals lower the held-out;
(c) an ambiguous npy state → the tool refuses by name."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_FORK = Path(__file__).resolve().parents[1]
_SERVER = _FORK.parents[1] / "server"
for p in (_FORK, _SERVER, _SERVER / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from loop_utils import seam_radial as SR                                   # noqa: E402
from synth_metric import (corridor_scene, fork_graph_cfg, look_c2w, make_chunks,   # noqa: E402
                          make_session)

STAMP = "a1b2c3d4e5f6" * 4


def _reunproject_with(chunk: dict, K: np.ndarray) -> dict:
    """The same depth and cameras, unprojected with another K and declaring it: what a second
    inference of the same frames with a different focal would have written."""
    d = dict(chunk)
    d["world_points"] = SR.unproject_chunk(chunk, K).astype(np.float32)
    d["intrinsic"] = np.tile(np.asarray(K, np.float64)[None], (chunk["intrinsic"].shape[0], 1, 1))
    return d


def _write_run(root: Path, chunks, ci, *, delta=None, probe_K=None, native_wh=None,
               stamp_npy=STAMP, lock_all=True, with_sim3=True) -> Path:
    """A finished run's directory as the fork leaves it with Model.keep_unaligned_chunks:
    the chunk npys (stamped, in their final state), fork_stamp.json, metric_lock.json,
    chunk_sim3.json, optionally scale_graph.json, and the run config."""
    run = root / "output" / "maplong_run"
    (run / "_tmp_results_unaligned").mkdir(parents=True)
    for k, c in enumerate(chunks):
        d = dict(c)
        if stamp_npy is not None:
            d[SR.STAMP_KEY] = stamp_npy
        d[SR.RANGE_KEY] = [int(ci[k][0]), int(ci[k][1])]
        if delta is not None:
            d[SR.LOOP_SCALE_KEY] = float(delta[k])
        np.save(run / "_tmp_results_unaligned" / f"chunk_{k}.npy", d)
    (run / "fork_stamp.json").write_text(json.dumps({"sha256": STAMP, "stamp_version": 1}))
    lock = {"fork_stamp": STAMP, "chunk_indices": [list(x) for x in ci],
            "chunks": {str(k): ({"s": 1.0, "n_anchors": 3} if lock_all or k == 0 else {"s": None, "n_anchors": 0})
                       for k in range(len(ci))},
            "drift": {"verdict": "SKIP"}}
    (run / "metric_lock.json").write_text(json.dumps(lock))
    if with_sim3:
        (run / "chunk_sim3.json").write_text(json.dumps(
            {"chunk_indices": [list(x) for x in ci],
             "sim3": [{"s": 1.0, "R": np.eye(3).tolist(), "t": [0, 0, 0]} for _ in ci[1:]]}))
    if delta is not None:
        (run / "scale_graph.json").write_text(json.dumps(
            {"fork_stamp": STAMP, "delta_applied": {str(k): float(v) for k, v in enumerate(delta)}}))
    cfg = {"Model": {"chunk_ranges": [list(x) for x in ci], "graph": fork_graph_cfg(),
                     "exact_seam_align": True, "frame_ownership": True, "pose_fit_conf_min_norm": 0.0,
                     "loops": {"anchors_per_bridge": 3}}}
    (run / "vggt_omega_config.yaml").write_text(yaml.safe_dump(cfg))
    if probe_K is not None:
        (root / "intake").mkdir(exist_ok=True)
        (root / "intake" / "focal_probe.json").write_text(json.dumps(
            {"version": 2, "K": np.asarray(probe_K).tolist(), "fx": float(probe_K[0, 0]),
             "fy": float(probe_K[1, 1]), "cx": float(probe_K[0, 2]), "cy": float(probe_K[1, 2]),
             "native_wh": list(native_wh)}))
    return run


@pytest.fixture(scope="module")
def synth():
    sess = make_session(n_kf=90, H=40, W=56)
    chunks, ci, _ = make_chunks(sess, chunk_size=60, overlap=30, depth_noise_rel=0.003)
    assert len(ci) == 2 and ci == [(0, 60), (30, 90)]
    return sess, chunks, ci


@pytest.fixture(scope="module")
def synth_exact():
    """The same session without depth noise: two inferences of the same frames with the same K
    and the same depth agree exactly (up to float32)."""
    sess = make_session(n_kf=90, H=40, W=56)
    chunks, ci, _ = make_chunks(sess, chunk_size=60, overlap=30, depth_noise_rel=0.0,
                                scale_err=[1.0, 1.04], yaw_err_deg=[0.0, 2.0], t_err=[(0, 0, 0), (0.3, 0.1, -0.2)])
    return sess, chunks, ci


@pytest.fixture(scope="module")
def synth_straight():
    """A straight corridor walked along the camera's own z axis, every camera parallel: the
    paper's affine model (one A in the middle shared frame's camera) is then EXACT for every
    shared frame — (I − A) acts on the camera offsets, which lie along z where A is 1."""
    n = 90
    poses = np.stack([look_c2w((1.0 + 33.0 * i / (n - 1), 1.5, 0.0), np.array([1.0, 0.0, 0.0]))
                      for i in range(n)])
    sess = make_session(n_kf=n, H=40, W=56, scene=corridor_scene(length_m=40.0), poses=poses)
    chunks, ci, _ = make_chunks(sess, chunk_size=60, overlap=30, depth_noise_rel=0.003)
    assert ci == [(0, 60), (30, 90)]
    return sess, chunks, ci


def _lat_ax_by_radius(seam_rec, state="before", angular=False):
    rows = [r for r in seam_rec["states"][state]["radius_bins"] if r["n"] >= 200]
    if angular:
        return ([r["lateral_over_depth_median"] for r in rows], [r["axial_abs_over_depth_median"] for r in rows],
                [0.5 * (r["lo"] + r["hi"]) for r in rows])
    return ([r["lateral_median_m"] for r in rows], [r["axial_abs_median_m"] for r in rows],
            [0.5 * (r["lo"] + r["hi"]) for r in rows])


def test_same_K_exact_depth_flat_zero_residual_and_zero_offset(tmp_path, synth_exact):
    """(a) the same K in both inferences, the same depth: whatever the chunk gauges (the rigid
    fit absorbs the injected yaw / translation; the metric lock would have removed the scale),
    the held-out residual is flat in r at the float32 floor and the offset is exactly 0."""
    sess, chunks, ci = synth_exact
    # the lock's job, done here: chunk 1's injected scale removed (depth, points, translations)
    from loop_utils.metric_lock import apply_scale
    chunks = [chunks[0], apply_scale(dict(chunks[1]), 1.0 / 1.04)]
    run = _write_run(tmp_path, chunks, ci)
    doc = SR.run_instrument(run, judge_samples=5000, png=False, log=lambda *a: None)
    s = doc["seams"][0]
    assert not s.get("starved") and s["shared_frames"] == 30
    off = s["intrinsics"]["offset_median_px"]
    assert all(v == 0.0 for v in off.values())
    lat, ax, r = _lat_ax_by_radius(s)
    assert len(lat) >= 3, s["states"]["before"]["radius_bins"]
    assert max(lat) < 1e-4 and max(ax) < 1e-4, (lat, ax, r)          # flat: the float32 floor


def test_same_K_noisy_depth_common_K_is_the_identity(tmp_path, synth):
    """(a) with depth noise along the rays the residual follows the ray geometry (its lateral part
    grows with r like the ray's tilt); the same K is the session median, so the common-K
    counterfactual reproduces the state exactly and the rule says no improvement."""
    sess, chunks, ci = synth
    run = _write_run(tmp_path, chunks, ci)
    doc = SR.run_instrument(run, judge_samples=5000, png=False, log=lambda *a: None)
    s = doc["seams"][0]
    assert not s.get("starved") and s["shared_frames"] == 30
    off = s["intrinsics"]["offset_median_px"]
    assert all(v == 0.0 for v in off.values()) and all(v == 0.0 for v in s["intrinsics"]["offset_spread_sigma_px"].values())
    before = s["states"]["before"]
    # the same K is the session median → the counterfactual reproduces the state exactly
    cm = s["states"]["common_K_session_median"]
    assert cm["run"] and np.allclose(cm["K"], sess.K)
    # the stored points are float32, the re-unprojection float64 from the float32 depth: the
    # state is reproduced to the float32 rounding of the points (≈ 1e-7 relative), no more
    assert cm["rule"]["median_before_m"] == pytest.approx(cm["rule"]["median_after_m"], rel=0, abs=1e-6)
    assert cm["rule"]["improves_by_rule"] is False
    assert before["held_norm_median_m"] < 0.05       # 0.3 % depth noise on a ~5 m corridor
    # no intake probe for this session: declared, not run
    assert s["states"]["common_K_intake_probe"]["run"] is False
    assert "does not exist" in doc["intake_probe_K"]["reason"]
    # the pairs inside each chunk did not worsen (the points did not move)
    for kk, pr in cm["intra_chunk_pairs"].items():
        assert pr["n_pairs_judged"] >= 5, pr
        assert pr["worsens_by_rule"] is False


def test_focal_error_shows_in_the_lateral_residual_and_both_counterfactuals_lower_it(tmp_path, synth_straight):
    sess, chunks, ci = synth_straight
    Kbad = sess.K.copy()
    Kbad[0, 0] *= 1.03
    Kbad[1, 1] *= 1.03
    chunks2 = [chunks[0], _reunproject_with(chunks[1], Kbad)]
    run = _write_run(tmp_path, chunks2, ci, probe_K=sess.K, native_wh=(sess.W, sess.H))
    doc = SR.run_instrument(run, judge_samples=5000, png=True, log=lambda *a: None)
    s = doc["seams"][0]
    off = s["intrinsics"]["offset_median_px"]
    assert off["fx"] == pytest.approx(0.03 * sess.K[0, 0], rel=1e-9)
    assert off["fy"] == pytest.approx(0.03 * sess.K[1, 1], rel=1e-9)
    assert off["cx"] == 0.0 and off["cy"] == 0.0
    # the ANGULAR lateral residual (lateral / depth) grows with the radius — the focal error moves
    # a point by ≈ z·r·3 % whatever the scene puts at each depth (a corridor seen along its axis
    # has its FAR points at the image centre) — far beyond what the depth noise alone does there;
    # the axial component does not grow
    lat, ax, r = _lat_ax_by_radius(s, angular=True)
    assert len(lat) >= 3
    run_same = _write_run(tmp_path / "same", chunks, ci)
    doc_same = SR.run_instrument(run_same, judge_samples=5000, png=False, log=lambda *a: None)
    lat_same, ax_same, _ = _lat_ax_by_radius(doc_same["seams"][0], angular=True)
    assert lat[-1] > 2.0 * lat[0], (lat, r)
    assert lat[-1] > 5.0 * lat_same[-1], (lat[-1], lat_same[-1])
    assert all(b >= a * 0.8 for a, b in zip(lat[:-1], lat[1:])), lat       # monotone within noise
    assert ax[-1] < 1.5 * max(ax[0], 1e-5), (ax, r)
    assert ax[-1] < 2.0 * ax_same[-1], (ax[-1], ax_same[-1])
    # the joint table says the same at ONE depth band
    cells = [c for c in s["states"]["before"]["radius_depth_cells"] if c["z_lo"] == 4.0 and c["n"] >= 100]
    if len(cells) >= 2:
        assert cells[-1]["lateral_median_m"] > cells[0]["lateral_median_m"]
    before = s["states"]["before"]
    for name in ("common_K_session_median", "common_K_intake_probe", "affine_seam"):
        st = s["states"][name]
        assert st["run"] and not st.get("starved"), name
        assert st["held_norm_median_m"] < 0.5 * before["held_norm_median_m"], (name, st["held_norm_median_m"], before["held_norm_median_m"])
        assert st["rule"]["median_after_m"] < st["rule"]["median_before_m"]
        assert st["rule"]["decide_change"]["significant"] is True, st["rule"]
        assert st["rule"]["n_judges_frames"] == 30
    # the session median K by owner is the TRUE K here (chunk 0 owns 46 frames, chunk 1 44)
    assert np.allclose(s["states"]["common_K_session_median"]["K"], sess.K)
    assert np.allclose(s["states"]["common_K_intake_probe"]["K"], sess.K)
    assert doc["intake_probe_K"]["scale"] == [1.0, 1.0]
    # the whole-chunk re-unprojection with the common K does not worsen chunk 1's own pairs
    pr = s["states"]["common_K_session_median"]["intra_chunk_pairs"]["1"]
    assert pr["n_pairs_judged"] >= 5 and pr["worsens_by_rule"] is False
    # outputs: JSON + CSV + PNG in the run directory
    out = Path(doc["outputs"]["dir"])
    assert (out / "seam_radial.json").is_file() and (out / "seam_0.csv").is_file() and (out / "seam_0.png").is_file()
    rows = (out / "seam_0.csv").read_text().splitlines()
    assert rows[0].startswith("state,kind,lo,hi,n,lateral_median_m")
    assert any(r.startswith("affine_seam,radius,") for r in rows)


@pytest.mark.parametrize("break_", ["stamp", "no_lock", "mark_without_graph", "graph_without_mark",
                                    "no_sim3", "wrong_mark", "missing_npy"])
def test_an_ambiguous_state_is_refused_by_name(tmp_path, synth, break_):
    sess, chunks, ci = synth
    if break_ == "stamp":
        run = _write_run(tmp_path, chunks, ci, stamp_npy="ffff" * 16)
        expect = "chunk_0.npy carries fork stamp"
    elif break_ == "no_lock":
        run = _write_run(tmp_path, chunks, ci, lock_all=False)
        expect = "does not record chunk 1 as locked"
    elif break_ == "mark_without_graph":
        run = _write_run(tmp_path, chunks, ci, delta=[1.0, 1.001])
        (run / "scale_graph.json").unlink()
        expect = "carries a loop-scale mark"
    elif break_ == "graph_without_mark":
        run = _write_run(tmp_path, chunks, ci)
        (run / "scale_graph.json").write_text(json.dumps(
            {"fork_stamp": STAMP, "delta_applied": {"0": 1.0, "1": 1.0004}}))
        expect = "carries no loop-scale mark"
    elif break_ == "no_sim3":
        run = _write_run(tmp_path, chunks, ci, with_sim3=False)
        expect = "chunk_sim3.json"
    elif break_ == "wrong_mark":
        run = _write_run(tmp_path, chunks, ci, delta=[1.0, 1.001])
        (run / "scale_graph.json").write_text(json.dumps(
            {"fork_stamp": STAMP, "delta_applied": {"0": 1.0, "1": 1.002}}))
        expect = "carries loop-scale mark 1.001"
    else:
        run = _write_run(tmp_path, chunks, ci)
        (run / "_tmp_results_unaligned" / "chunk_1.npy").unlink()
        expect = "chunk_1.npy is missing"
    with pytest.raises(SR.SeamRadialError) as e:
        SR.verify_state(run)
    assert expect in str(e.value), str(e.value)


def test_a_consistent_closed_state_is_accepted(tmp_path, synth):
    sess, chunks, ci = synth
    run = _write_run(tmp_path, chunks, ci, delta=[1.0, 1.0004])
    st = SR.verify_state(run)
    assert st["delta_applied"] == {"0": 1.0, "1": 1.0004}
    assert [c["loop_scale_mark"] for c in st["chunks"]] == [1.0, 1.0004]


def test_the_cli_refuses_with_exit_2_and_names_the_reason(tmp_path, synth):
    sess, chunks, ci = synth
    run = _write_run(tmp_path, chunks, ci, stamp_npy=None)
    env = dict(os.environ, OMP_NUM_THREADS="2", PYTHONHASHSEED="0")
    env.pop("PYTHONPATH", None)
    r = subprocess.run([sys.executable, "-m", "loop_utils.seam_radial", "--run", str(run), "--no-png"],
                       cwd=str(_FORK), env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 2
    assert "REFUSED" in r.stderr and "carries no fork stamp" in r.stderr


def test_session_median_K_counts_every_frame_once_by_owner():
    ci = [(0, 60), (30, 90)]
    K0 = np.array([[100.0, 0, 20], [0, 100.0, 30], [0, 0, 1]])
    K1 = K0.copy(); K1[0, 0] = K1[1, 1] = 130.0
    Ks = {0: np.tile(K0[None], (60, 1, 1)), 1: np.tile(K1[None], (60, 1, 1))}
    sm = SR.session_median_K(Ks, ci, 90)
    # chunk 0 owns frames 0..45 (46), chunk 1 owns 46..89 (44): the median is chunk 0's focal
    assert sm["n_frames"] == 90 and sm["fx"] == 100.0 and sm["cx"] == 20.0
