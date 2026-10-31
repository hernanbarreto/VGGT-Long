"""claude_stac_slam2.txt, phase 2 — ONE calibration for every Ω inference (loop_utils.common_k,
VGGT_Long._stac_common_K_chunks / _bridges), on the synthetic scene of phase 1. No GPU, no model."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_FORK = Path(__file__).resolve().parents[1]
_SERVER = _FORK.parents[1] / "server"
for p in (_FORK, _SERVER, _SERVER / "tests", _FORK / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from loop_utils import common_k as CK                                            # noqa: E402
from slam2_scenario import make_runner                                           # noqa: E402
from synth_metric import (corridor_scene, look_c2w, make_bridge, make_chunks,     # noqa: E402
                          make_session, render)

SEAMS_MEDIAN = {"focal_model": "common_K", "common_K_source": "session_median", "judge_samples": 4000}
SEAMS_PROBE = {"focal_model": "common_K", "common_K_source": "intake_probe", "judge_samples": 4000}


def _straight_session(n=90, length=33.0):
    poses = np.stack([look_c2w((1.0 + length * i / (n - 1), 1.5, 0.0), np.array([1.0, 0.0, 0.0]))
                      for i in range(n)])
    return make_session(n_kf=n, H=40, W=56, scene=corridor_scene(length_m=40.0), poses=poses)


def _with_focal(chunk: dict, K: np.ndarray) -> dict:
    """The same depth and cameras unprojected with another K, declaring it (a second inference of
    the same frames that estimated another focal)."""
    d = dict(chunk)
    d["world_points"] = CK.unproject_chunk(chunk, K).astype(np.float32)
    d["intrinsic"] = np.tile(np.asarray(K, np.float64)[None], (chunk["intrinsic"].shape[0], 1, 1))
    return d


def _bad_K(K, f=1.03):
    Kb = np.asarray(K, np.float64).copy()
    Kb[0, 0] *= f
    Kb[1, 1] *= f
    return Kb


def _runner(tmp_path, sess, chunks, ci, seams):
    r, save_dir = make_runner(tmp_path, sess, chunks, ci, seams=seams)
    r.all_camera_intrinsics = [(tuple(c), np.asarray(ch["intrinsic"]).copy()) for c, ch in zip(ci, chunks)]
    r.all_camera_poses = [(tuple(c), np.asarray(ch["extrinsic"]).copy()) for c, ch in zip(ci, chunks)]
    return r, save_dir


def _npy(save_dir, k):
    return np.load(save_dir / "_tmp_results_unaligned" / f"chunk_{k}.npy", allow_pickle=True).item()


def _seam_held_median(r, k=0):
    """The exact seam's held-out residual median on the chunks as they are on disk."""
    from loop_utils import seam_radial as SR
    d1, d2 = _npy(Path(r.output_dir), k), _npy(Path(r.output_dir), k + 1)
    layout = [tuple(c) for c in r.chunk_indices]
    corr = SR.seam_correspondences(layout, k, d1, d2)
    fit_idx, held_idx = SR.split_fit_heldout(corr["keys"])
    P1 = SR._points_at(SR._wp(d1)[-corr["ov"]:], corr["flat"])
    P2 = SR._points_at(SR._wp(d2)[:corr["ov"]], corr["flat"])
    f = SR.fit_seam(P1, P2, corr["keys"], fit_idx, held_idx)
    return float(np.median(np.linalg.norm(f["held_residual"], axis=1)))


@pytest.fixture(scope="module")
def straight():
    sess = _straight_session()
    chunks, ci, _ = make_chunks(sess, chunk_size=60, overlap=30, depth_noise_rel=0.003)
    return sess, chunks, ci


def test_common_K_recovers_the_injected_focal_and_lowers_the_seam(tmp_path, straight):
    sess, chunks, ci = straight
    Kb = _bad_K(sess.K)
    chunks2 = [chunks[0], _with_focal(chunks[1], Kb)]
    r, save_dir = _runner(tmp_path, sess, chunks2, ci, SEAMS_MEDIAN)
    before = _seam_held_median(r)
    r._stac_metric_lock()                                   # → _stac_common_K_chunks at its end
    doc = json.loads((save_dir / CK.REPORT_NAME).read_text())
    assert doc["run"] and doc["applied"], doc.get("reason")
    assert np.allclose(doc["K"], sess.K)                    # chunk 0 owns 46 frames, chunk 1 44: the median is K
    assert doc["seams_rule"]["improves_by_rule"] and doc["seams_rule"]["decide_change"]["significant"]
    assert doc["pairs_rule"]["worsens_by_rule"] is False and doc["pairs_rule"]["n_pairs_judged"] >= 5
    assert doc["chunks_excluded_zoom"] == []
    for k in (0, 1):
        d = _npy(save_dir, k)
        assert d[CK.MARK_KEY]["applied"] and np.allclose(d["intrinsic"], sess.K)
        # the depth and the cameras did not move; the points are the depth re-lifted along the new rays
        assert np.array_equal(d["depth"], _npy_locked_depth(chunks2[k], r, k))
        ref = CK.unproject_chunk(d, sess.K)
        assert np.allclose(d["world_points"], ref, atol=2e-4)
    after = _seam_held_median(r)
    assert after < 0.5 * before, (before, after)
    # the intrinsics of record (what save_camera_poses writes into intrinsic.txt for F0) follow
    assert all(np.allclose(Ks, sess.K) for _rng, Ks in r.all_camera_intrinsics)


def _npy_locked_depth(chunk, r, k):
    """The chunk's depth after the metric lock (scaled by the lock's factor) — read back from the
    lock's own record so the test does not re-derive the lock."""
    rec = json.load(open(Path(r.output_dir) / "metric_lock.json"))["chunks"][str(k)]
    s = rec.get("s_applied", rec.get("s_graph", rec.get("s")))
    return (np.asarray(chunk["depth"]) * float(s)).astype(np.asarray(chunk["depth"]).dtype)


def test_equal_K_is_the_exact_identity(tmp_path, straight):
    sess, chunks, ci = straight
    r, save_dir = _runner(tmp_path, sess, chunks, ci, SEAMS_MEDIAN)
    r._stac_metric_lock()
    locked = {k: _npy(save_dir, k) for k in (0, 1)}
    doc = json.loads((save_dir / CK.REPORT_NAME).read_text())
    assert doc["run"] and doc["applied"] is False
    assert "does not improve" in doc["reason"]
    for k in (0, 1):
        d = _npy(save_dir, k)
        assert CK.MARK_KEY not in d
        for key in ("world_points", "intrinsic", "depth", "extrinsic"):
            assert np.array_equal(np.asarray(d[key]), np.asarray(locked[k][key]))
    assert all(np.allclose(Ks, sess.K) for _rng, Ks in r.all_camera_intrinsics)


def test_a_common_K_that_worsens_the_chunks_own_pairs_stays_identity(tmp_path):
    """Chunk 1: its shared frames 30–59 with a 3 % focal error (the seam would improve), its own
    frames 60–89 TRULY zoomed (depth rendered with a 15 % longer focal, points right under it). The
    common K fixes the seam and breaks the chunk's own geometry: ITS held-out pairs worsen by the
    mirror of the rule (judged chunk by chunk — a pooled median over the session would hide one
    chunk) → identity, declared."""
    sess = _straight_session()
    chunks, ci, _ = make_chunks(sess, chunk_size=60, overlap=30, depth_noise_rel=0.003)
    Kb = _bad_K(sess.K)
    Kz = _bad_K(sess.K, 1.15)
    c1 = dict(chunks[1])
    wp = CK.unproject_chunk(c1, Kb).astype(np.float32)
    dep = np.asarray(c1["depth"]).copy()
    Ks = np.tile(Kb[None], (60, 1, 1))
    ext = np.asarray(c1["extrinsic"])
    for local in range(30, 60):                            # global 60..89: a real zoom
        g = ci[1][0] + local
        z, _oid, pts = render(sess.scene, sess.poses[g], Kz, sess.H, sess.W)
        valid = z > 0
        dep[local] = np.where(valid, z, 0.0).astype(dep.dtype)
        c1_frame = {"depth": dep[local:local + 1], "extrinsic": ext[local:local + 1], "intrinsic": Kz[None]}
        wp[local] = CK.unproject_chunk(c1_frame, Kz)[0].astype(np.float32)
        wp[local][~valid] = 0.0
        Ks[local] = Kz
    c1.update({"world_points": wp, "depth": dep, "intrinsic": Ks})
    r, save_dir = _runner(tmp_path, sess, [chunks[0], c1], ci, SEAMS_MEDIAN)
    r._stac_metric_lock()
    doc = json.loads((save_dir / CK.REPORT_NAME).read_text())
    assert doc["run"] and doc["applied"] is False
    assert doc["seams_rule"]["improves_by_rule"] is True          # the seam alone would have passed
    assert doc["pairs_rule"]["worsens_by_rule"] is True and doc["pairs_rule"]["worsening_chunks"] == [1]
    assert doc["pairs_rule"]["by_chunk"]["0"]["worsens_by_rule"] is False
    assert "worsen by the mirror" in doc["reason"]
    assert CK.MARK_KEY not in _npy(save_dir, 1)


def test_a_zoomed_chunk_keeps_its_own_K(tmp_path):
    sess = _straight_session(n=120, length=36.0)
    chunks, ci, _ = make_chunks(sess, chunk_size=60, overlap=30, depth_noise_rel=0.003)
    assert ci == [(0, 60), (30, 90), (60, 120)]
    Kb = _bad_K(sess.K)
    chunks2 = [chunks[0], _with_focal(chunks[1], Kb), chunks[2]]
    r, save_dir = _runner(tmp_path, sess, chunks2, ci, SEAMS_MEDIAN)
    # the lock's zoom detector is its own (tested elsewhere); here chunk 2 is DECLARED zoomed
    orig_lock = r._stac_metric_lock
    import vggt_long as vl
    from loop_utils import common_k as _CK
    state = {}

    def _judge_with_zoom(layout, load_chunk, **kw):
        kw["zoom_chunks"] = sorted(set(kw.get("zoom_chunks", ())) | {2})
        state["zoom"] = kw["zoom_chunks"]
        return _orig_judge(layout, load_chunk, **kw)
    _orig_judge = _CK.judge_and_decide
    _CK.judge_and_decide = _judge_with_zoom
    try:
        orig_lock()
    finally:
        _CK.judge_and_decide = _orig_judge
    doc = json.loads((save_dir / CK.REPORT_NAME).read_text())
    assert state["zoom"] == [2] and doc["chunks_excluded_zoom"] == [2]
    assert doc["applied"], doc["reason"]
    assert np.allclose(doc["K"], sess.K)
    d2 = _npy(save_dir, 2)
    assert CK.MARK_KEY not in d2 and np.allclose(d2["intrinsic"], sess.K)      # its own K, untouched
    assert _npy(save_dir, 1)[CK.MARK_KEY]["applied"] and np.allclose(_npy(save_dir, 1)["intrinsic"], sess.K)
    assert any(j == "zoom_side" and v == [2] for s_ in doc["seams"] for j, v in s_.items())


def test_intake_probe_source_runs_only_with_the_probe_K(tmp_path, straight):
    sess, chunks, ci = straight
    Kb = _bad_K(sess.K)
    chunks2 = [chunks[0], _with_focal(chunks[1], Kb)]
    # without the probe: declared not run, nothing touched
    r, save_dir = _runner(tmp_path / "noprobe", sess, chunks2, ci, SEAMS_PROBE)
    r._stac_metric_lock()
    doc = json.loads((save_dir / CK.REPORT_NAME).read_text())
    assert doc["run"] is False and doc["applied"] is False and "not run" in doc["reason"]
    assert CK.MARK_KEY not in _npy(save_dir, 1)
    # with the probe's K (the map worker put it in the section, on the Ω grid): judged and applied
    seams = dict(SEAMS_PROBE, intake_probe_K=sess.K.tolist(), intake_probe={"source": "intake/focal_probe.json"})
    r, save_dir = _runner(tmp_path / "probe", sess, chunks2, ci, seams)
    r._stac_metric_lock()
    doc = json.loads((save_dir / CK.REPORT_NAME).read_text())
    assert doc["run"] and doc["applied"] and doc["source"] == "intake_probe"
    assert np.allclose(doc["K"], sess.K) and doc["K_source_detail"]["source"] == "intake/focal_probe.json"
    assert np.allclose(_npy(save_dir, 1)["intrinsic"], sess.K)


def test_bridges_take_the_applied_K_after_their_lock(tmp_path, straight):
    sess, chunks, ci = straight
    Kb = _bad_K(sess.K)
    chunks2 = [chunks[0], _with_focal(chunks[1], Kb)]
    r, save_dir = _runner(tmp_path, sess, chunks2, ci, SEAMS_MEDIAN)
    r._stac_metric_lock()
    assert r._stac_common_K_state["applied"]
    from loop_utils.loop_bridges import bridge_layout
    item = (1, (70, 80), 0, (5, 15))
    pred = make_bridge(sess, item)
    pred = _with_focal(pred, Kb)                                 # the bridge inferred Kb too
    r.loop_predict_list = [(item, pred, {"candidate": {"i": 75, "j": 10}, "gate": {}, "layout": bridge_layout(item, ci)})]
    r._stac_lock_bridges()
    r._stac_common_K_bridges()
    assert pred[CK.MARK_KEY]["applied"] and np.allclose(pred["intrinsic"], sess.K)
    assert np.allclose(pred["world_points"], CK.unproject_chunk(pred, sess.K), atol=2e-4)
    assert r.loop_predict_list[0][2]["common_K"] == {"frames": 20, "of": 20}
    # a zoomed parent's frames keep their own K
    r._stac_common_K_state = dict(r._stac_common_K_state, chunks_excluded_zoom=[0])
    pred2 = _with_focal(make_bridge(sess, item), Kb)
    r.loop_predict_list = [(item, pred2, {"candidate": {}, "gate": {}, "layout": bridge_layout(item, ci)})]
    r._stac_common_K_bridges()
    assert pred2[CK.MARK_KEY]["frames"] == list(range(0, 10))
    assert np.allclose(pred2["intrinsic"][:10], sess.K) and np.allclose(pred2["intrinsic"][10:], Kb)


def test_F0_reads_the_common_K_when_applied_and_omegas_when_not(tmp_path, straight):
    """intrinsic.txt (what precision.camera reads) carries the per-keyframe K of the chunk that OWNS
    each frame: the common K after an applied correction, Ω's own K otherwise."""
    from precision.camera import read_omega_intrinsics
    sess, chunks, ci = straight
    Kb = _bad_K(sess.K)
    chunks2 = [chunks[0], _with_focal(chunks[1], Kb)]
    for name, seams in (("on", SEAMS_MEDIAN), ("off", None)):
        root = tmp_path / name
        r, save_dir = _runner(root, sess, chunks2, ci, seams)
        r._stac_metric_lock()
        for k in range(2):                                     # the aligned copies save_camera_poses refreshes from
            np.save(save_dir / "_tmp_results_aligned" / f"chunk_{k}.npy", _npy(save_dir, k))
        r.sim3_list = [(1.0, np.eye(3), np.zeros(3))]
        r.save_camera_poses()
        rows = read_omega_intrinsics(root)                      # finds <root>/maplong_run/intrinsic.txt
        assert rows.shape[0] == 90
        own = rows[70]                                          # frame 70: owned by chunk 1
        if name == "on":
            assert np.allclose(own[:2], [sess.K[0, 0], sess.K[1, 1]]), own
        else:
            assert np.allclose(own[:2], [Kb[0, 0], Kb[1, 1]]), own
