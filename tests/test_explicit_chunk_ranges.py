"""Model.chunk_ranges inside VGGT_Long, without the model (CPU): an EXPLICIT,
variable-size layout (the co-visibility planner's — unequal chunks, every seam its own
overlap) builds chunk_indices as given, and every seam stage reads THAT seam's shared
frames: the seam copies, the exact seam chain, the elastic seam fits, the two-copy
uncertainty and the ensemble pass. Without chunk_ranges the uniform layout is the
vendor's. Explicit-layout products carry their layout and a product of another plan
stops the run."""

import json

import numpy as np
import pytest
import torch  # noqa: F401 — vggt_long imports it at module level

import synth_metric
from synth_metric import make_session, make_chunks, yaw_R

# n = 150 keyframes in blocks 20/20/55/25/30 → chunks [b_k, b_k+2): seams 20, 55, 25
EXPLICIT = [(0, 40), (20, 95), (40, 120), (95, 150)]
SEAMS = [20, 55, 25]
YAW = [0.0, 1.5, -1.0, 2.0]
T_ERR = [(0, 0, 0), (0.3, 0, 0.1), (-0.2, 0.05, 0.3), (0.1, 0, -0.4)]


@pytest.fixture(scope="module")
def synth():
    sess = make_session(n_kf=150, H=40, W=56)
    orig = synth_metric.chunk_ranges
    synth_metric.chunk_ranges = lambda n, size, ov: list(EXPLICIT)
    try:
        chunks, ci, errors = make_chunks(sess, scale_err=[1.0] * 4, yaw_err_deg=YAW,
                                         t_err=T_ERR)
    finally:
        synth_metric.chunk_ranges = orig
    assert ci == EXPLICIT
    return sess, chunks, errors


def _runner(tmp_path, sess, chunk_ranges=None, chunk_size=None, overlap=None, extra=None):
    """A VGGT_Long without __init__ (no model) — the state process_long_sequence sees."""
    import vggt_long as vl
    save_dir = tmp_path / "maplong_run"
    for d in ("_tmp_results_unaligned", "_tmp_results_aligned", "_tmp_results_loop", "pcd"):
        (save_dir / d).mkdir(parents=True, exist_ok=True)
    r = object.__new__(vl.VGGT_Long)
    model = {"loop_chunk_size": 20, "loop_enable": False, "using_sim3": False,
             "align_method": "numpy", "IRLS": {"delta": 0.1, "max_iters": 5, "tol": "1e-9"},
             "Pointcloud_Save": {"use_conf_filter": True, "conf_percentile": 20.0,
                                 "sample_ratio": 1.0},
             "exact_seam_align": True}
    if chunk_ranges is not None:
        model["chunk_ranges"] = [list(x) for x in chunk_ranges]
    if chunk_size is not None:
        model["chunk_size"], model["overlap"] = chunk_size, overlap
    model.update(extra or {})
    r.config = {"Model": model}
    r.chunk_ranges_cfg = model.get("chunk_ranges")
    r.chunk_size, r.overlap = model.get("chunk_size"), model.get("overlap")
    r.chunk_indices = None
    r.sky_mask = False
    r.img_dir = str(tmp_path / "frames")
    r.img_list = [str(tmp_path / "frames" / f"{int(n):06d}.jpg") for n in sess.frame_numbers]
    r.output_dir = str(save_dir)
    r.result_unaligned_dir = str(save_dir / "_tmp_results_unaligned")
    r.result_aligned_dir = str(save_dir / "_tmp_results_aligned")
    r.result_loop_dir = str(save_dir / "_tmp_results_loop")
    r.pcd_dir = str(save_dir / "pcd")
    r.all_camera_poses, r.all_camera_intrinsics = [], []
    return r, save_dir


def _write(save_dir, chunks, sub):
    for k, c in enumerate(chunks):
        np.save(save_dir / sub / f"chunk_{k}.npy", c)


# ── the layout ─────────────────────────────────────────────────────────────

def test_explicit_ranges_replace_the_uniform_construction(tmp_path, synth):
    sess, _, _ = synth
    r, _ = _runner(tmp_path, sess, chunk_ranges=EXPLICIT, chunk_size=60, overlap=30)
    assert r._stac_build_layout() == 4
    assert r.chunk_indices == EXPLICIT          # NOT the 60/30 layout the keys describe
    assert all(type(x) is int for c in r.chunk_indices for x in c)
    assert r._stac_layout_text() == "4 chunk(s), lengths 40-80, seams 20-55 shared frames"


def test_without_chunk_ranges_the_layout_is_the_vendors(tmp_path, synth):
    sess, _, _ = synth
    r, _ = _runner(tmp_path, sess, chunk_size=60, overlap=30)
    assert r._stac_build_layout() == 4
    assert r.chunk_indices == [(0, 60), (30, 90), (60, 120), (90, 150)]
    assert r._stac_layout_stamp() is None


def test_ranges_planned_for_another_frame_list_fail_loudly(tmp_path, synth):
    from loop_utils.metric_lock import ChunkRangesError
    sess, _, _ = synth
    r, _ = _runner(tmp_path, sess, chunk_ranges=EXPLICIT)
    r.img_list = r.img_list[:-1]                # e.g. another stride: 149 frames
    with pytest.raises(ChunkRangesError, match="ends at 150 but this run has 149 frames"):
        r._stac_build_layout()


# ── every seam stage reads THAT seam's frames ──────────────────────────────

def test_seam_copies_are_each_seams_own_shared_frames(tmp_path, synth):
    sess, chunks, _ = synth
    r, _ = _runner(tmp_path, sess, chunk_ranges=EXPLICIT)
    r._stac_build_layout()
    for k in range(len(EXPLICIT) - 1):
        (s0, e0), (s1, _) = EXPLICIT[k], EXPLICIT[k + 1]
        ov, pm1, pm2, c1, c2 = r._stac_seam_copies(k, chunks[k], chunks[k + 1])
        assert ov == SEAMS[k] == e0 - s1
        assert np.array_equal(pm1, chunks[k]['world_points'][s1 - s0:e0 - s0])
        assert np.array_equal(pm2, chunks[k + 1]['world_points'][:e0 - s1])
        assert np.array_equal(c1, chunks[k]['world_points_conf'][s1 - s0:e0 - s0])
        assert np.array_equal(c2, chunks[k + 1]['world_points_conf'][:e0 - s1])
        # the same frames: both copies show the same surface (each chunk's own gauge)
        R0, t0 = yaw_R(YAW[k]), np.asarray(T_ERR[k], float)
        R1, t1 = yaw_R(YAW[k + 1]), np.asarray(T_ERR[k + 1], float)
        ok = (c1 > 0) & (c2 > 0)
        X0 = (pm1[ok].astype(np.float64) - t0) @ R0
        X1 = (pm2[ok].astype(np.float64) - t1) @ R1
        assert np.median(np.linalg.norm(X0 - X1, axis=1)) < 0.05


def test_exact_seam_chain_recovers_every_uneven_seam(tmp_path, synth):
    sess, chunks, _ = synth
    r, save_dir = _runner(tmp_path, sess, chunk_ranges=EXPLICIT)
    r._stac_build_layout()
    _write(save_dir, chunks, "_tmp_results_unaligned")
    seq, report = r._stac_seam_chain("test")
    assert len(seq) == 3
    for k, (s, R, t) in enumerate(seq):
        assert report[str(k)]["exact"] is True
        # chunk k+1 -> chunk k = E_k o E_{k+1}^-1 (rigid: every injected scale is 1)
        R0, t0 = yaw_R(YAW[k]), np.asarray(T_ERR[k], float)
        R1, t1 = yaw_R(YAW[k + 1]), np.asarray(T_ERR[k + 1], float)
        R_true = R0 @ R1.T
        t_true = t0 - R_true @ t1
        ang = np.degrees(np.arccos(np.clip((np.trace(np.asarray(R).T @ R_true) - 1) / 2,
                                           -1, 1)))
        assert ang < 0.05, (k, ang)
        assert np.linalg.norm(np.asarray(t) - t_true) < 0.01, k


def test_elastic_fits_and_uncertainty_count_each_seams_frames(tmp_path, synth):
    sess, chunks, _ = synth
    r, save_dir = _runner(tmp_path, sess, chunk_ranges=EXPLICIT)
    r._stac_build_layout()
    _write(save_dir, chunks, "_tmp_results_aligned")
    fits, rep = r._stac_elastic_fit_seams()
    for j in range(len(EXPLICIT) - 1):
        shared = list(range(EXPLICIT[j + 1][0], EXPLICIT[j][1]))
        assert len(shared) == SEAMS[j]
        assert sorted(int(g) for g in rep["seams"][str(j)]) == shared
        assert sorted(fits[j]) == shared
    assert rep["chunk_indices"] == [list(c) for c in EXPLICIT]
    r._stac_uncertainty()
    doc = json.loads((save_dir / "uncertainty.json").read_text())
    assert doc["n_shared_frames"] == sum(SEAMS)
    assert sorted(int(g) for g in doc["frames"]) == list(range(20, 120))
    assert all(v["witnesses"] == 2 for v in doc["frames"].values())


class _StubModel:
    def infer_chunk(self, paths):
        return {"depth": np.zeros((1, len(paths), 2, 2), np.float32)}


def _old_ensemble_ranges(N, chunk_size, overlap, off):
    """The vendor-era ensemble layout, verbatim."""
    step = chunk_size - overlap
    ranges = []
    start = off
    while start + 2 < N:
        end = min(start + chunk_size, N)
        ranges.append((start, end))
        if end >= N:
            break
        start += step
    return ranges


def test_ensemble_shifts_this_runs_chunks(tmp_path, synth):
    sess, _, _ = synth
    cert = {"certify": {"ensemble_offset_frames": 7}}
    r, save_dir = _runner(tmp_path, sess, chunk_ranges=EXPLICIT, extra=cert)
    r._stac_build_layout()
    r.model = _StubModel()
    r._stac_ensemble_uncertainty()
    assert r._stac_ensemble_ranges == [(7, 47), (27, 102), (47, 127), (102, 150)]
    stamped = [np.load(save_dir / "_tmp_results_ensemble" / f"chunk_{i}.npy",
                       allow_pickle=True).item()["_stac_range"] for i in range(4)]
    assert stamped == [list(x) for x in r._stac_ensemble_ranges]
    # uniform: the vendor-era formula, unchanged
    r2, _ = _runner(tmp_path / "u", sess, chunk_size=60, overlap=30, extra=cert)
    r2._stac_build_layout()
    r2.model = _StubModel()
    r2._stac_ensemble_uncertainty()
    assert r2._stac_ensemble_ranges == _old_ensemble_ranges(150, 60, 30, 7)


# ── products of another plan stop the run ──────────────────────────────────

class _InferModel:
    def infer_chunk(self, paths):
        S = len(paths)
        return {"depth": np.ones((S, 4, 4), np.float32),
                "world_points": np.zeros((S, 4, 4, 3), np.float32),
                "world_points_conf": np.ones((S, 4, 4), np.float32),
                "extrinsic": np.tile(np.eye(4), (S, 1, 1)),
                "intrinsic": np.tile(np.eye(3), (S, 1, 1)),
                "images": np.zeros((S, 3, 4, 4), np.float32)}


def test_explicit_chunks_carry_their_range_and_another_plans_chunk_stops(tmp_path, synth):
    import vggt_long as vl
    sess, _, _ = synth
    r, save_dir = _runner(tmp_path, sess, chunk_ranges=EXPLICIT)
    r._stac_build_layout()
    r.model = _InferModel()
    r.process_single_chunk(EXPLICIT[1], chunk_idx=1)
    saved = np.load(save_dir / "_tmp_results_unaligned" / "chunk_1.npy", allow_pickle=True).item()
    assert saved["_stac_range"] == [20, 95]
    # resume, same plan: inference skipped
    r.all_camera_poses = []
    r.process_single_chunk(EXPLICIT[1], chunk_idx=1)
    assert r.all_camera_poses[0][0] == (20, 95)
    # another plan whose chunk 1 has the SAME length (75 frames) somewhere else: stop
    other = [(0, 45), (25, 100), (45, 120), (100, 150)]
    r2, _ = _runner(tmp_path, sess, chunk_ranges=other)
    r2._stac_build_layout()
    r2.model = _InferModel()
    with pytest.raises(vl._StacPlanMismatch, match=r"is frames \[20, 95\]"):
        r2.process_single_chunk(other[1], chunk_idx=1)


def test_uniform_chunks_are_written_as_before(tmp_path, synth):
    sess, _, _ = synth
    r, save_dir = _runner(tmp_path, sess, chunk_size=60, overlap=30)
    r._stac_build_layout()
    r.model = _InferModel()
    r.process_single_chunk(r.chunk_indices[1], chunk_idx=1)
    saved = np.load(save_dir / "_tmp_results_unaligned" / "chunk_1.npy", allow_pickle=True).item()
    assert "_stac_range" not in saved


def test_reports_of_another_layout_stop_the_run(tmp_path, synth):
    import vggt_long as vl
    sess, _, _ = synth
    r, _ = _runner(tmp_path, sess, chunk_ranges=EXPLICIT)
    r._stac_build_layout()
    assert r._stac_layout_stamp() == [list(c) for c in EXPLICIT]
    r._stac_check_layout_stamp({"chunk_indices": [list(c) for c in EXPLICIT]}, "x.json")
    r._stac_check_layout_stamp({"chunks": {}}, "x.json")          # unstamped: passes
    r._stac_check_layout_stamp(None, "x.json")                   # unreadable: passes
    with pytest.raises(vl._StacPlanMismatch, match="metric_lock.json was written for"):
        r._stac_check_layout_stamp({"chunk_indices": [[0, 60], [30, 90], [60, 120],
                                                      [90, 150]]}, "metric_lock.json")
