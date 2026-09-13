"""claude_stac.txt §7: the single-variable elastic_seam A/B runs both variants
on copies of the aligned chunks, reports the three instruments and their
deltas, and never touches the source chunks or the config default."""

import json
from pathlib import Path

import numpy as np
import torch  # noqa: F401

from synth_metric import (make_session, make_chunks, drift_field, fork_loops_cfg,
                          fork_scale_cfg, fork_graph_cfg, fork_authority_cfg)
from test_stac_loop_stage import _make_runner
from test_stac_pose_graph_stage import _align_and_write

_SERVER = Path(__file__).resolve().parents[2] / "server"


def test_ab_elastic_reports_both_variants(tmp_path):
    import sys
    if str(_SERVER) not in sys.path:
        sys.path.insert(0, str(_SERVER))
    from reconstruction.quality.ab_elastic import run_ab
    sess = make_session(n_kf=150, H=40, W=56)
    D = drift_field(sess.n_kf, t_per_kf=(0.002, 0.0, 0.001))
    chunks, ci, _ = make_chunks(sess, scale_err=[1.0, 1.02, 0.99, 1.01], drift=D)
    r, save_dir = _make_runner(tmp_path, sess, chunks, ci)
    r._stac_metric_lock()
    _align_and_write(r, save_dir, ci)
    src = save_dir / "_tmp_results_aligned"
    before = {p.name: p.stat().st_mtime for p in src.glob("chunk_*.npy")}
    model_cfg = dict(r.config["Model"], elastic_seam=True, intra_chunk=False, depth_graph=False,
                     blend_copies=False, elastic_smooth_win=5, elastic_max_t_m=0.30)
    rep = run_ab(src, model_cfg, r.img_list, [list(c) for c in ci], r.sim3_list, r.img_dir,
                 fork_graph_cfg(), tmp_path / "quality", work=tmp_path / "ab_work",
                 log=lambda m: None)
    assert rep["off"]["elastic_seam"] is False and rep["on"]["elastic_seam"] is True
    for k in ("two_copy_disagreement_median_m", "holdout_surface_pairs_median_m",
              "depth_pair_disagreement_median_pct"):
        assert np.isfinite(rep["off"][k]) and np.isfinite(rep["on"][k])
    assert set(rep["delta_on_minus_off"]) == {"two_copy_disagreement_median_m",
                                              "holdout_surface_pairs_median_m",
                                              "depth_pair_disagreement_median_pct"}
    # the elastic consensus makes the two copies of every shared frame coincide
    assert rep["on"]["two_copy_disagreement_median_m"] <= rep["off"]["two_copy_disagreement_median_m"]
    # the source chunks are untouched and the report is on disk
    after = {p.name: p.stat().st_mtime for p in src.glob("chunk_*.npy")}
    assert after == before
    assert json.loads((tmp_path / "quality" / "ab_elastic.json").read_text())["variable"] == "Model.elastic_seam"
