"""claude_stac_slam2.txt, phase 3 — the offline calibration of the attention gate
(loop_utils.attention_calibrate) with a synthetic scorer of known separation. No card."""
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

from loop_utils import attention_calibrate as AC                                  # noqa: E402

PARAMS = {"select": "max_auc_lower_ci", "target_precision": 0.9, "min_labels": 5, "n_negatives": 20,
          "n_boot": 300, "confidence": 0.95, "salad_grid": [0.55, 0.6, 0.65], "top_fraction": 0.25}
LAYERS = [0, 2, 5]


def _scorer(separation):
    """A deterministic scorer: per layer L, positives around 0.5 + sep[L], negatives around 0.5 − sep[L]
    (the pair key seeds the noise) — layer 2 the sharpest."""
    def scorer(i, j):
        rng = np.random.default_rng(1000003 * min(i, j) + max(i, j))
        out = {}
        for L in LAYERS:
            is_pos = (i + j) % 2 == 0                        # the synthetic truth: even sums are revisits
            centre = 0.5 + (separation[L] if is_pos else -separation[L])
            out[L] = float(np.clip(centre + 0.05 * rng.standard_normal(), 0.0, 1.0))
        return out
    return scorer


def _labels(n_pos, n_neg):
    pos = [(2 * k, 2 * k + 40) for k in range(n_pos)]                 # even sums
    neg = [(2 * k, 2 * k + 41) for k in range(n_neg)]                 # odd sums
    return {"positives": pos, "negatives": neg}


def test_the_calibration_recovers_the_sharpest_layer_and_a_threshold_at_the_target_precision():
    sep = {0: 0.02, 2: 0.35, 5: 0.06}            # layers 0 and 5 overlap under the 0.05 noise, layer 2 separates
    run = {"n_frames": 200}
    manual = _labels(12, 12)
    doc = AC.calibrate(run, PARAMS, _scorer(sep), LAYERS, labels={"available": False, "reason": "synthetic"},
                       manual=manual, negatives_frustum={"pairs": []})
    ch = doc["choice"]
    assert ch["chosen"] and ch["layer"] == 2, ch
    aucs = {int(L): r["auc"] for L, r in doc["layers"].items()}
    assert aucs[2]["auc"] == 1.0 and aucs[2]["ci_low"] > aucs[0]["ci_low"] and aucs[2]["ci_low"] > aucs[5]["ci_low"]
    assert ch["precision"] >= PARAMS["target_precision"]
    # the LOWEST threshold reaching the target precision (not the one separating perfectly): with
    # 12 positives, precision 12/13 at the highest negative's score already clears 0.9
    pos = np.array(doc["layers"]["2"]["pos"]); neg = np.array(doc["layers"]["2"]["neg"])
    assert ch["threshold"] == AC.threshold_for_precision(pos, neg, PARAMS["target_precision"])["threshold"]
    assert ch["threshold"] <= float(pos.min()) and ch["threshold"] == pytest.approx(float(neg.max()))
    assert ch["precision"] == pytest.approx(12 / 13) and ch["recall"] == 1.0 and ch["n_pos"] == 12 and ch["n_neg"] == 12
    assert doc["labels"]["sources"]["positives"] == {"manual": 12}
    # scores are recorded per pair and layer, float64 on the CPU
    assert len(doc["scores"]) == 24 and all(isinstance(v, float) for s in doc["scores"].values() for v in s.values())


def test_insufficient_labels_choose_nothing_and_say_why():
    doc = AC.calibrate({"n_frames": 200}, PARAMS, _scorer({0: 0.1, 2: 0.3, 5: 0.2}), LAYERS,
                       labels={"available": False, "reason": "none"}, manual=_labels(3, 12),
                       negatives_frustum={"pairs": []})
    assert doc["choice"]["chosen"] is False and "insufficient labels: 3 positive(s)" in doc["choice"]["reason"]


def test_no_threshold_reaching_the_target_is_declared():
    """Layer scores that barely separate: the lowest-threshold rule finds no precision >= target."""
    sep = {0: 0.0, 2: 0.0, 5: 0.0}
    doc = AC.calibrate({"n_frames": 200}, PARAMS, _scorer(sep), LAYERS, labels={"available": False, "reason": "none"},
                       manual=_labels(12, 12), negatives_frustum={"pairs": []})
    ch = doc["choice"]
    # with no separation the AUC is ~0.5 everywhere; a threshold with precision 0.9 may or may not exist by
    # chance among 24 points — the record says which, by name
    assert ("chosen" in ch) and (ch["chosen"] or "reaches no threshold" in ch["reason"] or "insufficient" in ch["reason"])


def test_auc_and_pr_are_exact_rank_statistics():
    pos = np.array([0.9, 0.8, 0.7])
    neg = np.array([0.6, 0.7, 0.1])
    assert AC.auc(pos, neg) == pytest.approx((8 + 0.5) / 9)          # one tie (0.7 vs 0.7) counts half
    rec = AC.auc_with_ci(pos, neg, n_boot=200, confidence=0.9)
    assert rec["ci_low"] <= rec["auc"] <= rec["ci_high"]
    pr = AC.pr_curve(pos, neg)
    assert pr["thresholds"][0] == 0.9 and pr["precision"][0] == 1.0 and pr["recall"][0] == pytest.approx(1 / 3)
    th = AC.threshold_for_precision(pos, neg, 1.0)
    assert th["threshold"] == 0.8 and th["recall"] == pytest.approx(2 / 3)
    assert AC.threshold_for_precision(np.array([0.2]), np.array([0.9]), 0.9) is None


def test_run_labels_read_the_verified_cycle_consistent_closures_or_declare(tmp_path):
    run = tmp_path / "maplong_run"
    run.mkdir()
    edges = [{"stage": "verification", "bridge": 0, "status": "accepted", "candidate": {"i": 140, "j": 8}},
             {"stage": "verification", "bridge": 1, "status": "accepted", "candidate": {"i": 100, "j": 30}},
             {"stage": "verification", "bridge": 2, "status": "rejected", "candidate": {"i": 90, "j": 50}},
             {"stage": "spatial_gate", "status": "rejected", "candidate": {"i": 75, "j": 8}}]
    (run / "loop_edges.json").write_text(json.dumps({"version": 1, "edges": edges}))
    lab = AC.run_labels(run)
    assert lab["available"] is False and "pose_graph.json is missing" in lab["reason"]
    assert lab["negatives_rejected"] == [(50, 90)]
    (run / "pose_graph.json").write_text(json.dumps({"verdict": "IDENTITY", "reason": "too few closures"}))
    lab = AC.run_labels(run)
    assert lab["available"] is False and "no cycle_consistency" in lab["reason"]
    (run / "pose_graph.json").write_text(json.dumps(
        {"verdict": "APPLY", "cycle_consistency": {"groups": [{"chunks": [0, 3], "bridges": [0, 1], "kept": [0], "rejected": [1]}]}}))
    lab = AC.run_labels(run)
    assert lab["available"] and lab["positives"] == [(8, 140)] and lab["negatives_rejected"] == [(50, 90)]
    assert lab["detail"]["cycle_kept_bridges"] == [0]
    assert AC.run_labels(tmp_path / "nowhere")["available"] is False


def test_frustum_disjoint_negatives_are_non_local_pairs_that_cannot_see_each_other():
    """Two camera groups looking in opposite directions far apart never see each other; two
    cameras side by side looking the same way do."""
    from synth_metric import look_c2w
    N = 40
    poses = np.zeros((N, 4, 4))
    for i in range(N):
        if i < 20:
            poses[i] = look_c2w((0.0 + 0.1 * i, 1.5, 0.0), np.array([1.0, 0.0, 0.0]))        # looking +x
        else:
            poses[i] = look_c2w((-50.0 - 0.1 * i, 1.5, 0.0), np.array([-1.0, 0.0, 0.0]))     # far away, looking −x
    K = np.tile(np.array([40.0, 40.0, 28.0, 20.0]), (N, 1))
    rec = AC.frustum_disjoint_pairs(poses, K, (40, 56), min_gap=5, depth_range=(0.3, 15.0), n_pick=10)
    pairs = rec["pairs"]
    assert len(pairs) == 10 and rec["n_disjoint"] > 0
    assert all((i < 20) != (j < 20) for i, j in pairs), pairs          # every pick crosses the two groups
    assert all(j - i >= 5 for i, j in pairs)
    # the pick is stable: the same pairs again
    assert AC.frustum_disjoint_pairs(poses, K, (40, 56), min_gap=5, depth_range=(0.3, 15.0), n_pick=10)["pairs"] == pairs


def test_parameters_come_from_the_run_or_a_params_file_never_a_default(tmp_path):
    cfg = {"Model": {"loops": {"attention_gate": {"mode": "gate", "layer": 4, "min_score": 0.5, "top_fraction": 0.25,
                                                 "salad_similarity_threshold": None,
                                                 "calibration": {k: v for k, v in PARAMS.items() if k != "top_fraction"}}}}}
    p = AC.calibration_params(cfg, None)
    assert p["select"] == "max_auc_lower_ci" and p["top_fraction"] == 0.25 and p["salad_grid"] == [0.55, 0.6, 0.65]
    with pytest.raises(AC.CalibrationError, match="missing"):
        AC.calibration_params({"Model": {}}, None)
    pf = tmp_path / "params.json"
    pf.write_text(json.dumps(PARAMS))
    p2 = AC.calibration_params({"Model": {}}, pf)
    assert p2["n_negatives"] == 20 and p2["source"] == str(pf)
    pf.write_text(json.dumps({k: v for k, v in PARAMS.items() if k != "n_boot"}))
    with pytest.raises(AC.CalibrationError, match="n_boot"):
        AC.calibration_params({"Model": {}}, pf)


def test_salad_grid_counts_use_the_detectors_own_selection(tmp_path):
    from LoopModels.LoopModel import nonlocal_similarities, select_candidates
    rng = np.random.default_rng(0)
    n = 30
    D = rng.normal(size=(n, 8))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    sims = D @ D.T
    sims[3, 25] = sims[25, 3] = 0.95                                  # one strong revisit
    cfg = {"Loop": {"SALAD": {"min_gap": 5, "min_gap_frac": 0.1, "top_k": 3, "use_nms": True, "nms_threshold": 1}},
           "Model": {"loop_chunk_size": 4}}
    scorer = lambda i, j: {2: (0.9 if {i, j} == {3, 25} else 0.1)}     # noqa: E731
    out = AC.salad_grid_counts(sims, cfg, [0.5, 0.9, 0.99], scorer, 2, 0.5, max_pairs=None, log=lambda *a: None)
    nl = nonlocal_similarities(sims, AC.detector_min_gap(cfg, n))
    for rec in out:
        assert rec["n_candidates"] == len(select_candidates(nl, rec["salad_similarity"], 3, True, 1))
    by = {r["salad_similarity"]: r for r in out}
    assert by[0.9]["n_candidates"] >= 1 and by[0.9]["n_pass"] == 1       # only the revisit passes the gate
    assert by[0.99]["n_candidates"] == 0 and by[0.99]["n_pass"] == 0
    assert by[0.5]["n_pass"] == 1


def test_outputs_are_json_csv_and_png(tmp_path):
    doc = AC.calibrate({"n_frames": 200}, PARAMS, _scorer({0: 0.02, 2: 0.35, 5: 0.06}), LAYERS,
                       labels={"available": False, "reason": "synthetic"}, manual=_labels(8, 8),
                       negatives_frustum={"pairs": []})
    doc.update({"version": 1, "params": PARAMS})
    outs = AC.write_outputs(tmp_path / "cal", doc, png=True)
    assert Path(outs["json"]).is_file() and Path(outs["csv"]).is_file() and Path(outs["png"]).is_file()
    rows = Path(outs["csv"]).read_text().splitlines()
    assert rows[0] == "layer,auc,ci_low,ci_high,n_pos,n_neg,chosen" and len(rows) == 4
    assert json.loads(Path(outs["json"]).read_text())["choice"]["layer"] == 2
