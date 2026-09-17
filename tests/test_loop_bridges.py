"""claude_stac.txt §12.1 / §12.2 / §12.4 on the fork: exact bridges, loop
verification, closed scale graph + scale-break localisation. Synthetic
corridor loop (server/tests/synth_metric.py), no GPU."""

import numpy as np
import pytest

from loop_utils import loop_bridges as lb
from loop_utils.metric_lock import (solve_scale_graph, scale_break_diagnosis,
                                    seam_relative_scale)
from synth_metric import (make_session, make_chunks, make_bridge, fork_loops_cfg,
                                yaw_R)

_SESS = None


def session():
    global _SESS
    if _SESS is None:
        _SESS = make_session(n_kf=150, H=40, W=56)
    return _SESS


def _item(ci, ka, a_mid, kb, b_mid, half=10):
    a0, a1 = max(ci[ka][0], a_mid - half), min(ci[ka][1], a_mid + half)
    b0, b1 = max(ci[kb][0], b_mid - half), min(ci[kb][1], b_mid + half)
    return (ka, (a0, a1), kb, (b0, b1))


# ── §12.1 exact bridges ─────────────────────────────────────────────────────

def test_robust_sim3_recovers_injected_similarity():
    rng = np.random.default_rng(0)
    src = rng.uniform(-5, 5, (5000, 3))
    R = yaw_R(7.0)
    dst = 1.08 * (src @ R.T) + np.array([0.3, -0.1, 0.5])
    dst[:200] += rng.normal(0, 2.0, (200, 3))          # gross outliers (Cauchy kills them)
    s, R_, t_, res, n = lb.robust_sim3(src, dst, min_points=100)
    assert abs(s - 1.08) < 1e-6
    assert np.allclose(R_, R, atol=1e-6)
    assert np.allclose(t_, [0.3, -0.1, 0.5], atol=1e-5)


def test_exact_bridge_edge_residual_at_noise_level():
    sess = session()
    chunks, ci, errors = make_chunks(sess, scale_err=[1.0, 1.1, 0.95, 1.05, 1.02][:20],
                                     yaw_err_deg=[0, 2.0, -1.5, 3.0, 1.0][:20],
                                     t_err=[(0, 0, 0), (0.5, 0, 0.2), (-0.3, 0.1, 0.4),
                                            (0.2, 0, -0.6), (0.1, 0, 0.1)][:20])
    # chunk 0 frames 0..59, last chunk holds the revisit past the start
    ka, kb = 0, len(ci) - 1
    item = _item(ci, ka, 20, kb, ci[kb][0] + 15)
    bridge = make_bridge(sess, item, s_L=1.3, yaw_L_deg=10.0, t_L=(1.0, 0.2, -0.5))
    layout = lb.bridge_layout(item, ci)
    cfg = fork_loops_cfg()
    meas = lb.measure_bridge(bridge, layout, chunks[ka], chunks[kb], cfg, rigid=False)
    assert meas["ok"]
    # s_ab = s_Eb / s_Ea exactly (bridge gauge cancels)
    s_true = errors[kb][0] / errors[ka][0]
    assert abs(meas["s_ab"] / s_true - 1.0) < 0.01
    # residual ≈ depth noise (0.3 % of a few metres)
    assert meas["residual_m"] < 0.05
    assert meas["n_corr"] >= cfg["min_correspondences"]
    v = lb.verify_loop(meas, cfg)
    assert v["status"] in ("accepted", "scale_break")
    assert v["sigma_m"] >= meas["residual_m"]


def test_starved_bridge_is_declared_not_hidden():
    sess = session()
    chunks, ci, _ = make_chunks(sess)
    item = _item(ci, 0, 20, len(ci) - 1, ci[-1][0] + 15)
    bridge = make_bridge(sess, item)
    layout = lb.bridge_layout(item, ci)
    cfg = fork_loops_cfg(min_correspondences=10 ** 9)   # force starvation
    meas = lb.measure_bridge(bridge, layout, chunks[0], chunks[-1], cfg, rigid=True)
    assert not meas["ok"]
    assert meas["sides"]["a"]["starved"] and meas["sides"]["b"]["starved"]
    v = lb.verify_loop(meas, cfg)
    assert v["status"] == "rejected" and "starved" in v["reasons"][0]


# ── §12.2 verification ──────────────────────────────────────────────────────

def test_scale_out_of_tolerance_is_a_scale_break_not_a_discard():
    sess = session()
    chunks, ci, errors = make_chunks(sess, scale_err=[1.0] + [1.15] * 20)
    ka, kb = 0, len(ci) - 1
    item = _item(ci, ka, 20, kb, ci[kb][0] + 15)
    bridge = make_bridge(sess, item)
    meas = lb.measure_bridge(bridge, lb.bridge_layout(item, ci), chunks[ka], chunks[kb],
                             fork_loops_cfg(), rigid=False)
    cfg = fork_loops_cfg()
    v = lb.verify_loop(meas, cfg)
    assert v["status"] == "scale_break"
    # σ is the SPLIT-HALF held-out residual (USER 2026-09-16), not the fit's
    # own optimistic residual — the break factor multiplies THAT
    _base = meas.get("holdout_residual_m", meas["residual_m"])
    assert v["sigma_m"] == pytest.approx(_base * cfg["scale_break_sigma_factor"])
    assert v["checks"]["scale"]["passed"] is False


def test_a_broken_bridge_is_kept_and_pays_in_sigma():
    """USER 2026-09-16: a bad fit is weak evidence, not absent evidence.

    The old rule dropped any bridge whose residual passed `max_residual_m`, and
    on pccr that deleted the only two edges able to close a 44 m walk. A broken
    window now still produces an edge — with a σ big enough that the pose graph
    barely listens to it, and with the reason declared."""
    sess = session()
    chunks, ci, _ = make_chunks(sess)
    item = _item(ci, 0, 20, len(ci) - 1, ci[-1][0] + 15)
    bridge = make_bridge(sess, item, corrupt_b=0.6)      # window b is not a rigid copy
    cfg = fork_loops_cfg()
    meas = lb.measure_bridge(bridge, lb.bridge_layout(item, ci), chunks[0], chunks[-1],
                             cfg, rigid=True)
    clean = make_bridge(sess, item)
    meas_ok = lb.measure_bridge(clean, lb.bridge_layout(item, ci), chunks[0], chunks[-1],
                                cfg, rigid=True)
    v = lb.verify_loop(meas, cfg, reference_m=meas_ok["residual_m"])
    v_ok = lb.verify_loop(meas_ok, cfg, reference_m=meas_ok["residual_m"])
    assert v["status"] in ("accepted", "scale_break")
    assert v["checks"]["geometric"]["passed"] is True      # the fit EXISTS
    # …and the broken one is the one that gets distrusted, by measurement
    assert v["sigma_m"] > v_ok["sigma_m"]
    assert v.get("evidence") == "weak"
    assert any("weak evidence" in r for r in v["reasons"])


def test_only_starvation_rejects():
    sess = session()
    chunks, ci, _ = make_chunks(sess)
    item = _item(ci, 0, 20, len(ci) - 1, ci[-1][0] + 15)
    bridge = make_bridge(sess, item)
    cfg = fork_loops_cfg(min_correspondences=10 ** 9)     # nothing can feed the fit
    meas = lb.measure_bridge(bridge, lb.bridge_layout(item, ci), chunks[0], chunks[-1],
                             cfg, rigid=True)
    v = lb.verify_loop(meas, cfg)
    assert v["status"] == "rejected"
    assert "sigma_m" not in v


def test_semantic_and_ambiguous_checks():
    sess = session()
    chunks, ci, _ = make_chunks(sess)
    item = _item(ci, 0, 20, len(ci) - 1, ci[-1][0] + 15)
    bridge = make_bridge(sess, item)
    cfg = fork_loops_cfg()
    meas = lb.measure_bridge(bridge, lb.bridge_layout(item, ci), chunks[0], chunks[-1],
                             cfg, rigid=True)
    ok = lb.verify_loop(meas, cfg, semantic={"a": ["wall", "box"], "b": ["wall", "column"]})
    assert ok["status"] == "accepted" and ok["checks"]["semantic"]["shared_structural"] == ["wall"]
    bad = lb.verify_loop(meas, cfg, semantic={"a": ["box"], "b": ["box", "column"]})
    assert bad["status"] == "rejected"                   # movable labels never count
    amb = lb.verify_loop(meas, cfg, spatial={"verdict": "ambiguous"})
    _base = meas.get("holdout_residual_m", meas["residual_m"])
    assert amb["sigma_m"] == pytest.approx(_base * cfg["ambiguous_sigma_factor"])


def test_candidate_file_round_trip(tmp_path):
    p = str(tmp_path / "loop_closures.txt")
    lb.write_loop_candidates(p, [{"i": 140, "j": 12, "sim": 0.91, "source": "salad"},
                                 {"i": 131, "j": 5, "sim": None, "source": "instance"}])
    back = lb.load_loop_candidates(p)
    assert [(c["i"], c["j"], c["source"]) for c in back] == [(140, 12, "salad"), (131, 5, "instance")]
    assert back[0]["sim"] == pytest.approx(0.91) and back[1]["sim"] is None


# ── §12.4 scale graph ───────────────────────────────────────────────────────

def _chain_measurements(errors, ci, rng, anchor_noise=0.08, seam_noise=0.003):
    """Raw-chunk measurements as metric_lock sees them: DA3 anchors (noisy
    absolute), seams (precise relative)."""
    n = len(errors)
    s_true = np.array([1.0 / e[0] for e in errors])
    s_da3 = {k: float(s_true[k] * (1 + anchor_noise * rng.standard_normal())) for k in range(n)}
    n_anch = {k: 1 for k in range(n)}
    seam = {k: float(s_true[k + 1] / s_true[k] * (1 + seam_noise * rng.standard_normal()))
            for k in range(n - 1)}
    return s_true, s_da3, n_anch, seam


def test_loop_rows_close_the_scale_walk():
    rng = np.random.default_rng(3)
    n = 12
    errors = [(float(np.exp(0.03 * rng.standard_normal())), np.eye(3), np.zeros(3)) for _ in range(n)]
    ci = [(k * 30, k * 30 + 60) for k in range(n)]
    s_true, s_da3, n_anch, seam = _chain_measurements(errors, ci, rng)
    # a single DA3 anchor per chunk is weak → the chain random-walks; the loop
    # 0↔11 measured exactly by a bridge (s_ab = s_E11/s_E0) pins it
    chain = solve_scale_graph(s_da3, n_anch, seam, n, sigma_seam=0.003, sigma_anchor=0.08)
    s_ab = errors[n - 1][0] / errors[0][0]
    loop_rel = {(0, n - 1): (lb.loop_scale_row(s_ab), 0.005)}
    closed = solve_scale_graph(s_da3, n_anch, seam, n, sigma_seam=0.003, sigma_anchor=0.08,
                               loop_rel=loop_rel)
    # relative scale between the loop ends: the closed solve reproduces the
    # measured loop ratio, the chain does not have to
    err_chain = abs(np.log(chain[n - 1] / chain[0]) - np.log(s_true[n - 1] / s_true[0]))
    err_closed = abs(np.log(closed[n - 1] / closed[0]) - np.log(s_true[n - 1] / s_true[0]))
    assert err_closed < 0.01
    assert err_closed <= err_chain + 1e-9


def test_absolute_rows_pin_the_metre():
    n = 5
    s_true = np.array([2.0, 2.1, 2.05, 1.95, 2.0])
    s_da3 = {k: float(s_true[k] * 1.10) for k in range(n)}     # DA3 10 % biased
    seam = {k: float(s_true[k + 1] / s_true[k]) for k in range(n - 1)}
    absolute = [(2, float(np.log(s_true[2])), 0.005, "vio")]
    s = solve_scale_graph(s_da3, {k: 1 for k in range(n)}, seam, n, sigma_seam=0.003,
                          sigma_anchor=0.08, absolute=absolute)
    assert abs(s[2] / s_true[2] - 1.0) < 0.01           # the precise source wins
    assert abs(s[0] / s_true[0] - 1.0) < 0.01           # seams carry it along the chain


def test_scale_break_localises_the_corrupt_seam():
    rng = np.random.default_rng(5)
    n = 8
    errors = [(1.0, np.eye(3), np.zeros(3)) for _ in range(n)]
    ci = [(k * 30, k * 30 + 60) for k in range(n)]
    s_true, s_da3, n_anch, seam = _chain_measurements(errors, ci, rng, anchor_noise=0.03)
    n_anch = {k: 9 for k in range(n)}                        # 9 anchors per chunk (σ 0.027): enough witnesses
    seam[4] *= 1.08                                          # the corrupt seam 4→5
    loop_rel = {(1, 7): (lb.loop_scale_row(errors[7][0] / errors[1][0]), 0.005)}
    d = scale_break_diagnosis(s_da3, n_anch, seam, n, loop_rel, (1, 7),
                              sigma_seam=0.003, sigma_anchor=0.08)
    assert d["suspect_seam"] == 4 and d["localised"] is True
    assert d["jump_pct"] > 5.0
    assert d["ranking"][0]["seam"] == 4
    # one cycle and NO absolute witness cannot place the jump — and says so
    d2 = scale_break_diagnosis({}, {}, seam, n, loop_rel, (1, 7),
                               sigma_seam=0.003, sigma_anchor=0.08)
    assert d2["localised"] is False


def test_seam_ratio_convention_matches_loop_row():
    sess = session()
    chunks, ci, errors = make_chunks(sess, scale_err=[1.0, 1.2, 0.9, 1.0, 1.0][:20])
    # seam 0→1 measured on the shared frame depths
    g = ci[1][0]
    r = seam_relative_scale(chunks[0]["depth"][g - ci[0][0]], chunks[1]["depth"][0])
    assert abs(r - (errors[0][0] / errors[1][0])) < 0.01     # r = s_1/s_0 (factors)
    # a bridge between chunks 0 and 1 must give the same row
    item = (0, (g, g + 10), 1, (g + 20, g + 30))
    bridge = make_bridge(sess, item)
    meas = lb.measure_bridge(bridge, lb.bridge_layout(item, ci), chunks[0], chunks[1],
                             fork_loops_cfg(), rigid=False)
    assert abs(lb.loop_scale_row(meas["s_ab"]) - np.log(r)) < 0.01
