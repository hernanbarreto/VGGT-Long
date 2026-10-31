"""claude_stac_slam2.txt, phase 3 — the loop verifier by attention (loop_utils.attention_gate,
VGGT_Long._stac_attention_gate) on a tiny stand-in of Ω's aggregator with random weights (CPU).
No card, no model weights."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_FORK = Path(__file__).resolve().parents[1]
_SERVER = _FORK.parents[1] / "server"
for p in (_FORK, _SERVER, _SERVER / "tests", _FORK / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from loop_utils import attention_gate as AG                                      # noqa: E402
from slam2_scenario import GATE_ON, FakeAttentionModel, compare_products, make_runner   # noqa: E402
from tiny_omega import FakeAggregator, tiny_block                                # noqa: E402


def test_the_capture_reproduces_the_modules_attention_exactly():
    """q, k, v from the captured qkv → q_norm / k_norm → SDPA → proj equals the module's output."""
    blk = tiny_block(dim=16, heads=2)
    agg = FakeAggregator(dim=16, heads=2)
    agg.inter_frame_blocks[0] = blk
    x = torch.randn(1, 30, 16)
    with AG.capture_qkv(agg, [0]) as cap:
        with torch.no_grad():
            y = blk.attn(blk.norm1(x))                    # the block feeds attn the normed tokens
    assert 0 in cap and cap[0].shape == (1, 30, 48)
    with torch.no_grad():
        y2 = AG.recompute_attention(blk.attn, cap[0])
    assert torch.allclose(y, y2, atol=1e-6, rtol=1e-6)
    # q, k in float64 follow the same construction (heads first, q_norm / k_norm of the module)
    q, k = AG.qk_from_qkv(blk.attn, cap[0])
    assert q.shape == (2, 30, 8) and k.shape == (2, 30, 8) and q.dtype == np.float64
    qkv = cap[0].reshape(1, 30, 3, 2, 8)
    q_ref = blk.attn.q_norm(qkv[0, :, 0].transpose(0, 1)).detach().double().numpy()
    assert np.allclose(q, q_ref, atol=1e-6)


def test_hooks_are_removed_and_the_output_is_identical_with_and_without_them():
    agg = FakeAggregator()
    tokens = torch.randn(1, 2, 15, 16)
    with torch.no_grad():
        ref = agg(tokens.clone())
        with AG.capture_qkv(agg, [0, 2]) as cap:
            out = agg(tokens.clone())
    assert torch.equal(ref, out)
    assert set(cap) == {0, 2} and AG.hooks_installed(agg) == 0
    with torch.no_grad():
        assert torch.equal(agg(tokens.clone()), ref)
    # a failing body still removes the hooks
    with pytest.raises(RuntimeError, match="boom"):
        with AG.capture_qkv(agg, [0]):
            raise RuntimeError("boom")
    assert AG.hooks_installed(agg) == 0


def test_a_register_layer_is_refused_by_name():
    agg = FakeAggregator()
    assert AG.global_layers(agg) == [0, 2] and AG.register_layers(agg) == [1]
    with pytest.raises(AG.AttentionGateError) as e:
        AG.check_layer(agg, 1)
    assert "'register' block" in str(e.value) and "[1]" in str(e.value) and "[0, 2]" in str(e.value)
    with pytest.raises(AG.AttentionGateError):
        with AG.capture_qkv(agg, [1]):
            pass
    with pytest.raises(AG.AttentionGateError, match="outside"):
        AG.check_layer(agg, 7)


def _qk(features_a, features_b):
    """q = k = the token features of frames A and B (heads split from the feature axis)."""
    x = np.concatenate([features_a, features_b], axis=0)            # [2N, C]
    H = 2
    q = np.transpose(x.reshape(x.shape[0], H, -1), (1, 0, 2))
    return q, q.copy()


def test_the_score_separates_a_repeated_pair_from_a_noise_pair_and_the_min_is_symmetric():
    rng = np.random.default_rng(0)
    N, C, pts = 20, 64, 3            # 32-d heads: a self-match ⟨x, x⟩ ≈ 32 against random dots of σ ≈ 5.7
    A = rng.normal(size=(N, C))
    q, k = _qk(A, A.copy())                                          # the same frame twice
    rep = AG.pair_scores(q, k, n_frames=2, patch_token_start=pts, top_fraction=0.25)
    q2, k2 = _qk(A, rng.normal(size=(N, C)))                          # another, unrelated frame
    noise = AG.pair_scores(q2, k2, n_frames=2, patch_token_start=pts, top_fraction=0.25)
    assert rep["score_min"] > 0.99 and noise["score_min"] < 0.2, (rep["score_min"], noise["score_min"])
    assert rep["score_min"] == min(rep["score_ab"], rep["score_ba"])
    # swapping the frames swaps the two directed scores exactly
    B = rng.normal(size=(N, C))
    q3, k3 = _qk(A, B)
    q4, k4 = _qk(B, A)
    s34 = AG.pair_scores(q3, k3, n_frames=2, patch_token_start=pts, top_fraction=0.25)
    s43 = AG.pair_scores(q4, k4, n_frames=2, patch_token_start=pts, top_fraction=0.25)
    assert s34["score_ab"] == s43["score_ba"] and s34["score_ba"] == s43["score_ab"]
    assert s34["score_min"] == s43["score_min"]
    assert rep["detail_ab"]["n_keys_a"] == N - pts and rep["detail_ab"]["n_top"] == 5
    # float64 on the CPU, from the captured tensors
    assert isinstance(rep["score_ab"], float)


def test_the_scores_through_the_stand_in_pass_an_identical_pair_and_fail_a_noise_pair():
    fake = FakeAttentionModel()
    cap = fake.capture(["a.jpg", "b.jpg"], [0, 2])
    scores = AG.scores_from_captures(fake.model.aggregator, cap, top_fraction=0.25)
    assert set(scores) == {0, 2}
    for L, sc in scores.items():
        assert sc["score_min"] > 0.99, (L, sc)
        assert sc["tokens_per_frame"] == 1 + 2 + 12 and sc["patch_token_start"] == 3
    noise = AG.scores_from_captures(fake.model.aggregator,
                                    fake.capture_pair(fake.frame_features(7), fake.frame_features(8), [0, 2]),
                                    top_fraction=0.25)
    assert all(sc["score_min"] < 0.2 for sc in noise.values()), noise


@pytest.fixture(scope="module")
def synth():
    from synth_metric import make_chunks, make_session
    sess = make_session(n_kf=150, H=40, W=56)
    chunks, ci, _ = make_chunks(sess, scale_err=[1.0, 1.06, 0.97, 1.04], yaw_err_deg=[0.0, 1.5, -1.0, 2.0],
                                t_err=[(0, 0, 0), (0.3, 0, 0.1), (-0.2, 0.05, 0.3), (0.1, 0, -0.4)])
    return sess, chunks, ci


def _paired(r, ci, pairs):
    from loop_utils.sim3utils import process_loop_list
    r.loop_list = list(pairs)
    r.loop_cands = [{"i": i, "j": j, "sim": 0.9, "source": "salad"} for i, j in pairs]
    return [tuple(res) + (pair,) for res, pair in zip(process_loop_list(ci, r.loop_list, half_window=10), pairs)]


def test_the_gate_keeps_passing_candidates_drops_failing_ones_and_records_everything(tmp_path, synth):
    sess, chunks, ci = synth
    r, save_dir = make_runner(tmp_path, sess, chunks, ci, gate=GATE_ON)
    fake = FakeAttentionModel()
    r.model = fake
    calls = []

    def capture(paths, layers):
        calls.append(tuple(paths))
        if paths[0].endswith("004200.jpg"):            # keyframe 140 → a NOISE second frame
            return fake.capture_pair(fake.frame_features(7), fake.frame_features(8), layers)
        return fake.capture(paths, layers)
    r._stac_attention_capture = capture
    paired = _paired(r, ci, [(140, 8), (120, 20)])
    kept = r._stac_attention_gate(paired)
    assert len(calls) == 2
    assert [k[-1] for k in kept] == [(120, 20)]
    doc = json.loads((save_dir / AG.GATE_NAME).read_text())
    assert doc["mode"] == "gate" and doc["layer"] == 2 and doc["layer_type"] == "global"
    assert doc["n_candidates"] == 2 and doc["n_passed"] == 1
    by = {(c["i"], c["j"]): c for c in doc["candidates"]}
    assert by[(140, 8)]["pass"] is False and by[(120, 20)]["pass"] is True
    assert by[(140, 8)]["score_min"] < doc["min_score"] <= by[(120, 20)]["score_min"]
    assert doc["global_layers"] == [0, 2] and doc["register_layers"] == [1]
    # the rejection reaches the loop report through _stac_plan_bridges
    r._stac_metric_lock()
    r._stac_plan_bridges(kept)
    rej = [x for x in r._stac_loops_rejected if x["stage"] == "attention_gate"]
    assert len(rej) == 1 and rej[0]["candidate"]["i"] == 140 and rej[0]["attention"]["layer"] == 2
    # resume: the verdicts of this run are read back, no capture runs again
    kept2 = r._stac_attention_gate(paired)
    assert [k[-1] for k in kept2] == [(120, 20)] and len(calls) == 2


def test_a_register_layer_in_the_config_stops_the_gate_by_name(tmp_path, synth):
    sess, chunks, ci = synth
    r, save_dir = make_runner(tmp_path, sess, chunks, ci, gate=dict(GATE_ON, layer=1))
    r.model = FakeAttentionModel()
    r.model.model.aggregator.inter_frame_attention_types = ["global", "register", "global"]
    r.config["Model"]["loops"]["attention_gate"]["layer"] = 1
    paired = _paired(r, ci, [(140, 8)])
    with pytest.raises(AG.AttentionGateError, match="register"):
        r._stac_attention_gate(paired)


def test_off_or_absent_returns_the_list_untouched_and_writes_nothing(tmp_path, synth):
    sess, chunks, ci = synth
    for gate in (None, dict(GATE_ON, mode="off")):
        r, save_dir = make_runner(tmp_path / ("absent" if gate is None else "off"), sess, chunks, ci, gate=gate)
        paired = _paired(r, ci, [(140, 8)])
        assert r._stac_attention_gate(paired) is paired
        assert not (save_dir / AG.GATE_NAME).exists()


# ── bit for bit: the gate with every candidate passing = the branch without the gate ────────

def _env():
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    env.pop("PYTHONPATH", None)
    return env


def _run(out: Path, *flags) -> Path:
    r = subprocess.run([sys.executable, str(_FORK / "tests" / "slam2_scenario.py"), str(out), *flags],
                       cwd=str(_FORK), env=_env(), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"scenario failed:\n{r.stdout[-3000:]}\n{r.stderr[-6000:]}"
    return out / "maplong_run"


def test_the_gate_with_every_candidate_passing_leaves_the_products_byte_for_byte(tmp_path):
    plain = _run(tmp_path / "plain")
    gated = _run(tmp_path / "gated", "--gate-pass-all")
    assert (gated / AG.GATE_NAME).is_file() and not (plain / AG.GATE_NAME).is_file()
    doc = json.loads((gated / AG.GATE_NAME).read_text())
    assert doc["n_candidates"] == 1 and doc["n_passed"] == 1
    diffs = compare_products(plain, gated, ignore_names=[AG.GATE_NAME])
    assert diffs == [], "\n".join(diffs)
    gated2 = _run(tmp_path / "gated2", "--gate-pass-all")
    assert compare_products(gated, gated2) == []
