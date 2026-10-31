"""Loop verifier by attention, training-free — claude_stac_slam2.txt, phase 3 (idea 2 of VGGT-SLAM 2.0,
MIT-SPARK arXiv 2601.19887: the backbone's global attention over a candidate pair says whether the
two frames look at the same scene). Implemented from the definition in the prompt, never from
MIT-SPARK/VGGT_SPARK (CC BY-NC 4.0).

WHERE. In the STAC branch of process_long_sequence, after the loop candidates are built (owners,
shared windows, duplicates) and before the metric lock / the bridges, while the Ω model is still on
the card: one Ω pass over the candidate's TWO keyframes with the run's own preprocessing and
resolution. A forward hook on ``aggregator.inter_frame_blocks[L].attn.qkv`` (LinearKMaskedBias)
captures its output for the GLOBAL block L; the hook is removed afterwards (the model's output with
and without it is identical bit for bit — tested). q and k are built from the capture exactly as
SelfAttention.compute_attention does: reshape [B, N, 3, H, D], unbind, q_norm / k_norm of the
module itself; a global block carries no RoPE. Only a block of type "global" per
``aggregator.inter_frame_attention_types`` is accepted — a "register" block is refused by name.
The token layout per frame — 1 camera token + num_register_tokens registers, the patches from
``aggregator.patch_token_start`` — is read from the model, nothing is hard-coded.

THE SCORE (the prompt's definition). For the ordered pair (A, B), at layer L, per head h:
Q = the queries of every token of A and B; K_A = the keys of A's PATCH tokens;
P_h(k, t) = softmax over t ∈ Q of ⟨q_t, k_k⟩, without scale, for each k ∈ K_A; P = the mean of P_h
over the heads; m_A(k) = max over t ∈ tokens of A of P(k, t); r(t) = max over k ∈ K_A of
P(k, t) / m_A(k) for t ∈ tokens of B; score(A, B) = the mean of the top ``top_fraction`` of r
(0.25 in the paper). The candidate passes when min(score(A, B), score(B, A)) >= min_score.

Every score is computed on the CPU in float64 from the captured tensors (prompt rule 8).
``mode: gate``: the candidates that fail never get a bridge; attention_gate.json records every
candidate with both scores, the layer, the threshold and the verdict.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

GATE_NAME = "attention_gate.json"
CALIBRATION_NAME = "attention_calibration.json"
GLOBAL = "global"
REGISTER = "register"


class AttentionGateError(RuntimeError):
    """The gate cannot run as configured — with the exact reason (a register layer, a model
    without the expected aggregator, a capture that did not happen)."""


# ── the model's own layout ────────────────────────────────────────────────────────────────

def aggregator_of(model) -> Any:
    """The aggregator of an Ω model (VGGTOmega.aggregator) or of anything with the same interface."""
    agg = getattr(model, "aggregator", None)
    if agg is None:
        raise AttentionGateError("the model has no 'aggregator' — the attention gate reads the global "
                                 "inter-frame blocks of VGGT-Omega's aggregator")
    for attr in ("inter_frame_blocks", "inter_frame_attention_types", "patch_token_start"):
        if not hasattr(agg, attr):
            raise AttentionGateError(f"the aggregator has no '{attr}' — not the interface the gate reads")
    return agg


def layer_types(agg) -> List[str]:
    return [str(t) for t in agg.inter_frame_attention_types]


def global_layers(agg) -> List[int]:
    return [i for i, t in enumerate(layer_types(agg)) if t == GLOBAL]


def register_layers(agg) -> List[int]:
    return [i for i, t in enumerate(layer_types(agg)) if t == REGISTER]


def check_layer(agg, layer: int) -> None:
    """A GLOBAL inter-frame block, or a refusal naming the layer's type and the register layers."""
    types = layer_types(agg)
    layer = int(layer)
    if layer < 0 or layer >= len(types):
        raise AttentionGateError(f"attention_gate.layer {layer} is outside the aggregator's {len(types)} "
                                 f"inter-frame blocks")
    if types[layer] != GLOBAL:
        raise AttentionGateError(f"attention_gate.layer {layer} is a '{types[layer]}' block — the gate reads "
                                 f"GLOBAL attention only; the register blocks of this model are "
                                 f"{register_layers(agg)}, the global ones {global_layers(agg)}")


def num_register_tokens(agg) -> int:
    rt = getattr(agg, "register_token", None)
    if rt is not None:
        return int(rt.shape[-2])
    return int(agg.patch_token_start) - 1


# ── the capture ───────────────────────────────────────────────────────────────────────────

@contextmanager
def capture_qkv(agg, layers: Iterable[int]):
    """Forward hooks on ``inter_frame_blocks[L].attn.qkv`` for every L in ``layers`` (global blocks
    only — checked), filling the yielded dict {L: qkv output (detached, on the CPU, as captured)};
    the hooks are removed on exit whatever happens."""
    import torch
    layers = [int(L) for L in layers]
    for L in layers:
        check_layer(agg, L)
    captured: Dict[int, "torch.Tensor"] = {}
    handles = []

    def _make(L):
        def _hook(_module, _inputs, output):
            captured[L] = output.detach().to("cpu")
        return _hook

    try:
        for L in layers:
            qkv = agg.inter_frame_blocks[L].attn.qkv
            handles.append(qkv.register_forward_hook(_make(L)))
        yield captured
    finally:
        for h in handles:
            h.remove()


def hooks_installed(agg) -> int:
    """How many forward hooks sit on the inter-frame qkv modules (0 after a clean capture)."""
    n = 0
    for blk in agg.inter_frame_blocks:
        n += len(getattr(blk.attn.qkv, "_forward_hooks", {}))
    return n


def _layernorm64(x: np.ndarray, ln) -> np.ndarray:
    """nn.LayerNorm over the last axis in float64 with the module's own weight, bias and eps
    (None → identity, as SelfAttention without use_qk_norm)."""
    if ln is None:
        return x
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + float(ln.eps))
    w = ln.weight.detach().to("cpu").double().numpy() if ln.weight is not None else None
    b = ln.bias.detach().to("cpu").double().numpy() if ln.bias is not None else None
    if w is not None:
        y = y * w
    if b is not None:
        y = y + b
    return y


def qk_from_qkv(attn, qkv) -> Tuple[np.ndarray, np.ndarray]:
    """q and k ([H, N, D] float64) from a captured qkv output ([1, N, 3C]) exactly as
    SelfAttention.compute_attention forms them: reshape [B, N, 3, H, C // H], unbind, heads first,
    q_norm / k_norm of the module (no RoPE: a global block)."""
    x = np.asarray(qkv.detach().to("cpu").double().numpy() if hasattr(qkv, "detach") else qkv, np.float64)
    if x.ndim != 3 or x.shape[0] != 1:
        raise AttentionGateError(f"a captured qkv must be [1, N, 3C], got {x.shape}")
    H = int(attn.num_heads)
    C = int(attn.qkv.in_features)
    N = x.shape[1]
    if x.shape[2] != 3 * C:
        raise AttentionGateError(f"captured qkv has {x.shape[2]} features, the module's in_features is {C} (3C expected)")
    x = x.reshape(1, N, 3, H, C // H)
    q, k = x[0, :, 0], x[0, :, 1]                         # [N, H, D]
    q = np.transpose(q, (1, 0, 2))                        # [H, N, D]
    k = np.transpose(k, (1, 0, 2))
    if getattr(attn, "use_qk_norm", False):
        q = _layernorm64(q, attn.q_norm)
        k = _layernorm64(k, attn.k_norm)
    return q, k


def recompute_attention(attn, qkv):
    """The module's attention output recomputed from a captured qkv (torch, the module's dtype):
    reshape, unbind, q_norm / k_norm, SDPA, transpose, proj — the fidelity check of the capture
    (the result must equal attn(x) for the x that produced qkv)."""
    import torch
    B, N, _ = qkv.shape
    C = attn.qkv.in_features
    x = qkv.reshape(B, N, 3, attn.num_heads, C // attn.num_heads)
    q, k, v = torch.unbind(x, 2)
    q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
    if getattr(attn, "use_qk_norm", False):
        q = attn.q_norm(q)
        k = attn.k_norm(k)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):          # the reference kernel: this is a check, not a pass
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    y = y.transpose(1, 2).reshape(B, N, C)
    return attn.proj(y)


# ── the score ─────────────────────────────────────────────────────────────────────────────

def frame_token_ranges(n_total: int, n_frames: int) -> List[Tuple[int, int]]:
    if n_frames <= 0 or n_total % n_frames != 0:
        raise AttentionGateError(f"{n_total} tokens do not split into {n_frames} frames")
    n = n_total // n_frames
    return [(f * n, (f + 1) * n) for f in range(n_frames)]


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def directed_score(q: np.ndarray, k: np.ndarray, *, tokens_a: Tuple[int, int], tokens_b: Tuple[int, int],
                   patch_token_start: int, top_fraction: float) -> Dict[str, Any]:
    """score(A, B) of the prompt: q, k [H, N, D] float64 over the tokens of BOTH frames
    (N = 2 · tokens per frame); A's and B's token ranges; K_A = A's patch tokens."""
    H, N, D = q.shape
    a0, a1 = tokens_a
    b0, b1 = tokens_b
    ka0 = a0 + int(patch_token_start)
    if ka0 >= a1:
        raise AttentionGateError("frame A has no patch tokens after patch_token_start")
    P = np.zeros((a1 - ka0, N), np.float64)
    for h in range(H):
        logits = k[h, ka0:a1] @ q[h].T                   # ⟨q_t, k_k⟩ for k ∈ K_A, t ∈ Q — no scale
        P += _softmax_rows(logits)
    P /= float(H)
    m_a = P[:, a0:a1].max(axis=1)                        # m_A(k): the key's best query inside A
    m_a = np.where(m_a > 0.0, m_a, np.finfo(np.float64).tiny)
    r = (P[:, b0:b1] / m_a[:, None]).max(axis=0)        # r(t) for t ∈ tokens of B
    n_top = max(1, int(math.ceil(float(top_fraction) * r.size)))
    top = np.sort(r)[::-1][:n_top]
    return {"score": float(top.mean()), "n_top": int(n_top), "n_tokens_b": int(r.size),
            "n_keys_a": int(a1 - ka0), "r_median": float(np.median(r)), "r_max": float(r.max())}


def pair_scores(q: np.ndarray, k: np.ndarray, *, n_frames: int, patch_token_start: int, top_fraction: float
                ) -> Dict[str, Any]:
    """Both directions of a two-frame capture: score(A, B), score(B, A) and their minimum."""
    if n_frames != 2:
        raise AttentionGateError(f"the gate scores a PAIR: {n_frames} frames were captured")
    ra, rb = frame_token_ranges(q.shape[1], 2)
    ab = directed_score(q, k, tokens_a=ra, tokens_b=rb, patch_token_start=patch_token_start, top_fraction=top_fraction)
    ba = directed_score(q, k, tokens_a=rb, tokens_b=ra, patch_token_start=patch_token_start, top_fraction=top_fraction)
    return {"score_ab": ab["score"], "score_ba": ba["score"], "score_min": float(min(ab["score"], ba["score"])),
            "detail_ab": ab, "detail_ba": ba, "tokens_per_frame": int(ra[1] - ra[0]),
            "patch_token_start": int(patch_token_start)}


def scores_from_captures(agg, captures: Dict[int, Any], *, top_fraction: float, n_frames: int = 2
                         ) -> Dict[int, Dict[str, Any]]:
    """Per captured layer: the pair's scores (the layer's own attn module forms q, k)."""
    out = {}
    for L, qkv in sorted(captures.items()):
        attn = agg.inter_frame_blocks[int(L)].attn
        q, k = qk_from_qkv(attn, qkv)
        out[int(L)] = pair_scores(q, k, n_frames=n_frames, patch_token_start=int(agg.patch_token_start),
                                  top_fraction=top_fraction)
    return out


# ── the Ω pass over two keyframes ─────────────────────────────────────────────────────────

def omega_pair_pass(adapter, image_paths: Sequence[str], layers: Iterable[int]) -> Dict[int, Any]:
    """One Ω forward over ``image_paths`` (two keyframes) with the run's own preprocessing
    (adapter.preproc_mode / image_resolution) and autocast dtype, capturing the qkv of the
    requested global layers. Returns {L: qkv on the CPU}. The hooks are gone when it returns."""
    import torch
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    model = adapter.model
    agg = aggregator_of(model)
    images = load_and_preprocess_images(list(image_paths), mode=adapter.preproc_mode,
                                        image_resolution=adapter.image_resolution).to(adapter.device)
    if images.dim() == 5:
        images = images[0]
    with capture_qkv(agg, layers) as cap:
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=adapter.dtype):
                model(images)
    for L in layers:
        if int(L) not in cap:
            raise AttentionGateError(f"the forward did not reach inter_frame_blocks[{int(L)}].attn.qkv — "
                                     f"nothing was captured for layer {int(L)}")
    torch.cuda.empty_cache()
    return cap


def gate_record(cands: Sequence[Dict[str, Any]], *, layer: int, min_score: float, top_fraction: float,
                mode: str, agg=None) -> Dict[str, Any]:
    """attention_gate.json: every candidate with both scores and its verdict."""
    rec = {"version": 1, "mode": str(mode), "layer": int(layer), "min_score": float(min_score),
           "top_fraction": float(top_fraction),
           "rule": "pass when min(score(A,B), score(B,A)) >= min_score; score = mean of the top "
                   "top_fraction of r(t) = max_k P(k,t) / m_A(k), P the head-mean softmax over the "
                   "queries of A and B of <q_t, k_k> for the patch keys k of A (no scale)",
           "n_candidates": len(cands), "n_passed": int(sum(1 for c in cands if c.get("pass"))),
           "candidates": list(cands)}
    if agg is not None:
        rec["layer_type"] = layer_types(agg)[int(layer)]
        rec["global_layers"] = global_layers(agg)
        rec["register_layers"] = register_layers(agg)
        rec["patch_token_start"] = int(agg.patch_token_start)
        rec["num_register_tokens"] = num_register_tokens(agg)
    return rec
