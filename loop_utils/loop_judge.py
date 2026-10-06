"""The judge of the in-run pose graph: HELD-OUT LOOP CLOSURES (USER 2026-10-06).

A loop closure corrects a GLOBAL drift — metres between two visits of the same place
over a long walk. The local held-out surface pairs (intra-chunk, a few frames apart)
measure smoothness and cannot see that; on pccr 2408 they vetoed the only correction
that mattered, and without them bridges disagreeing by metres would have been applied
unjudged. So the judge is a share of the closures THEMSELVES: held out of the solve,
measured at the chain before and at the solution, paired, bootstrapped over the
closures (metric_lock.heldout_change). The graph is applied only when they improve
beyond their own noise. Deterministic: the same edges give the same split."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from loop_utils.lie import se3_inv, se3_log


def split_loop_edges(edges: Sequence[dict], owner: Sequence[int], frac: float,
                     min_edges: int) -> Tuple[List[dict], List[dict], dict]:
    """(fit, judge, report): every k-th edge (k = round(1/frac)) of the edges sorted by
    chunk pair, then keyframes, is held out — and every chunk pair keeps at least one
    FIT edge, so the judge measures both direct and transitive consistency. Fewer than
    ``min_edges`` edges: nothing is held out (declared in the report)."""
    edges = list(edges)
    n = len(edges)
    rep: dict = {"n_edges": n, "frac": float(frac), "min_edges": int(min_edges)}
    if n < int(min_edges) or float(frac) <= 0.0:
        rep.update({"n_fit": n, "n_judge": 0, "every_kth": None, "pairs_kept_in_fit": 0,
                    "judge_bridges": [], "judge_chunk_pairs": [],
                    "reason": (f"only {n} loop edge(s) (< {int(min_edges)}): nothing can be held out"
                               if n < int(min_edges) else "loop_holdout_frac is 0: nothing held out")})
        return edges, [], rep

    def pair(e: dict) -> Tuple[int, int]:
        a, b = int(owner[int(e["i"])]), int(owner[int(e["j"])])
        return (min(a, b), max(a, b))

    order = sorted(range(n), key=lambda k_: (pair(edges[k_]), int(edges[k_]["i"]),
                                             int(edges[k_]["j"]), int(edges[k_].get("bridge", k_))))
    k = max(2, int(round(1.0 / float(frac))))
    judge_idx = {order[pos] for pos in range(n) if pos % k == k - 1}
    by_pair: Dict[Tuple[int, int], List[int]] = {}
    for idx in order:
        by_pair.setdefault(pair(edges[idx]), []).append(idx)
    moved = 0
    for idxs in by_pair.values():
        if all(i_ in judge_idx for i_ in idxs):
            judge_idx.discard(idxs[0])
            moved += 1
    fit = [edges[i_] for i_ in order if i_ not in judge_idx]
    judge = [edges[i_] for i_ in order if i_ in judge_idx]
    rep.update({"n_fit": len(fit), "n_judge": len(judge), "every_kth": k, "pairs_kept_in_fit": moved,
                "judge_bridges": [int(e.get("bridge", -1)) for e in judge],
                "judge_chunk_pairs": [list(pair(e)) for e in judge]})
    if not judge:
        rep["reason"] = ("every held-out candidate was the only edge of its chunk pair — "
                         "nothing can be held out")
    return fit, judge, rep


def loop_residuals_m(edges: Sequence[dict], T0: np.ndarray, X: np.ndarray = None) -> List[float]:
    """Translation residual (m) of each loop edge at the poses X·T0 (X None = the chain
    as it is): the translation part of log(Z⁻¹ · Tᵢ⁻¹ · Tⱼ) — the graph's own
    relative-edge residual (pose_graph._res_rel), so before/after are comparable with
    what the fitted edges report."""
    out: List[float] = []
    for e in edges:
        i, j = int(e["i"]), int(e["j"])
        Ti = np.asarray(T0[i], np.float64) if X is None else np.asarray(X[i]) @ np.asarray(T0[i])
        Tj = np.asarray(T0[j], np.float64) if X is None else np.asarray(X[j]) @ np.asarray(T0[j])
        xi = se3_log(se3_inv(np.asarray(e["Z"], np.float64)) @ se3_inv(Ti) @ Tj)
        out.append(float(np.linalg.norm(np.asarray(xi)[3:])))
    return out
