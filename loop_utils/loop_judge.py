"""The judge of the in-run pose graph: the LOOP CLOSURES THEMSELVES, leave-one-out
(USER 2026-10-06, made leave-one-out 2026-10-07 — docs/plan_determinismo.md points 1-2).

A loop closure corrects a GLOBAL drift — metres between two visits of the same place
over a long walk. The local held-out surface pairs (intra-chunk, a few frames apart)
measure smoothness and cannot see that; on pccr 2408 they vetoed the only correction
that mattered, and without them bridges disagreeing by metres would have been applied
unjudged. So the judges are the closures: each one is left out of ONE solve and its
residual measured at the chain before and at the solution that never saw it
(:func:`judge_leave_one_out`, called by vggt_long._stac_pose_graph); paired, bootstrapped
and decided by metric_lock.decide_change (significant, >= min_judge_closures(confidence)
judges, median improvement >= the user's factor x the largest bridge σ). n judges = n closures:
no held-out share that changed with the edge count and the chunk pairs (pccr: 10 edges
in 7 chunk pairs could never hold out 5). Deterministic: the same edges give the same
judges."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from loop_utils.lie import se3_inv, se3_log


def min_judge_closures(confidence: float) -> int:
    """The fewest held-out closures that can testify (USER 2026-10-06): if a correction did
    nothing real, each held-out closure improves or worsens like a coin; all n improving by chance
    has probability 0.5**n, which must fall below 1 - confidence — n = ceil(log(1-c)/log(0.5)).
    At the declared 0.95: 5. Fewer cannot tell a correction from luck (pccr 2026-08-31: ONE
    held-out closure, 34.5 → 34.5 cm, read as 'improves')."""
    import math
    c = float(confidence)
    return int(math.ceil(math.log(1.0 - c) / math.log(0.5) - 1e-12))


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


def build_keyframe_graph(T0: np.ndarray, owner: Sequence[int], sig_t: Dict[int, float],
                         sig_r: Dict[int, float], seam_res: Dict[int, float], gcfg: dict,
                         loop_edges: Sequence[dict], device: str):
    """The keyframe SE(3) graph: the chain's odometry (σ measured per chunk from the held-out
    surface pairs; the link that crosses chunk ownership adds that seam's own measured
    residual in quadrature) plus the given loop edges, each with its own measured σ (Huber).
    Returns (PoseGraph, [(edge id, loop edge)]). ``device``: where the solver runs — the
    fork passes the card (no CPU fallback, plan point 12); tests pass 'cpu'."""
    from loop_utils.pose_graph import PoseGraph
    N = len(T0)
    pg = PoseGraph(T0, gcfg, device=device)
    for g in range(N - 1):
        Z = se3_inv(T0[g]) @ T0[g + 1]
        k_o = int(owner[g])
        s_t, s_deg = float(sig_t[k_o]), float(sig_r[k_o])
        if owner[g] != owner[g + 1]:
            s_t = float(np.sqrt(s_t ** 2 + float(seam_res.get(k_o, 0.0)) ** 2))
        pg.add_relative(g, g + 1, Z, s_deg, s_t, huber=False, tag="odo")
    loop_ids = []
    for e in loop_edges:
        eid = pg.add_relative(int(e["i"]), int(e["j"]), np.asarray(e["Z"]),
                              float(e["sigma_deg"]), float(e["sigma_m"]), huber=True,
                              tag=f"loop:{e['bridge']}")
        loop_ids.append((eid, e))
    return pg, loop_ids


def judge_leave_one_out(edges: Sequence[dict], T0: np.ndarray,
                        solve: Callable[[List[dict]], Tuple[np.ndarray, bool]], *,
                        error_factor: float, confidence: float,
                        owner: Optional[Sequence[int]] = None,
                        log: Callable[[str], object] = print) -> dict:
    """THE JUDGE of the in-run pose graph (USER 2026-10-07, docs/plan_determinismo.md points
    1 and 2): EVERY loop closure judges once.

    With n closures and n < min_judge_closures(confidence) (5 at 0.95) the graph cannot be
    judged: that is DECLARED BEFORE SOLVING — ``solve`` is never called, nothing is applied.
    Otherwise the graph is solved n times, each time WITHOUT one closure (``solve(edges
    minus closure q)`` → (corrections X (N,4,4), converged)), and closure q's residual is
    measured at the chain (X = identity) and at the solution that never saw it. A solve that
    did not converge is not a solution: its closure cannot testify (discarded and COUNTED).
    The paired residuals are decided by metric_lock.decide_change — all three at once:
    (a) the whole ``confidence`` bootstrap interval of the median improvement above zero,
    (b) at least min_judge_closures(confidence) judges, (c) the median improvement >=
    ``error_factor`` (the user's 2) x the LARGEST σ of the closures (the measured error of the
    bridges involved). Only then may the caller solve the final graph with ALL the closures
    and apply it. Every margin is in the returned dict (JSON-able) and in the log."""
    from loop_utils.metric_lock import decide_change
    conf = float(confidence)
    min_judge = min_judge_closures(conf)
    edges = list(edges)
    n = len(edges)
    for e in edges:
        s = float(e["sigma_m"])
        if not (np.isfinite(s) and s > 0.0):
            raise ValueError(f"loop closure {e.get('bridge')} ({e.get('i')}<->{e.get('j')}) "
                             f"has σ {e.get('sigma_m')!r} — a closure enters the graph only with "
                             f"its measured σ")
    # USER 2026-10-07: the error is that of the closures that PASSED the cycle-consistency
    # filter (consistent_closures), each with its own σ measured at its place — their median,
    # never the largest of all (one bad bridge set pccr's bar at 2 x 556 cm)
    error_m = float(np.median([float(e["sigma_m"]) for e in edges])) if edges else None
    out = {"rule": ("USER 2026-10-07: leave-one-out — each closure judges once; applied only if "
                    "significant, >= min_judges judges and median improvement >= "
                    "error_factor x the median own σ of the cycle-consistent closures"),
           "n_closures": n, "min_judges": int(min_judge), "confidence": conf,
           "error_factor": float(error_factor), "error_m": error_m,
           "error_source": "the median own σ (sigma_m) of the closures that passed the cycle "
                           "consistency filter"}
    if n < min_judge:
        out.update({"solved": False, "declared_before_solving": True, "improves": False,
                    "n_judges": 0, "n_discarded": 0, "per_edge": [], "decision": None,
                    "reason": ("no verified loop closure — nothing closes; odometry alone "
                               "would only smooth what the seams already fixed" if n == 0 else
                               f"only {n} verified loop closure(s) — {min_judge} are needed to "
                               f"judge the graph at {conf:g} (each closure judges once, left "
                               f"out of its own solve): declared before solving, NOT solved, "
                               f"NOT applied")})
        return out
    per = []
    for q, e in enumerate(edges):
        rest = edges[:q] + edges[q + 1:]
        X_q, conv_q = solve(rest)
        b_q = float(loop_residuals_m([e], T0)[0])
        a_q = (float(loop_residuals_m([e], T0, X_q)[0]) if X_q is not None
               else float("nan"))
        valid = bool(conv_q) and np.isfinite(b_q) and np.isfinite(a_q)
        rec = {"bridge": int(e.get("bridge", -1)), "i": int(e["i"]), "j": int(e["j"]),
               "sigma_m": float(e["sigma_m"]), "before_m": b_q, "after_m": a_q,
               "improvement_m": (b_q - a_q) if valid else None,
               "converged": bool(conv_q), "judges": bool(valid)}
        if owner is not None:
            rec["chunks"] = [int(owner[int(e["i"])]), int(owner[int(e["j"])])]
        if not valid:
            rec["discarded"] = ("its leave-one-out solve did not converge — not a solution, it "
                                "cannot testify")
        per.append(rec)
        log(f"[pose-graph] judge {q + 1}/{n} (bridge {e.get('bridge')}, {int(e['i'])}<->"
            f"{int(e['j'])}, σ {float(e['sigma_m']) * 100:.2f} cm): left out of its solve → "
            f"{b_q * 100:.2f} → {a_q * 100:.2f} cm"
            + ("" if valid else " — NOT converged, discarded"))
    judges = [r for r in per if r["judges"]]
    decision = decide_change([r["before_m"] for r in judges], [r["after_m"] for r in judges],
                             error=error_m, error_factor=float(error_factor), confidence=conf,
                             min_judges=min_judge)
    decision["error_source"] = out["error_source"]
    out.update({"solved": True, "declared_before_solving": False,
                "improves": bool(decision["improves"]), "n_judges": len(judges),
                "n_discarded": n - len(judges), "per_edge": per, "decision": decision,
                "median_before_m": (float(np.median([r["before_m"] for r in judges]))
                                    if judges else None),
                "median_after_m": (float(np.median([r["after_m"] for r in judges]))
                                   if judges else None),
                "reason": decision["reason"]})
    log(f"[pose-graph] leave-one-out judge ({len(judges)} of {n} closure(s) testify, "
        f"{n - len(judges)} discarded): {decision['reason']} (margins: CI {decision['ci_margin']:+.6g} m, "
        f"judges {decision['judges_margin']:+d}, error {decision['error_margin']:+.6g} m)")
    return out


# ── cycle consistency of the closures (USER 2026-10-07) ─────────────────────────────────────

def _chi2_quantile(confidence: float, dof: int) -> float:
    """The ``confidence`` quantile of a chi-square with an EVEN number of degrees of freedom,
    by bisection on its closed-form CDF 1 - e^(-x/2) Σ_{k<dof/2} (x/2)^k / k! — derived from
    the declared confidence, no table."""
    import math
    if dof <= 0 or dof % 2:
        raise ValueError(f"chi-square quantile: even dof required, got {dof}")
    c = float(confidence)

    def cdf(x: float) -> float:
        h = x / 2.0
        s, term = 0.0, 1.0
        for k in range(dof // 2):
            if k:
                term *= h / k
            s += term
        return 1.0 - math.exp(-h) * s

    lo, hi = 0.0, 1.0
    while cdf(hi) < c:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < c:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _rot_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(np.asarray(R)[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))


def consistent_closures(edges: Sequence[dict], T0: np.ndarray, owner: Sequence[int],
                        sig_t: Dict[int, float], sig_r: Dict[int, float], confidence: float,
                        log: Callable[[str], object] = print) -> Tuple[List[dict], dict]:
    """The closures that AGREE WITH EACH OTHER, measured at their own place (USER 2026-10-07).

    Two closures A and B that join the same pair of chunks (p, q) close a cycle through the
    chain INSIDE each chunk — A, the chain from A to B in chunk p, B, the chain back in chunk q:
    C = (T_iA⁻¹ T_iB) · Z_B · (T_jB⁻¹ T_jA) · Z_A⁻¹, identity when both are right. Inside a chunk
    the chain is ONE Omega pass, measured per link (sig_t / sig_r, the held-out odometry), so
    what does not close is the two closures disagreeing — read in centimetres at the closures'
    place, never through a chunk-to-chunk transform at the chunk's origin (pccr 2026-10-07: that
    turned 8°–45° of disagreement into a 556 cm σ for all three start↔end bridges).

    The pair is consistent when the cycle's Mahalanobis distance — translation over
    sqrt(σ_A² + σ_B² + chain² + (σrot_A² + σrot_B²)·lever²), rotation over the closures' and the
    chain's rotation σ — is within the chi-square quantile (6 dof) at ``confidence`` (pairwise
    consistency maximisation, PCM: Mangelson et al., ICRA 2018). Per chunk pair the LARGEST set
    of mutually consistent closures is kept; ties go to the most informative set (Σ 1/σ²), then
    to the lower bridge ids. A closure outside it never enters the graph (declared, with its
    cycles); a closure alone on its chunk pair is kept, declared uncorroborated. Every closure
    keeps its OWN σ. Returns (kept closures in their input order, report)."""
    import itertools
    edges = list(edges)
    T0 = np.asarray(T0, np.float64)
    bar = _chi2_quantile(float(confidence), 6)

    def oriented(e):
        i, j, Z = int(e["i"]), int(e["j"]), np.asarray(e["Z"], np.float64)
        p, q = int(owner[i]), int(owner[j])
        if (p, i) > (q, j):
            i, j, Z, p, q = j, i, se3_inv(Z), q, p
        return i, j, Z, p, q

    groups: Dict[Tuple[int, int], List[int]] = {}
    orient = []
    for k, e in enumerate(edges):
        o = oriented(e)
        orient.append(o)
        groups.setdefault((o[3], o[4]), []).append(k)
    keep = set()
    rep_groups = []
    for (p, q), ks in sorted(groups.items()):
        g = {"chunks": [p, q], "bridges": [int(edges[k].get("bridge", -1)) for k in ks]}
        if len(ks) == 1:
            keep.add(ks[0])
            g.update({"kept": g["bridges"], "rejected": [],
                      "note": "the only closure of its chunk pair — kept, uncorroborated"})
            rep_groups.append(g)
            continue
        cons = {}
        pairs = []
        for a, b in itertools.combinations(ks, 2):
            iA, jA, ZA, _, _ = orient[a]
            iB, jB, ZB, _, _ = orient[b]
            C = (se3_inv(T0[iA]) @ T0[iB]) @ ZB @ (se3_inv(T0[jB]) @ T0[jA]) @ se3_inv(ZA)
            ct = float(np.linalg.norm(C[:3, 3]))
            cr = _rot_angle(C)
            sA, sB = float(edges[a]["sigma_m"]), float(edges[b]["sigma_m"])
            rA, rB = np.radians(float(edges[a]["sigma_deg"])), np.radians(float(edges[b]["sigma_deg"]))
            n_p, n_q = abs(iB - iA), abs(jB - jA)
            chain_t2 = float(sig_t[p]) ** 2 * n_p + float(sig_t[q]) ** 2 * n_q
            chain_r2 = np.radians(float(sig_r[p])) ** 2 * n_p + np.radians(float(sig_r[q])) ** 2 * n_q
            lever = max(float(np.linalg.norm(T0[iA][:3, 3] - T0[iB][:3, 3])),
                        float(np.linalg.norm(T0[jA][:3, 3] - T0[jB][:3, 3])))
            st = float(np.sqrt(sA ** 2 + sB ** 2 + chain_t2 + (rA ** 2 + rB ** 2) * lever ** 2))
            sr = float(np.sqrt(rA ** 2 + rB ** 2 + chain_r2))
            m2 = (ct / st) ** 2 + (cr / sr) ** 2
            ok = bool(m2 <= bar)
            cons[(a, b)] = cons[(b, a)] = ok
            pairs.append({"bridges": [int(edges[a].get("bridge", -1)), int(edges[b].get("bridge", -1))],
                          "cycle_t_m": ct, "cycle_deg": float(np.degrees(cr)), "sigma_t_m": st,
                          "sigma_deg": float(np.degrees(sr)), "mahalanobis2": float(m2),
                          "bar": float(bar), "margin": float(bar - m2), "consistent": ok})
            log(f"[pose-graph] cycle bridges {pairs[-1]['bridges'][0]}<->{pairs[-1]['bridges'][1]} "
                f"(chunks {p}<->{q}): {ct * 100:.1f} cm, {np.degrees(cr):.2f}° vs σ "
                f"{st * 100:.1f} cm / {np.degrees(sr):.2f}° → m² {m2:.2f} "
                f"{'≤' if ok else '>'} {bar:.2f} ({confidence:g}, 6 dof): "
                f"{'consistent' if ok else 'INCONSISTENT'}")
        best = None
        for size in range(len(ks), 0, -1):
            for sub in itertools.combinations(ks, size):
                if all(cons[(a, b)] for a, b in itertools.combinations(sub, 2)):
                    info = float(sum(1.0 / float(edges[k]["sigma_m"]) ** 2 for k in sub))
                    key = (-info, [int(edges[k].get("bridge", -1)) for k in sub])
                    if best is None or key < best[0]:
                        best = (key, sub)
            if best is not None:
                break
        kept = set(best[1])
        keep |= kept
        g.update({"pairs": pairs, "kept": [int(edges[k].get("bridge", -1)) for k in ks if k in kept],
                  "rejected": [int(edges[k].get("bridge", -1)) for k in ks if k not in kept]})
        rep_groups.append(g)
        if g["rejected"]:
            log(f"[pose-graph] chunks {p}<->{q}: bridge(s) {g['rejected']} disagree with the "
                f"largest consistent set {g['kept']} — left out of the graph (declared)")
    kept_edges = [e for k, e in enumerate(edges) if k in keep]
    report = {"rule": "USER 2026-10-07: pairwise cycle consistency per chunk pair at the closures' "
                      "own place (PCM), chi-square 6 dof at the declared confidence; the largest "
                      "consistent set is kept, each closure with its own σ",
              "confidence": float(confidence), "chi2_bar": float(bar),
              "n_in": len(edges), "n_kept": len(kept_edges),
              "rejected_bridges": sorted(int(e.get("bridge", -1)) for k, e in enumerate(edges)
                                         if k not in keep),
              "groups": rep_groups}
    log(f"[pose-graph] cycle consistency: {len(kept_edges)} of {len(edges)} closure(s) kept"
        + (f", bridge(s) {report['rejected_bridges']} left out" if report["rejected_bridges"] else ""))
    return kept_edges, report

