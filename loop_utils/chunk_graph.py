"""STAC (USER 2026-11-04, "ok, Sim3 … lo tenés que implementar correctamente"): the CHUNK-level
Sim(3) graph — every chunk one piece with its own rotation, translation and scale, tied by

* its SEAMS with the next chunk: the exact Sim(3) fit of the two copies of the shared frames (the
  same pixels in both chunks), σ = the fit's own median residual (how far the two copies still
  disagree), read at the surfaces' lever arm;
* the verified SALAD LOOP closures: the bridge's Sim(3) between the two chunks, σ = its split-half
  held-out residual (loop_bridges.verify_loop, WITHOUT the scale disagreement in quadrature — in an
  SE(3) graph a scale disagreement cannot be absorbed and widened σ; here scale is a variable of the
  graph and that disagreement is exactly what it corrects);
* DA3 per chunk as a WEAK scale prior: log s_k toward the chunk's own DA3 estimate, σ =
  sigma_anchor / √anchors (metric_lock.solve_scale_graph's weight) — so a walk without loops does
  not chain its scale freely.

The VGGT-SLAM structure (short pieces tied in a graph, loops applied), with Sim(3) instead of its
SL(4): SL(4) also shears and stretches each piece (on station each keyframe ended with its own
camera — focal ±9 %, principal point 40 px, skew 29 px) and a measured model cannot.

Conventions: a Sim(3) is (s, R, t) with x' = s·R·x + t; a chunk's pose C_k maps chunk-k
coordinates (metric-locked) to the world (chunk 0's frame); an edge (a, b, M) maps chunk-a
coordinates onto chunk-b coordinates (loop_bridges.compose_ab's convention; a seam k→k+1 is
(k+1, k, seam fit)). The residual of an edge is read at the place it was measured: E = (C_b⁻¹∘C_a)∘M⁻¹
is the identity when the edge agrees; its displacement of the edge's centroid c (chunk-b
coordinates, metres), its rotation and its log-scale times the edge's lever arm L (the surfaces'
distance to the cameras — the same reading as σ_rot = σ_t / range of the SE(3) graph), all over σ.
The gauge: chunk 0 whole — rotation, translation AND scale, at its metric lock, as the vendor's
alignment always had it (every chunk is placed in chunk 0's frame). Its DA3 prior is therefore not
a variable's (recorded, not used): a free chunk-0 scale would move the whole session's metre away
from DA3's, which scale_align then has to bring back and its verifier (|s − 1| ≤
scale.verify_max_dev) refuses beyond 10 %. Solved by scipy's trust-region least squares with a Cauchy loss on the
whitened residuals, float64, deterministic."""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Sim3 = Tuple[float, np.ndarray, np.ndarray]


def _rodrigues(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, np.float64)
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
        return np.eye(3) + K
    k = w / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _rotvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, np.float64)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = float(np.arccos(c))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-7:
        return 0.5 * v
    if np.pi - th < 1e-6:                        # near π: the axis from the symmetric part
        A = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(k))
        axis = A[:, i] / max(k[i], 1e-15)
        axis /= max(np.linalg.norm(axis), 1e-15)
        return th * axis
    return th / (2.0 * np.sin(th)) * v


def compose(A: Sim3, B: Sim3) -> Sim3:
    """A∘B (x ↦ A(B(x)))."""
    sa, Ra, ta = A
    sb, Rb, tb = B
    Ra, Rb = np.asarray(Ra, np.float64), np.asarray(Rb, np.float64)
    return (float(sa) * float(sb), Ra @ Rb,
            float(sa) * (Ra @ np.asarray(tb, np.float64)) + np.asarray(ta, np.float64))


def inverse(A: Sim3) -> Sim3:
    s, R, t = A
    R = np.asarray(R, np.float64)
    return (1.0 / float(s), R.T, -(R.T @ np.asarray(t, np.float64)) / float(s))


def identity() -> Sim3:
    return (1.0, np.eye(3), np.zeros(3))


def apply(A: Sim3, x: np.ndarray) -> np.ndarray:
    s, R, t = A
    return float(s) * (np.asarray(x, np.float64) @ np.asarray(R, np.float64).T) + np.asarray(t)


def edge_residual(Ca: Sim3, Cb: Sim3, edge: dict) -> np.ndarray:
    """The 7 whitened residuals of one edge at the chunk poses Ca, Cb (see the module doc)."""
    M = (float(edge["s"]), np.asarray(edge["R"], np.float64), np.asarray(edge["t"], np.float64))
    E = compose(compose(inverse(Cb), Ca), inverse(M))
    c = np.asarray(edge["centroid"], np.float64)
    d = apply(E, c[None])[0] - c
    L, sig = float(edge["lever_m"]), float(edge["sigma_m"])
    return np.concatenate([d, _rotvec(E[1]) * L, [np.log(float(E[0])) * L]]) / sig


def _unpack(x: np.ndarray, n: int) -> List[Sim3]:
    """x = [(ω_k, t_k, λ_k) for k = 1..n-1] → the chunk poses (chunk 0: the identity, the gauge)."""
    out = [identity()]
    for k in range(1, n):
        o = 7 * (k - 1)
        out.append((float(np.exp(x[o + 6])), _rodrigues(x[o:o + 3]), np.asarray(x[o + 3:o + 6])))
    return out


def _pack(poses: Sequence[Sim3]) -> np.ndarray:
    x = []
    for s, R, t in poses[1:]:
        x.extend(list(_rotvec(R)) + list(np.asarray(t, np.float64)) + [np.log(float(s))])
    return np.asarray(x, np.float64)


def chain_poses(n: int, seams: Sequence[dict]) -> List[Sim3]:
    """The chunk poses the seams alone give (C_0 = identity, C_{k+1} = C_k ∘ seam(k+1→k)) — the
    graph's starting point and the 'chain' every judge compares with."""
    by = {(int(e["a"]), int(e["b"])): e for e in seams}
    poses = [identity()]
    for k in range(n - 1):
        e = by.get((k + 1, k))
        if e is None:
            raise ValueError(f"chunk_graph: no seam between chunks {k} and {k + 1}")
        poses.append(compose(poses[k], (float(e["s"]), np.asarray(e["R"]), np.asarray(e["t"]))))
    return poses


def solve(n: int, seams: Sequence[dict], loops: Sequence[dict], priors: Dict[int, dict],
          x0: Optional[Sequence[Sim3]] = None, max_nfev: int = 2000) -> dict:
    """The chunk poses (relative to chunk 0) that best agree with ``seams``, ``loops`` and the
    scale ``priors`` ({k: {"log_s": target log scale, "sigma": σ log}}). Returns {"poses",
    "converged", "cost", "nfev", "residuals": per edge whitened norm}."""
    from scipy.optimize import least_squares
    edges = list(seams) + list(loops)
    start = chain_poses(n, seams) if x0 is None else list(x0)
    pk = sorted(int(k) for k in priors if 0 < int(k) < n)        # chunk 0 is the gauge

    def fun(x):
        P = _unpack(x, n)
        r = [edge_residual(P[int(e["a"])], P[int(e["b"])], e) for e in edges]
        for k in pk:
            pr = priors[k]
            r.append(np.array([(np.log(P[k][0]) - float(pr["log_s"])) / float(pr["sigma"])]))
        return np.concatenate(r) if r else np.zeros(1)

    if n == 1:
        P = [identity()]
        return {"poses": P, "converged": True, "cost": 0.0, "nfev": 0, "residuals": []}
    res = least_squares(fun, _pack(start), method="trf", loss="cauchy", f_scale=1.0,
                        x_scale="jac", max_nfev=int(max_nfev), xtol=1e-12, ftol=1e-12,
                        gtol=1e-12)
    P = _unpack(res.x, n)
    per_edge = [float(np.linalg.norm(edge_residual(P[int(e["a"])], P[int(e["b"])], e)))
                for e in edges]
    return {"poses": P, "converged": bool(res.status > 0), "cost": float(res.cost),
            "nfev": int(res.nfev), "status": int(res.status), "residuals": per_edge}


def edge_error_m(Ca: Sim3, Cb: Sim3, edge: dict) -> float:
    """The displacement (m) an edge's centroid suffers between what the edge measured and the
    poses — the judges' unit."""
    M = (float(edge["s"]), np.asarray(edge["R"], np.float64), np.asarray(edge["t"], np.float64))
    E = compose(compose(inverse(Cb), Ca), inverse(M))
    c = np.asarray(edge["centroid"], np.float64)
    return float(np.linalg.norm(apply(E, c[None])[0] - c))


def judge_leave_one_out(n: int, seams: Sequence[dict], loops: Sequence[dict],
                        priors: Dict[int, dict], *, error_factor: float, confidence: float,
                        log: Callable[[str], object] = print) -> dict:
    """THE JUDGE (USER 2026-10-07, the in-run pose graph's rule, here on the chunk graph): every
    loop closure judges once — the graph is solved without it, and its error at the CHAIN (the
    seams + priors alone, no loop) and at that solution are paired; metric_lock.decide_change
    decides (significant at ``confidence``, enough judges, the median improvement ≥
    ``error_factor`` × the MEDIAN σ of the closures). A solve that does not converge cannot
    testify (counted)."""
    from loop_utils.loop_judge import min_judge_closures
    from loop_utils.metric_lock import decide_change
    loops = list(loops)
    nmin = min_judge_closures(confidence)
    base = solve(n, seams, [], priors)
    out = {"n_closures": len(loops), "min_judges": nmin, "judges": [], "discarded": 0}
    if len(loops) < nmin:
        out.update({"improves": False, "reason": f"{len(loops)} closure(s), {nmin} needed to "
                    f"judge — declared before solving, not applied"})
        log(f"[chunk-graph] judge: {out['reason']}")
        return out
    before, after = [], []
    for q, e in enumerate(loops):
        rest = loops[:q] + loops[q + 1:]
        sol = solve(n, seams, rest, priors)
        b = edge_error_m(base["poses"][int(e["a"])], base["poses"][int(e["b"])], e)
        if not sol["converged"]:
            out["discarded"] += 1
            out["judges"].append({"bridge": e.get("bridge"), "chain_m": b, "converged": False})
            continue
        a = edge_error_m(sol["poses"][int(e["a"])], sol["poses"][int(e["b"])], e)
        before.append(b)
        after.append(a)
        out["judges"].append({"bridge": e.get("bridge"), "chunks": [int(e["a"]), int(e["b"])],
                              "chain_m": b, "without_it_m": a, "sigma_m": float(e["sigma_m"])})
        log(f"[chunk-graph] judge {q + 1}/{len(loops)} (bridge {e.get('bridge')}, chunks "
            f"{e['a']}<->{e['b']}, σ {float(e['sigma_m']) * 100:.1f} cm): left out of its solve "
            f"→ {b * 100:.2f} → {a * 100:.2f} cm")
    err = float(np.median([float(e["sigma_m"]) for e in loops]))
    if not before:
        out.update({"improves": False, "reason": "no closure could testify"})
        return out
    dec = decide_change(np.asarray(before), np.asarray(after), error=err,
                        error_factor=float(error_factor), confidence=float(confidence))
    out.update({"decision": dec, "improves": bool(dec["improves"]), "error_m": err,
                "reason": dec["reason"]})
    log(f"[chunk-graph] leave-one-out judge ({len(before)} of {len(loops)} closure(s) testify, "
        f"{out['discarded']} discarded): {dec['reason']}")
    return out


def sequential(poses: Sequence[Sim3]) -> List[Sim3]:
    """The poses as the fork's sim3_list: entry k maps chunk k+1 onto chunk k (C_k⁻¹ ∘ C_{k+1}),
    so sim3utils.accumulate_sim3_transforms gives back C_1 … C_{n-1} (C_0 = identity)."""
    return [compose(inverse(poses[k]), poses[k + 1]) for k in range(len(poses) - 1)]
