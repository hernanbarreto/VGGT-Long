"""claude_stac.txt §4.3 solver tests: Lie maps (round-trip, limits), autograd
Jacobians vs finite differences, exact closure of a noise-free synthetic graph,
gauge invariance, λ→∞ → identity, PCG path == dense path."""

import numpy as np
import pytest
import torch

from loop_utils import lie
from loop_utils.pose_graph import PoseGraph, _res_rel, _res_plane, _res_axis


def graph_cfg(**over):
    d = {"huber_delta_m": 0.10, "huber_delta_deg": 2.0, "dense_max_unknowns": 12000,
         "lambda_init": 1e-4, "lambda_max": 1e12, "lm_diag_floor": 1e-9,
         "tol": 1e-10, "rel_tol": 1e-12, "max_iters": 50,
         "pcg_tol": 1e-12, "pcg_max_iters": 2000}
    d.update(over)
    return d


# ── Lie maps ────────────────────────────────────────────────────────────────

def test_so3_se3_round_trip_and_limits():
    rng = np.random.default_rng(0)
    for scale in (1e-9, 1e-7, 1e-4, 0.1, 1.0, 3.0):
        w = rng.standard_normal(3)
        w = w / np.linalg.norm(w) * scale
        R = lie.so3_exp(w)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.allclose(lie.so3_log(R), w, atol=1e-8)
        xi = np.concatenate([w, rng.standard_normal(3)])
        T = lie.se3_exp(xi)
        assert np.allclose(lie.se3_log(T), xi, atol=1e-8)
        assert np.allclose(lie.se3_exp(lie.se3_log(T)), T, atol=1e-10)
    # near π
    w = np.array([0.0, np.pi - 1e-6, 0.0])
    assert abs(np.linalg.norm(lie.so3_log(lie.so3_exp(w))) - np.linalg.norm(w)) < 1e-5
    # identity exactly
    assert np.allclose(lie.so3_log(np.eye(3)), 0.0)
    assert np.allclose(lie.se3_exp(np.zeros(6)), np.eye(4))
    # torch == numpy
    xi = torch.tensor([0.3, -0.2, 0.1, 1.0, -2.0, 0.5], dtype=torch.float64)
    assert np.allclose(lie.t_se3_exp(xi).numpy(), lie.se3_exp(xi.numpy()), atol=1e-12)
    assert np.allclose(lie.t_se3_log(lie.t_se3_exp(xi)).numpy(), xi.numpy(), atol=1e-9)
    tiny = torch.tensor([1e-9, 0.0, 0.0, 0.1, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    lie.t_se3_exp(tiny).sum().backward()
    assert torch.isfinite(tiny.grad).all()


# ── Jacobians vs finite differences ────────────────────────────────────────

def _fd(f, x, eps=1e-6):
    x = x.clone()
    J = []
    for k in range(x.numel()):
        xp = x.clone(); xp[k] += eps
        xm = x.clone(); xm[k] -= eps
        J.append((f(xp) - f(xm)) / (2 * eps))
    return torch.stack(J, dim=1)


def test_autograd_jacobians_match_finite_differences():
    rng = np.random.default_rng(1)
    Ti = torch.as_tensor(lie.se3_exp(rng.standard_normal(6)), dtype=torch.float64)
    Tj = torch.as_tensor(lie.se3_exp(rng.standard_normal(6)), dtype=torch.float64)
    Z = lie.se3_inv(Ti.numpy()) @ Tj.numpy() @ lie.se3_exp(0.05 * rng.standard_normal(6))
    Zinv = torch.as_tensor(lie.se3_inv(Z).reshape(-1), dtype=torch.float64)
    d0 = torch.zeros(6, dtype=torch.float64)
    for kern, params in ((_res_rel, Zinv),
                         (_res_plane, torch.tensor([0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.5],
                                                   dtype=torch.float64)),
                         (_res_axis, torch.tensor([0.0, -1.0, 0.0, 0.0, 1.0, 0.0],
                                                  dtype=torch.float64))):
        Ja, Jb = torch.func.jacrev(kern, argnums=(0, 1))(d0, d0, Ti, Tj, params)
        Fa = _fd(lambda d: kern(d, d0, Ti, Tj, params), d0)
        Fb = _fd(lambda d: kern(d0, d, Ti, Tj, params), d0)
        assert torch.allclose(Ja, Fa, atol=1e-6), kern.__name__
        assert torch.allclose(Jb, Fb, atol=1e-6), kern.__name__


# ── synthetic graph ─────────────────────────────────────────────────────────

def _ring(n=40, radius=5.0, seed=2):
    """Ground-truth poses on a circle (the walk returns to the start)."""
    rng = np.random.default_rng(seed)
    T = []
    for k in range(n):
        a = 2 * np.pi * k / n
        R = lie.so3_exp(np.array([0.0, a, 0.0]))
        t = np.array([radius * np.cos(a), 0.1 * rng.standard_normal(), radius * np.sin(a)])
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
        T.append(M)
    return np.stack(T)


def _drifted(T_gt, yaw_per_step_deg=0.3, t_per_step=0.02):
    """Odometry chain integrating a small constant bias: the classic drift."""
    n = len(T_gt)
    T = [T_gt[0].copy()]
    bias = lie.se3_exp(np.array([0.0, np.radians(yaw_per_step_deg), 0.0, t_per_step, 0.0, 0.0]))
    for k in range(1, n):
        Z = lie.se3_inv(T_gt[k - 1]) @ T_gt[k]
        T.append(T[-1] @ Z @ bias)
    return np.stack(T)


def test_exact_closure_noise_free():
    T_gt = _ring()
    T_init = _drifted(T_gt)
    n = len(T_gt)
    pg = PoseGraph(T_init, graph_cfg())
    for k in range(n - 1):
        pg.add_relative(k, k + 1, lie.se3_inv(T_gt[k]) @ T_gt[k + 1], 0.5, 0.02, tag="odo")
    # three exact loops
    for i, j in ((0, n - 1), (5, n // 2 + 5), (10, n - 5)):
        pg.add_relative(i, j, lie.se3_inv(T_gt[i]) @ T_gt[j], 0.5, 0.02, huber=True, tag="loop")
    rep = pg.solve(log=lambda m: None)
    err = [np.linalg.norm(lie.se3_log(lie.se3_inv(T_gt[k]) @ pg.poses[k])) for k in range(n)]
    assert max(err) < 1e-6, max(err)
    assert rep["cost_final"] < 1e-10


def test_gauge_invariance():
    T_gt = _ring(n=25)
    T_init = _drifted(T_gt)
    n = len(T_gt)

    def solve(T0):
        pg = PoseGraph(T0, graph_cfg())
        for k in range(n - 1):
            pg.add_relative(k, k + 1, lie.se3_inv(T_gt[k]) @ T_gt[k + 1], 0.5, 0.02)
        pg.add_relative(0, n - 1, lie.se3_inv(T_gt[0]) @ T_gt[n - 1], 0.5, 0.02, huber=True, tag="loop")
        pg.solve(log=lambda m: None)
        return pg.poses

    G = lie.se3_exp(np.array([0.0, 0.7, 0.0, 3.0, 0.0, -1.0]))     # a yaw + translation
    A = solve(T_init)
    B = solve(np.stack([G @ M for M in T_init]))
    for k in range(n):
        assert np.allclose(B[k], G @ A[k], atol=1e-6)


def test_lambda_to_infinity_keeps_identity():
    T_gt = _ring(n=20)
    T_init = _drifted(T_gt)
    n = len(T_gt)
    pg = PoseGraph(T_init, graph_cfg(lambda_init=1e12, lambda_max=1e13, max_iters=3))
    for k in range(n - 1):
        pg.add_relative(k, k + 1, lie.se3_inv(T_gt[k]) @ T_gt[k + 1], 0.5, 0.02)
    pg.add_relative(0, n - 1, lie.se3_inv(T_gt[0]) @ T_gt[n - 1], 0.5, 0.02, huber=True, tag="loop")
    pg.solve(log=lambda m: None)
    X = pg.corrections()
    assert max(np.linalg.norm(lie.se3_log(M)) for M in X) < 1e-6


def test_pcg_matches_dense():
    T_gt = _ring(n=30)
    T_init = _drifted(T_gt)
    n = len(T_gt)

    def run(cfg):
        pg = PoseGraph(T_init, cfg)
        for k in range(n - 1):
            pg.add_relative(k, k + 1, lie.se3_inv(T_gt[k]) @ T_gt[k + 1], 0.5, 0.02)
        pg.add_relative(0, n - 1, lie.se3_inv(T_gt[0]) @ T_gt[n - 1], 0.5, 0.02, huber=True, tag="loop")
        pg.solve(log=lambda m: None)
        return pg.poses

    A = run(graph_cfg())
    B = run(graph_cfg(dense_max_unknowns=12))          # forces PCG
    assert np.allclose(A, B, atol=1e-6)


def test_huber_bounds_a_liar_and_the_veto_removes_it():
    """Huber cannot make a contradictory loop harmless — its tail is linear,
    so a liar keeps a CONSTANT pull — but it bounds the damage against the
    quadratic case; the authority veto (§4.7: an edge demanding more than the
    drift budget is deactivated and the graph re-solved) removes it."""
    T_gt = _ring(n=30)
    T_init = _drifted(T_gt)
    n = len(T_gt)

    def build(huber):
        pg = PoseGraph(T_init, graph_cfg())
        for k in range(n - 1):
            pg.add_relative(k, k + 1, lie.se3_inv(T_gt[k]) @ T_gt[k + 1], 0.5, 0.02)
        pg.add_relative(0, n - 1, lie.se3_inv(T_gt[0]) @ T_gt[n - 1], 0.5, 0.02, huber=huber, tag="loop")
        bad = lie.se3_inv(T_gt[3]) @ T_gt[20] @ lie.se3_exp(np.array([0, 0.5, 0, 6.0, 0, 0]))
        e_bad = pg.add_relative(3, 20, bad, 0.5, 0.02, huber=huber, tag="loop_bad")
        return pg, e_bad

    def max_err(pg):
        return max(np.linalg.norm(lie.se3_log(lie.se3_inv(T_gt[k]) @ pg.poses[k])[3:])
                   for k in range(n))

    pg_q, _ = build(huber=False)
    pg_q.solve(log=lambda m: None)
    pg_h, e_bad = build(huber=True)
    pg_h.solve(log=lambda m: None)
    res = pg_h.edge_residuals("loop")
    assert res[e_bad]["t_m"] > 3.0                         # the liar stays a liar
    assert max_err(pg_h) < 0.5 * max_err(pg_q)             # and drags far less than quadratic
    # authority veto: the liar demands more than any drift budget → out, re-solve
    pg_h.deactivate(e_bad)
    pg_h.solve(log=lambda m: None)
    assert max_err(pg_h) < 1e-6


def test_structural_priors_pull_a_tilted_frame():
    T_gt = _ring(n=10)
    T_init = T_gt.copy()
    tilt = lie.se3_exp(np.array([np.radians(4.0), 0.0, 0.0, 0.0, 0.0, 0.0]))
    T_init[4] = tilt @ T_gt[4]
    pg = PoseGraph(T_init, graph_cfg())
    for k in range(9):
        pg.add_relative(k, k + 1, lie.se3_inv(T_init[k]) @ T_init[k + 1], 5.0, 0.5)   # weak odometry
    # a floor patch seen by frame 4: local plane of the world floor y=0 under the GT pose
    Rg, tg = T_gt[4][:3, :3], T_gt[4][:3, 3]
    n_l = Rg.T @ np.array([0.0, 1.0, 0.0])
    d_l = float(n_l @ (Rg.T @ (np.array([0.0, 0.0, 0.0]) - tg)))
    pg.add_plane_datum(4, n_l, d_l, np.array([0.0, 1.0, 0.0]), 0.0, 0.2, 0.01)
    pg.solve(log=lambda m: None)
    tilt_after = np.degrees(np.linalg.norm(lie.so3_log(T_gt[4][:3, :3].T @ pg.poses[4][:3, :3])))
    assert tilt_after < 0.5


def test_plane_node_is_solved_jointly_with_the_poses():
    """A wall seen from every frame of a drifted chain: the plane node and the
    poses are solved together; the plane converges to the true wall and the
    drift components the wall OBSERVES — translation along its normal,
    rotation of the normal — fall out. The chain drifts along x and yaws
    about y; the wall x = 8 (normal +x) observes both.

    The plane's absolute pose is pinned only by the drift-free start (node 0
    is the gauge): the least-squares compromise rotates the plane by
    B·σ_a²/(σ_a² + (n−1)·σ_r²) for a total odometric yaw bias B — the wall
    observations (a normal from thousands of points) must be the tighter
    evidence, σ_a ≪ √(n−1)·σ_r, for the truth to be the optimum."""
    T_gt = _ring(n=30)
    T_init = _drifted(T_gt)
    n = len(T_gt)
    pg = PoseGraph(T_init, graph_cfg())
    for k in range(n - 1):
        pg.add_relative(k, k + 1, lie.se3_inv(T_init[k]) @ T_init[k + 1], 0.5, 0.05)  # drifted odometry
    n_true = np.array([1.0, 0.0, 0.0]); d_true = 8.0
    # the plane node starts at a WRONG guess (yawed 3°, 30 cm off)
    n0 = lie.so3_exp(np.array([0.0, np.radians(3.0), 0.0])) @ n_true
    node = pg.add_plane_node(n0, n0 * (d_true + 0.3))
    for k in range(n):
        R, t = T_gt[k][:3, :3], T_gt[k][:3, 3]
        n_l = R.T @ n_true
        d_l = float(d_true - n_true @ t)
        pg.add_plane_edge(k, node, n_l, d_l, 0.1, 0.01, huber=True, tag="wall")
    rep = pg.solve(log=lambda m: None)
    assert rep["stop"] in ("rel_tol", "step_below_tol"), rep["stop"]
    n_s, d_s = pg.plane_of(node)
    assert abs(float(n_s @ n_true)) > np.cos(np.radians(0.05)), n_s
    assert abs(d_s - d_true) < 0.01, d_s
    # the observable components: distance to the wall and yaw of every frame
    dist_err = [abs(float(n_true @ pg.poses[k][:3, 3]) - float(n_true @ T_gt[k][:3, 3])) for k in range(n)]
    yaw_err = [np.degrees(np.linalg.norm(lie.so3_log(T_gt[k][:3, :3].T @ pg.poses[k][:3, :3])))
               for k in range(n)]
    assert max(dist_err) < 0.02, max(dist_err)
    assert max(yaw_err) < 0.1, max(yaw_err)
