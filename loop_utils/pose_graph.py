# STAC patch — KEYFRAME SE(3) pose graph (claude_stac.txt §4.3, fixed spec).
#
#   nodes      every keyframe, state T_i ∈ SE(3) stored as a 4x4 (R_i, t_i), c2w
#   update     LEFT perturbation in the algebra: T_i ← Exp(δ_i)·T_i, δ_i ∈ ℝ⁶ (ω, ν)
#   edges      relative (odometry / loop): measurement Z_ij, residual
#              r_ij = Log(Z_ij⁻¹ · T_i⁻¹ · T_j) ∈ ℝ⁶, weighted by Σ_ij⁻¹ (σ_rot, σ_t)
#              unary structural priors: gravity (camera down vs world down),
#              plane datum (a local plane must land on a FIXED target plane),
#              axis vertical (a local axis must be vertical);
#              PLANE NODES: a plane (wall, floor datum) can be a node itself — a
#              frame whose z axis is the normal — and every keyframe patch of
#              it is a binary edge pose→plane, so the plane is solved JOINTLY
#              with the poses (an alternation has a continuum of fixed points)
#   robust     Huber on loop and structural edges (never on odometry)
#   solver     Levenberg-Marquardt, normal equations (JᵀWJ + λ·diag)·δ = −JᵀWr;
#              Jacobians by torch autograd (torch.func.jacrev, vmapped per edge
#              type) on the perturbations at the current state — no analytic
#              Jacobians by hand (a test checks them against finite differences);
#              dense torch.linalg.cholesky/cholesky_solve when 6·n_kf ≤
#              dense_max_unknowns, block-Jacobi preconditioned conjugate gradient
#              beyond; λ adaptive (Nielsen); gauge δ_0 ≡ 0; float64 throughout
#   stop       ‖δ‖ < tol, relative cost improvement < rel_tol, or max_iters
#
# Only torch + torch.linalg — no g2o / GTSAM / Ceres. Every threshold comes from
# the caller's config dict (Model.graph.*); the module holds no decision literal.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch

from loop_utils.lie import (t_se3_exp, t_se3_log, t_se3_inv, se3_log, se3_exp,
                            se3_inv)

_REL = 0      # relative SE(3) edge (odometry / loop)
_GRAV = 1     # gravity prior (unary)
_PLANE = 2    # plane datum (unary, fixed target plane)
_AXIS = 3     # axis vertical (unary)
_PLANE_NODE = 4   # pose i's local plane must lie on the plane carried by NODE j (binary)
_BINARY = (_REL, _PLANE_NODE)


def _req(cfg: dict, key: str):
    if cfg is None or key not in cfg:
        raise RuntimeError(f"config key Model.graph.{key} is missing — the pose graph "
                           f"declares every threshold in config.yaml correction_graph.graph")
    return cfg[key]


# ── residual kernels (torch, one edge; vmapped by the solver) ──────────────

def _res_rel(di, dj, Ti, Tj, Zinv):
    # Zinv arrives flattened (16,) — edge params are one row per edge for vmap
    Ti2 = t_se3_exp(di) @ Ti
    Tj2 = t_se3_exp(dj) @ Tj
    return t_se3_log(Zinv.reshape(4, 4) @ t_se3_inv(Ti2) @ Tj2)


def _res_grav(di, dj, Ti, Tj, down):
    R = (t_se3_exp(di) @ Ti)[:3, :3]
    return R[:, 1] - down                  # OpenCV camera +Y is "down"


def _res_plane(di, dj, Ti, Tj, params):
    # params = [n_local(3), d_local, n_target(3), d_target] : n·p = d
    T = t_se3_exp(di) @ Ti
    R, t = T[:3, :3], T[:3, 3]
    n_w = R @ params[:3]
    d_w = params[3] + n_w @ t
    return torch.cat([n_w - params[4:7], (d_w - params[7]).reshape(1)])


def _res_axis(di, dj, Ti, Tj, params):
    # params = [axis_local(3), up_world(3)]
    R = (t_se3_exp(di) @ Ti)[:3, :3]
    a = R @ params[:3]
    return a - params[3:6]


def _res_plane_node(di, dj, Ti, Tj, params):
    # params = [n_local(3), d_local]; node j is a PLANE FRAME: its z axis is the
    # plane normal and its origin lies on the plane — the plane is solved
    # JOINTLY with the poses (an alternation plane↔poses has a continuum of
    # fixed points: whatever plane the poses were pulled to refits itself).
    T = t_se3_exp(di) @ Ti
    R, t = T[:3, :3], T[:3, 3]
    n_w = R @ params[:3]
    d_w = params[3] + n_w @ t
    P = t_se3_exp(dj) @ Tj
    n_p = P[:3, 2]
    d_p = n_p @ P[:3, 3]
    return torch.cat([n_w - n_p, (d_w - d_p).reshape(1)])


_KERNELS = {_REL: _res_rel, _GRAV: _res_grav, _PLANE: _res_plane, _AXIS: _res_axis,
            _PLANE_NODE: _res_plane_node}
_RES_DIM = {_REL: 6, _GRAV: 3, _PLANE: 4, _AXIS: 3, _PLANE_NODE: 4}
# residual layout per kind: (n angular components, n metric components) — the
# Huber δ of an angular component is huber_delta_deg, of a metric one huber_delta_m
_HUBER_LAYOUT = {_REL: (3, 3), _GRAV: (3, 0), _PLANE: (3, 1), _AXIS: (3, 0), _PLANE_NODE: (3, 1)}


class PoseGraph:
    """Build → solve → read `poses` (N,4,4) and `report`."""

    def __init__(self, poses: np.ndarray, cfg: dict, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        P = np.asarray(poses, np.float64)
        if P.ndim != 3 or P.shape[1:] != (4, 4):
            raise ValueError("poses must be (N,4,4) c2w")
        self.n = int(P.shape[0])
        self.T0 = torch.as_tensor(P, dtype=torch.float64, device=self.device)
        self.T = self.T0.clone()
        self._edges: List[dict] = []
        self.report: dict = {}

    # ── graph construction ───────────────────────────────────────────────
    def _add(self, kind, i, j, params, weight, huber, tag, huber_delta_m=None,
             huber_delta_deg=None, sqrt_info=None):
        # per-edge Huber widths override Model.graph.huber_delta_* (a wall's
        # tolerance is the wall's, not the loop's); NaN = the global value.
        # ``sqrt_info`` (m×m, L with L·Lᵀ = information) replaces the diagonal
        # weights when a measurement observes some DOF and not others (a
        # floor patch observes only its normal; a column only the plane
        # across its axis): the weighted residual is L·r, its covariance I.
        w = np.asarray(weight, np.float64).reshape(-1)
        self._edges.append({"kind": kind, "i": int(i), "j": int(j),
                            "params": np.asarray(params, np.float64).reshape(-1),
                            "w": w,
                            "W": (np.asarray(sqrt_info, np.float64) if sqrt_info is not None
                                  else np.diag(w)),
                            "huber": bool(huber), "tag": tag, "active": True,
                            "hd_m": float("nan") if huber_delta_m is None else float(huber_delta_m),
                            "hd_rad": (float("nan") if huber_delta_deg is None
                                       else math.radians(float(huber_delta_deg)))})
        return len(self._edges) - 1

    def add_relative(self, i, j, Z, sigma_rot_deg, sigma_t_m, huber=False, tag="odo",
                     huber_delta_m=None, huber_delta_deg=None, info_t=None, info_rot=None):
        """Measurement Z = T_i⁻¹ T_j (4x4). Returns the edge id. ``info_t`` /
        ``info_rot``: optional 3×3 information matrices of the translation /
        rotation residual (expressed in node i's frame, the residual's frame)
        for measurements that observe only some directions; the scalar σ's
        stay the diagonal reference for the Huber thresholds."""
        Zinv = se3_inv(np.asarray(Z, np.float64)).reshape(-1)
        w = np.concatenate([np.full(3, 1.0 / math.radians(max(float(sigma_rot_deg), 1e-9))),
                            np.full(3, 1.0 / max(float(sigma_t_m), 1e-9))])
        W = None
        if info_t is not None or info_rot is not None:
            W = np.zeros((6, 6))
            W[:3, :3] = (np.linalg.cholesky(np.asarray(info_rot, np.float64)) if info_rot is not None
                         else np.diag(w[:3]))
            W[3:, 3:] = (np.linalg.cholesky(np.asarray(info_t, np.float64)) if info_t is not None
                         else np.diag(w[3:]))
            # the diagonal reference of a general L: the per-axis σ it implies
            w = np.sqrt(np.clip(np.diag(W @ W.T), 1e-18, None))
        return self._add(_REL, i, j, Zinv, w, huber, tag, huber_delta_m, huber_delta_deg, W)

    def add_gravity(self, i, down_world, sigma_deg, tag="gravity"):
        d = np.asarray(down_world, np.float64)
        d = d / (np.linalg.norm(d) + 1e-12)
        w = np.full(3, 1.0 / max(math.sin(math.radians(float(sigma_deg))), 1e-9))
        return self._add(_GRAV, i, i, d, w, False, tag)

    def add_plane_datum(self, i, n_local, d_local, n_target, d_target,
                        sigma_angle_deg, sigma_offset_m, huber=True, tag="plane",
                        huber_delta_m=None, huber_delta_deg=None):
        """Local plane n_local·p_c = d_local (camera coords of node i) must land
        on the world plane n_target·p_w = d_target."""
        nl = np.asarray(n_local, np.float64); nl = nl / (np.linalg.norm(nl) + 1e-12)
        nt = np.asarray(n_target, np.float64); nt = nt / (np.linalg.norm(nt) + 1e-12)
        params = np.concatenate([nl, [float(d_local)], nt, [float(d_target)]])
        w = np.concatenate([np.full(3, 1.0 / max(math.sin(math.radians(float(sigma_angle_deg))), 1e-9)),
                            [1.0 / max(float(sigma_offset_m), 1e-9)]])
        return self._add(_PLANE, i, i, params, w, huber, tag, huber_delta_m, huber_delta_deg)

    def add_axis_vertical(self, i, axis_local, up_world, sigma_deg, huber=True, tag="axis",
                          huber_delta_deg=None):
        al = np.asarray(axis_local, np.float64); al = al / (np.linalg.norm(al) + 1e-12)
        up = np.asarray(up_world, np.float64); up = up / (np.linalg.norm(up) + 1e-12)
        w = np.full(3, 1.0 / max(math.sin(math.radians(float(sigma_deg))), 1e-9))
        return self._add(_AXIS, i, i, np.concatenate([al, up]), w, huber, tag, None, huber_delta_deg)

    def add_plane_node(self, normal, point_on_plane) -> int:
        """Append a PLANE node (a frame whose z axis is the normal and whose
        origin is on the plane). Returns its node id (≥ the number of poses).
        Its in-plane translation and the rotation about its normal are
        unobservable by construction — LM damping (lm_diag_floor) leaves
        them where they start."""
        n = np.asarray(normal, np.float64); n = n / (np.linalg.norm(n) + 1e-12)
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(a, n); x = x / (np.linalg.norm(x) + 1e-12)
        y = np.cross(n, x)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2] = x, y, n
        T[:3, 3] = np.asarray(point_on_plane, np.float64)
        T0 = torch.cat([self.T0, torch.as_tensor(T[None], dtype=torch.float64, device=self.device)])
        self.T0 = T0
        self.T = T0.clone()
        self.n = int(T0.shape[0])
        return self.n - 1

    def add_plane_edge(self, i, plane_node, n_local, d_local, sigma_angle_deg, sigma_offset_m,
                       huber=True, tag="wall", huber_delta_m=None, huber_delta_deg=None):
        """Local plane n_local·p_c = d_local of pose node i must lie on the
        plane carried by ``plane_node`` (joint plane estimation)."""
        nl = np.asarray(n_local, np.float64); nl = nl / (np.linalg.norm(nl) + 1e-12)
        params = np.concatenate([nl, [float(d_local)]])
        w = np.concatenate([np.full(3, 1.0 / max(math.sin(math.radians(float(sigma_angle_deg))), 1e-9)),
                            [1.0 / max(float(sigma_offset_m), 1e-9)]])
        return self._add(_PLANE_NODE, i, plane_node, params, w, huber, tag, huber_delta_m,
                         huber_delta_deg)

    def plane_of(self, plane_node: int):
        """(normal, offset d) of a plane node at the current state."""
        P = self.T[int(plane_node)].cpu().numpy()
        n = P[:3, 2]
        return n / (np.linalg.norm(n) + 1e-12), float(n @ P[:3, 3])

    def deactivate(self, edge_id: int):
        self._edges[int(edge_id)]["active"] = False

    @property
    def n_edges(self):
        return sum(1 for e in self._edges if e["active"])

    # ── residuals / jacobians ─────────────────────────────────────────────
    def _groups(self):
        groups: Dict[int, dict] = {}
        for eid, e in enumerate(self._edges):
            if not e["active"]:
                continue
            g = groups.setdefault(e["kind"], {"eid": [], "i": [], "j": [], "params": [],
                                              "w": [], "huber": [], "hd": []})
            g["eid"].append(eid); g["i"].append(e["i"]); g["j"].append(e["j"])
            g["params"].append(e["params"]); g["w"].append(e["w"]); g["huber"].append(e["huber"])
            g["hd"].append((e["hd_rad"], e["hd_m"]))
            g.setdefault("W", []).append(e["W"])
        out = {}
        for k, g in groups.items():
            out[k] = {"eid": np.asarray(g["eid"]),
                      "i": torch.as_tensor(np.asarray(g["i"]), device=self.device),
                      "j": torch.as_tensor(np.asarray(g["j"]), device=self.device),
                      "params": torch.as_tensor(np.stack(g["params"]), dtype=torch.float64,
                                                device=self.device),
                      "w": torch.as_tensor(np.stack(g["w"]), dtype=torch.float64,
                                           device=self.device),
                      "huber": torch.as_tensor(np.asarray(g["huber"]), device=self.device),
                      "hd": torch.as_tensor(np.asarray(g["hd"], np.float64), dtype=torch.float64,
                                            device=self.device),
                      "W": torch.as_tensor(np.stack(g["W"]), dtype=torch.float64, device=self.device)}
        return out

    def _huber_thresholds(self, kind, g):
        """(n_e, m) Huber δ per weighted-residual component: per-edge widths
        where given, else Model.graph.huber_delta_deg / huber_delta_m."""
        hd_rad = math.radians(float(_req(self.cfg, "huber_delta_deg")))
        hd_m = float(_req(self.cfg, "huber_delta_m"))
        per = g["hd"]
        a = torch.where(torch.isnan(per[:, 0]), torch.full_like(per[:, 0], hd_rad), per[:, 0])
        m = torch.where(torch.isnan(per[:, 1]), torch.full_like(per[:, 1], hd_m), per[:, 1])
        n_a, n_m = _HUBER_LAYOUT[kind]
        cols = [a[:, None].expand(-1, n_a)]
        if n_m:
            cols.append(m[:, None].expand(-1, n_m))
        return torch.cat(cols, dim=1) * g["w"]

    def _residuals(self, T, groups, with_jac):
        """Per group: weighted residuals (n_e, m), Huber IRLS weights (n_e,),
        and, if with_jac, weighted Jacobians (n_e, m, 12) w.r.t. (δ_i, δ_j)."""
        out = {}
        for kind, g in groups.items():
            kern = _KERNELS[kind]
            Ti, Tj = T[g["i"]], T[g["j"]]
            zero = torch.zeros(6, dtype=torch.float64, device=self.device)

            def f(di, dj, ti, tj, p):
                return kern(di, dj, ti, tj, p)

            r = torch.vmap(f)(zero.expand(len(Ti), 6), zero.expand(len(Ti), 6), Ti, Tj, g["params"])
            rw = torch.einsum("eab,eb->ea", g["W"], r)      # L·r (diagonal L = the σ weights)
            # Huber (IRLS): weight min(1, δ/‖r‖) on the WEIGHTED residual, with δ
            # expressed in σ units per component — angular components
            # (rotation, normal / axis differences ≈ radians) take
            # huber_delta_deg, metric ones (translation, plane offset) huber_delta_m
            thr = self._huber_thresholds(kind, g)
            nrm = torch.linalg.norm(rw, dim=1)
            # δ in σ units for the whole weighted residual: the tightest
            # component threshold (a 6-vector at δ per component has norm
            # √6·δ — taking the vector norm as the threshold would let a loop
            # lie 2.4× farther before it stops pulling quadratically)
            thr_n = thr.min(dim=1).values
            hub = torch.where(g["huber"] & (nrm > thr_n), thr_n / nrm.clamp(min=1e-18),
                              torch.ones_like(nrm))
            rec = {"r": rw, "hub": hub, "raw": r}
            if with_jac:
                jac = torch.vmap(torch.func.jacrev(f, argnums=(0, 1)))(
                    zero.expand(len(Ti), 6), zero.expand(len(Ti), 6), Ti, Tj, g["params"])
                J = torch.einsum("eab,ebc->eac", g["W"], torch.cat([jac[0], jac[1]], dim=2))  # (n_e, m, 12)
                rec["J"] = J
            out[kind] = rec
        return out

    def _cost(self, res):
        c = 0.0
        for kind, rec in res.items():
            nrm = torch.linalg.norm(rec["r"], dim=1)
            hub = rec["hub"]
            # Huber cost: ½‖r‖² inside, δ(‖r‖ − δ/2) outside  (δ = ‖r‖·hub)
            d = nrm * hub
            inside = hub >= 1.0
            c += float(torch.where(inside, 0.5 * nrm ** 2, d * (nrm - 0.5 * d)).sum())
        return c

    # ── normal equations ─────────────────────────────────────────────────
    def _assemble(self, groups, res):
        """H (6n×6n dense) and g (6n) of the Gauss-Newton system; node 0 rows
        and columns are removed by the caller (gauge)."""
        n6 = 6 * self.n
        H = torch.zeros((n6, n6), dtype=torch.float64, device=self.device)
        gvec = torch.zeros(n6, dtype=torch.float64, device=self.device)
        blocks = []
        for kind, g in groups.items():
            rec = res[kind]
            J, r, hub = rec["J"], rec["r"], rec["hub"]
            # IRLS for Huber: the edge enters the normal equations with weight
            # w = δ/‖r‖ (linear) — H += w·JᵀJ, g += w·Jᵀr — i.e. √w on both
            # (squaring the weight would turn Huber into a non-convex loss with
            # spurious minima that let a contradictory loop win)
            sq = torch.sqrt(hub)
            Jw = J * sq[:, None, None]
            rw = r * sq[:, None]
            Ji, Jj = Jw[:, :, :6], Jw[:, :, 6:]
            Hii = torch.einsum("emi,emj->eij", Ji, Ji)
            Hjj = torch.einsum("emi,emj->eij", Jj, Jj)
            Hij = torch.einsum("emi,emj->eij", Ji, Jj)
            gi = torch.einsum("emi,em->ei", Ji, rw)
            gj = torch.einsum("emi,em->ei", Jj, rw)
            ii = g["i"].cpu().numpy(); jj = g["j"].cpu().numpy()
            blocks.append((ii, jj, Hii, Hjj, Hij, gi, gj))
            for e in range(len(ii)):
                a, b = 6 * int(ii[e]), 6 * int(jj[e])
                H[a:a + 6, a:a + 6] += Hii[e]
                gvec[a:a + 6] += gi[e]
                if kind in _BINARY:
                    H[b:b + 6, b:b + 6] += Hjj[e]
                    H[a:a + 6, b:b + 6] += Hij[e]
                    H[b:b + 6, a:a + 6] += Hij[e].T
                    gvec[b:b + 6] += gj[e]
        return H, gvec, blocks

    def _solve_linear(self, H, gvec, lam):
        """δ (6n) with δ_0 ≡ 0: dense Cholesky on the reduced system, or PCG
        with a block-Jacobi preconditioner for large graphs."""
        n6 = 6 * self.n
        free = torch.arange(6, n6, device=self.device)
        A = H[free][:, free]
        b = -gvec[free]
        # LM damping on the observable freedoms; an ABSOLUTE Tikhonov floor
        # (lm_diag_floor × the largest diagonal entry, independent of λ) on the
        # unobservable ones — a plane node's in-plane translation and its
        # rotation about the normal have an exactly-zero diagonal, and a
        # floor that shrank with λ left them at 1e-13 against 1e4 entries:
        # Cholesky reported "not positive-definite" (certify smoke, 90 kf)
        dA = torch.diagonal(A).clamp(min=0)
        floor = float(_req(self.cfg, "lm_diag_floor")) * float(dA.max().clamp(min=1.0))
        A = A + lam * torch.diag(dA) + floor * torch.eye(len(free), dtype=A.dtype,
                                                          device=self.device)
        d = torch.zeros(n6, dtype=torch.float64, device=self.device)
        if len(free) <= int(_req(self.cfg, "dense_max_unknowns")):
            L = torch.linalg.cholesky(A)
            x = torch.cholesky_solve(b[:, None], L)[:, 0]
        else:
            x = self._pcg(A, b)
        d[free] = x
        return d

    def _pcg(self, A, b):
        """Preconditioned conjugate gradient, block-Jacobi (6x6 diagonal
        blocks) preconditioner. A is symmetric positive definite."""
        n = len(b)
        nb = n // 6
        blocks = torch.stack([A[6 * k:6 * k + 6, 6 * k:6 * k + 6] for k in range(nb)])
        Minv = torch.linalg.inv(blocks)

        def prec(v):
            return torch.einsum("kij,kj->ki", Minv, v.view(nb, 6)).reshape(-1)

        x = torch.zeros_like(b)
        r = b - A @ x
        z = prec(r)
        p = z.clone()
        rz = float(r @ z)
        tol = float(_req(self.cfg, "pcg_tol")) * float(torch.linalg.norm(b)) + 1e-30
        for _ in range(int(_req(self.cfg, "pcg_max_iters"))):
            Ap = A @ p
            alpha = rz / max(float(p @ Ap), 1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            if float(torch.linalg.norm(r)) < tol:
                break
            z = prec(r)
            rz_new = float(r @ z)
            p = z + (rz_new / max(rz, 1e-30)) * p
            rz = rz_new
        return x

    @staticmethod
    def _apply_delta(T, d):
        n = T.shape[0]
        dd = d.view(n, 6)
        return torch.stack([t_se3_exp(dd[k]) @ T[k] for k in range(n)])

    # ── the LM loop ──────────────────────────────────────────────────────
    def solve(self, log=print) -> dict:
        groups = self._groups()
        if not groups:
            raise RuntimeError("pose graph has no edges")
        lam = float(_req(self.cfg, "lambda_init"))
        nu = 2.0
        tol = float(_req(self.cfg, "tol"))
        rel_tol = float(_req(self.cfg, "rel_tol"))
        max_iters = int(_req(self.cfg, "max_iters"))
        T = self.T.clone()
        res = self._residuals(T, groups, with_jac=True)
        cost = self._cost(res)
        cost0 = cost
        history = [cost]
        stop = "max_iters"
        n_accept = 0
        for it in range(max_iters):
            H, gvec, _ = self._assemble(groups, res)
            d = self._solve_linear(H, gvec, lam)
            dn = float(torch.linalg.norm(d))
            if dn < tol:
                stop = "step_below_tol"
                break
            T_new = self._apply_delta(T, d)
            res_new = self._residuals(T_new, groups, with_jac=False)
            cost_new = self._cost(res_new)
            # gain ratio (Nielsen): actual vs predicted decrease
            pred = float(-(d @ gvec) - 0.5 * (d @ (H @ d)))
            rho = (cost - cost_new) / pred if pred > 0 else (1.0 if cost_new < cost else -1.0)
            if cost_new < cost:
                T = T_new
                improvement = (cost - cost_new) / max(cost, 1e-30)
                cost = cost_new
                res = self._residuals(T, groups, with_jac=True)
                history.append(cost)
                n_accept += 1
                lam = lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                nu = 2.0
                if improvement < rel_tol:
                    stop = "rel_tol"
                    break
            else:
                lam *= nu
                nu *= 2.0
                if lam > float(_req(self.cfg, "lambda_max")):
                    stop = "lambda_max"
                    break
        self.T = T
        self.poses = T.cpu().numpy()
        per_edge = {}
        for kind, rec in self._residuals(T, groups, with_jac=False).items():
            nrm = torch.linalg.norm(rec["r"], dim=1).cpu().numpy()
            raw = rec["raw"].cpu().numpy()
            for k, eid in enumerate(groups[kind]["eid"]):
                per_edge[int(eid)] = {"weighted_norm": float(nrm[k]),
                                      "raw": raw[k].tolist(), "tag": self._edges[eid]["tag"]}
        self.report = {"cost_initial": cost0, "cost_final": cost, "iterations": len(history) - 1,
                       "accepted_steps": n_accept, "stop": stop, "lambda_final": lam,
                       "n_nodes": self.n, "n_edges": self.n_edges,
                       "history": history, "per_edge": per_edge}
        log(f"[pose-graph] {self.n} nodes, {self.n_edges} edges: cost {cost0:.4g} → {cost:.4g} "
            f"in {len(history) - 1} accepted step(s), stop={stop}")
        return self.report

    # ── helpers for the caller ───────────────────────────────────────────
    def corrections(self) -> np.ndarray:
        """Per-node rigid correction X_i = T_i_new · T_i_old⁻¹ (N,4,4) — apply
        to that keyframe's points AND camera (rigid, depth invariant)."""
        Tn = self.T.cpu().numpy()
        T0 = self.T0.cpu().numpy()
        return np.stack([Tn[k] @ se3_inv(T0[k]) for k in range(self.n)])

    def edge_residuals(self, tag_prefix: Optional[str] = None) -> Dict[int, dict]:
        """Relative-edge residuals at the CURRENT state: raw (rot deg, t m)
        and, for edges with a per-DOF information matrix, the OBSERVED
        translation residual ``t_obs_m`` = √(r_tᵀ·info_t·r_t)·σ_t — what the
        measurement actually claims (a floor patch's in-plane components are
        not a residual, they were never measured); ``weighted_norm`` is the
        whole residual in σ units."""
        out = {}
        groups = self._groups()
        res = self._residuals(self.T, groups, with_jac=False)
        for kind, rec in res.items():
            raw = rec["raw"].cpu().numpy()
            rw = rec["r"].cpu().numpy()
            for k, eid in enumerate(groups[kind]["eid"]):
                e = self._edges[eid]
                tag = e["tag"]
                if tag_prefix is not None and not str(tag).startswith(tag_prefix):
                    continue
                if kind == _REL:
                    W_t = e["W"][3:, 3:]
                    sigma_t = 1.0 / max(float(np.max(e["w"][3:])), 1e-12)
                    t_obs = float(np.linalg.norm(W_t.T @ raw[k][3:])) * sigma_t
                    out[int(eid)] = {"rot_deg": float(np.degrees(np.linalg.norm(raw[k][:3]))),
                                     "t_m": float(np.linalg.norm(raw[k][3:])), "t_obs_m": t_obs,
                                     "weighted_norm": float(np.linalg.norm(rw[k])), "tag": tag}
                else:
                    out[int(eid)] = {"norm": float(np.linalg.norm(raw[k])),
                                     "weighted_norm": float(np.linalg.norm(rw[k])), "tag": tag}
        return out
