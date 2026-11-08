"""loop_utils/cube_run.CubeRun.closures (USER 2026-11-08: the revisits as SLAM closures of the cube graph): a bridge
pass seen in its own frame, cube a and cube b each in theirs — the closure maps cube b's coordinates onto cube a's
(the seam convention {"a": b, "b": a}), recovered exactly from synthetic Sim(3) frames."""
import os

import numpy as np
from scipy.spatial.transform import Rotation

from loop_utils import cube_run as CR


def _sim3(s, rv, t):
    return float(s), Rotation.from_rotvec(rv).as_matrix(), np.asarray(t, np.float64)


def _apply(T, X):
    s, R, t = T
    return s * X @ R.T + t


def _maps(world, frames, H=12, W=16):
    """Per-frame point maps: frame f sees a patch of the true world surface (a different patch per frame)."""
    out = []
    for f in frames:
        rng = np.random.default_rng(1000 + int(f))
        out.append(world[rng.choice(len(world), H * W, replace=False)].reshape(H, W, 3))
    return np.stack(out)


def test_the_closure_maps_cube_b_onto_cube_a(tmp_path):
    rng = np.random.default_rng(0)
    world = rng.uniform([-3, -1, 2], [3, 2, 8], size=(5000, 3))
    Ta = _sim3(1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])          # cube a's frame = world
    Tb = _sim3(1.15, [0.05, -0.2, 0.1], [0.4, -0.3, 1.2])       # cube b's frame: world → b
    Tbr = _sim3(0.9, [0.1, 0.05, -0.02], [-0.2, 0.1, 0.3])      # the bridge's own frame
    sets = [np.arange(0, 20), np.arange(40, 60)]
    i, j = 10, 50
    run = CR.CubeRun.__new__(CR.CubeRun)
    run.n, run.K = 60, 2
    run.owner = np.array([0] * 30 + [1] * 30)
    run.sets = sets
    run.local = [{int(g): k for k, g in enumerate(F)} for F in sets]
    run.pairs = CR.overlap_pairs(sets)
    run.revisits, run.bridge_half = [(i, j)], 4
    run.un = str(tmp_path)
    for k, (F, T) in enumerate(zip(sets, (Ta, Tb))):
        wp = _maps(world, F)
        np.save(CR._path(run.un, k), {"world_points": _apply(T, wp.reshape(-1, 3)).reshape(wp.shape),
                                      "world_points_conf": np.ones(wp.shape[:3]),
                                      "extrinsic": np.tile(np.eye(4), (len(F), 1, 1))})
    (a, b, _i, _j, wa, wb), = run._bridge_items()
    wbr = _maps(world, wa + wb)
    np.save(run._bridge_path(a, b, i, j), {"world_points": _apply(Tbr, wbr.reshape(-1, 3)).reshape(wbr.shape),
                                           "world_points_conf": np.ones(wbr.shape[:3]),
                                           "_stac_frames": wa + wb})
    (e,) = run.closures()
    assert e["a"] == 1 and e["b"] == 0
    Xb = _apply(Tb, world[:200])
    Xa_hat = float(e["s"]) * Xb @ np.asarray(e["R"]).T + np.asarray(e["t"])
    assert np.allclose(Xa_hat, _apply(Ta, world[:200]), atol=1e-6), "cube b's points land on cube a's"


def test_neighbours_sharing_frames_get_no_closure(tmp_path):
    run = CR.CubeRun.__new__(CR.CubeRun)
    sets = [np.arange(0, 20), np.arange(10, 30)]
    run.n, run.K, run.sets = 30, 2, sets
    run.owner = np.array([0] * 15 + [1] * 15)
    run.local = [{int(g): k for k, g in enumerate(F)} for F in sets]
    run.pairs = CR.overlap_pairs(sets)
    run.revisits, run.bridge_half, run.un = [(5, 25)], 4, str(tmp_path)
    assert run._bridge_items() == [], "a seam already ties cubes that share frames"
