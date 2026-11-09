"""STAC 2026-11-06 (USER: "hay que recortar la escena como hace por ejemplo genrecon, que toma cubos … en esa pasada armás
los cubos, de 64 imágenes solapadas al 50% con los vecinos … se arman los cubos siempre con tope de 64"): the fork's run
over CUBES — ``Model.chunk_sets``, the server's cube plan (server/reconstruction/cubes.py).

A cube is a SET of keyframes (positions in this run's frame list): two neighbour blocks of the scene, from any moment of
the walk. Two cubes that share a block share those keyframes. The run does with them what the chain does with its
chunks, on a graph instead of a chain:

1. each cube inferred whole with its CONTEXT views (``Model.chunk_context``: the turn frames between two of its keyframes
   consecutive in the walk), then its WITNESS pass (``Model.chunk_witnesses``: the interval frames, F6 only) — context
   and witness records exactly as the chain's (loop_utils.context_views);
2. the METRIC LOCK per cube: the median ratio of its DA3 anchors applied to the cube; a cube without an anchor takes
   the anchored cubes' median, said;
3. a SEAM for every two cubes that share keyframes: the exact Sim(3) of their two copies of the shared frames (the same
   pixels, stable keys), σ its median residual — the chain's exact seam, on every edge of the overlap graph;
4. ONE chunk Sim(3) graph (loop_utils.chunk_graph.solve): the seams and DA3 per cube as a weak scale prior, cube 0 the
   gauge, started from the spanning tree of the best seams;
5. the cubes placed; the two-copy uncertainty; the depth graph; OWNERSHIP — each keyframe's points, pose and K from the
   cube the plan names (``Model.chunk_owner``), its pixels the owner drops written by ONE other copy (backfill); the PLY
   + origins per cube; the poses.

Not run on cubes, declared: the SALAD loop bridges and the geometric revisit closures (a revisit is inside a cube), the
scale ladder and the per-chunk linear scale drift (a cube has no walk order to drift along), the zoom fix, the
far-contradiction cap, elastic / intra-chunk / two-copy blend (off in the configuration)."""

import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

LOG = "[cubes]"
SCALE_RANGE = (0.6, 1.6)            # (kept for the chain mode only; the cubes measure their own bound — scale_bound)
Z95 = 1.959963984540054             # the two-sided 95 % normal quantile (the project's 95 % interval)
FACTOR = 1.1                        # USER: "1.1 en todos los casos" (metric_lock's improvement_error_factor)


def scale_bound(scales) -> dict:
    """The session's OWN bound on a relative scale (USER 2026-11-08: "de dónde salieron 0.6 y 1.6 … que sea genérico y
    correcto"): over the seams' Sim(3) log-scales, the median and the robust spread σ = 1.4826·MAD; a scale is
    degenerate when |log s − median| exceeds FACTOR × Z95 × σ — the session's 95 % interval widened by the project's
    factor, no fixed number. Fewer than 3 scales (no spread to measure): no bound, declared."""
    ls = np.log(np.asarray([float(x) for x in scales if float(x) > 0], np.float64))
    if len(ls) < 3:
        return {"median": 0.0, "sigma": None, "bound": None, "n": int(len(ls))}
    med = float(np.median(ls))
    sig = float(1.4826 * np.median(np.abs(ls - med)))
    return {"median": med, "sigma": sig, "bound": float(FACTOR * Z95 * sig), "n": int(len(ls))}


def scale_ok(s: float, sb: dict) -> bool:
    return sb["bound"] is None or abs(float(np.log(float(s))) - sb["median"]) <= sb["bound"]


class CubeLayoutError(ValueError):
    """A Model.chunk_sets / chunk_owner the fork cannot run."""


class CubePlanMismatch(RuntimeError):
    """A cube file on disk belongs to another cube plan."""


# ── layout ────────────────────────────────────────────────────────────────────────────────────

def validate_chunk_sets(sets, n_frames: int, owner) -> Tuple[List[np.ndarray], np.ndarray]:
    """(the cubes as sorted int arrays, the owner cube of every frame): every cube ≥ 2 distinct frames of [0, n), every
    frame in a cube, every frame owned by a cube that holds it, every cube joined to cube 0 through shared frames."""
    if not isinstance(sets, (list, tuple)) or not sets:
        raise CubeLayoutError(f"Model.chunk_sets must be a non-empty list of frame-index lists, got {type(sets)}")
    n = int(n_frames)
    out = []
    for k, s in enumerate(sets):
        a = np.asarray(list(s), dtype=np.int64).ravel()
        if a.size < 2:
            raise CubeLayoutError(f"cube {k} holds {a.size} frame(s) — a cube is at least two")
        if (np.diff(a) <= 0).any():
            raise CubeLayoutError(f"cube {k}: its frames are not strictly increasing")
        if a[0] < 0 or a[-1] >= n:
            raise CubeLayoutError(f"cube {k}: frames outside [0, {n})")
        out.append(a)
    cover = np.zeros(n, bool)
    for a in out:
        cover[a] = True
    if not cover.all():
        raise CubeLayoutError(f"{int((~cover).sum())} frame(s) in no cube (first {int(np.flatnonzero(~cover)[0])})")
    ow = np.asarray(list(owner), dtype=np.int64).ravel()
    if ow.size != n:
        raise CubeLayoutError(f"Model.chunk_owner holds {ow.size} entries for {n} frames")
    for g in range(n):
        k = int(ow[g])
        if not 0 <= k < len(out) or not np.isin(g, out[k]):
            raise CubeLayoutError(f"frame {g} is owned by cube {k}, which does not hold it")
    pairs = overlap_pairs(out)
    seen, todo = {0}, [0]
    while todo:
        u = todo.pop()
        for (a, b) in pairs:
            v = b if a == u else (a if b == u else None)
            if v is not None and v not in seen:
                seen.add(v)
                todo.append(v)
    if len(seen) != len(out):
        raise CubeLayoutError(f"cube(s) {sorted(set(range(len(out))) - seen)} share no frame with the rest — the "
                              f"graph cannot place them")
    return out, ow


def overlap_pairs(sets: Sequence[np.ndarray]) -> Dict[Tuple[int, int], np.ndarray]:
    """{(a, b): the frames cubes a < b share} for every pair that shares any."""
    out = {}
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            sh = np.intersect1d(sets[a], sets[b], assume_unique=True)
            if sh.size:
                out[(a, b)] = sh
    return out


def check_cube_extras(extras, sets, img_list, img_dir, what: str) -> List[List[str]]:
    """``Model.chunk_context`` / ``chunk_witnesses`` resolved to paths per cube: every file a frame of ``img_dir``, not a
    keyframe, strictly between two keyframes CONSECUTIVE in the walk that are both in the cube (the frame Omega places
    between them, its anchor the earlier one); no file twice. None → nothing."""
    from loop_utils.context_views import frame_number
    if extras is None:
        return [[] for _ in sets]
    if not isinstance(extras, (list, tuple)) or len(extras) != len(sets):
        raise CubeLayoutError(f"Model.{what} must hold one list per cube ({len(sets)})")
    names = {os.path.basename(p) for p in img_list}
    nums = [frame_number(p) for p in img_list]
    out = []
    for k, (F, files) in enumerate(zip(sets, extras)):
        files = [str(f) for f in (files or [])]
        if len(set(files)) != len(files):
            raise CubeLayoutError(f"cube {k}: a {what} frame is listed twice")
        mine = set(int(g) for g in F)
        spans = [(nums[g], nums[g + 1]) for g in sorted(mine) if g + 1 < len(nums) and (g + 1) in mine]
        paths = []
        for fn in files:
            if os.path.basename(fn) != fn:
                raise CubeLayoutError(f"cube {k}: {what} frame {fn!r} must be a file name")
            if fn in names:
                raise CubeLayoutError(f"cube {k}: {what} frame {fn} is a keyframe")
            x = frame_number(fn)
            if not any(lo < x < hi for lo, hi in spans):
                raise CubeLayoutError(f"cube {k}: {what} frame {fn} is not between two consecutive keyframes of the "
                                      f"cube")
            p = os.path.join(img_dir, fn)
            if not os.path.isfile(p):
                raise CubeLayoutError(f"cube {k}: {what} frame {p} does not exist")
            paths.append(p)
        out.append(sorted(paths, key=frame_number))
    return out


def layout_digest(sets, owner) -> str:
    import hashlib
    h = hashlib.sha256(json.dumps([[int(g) for g in F] for F in sets]).encode())
    h.update(json.dumps([int(x) for x in owner]).encode())
    return h.hexdigest()


# ── the run ───────────────────────────────────────────────────────────────────────────────────

def _path(d, k):
    return os.path.join(d, f"chunk_{k}.npy")


def _wp(data):
    wp = np.asarray(data['world_points'])
    return wp[0] if wp.ndim == 5 else wp


def _ext(data):
    e = np.asarray(data['extrinsic'], np.float64)
    return e[0] if e.ndim == 4 else e


def _K(data):
    k = data.get('intrinsic')
    if k is None:
        return None
    k = np.asarray(k, np.float64)
    return k[0] if k.ndim == 4 else k


def _conf(data, shp):
    return np.asarray(data['world_points_conf']).reshape(shp)


def _sim3_matrix(s, R, t):
    S = np.eye(4)
    S[:3, :3] = float(s) * np.asarray(R, np.float64)
    S[:3, 3] = np.asarray(t, np.float64).reshape(3)
    return S


def aligned_pose(pose, ext_c2w) -> np.ndarray:
    """World c2w of a cube's frame: the cube's Sim(3) on its metric-locked extrinsic (the chain's composition)."""
    s, R, t = pose
    M = _sim3_matrix(s, R, t) @ np.asarray(ext_c2w, np.float64)
    M[:3, :3] /= float(s)
    return M


class CubeRun:
    """The cube run of a VGGT_Long instance ``vl`` (its model, img_list, directories, stamps and helpers)."""

    def __init__(self, vl, images_to_u8, images_u8, save_ply, apply_sim3_direct):
        self.vl = vl
        self.cfg = vl.config['Model']
        self.sets, self.owner = vl._stac_cube_sets, vl._stac_cube_owner
        self.K = len(self.sets)
        self.n = len(vl.img_list)
        self.local = [{int(g): i for i, g in enumerate(F)} for F in self.sets]
        self.pairs = overlap_pairs(self.sets)
        # the ONE other copy of a shared frame that backfills what its owner drops: the lowest cube holding it
        self.backfiller = np.full(self.n, -1, np.int64)
        for g in range(self.n):
            others = [k for k in range(self.K) if k != int(self.owner[g]) and g in self.local[k]]
            if others:
                self.backfiller[g] = min(others)
        self.images_to_u8, self.images_u8 = images_to_u8, images_u8
        self.save_ply, self.apply_sim3_direct = save_ply, apply_sim3_direct
        self.digest = layout_digest(self.sets, self.owner)
        self.un, self.al = vl.result_unaligned_dir, vl.result_aligned_dir
        self.poses = None
        # THE REVISITS AS SLAM CLOSURES (USER 2026-11-08): global keyframe-index pairs (i, j) the server's F4 rule
        # found co-visible — each bridged by a joint Omega pass, then a Cauchy-robust loop edge of the cube graph
        self.revisits = [(int(i), int(j)) for i, j in (self.cfg.get('cube_revisits') or [])]
        # each side of a bridge: F4's own loop window (precision.tracks.loop_half_window), handed by the server
        self.bridge_half = int(self.cfg.get('cube_bridge_half', 0))

    # 1b. the revisit bridges (model loaded) ──────────────────────────────────────────────────
    def _bridge_path(self, a, b, i, j):
        return os.path.join(self.un, f"bridge_{a}_{b}_{i}_{j}.npy")

    def _bridge_items(self):
        """(cube a, cube b, i, j, window a, window b) per revisit whose ends lie in two cubes that share no frame
        (a seam already ties cubes that do)."""
        out = []
        for i, j in self.revisits:
            if not (0 <= i < self.n and 0 <= j < self.n):
                continue
            a, b = int(self.owner[i]), int(self.owner[j])
            if a == b or (min(a, b), max(a, b)) in self.pairs:
                continue
            h = self.bridge_half
            wa = [g for g in sorted(self.local[a]) if abs(g - i) <= h]
            wb = [g for g in sorted(self.local[b]) if abs(g - j) <= h]
            if wa and wb:                       # a starved fit is refused by robust_sim3 itself
                out.append((a, b, i, j, wa, wb))
        return out

    def bridges(self):
        """One joint Omega pass per revisit (both windows together), written to disk — read back by the graph."""
        vl = self.vl
        items = self._bridge_items()
        print(f"{LOG} revisit bridges: {len(items)} of {len(self.revisits)} revisit(s) join two cubes that share "
              f"no frame")
        for a, b, i, j, wa, wb in items:
            p = self._bridge_path(a, b, i, j)
            if os.path.exists(p):
                try:
                    old = np.load(p, allow_pickle=True).item()
                    if list(old.get('_stac_frames') or []) == wa + wb:
                        vl._stac_check_fork_stamp(old, p)
                        continue
                except RuntimeError:
                    raise
                except Exception:  # noqa: BLE001 — unreadable: re-infer
                    pass
            views = [vl.img_list[int(g)] for g in wa + wb]
            pred = vl.model.infer_chunk(views)
            for key in pred.keys():
                if isinstance(pred[key], torch.Tensor):
                    pred[key] = pred[key].cpu().numpy().squeeze(0)
            pred['depth'] = np.squeeze(pred['depth'])
            vl._stac_mask_sky(pred, views)
            pred['_stac_frames'] = wa + wb
            pred['_stac_split'] = len(wa)
            if vl._stac_stamp_digest() is not None:
                pred['_stac_fork_stamp'] = vl._stac_stamp_digest()
            np.save(p, pred)
            del pred
            torch.cuda.empty_cache()
            print(f"{LOG} bridge {a}<->{b} (revisit {i}<->{j}): {len(wa)}+{len(wb)} keyframes")

    def closures(self) -> List[dict]:
        """The loop edges of the graph: per bridge, its fit onto cube a and onto cube b on EXACT pixel
        correspondences (the same frames in both passes), composed into cube b → cube a — the seam convention
        ({"a": b, "b": a}: the transform maps edge["a"]'s coordinates onto edge["b"]'s)."""
        from loop_utils.loop_bridges import robust_sim3
        from loop_utils.stable_sample import pixel_keys
        out = []
        for a, b, i, j, wa, wb in self._bridge_items():
            p = self._bridge_path(a, b, i, j)
            if not os.path.exists(p):
                continue
            br = np.load(p, allow_pickle=True).item()
            fits, side_pts = {}, {}
            for side, k, win, off in (("a", a, wa, 0), ("b", b, wb, len(wa))):
                dk = np.load(_path(self.un, k), allow_pickle=True).item()
                lb = np.arange(off, off + len(win))
                lk = np.asarray([self.local[k][int(g)] for g in win])
                pb, pk = _wp(br)[lb], _wp(dk)[lk]
                shp = pb.shape[:3]
                ok = (_conf(br, _wp(br).shape[:3])[lb] > 1e-5) & (_conf(dk, _wp(dk).shape[:3])[lk] > 1e-5)
                flat = np.flatnonzero(ok.reshape(-1))
                px = int(shp[1]) * int(shp[2])
                keys = pixel_keys(np.asarray(win)[flat // px], flat % px)
                f3 = robust_sim3(pb.reshape(-1, 3)[flat].astype(np.float64),
                                 pk.reshape(-1, 3)[flat].astype(np.float64), keys=keys)
                # each side's scale is the bridge's raw Omega scale against a cube already in the metric lock's
                # scale (~5–7× on pccr 2026-11-08: all 49 closures were refused here) — only the COMPOSED relative
                # scale between the two cubes is held to SCALE_RANGE, below
                if f3 is None or not (float(f3[0]) > 0):
                    fits = None
                    break
                fits[side] = f3
                if side == "a":
                    cams = _ext(dk)[lk][:, :3, 3]
                    q = pk.reshape(-1, 3)[flat].astype(np.float64)
                    side_pts = {"centroid": np.median(q[::max(1, len(q) // 20000)], axis=0).tolist(),
                                "lever_m": float(np.median(np.linalg.norm(
                                    q[::max(1, len(q) // 20000)][:, None, :] - cams[None], axis=2).min(1)))}
                del dk
            if not fits:
                print(f"{LOG} ⚠ closure {a}<->{b} (revisit {i}<->{j}): a side starved or degenerate — not an edge "
                      f"(declared)")
                continue
            sa, Ra, ta, ra, na = fits["a"]
            sb, Rb, tb, rb, nb = fits["b"]
            s = float(sa) / float(sb)
            if not scale_ok(s, getattr(self, "scale_bound", {"bound": None, "median": 0.0})):
                print(f"{LOG} ⚠ closure {a}<->{b} (revisit {i}<->{j}): relative scale {s:.4f} outside the session's "
                      f"bound — not an edge (declared)")
                continue
            R = np.asarray(Ra, np.float64) @ np.asarray(Rb, np.float64).T
            t = np.asarray(ta, np.float64).reshape(3) - s * (R @ np.asarray(tb, np.float64).reshape(3))
            sig = float(np.hypot(float(ra), float(rb)))
            out.append({"a": int(b), "b": int(a), "s": s, "R": R, "t": t, "centroid": side_pts["centroid"],
                        "lever_m": max(side_pts["lever_m"], 1e-3), "sigma_m": max(sig, 1e-3), "exact": False,
                        "fit": "bridge", "revisit": [int(i), int(j)], "correspondences": int(min(na, nb))})
            print(f"{LOG} closure {a}<->{b} (revisit {i}<->{j}): scale {s:.4f}, residual {sig * 100:.1f} cm")
        return out

    # 1. inference ─────────────────────────────────────────────────────────────────────────────
    def infer(self, k):
        from loop_utils.context_views import keep_keyframe_slots, merge_views
        vl, F = self.vl, self.sets[k]
        ctx = list(vl.chunk_context[k])
        names = [os.path.basename(p) for p in ctx]
        if os.path.exists(_path(self.al, k)):
            print(f"{LOG} cube {k}: aligned already — inference skipped")
            return
        sp = _path(self.un, k)
        if os.path.exists(sp):
            try:
                pred = np.load(sp, allow_pickle=True).item()
                vl._stac_check_fork_stamp(pred, sp)
                if (list(pred.get('_stac_frames') or []) != F.tolist()
                        or list(pred.get('_stac_context') or []) != names):
                    raise CubePlanMismatch(f"{sp} is another cube (frames / context views) — the outputs on disk are "
                                           f"of ANOTHER cube plan; re-run with replace, or let the map worker wipe "
                                           f"them")
                print(f"{LOG} cube {k}: on disk — inference skipped")
                return
            except RuntimeError:
                # a cube of another plan or another fork stamp (_StacPlanMismatch, CubePlanMismatch) is a broken
                # invariant — the run stops (plan point 9), it is never silently re-inferred
                raise
            except Exception as e:  # noqa: BLE001
                print(f"{LOG} cube {k}: {sp} unreadable ({e}) — re-inferring")
        paths = [vl.img_list[int(g)] for g in F]
        if ctx:
            views, keep = merge_views(paths, ctx)
        else:
            views, keep = paths, None
        pred = vl.model.infer_chunk(views)
        for key in pred.keys():
            if isinstance(pred[key], torch.Tensor):
                pred[key] = pred[key].cpu().numpy().squeeze(0)
        if keep is not None:
            vl._stac_mask_sky(pred, views)
            vl._stac_save_context_views(k, pred, views, keep, F.tolist())
            keep_keyframe_slots(pred, keep, len(views))
        pred['depth'] = np.squeeze(pred['depth'])
        if keep is None:
            vl._stac_mask_sky(pred, paths)
        self.images_to_u8(pred)
        pred['_stac_frames'] = F.tolist()
        pred['_stac_context'] = names
        if vl._stac_stamp_digest() is not None:
            pred['_stac_fork_stamp'] = vl._stac_stamp_digest()
        np.save(sp, pred)
        print(f"{LOG} cube {k}: {len(views)} views inferred ({len(paths)} keyframes + {len(ctx)} context)")

    def witness_pass(self, k):
        """The cube's interval frames with its keyframes, a pass of their own (the chain's _stac_witness_pass)."""
        from loop_utils.context_views import CONTEXT_DIR, context_records, frame_number, merge_views
        vl, F = self.vl, self.sets[k]
        wit = list((vl.chunk_witnesses or [[]] * self.K)[k] or [])
        if not wit:
            return
        kf_paths = [vl.img_list[int(g)] for g in F]
        d = os.path.join(vl.output_dir, CONTEXT_DIR)
        stamp = vl._stac_stamp_digest() or ""
        prefix = f"w{int(k):03d}_"
        want = {f"{prefix}{frame_number(p):06d}.npz" for p in wit}
        if os.path.isdir(d) and want <= set(os.listdir(d)):
            same = True
            for fn in sorted(want):
                with np.load(os.path.join(d, fn)) as z:
                    same = same and str(z["fork_stamp"]) == stamp
            if same:
                print(f"{LOG} cube {k}: {len(want)} witness view(s) of this run on disk — pass skipped")
                return
        views, keep = merge_views(kf_paths, wit)
        pred = vl.model.infer_chunk(views)
        for key in pred.keys():
            if isinstance(pred[key], torch.Tensor):
                pred[key] = pred[key].cpu().numpy().squeeze(0)
        vl._stac_mask_sky(pred, views)
        os.makedirs(d, exist_ok=True)
        for fn in os.listdir(d):
            if fn.startswith(prefix):
                os.remove(os.path.join(d, fn))
        recs = context_records(pred, views, keep, F.tolist())
        for r in recs:
            path = os.path.join(d, f"{prefix}{r['frame']:06d}.npz")
            np.savez(path + ".tmp.npz", chunk=int(k), fork_stamp=stamp, **{kk: v for kk, v in r.items()})
            os.replace(path + ".tmp.npz", path)
        print(f"{LOG} cube {k}: witness pass — {len(views)} views ({len(kf_paths)} keyframes + {len(wit)} "
              f"witnesses), {len(recs)} kept for F6")
        del pred

    # 2. metric lock ───────────────────────────────────────────────────────────────────────────
    def metric_lock(self):
        """Each cube at its DA3 metre: the median of its anchors' ratios (loop_utils.metric_lock.chunk_anchor_ratios,
        the chain's), applied in place; a cube without an anchor takes the anchored cubes' median (said). The scale
        each cube got is kept in its file (a resume never scales twice — its ratios are read back through it) and in
        metric_lock.json."""
        from loop_utils.metric_lock import apply_scale, chunk_anchor_ratios, real_frame_number
        vl = self.vl
        ml = self.cfg.get('metric_lock') or {}
        self.scale_inputs, self.scales_applied = {}, {}
        if not ml.get('enable'):
            print(f"{LOG} metric lock OFF (Model.metric_lock.enable) — the cubes keep Omega's units")
            return
        s_da3, n_anc, applied, deferred = {}, {}, {}, []
        rep = vl._stac_with_stamp({"layout": "cubes", "layout_sha256": self.digest, "chunks": {}})
        for k, data in vl._stac_iter_chunks(self.un, range(self.K)):
            nums = [real_frame_number(vl.img_list[int(g)]) for g in self.sets[k]]
            locs, ratios = chunk_anchor_ratios(data, nums, ml['anchor_dir'], near_frac=float(ml.get('near_frac', 0.25)))
            lock = data.get('_stac_cube_lock')
            if lock is not None:                      # a resumed cube: its ratios before the lock
                ratios = [float(r) * float(lock) for r in ratios]
            if ratios:
                s_da3[k] = float(np.median(ratios))
                n_anc[k] = len(ratios)
            rep["chunks"][str(k)] = {"s": s_da3.get(k), "n_anchors": len(ratios), "ratios": ratios,
                                     "anchor_locals": locs}
            if lock is not None:
                applied[k] = float(lock)
                print(f"{LOG} cube {k}: locked already (s {applied[k]:.4f})")
            elif k in s_da3:
                apply_scale(data, s_da3[k])
                data['_stac_cube_lock'] = float(s_da3[k])
                np.save(_path(self.un, k), data)
                applied[k] = float(s_da3[k])
                print(f"{LOG} cube {k}: DA3 s={s_da3[k]:.4f} from {n_anc[k]} anchor(s)")
            else:
                deferred.append(k)
            del data
        if not s_da3:
            raise RuntimeError(f"{LOG} no cube holds a DA3 anchor — the metric lock has nothing to lock with")
        fill = float(np.median(list(s_da3.values())))
        for k in deferred:
            data = np.load(_path(self.un, k), allow_pickle=True).item()
            apply_scale(data, fill)
            data['_stac_cube_lock'] = float(fill)
            np.save(_path(self.un, k), data)
            applied[k] = fill
            print(f"{LOG} cube {k}: no DA3 anchor — the anchored cubes' median s {fill:.4f} (its scale comes from its "
                  f"seams in the graph)")
            del data
        for k, s_ in applied.items():
            rep["chunks"][str(k)]["s_applied"] = float(s_)
        with open(os.path.join(vl.output_dir, "metric_lock.json"), "w") as f:
            json.dump(rep, f, indent=1, default=float)
        self.scale_inputs = {"s_da3": s_da3, "n_anchors": n_anc, "sigma_anchor": float(ml.get('sigma_anchor', 0.08))}
        self.scales_applied = applied

    # 3. seams ─────────────────────────────────────────────────────────────────────────────────
    def seams(self) -> List[dict]:
        """The exact Sim(3) of every two cubes that share frames (b's copies onto a's, a < b): an edge of the chunk
        graph {a: b, b: a, …} in loop_utils.chunk_graph's convention, read at the shared points in cube a with their
        lever arm, σ the fit's median residual."""
        from loop_utils.loop_bridges import robust_sim3
        from loop_utils.metric_lock import robust_rigid
        from loop_utils.stable_sample import pixel_keys
        cache = {}

        def load(k):
            if k not in cache:
                if len(cache) >= 2:
                    cache.pop(next(iter(cache)))
                cache[k] = np.load(_path(self.un, k), allow_pickle=True).item()
            return cache[k]
        edges, cand = [], []
        for (a, b), sh in sorted(self.pairs.items()):
            da, db = load(a), load(b)
            la = np.asarray([self.local[a][int(g)] for g in sh])
            lb = np.asarray([self.local[b][int(g)] for g in sh])
            pa, pb = _wp(da)[la], _wp(db)[lb]
            shp = pa.shape[:3]
            ca, cb = _conf(da, _wp(da).shape[:3])[la], _conf(db, _wp(db).shape[:3])[lb]
            ok = (ca > 1e-5) & (cb > 1e-5)
            flat = np.flatnonzero(ok.reshape(-1))
            px = int(shp[1]) * int(shp[2])
            keys = pixel_keys(sh[flat // px], flat % px)
            P_a = pa.reshape(-1, 3)[flat].astype(np.float64)
            P_b = pb.reshape(-1, 3)[flat].astype(np.float64)
            f3 = robust_sim3(P_b, P_a, keys=keys)
            fr = robust_rigid(P_b, P_a, keys=keys)
            fits_sim3 = (float(f3[0]), f3[1], f3[2], float(f3[3]), int(f3[4])) if f3 is not None else None
            fits_rigid = (1.0, fr[0], fr[1], float(fr[2]), int(fr[3])) if fr is not None else None
            if fits_sim3 is None and fits_rigid is None:
                print(f"{LOG} ⚠ seam {a}<->{b}: starved ({len(sh)} shared frame(s), {len(flat):,} pixels) — not an "
                      f"edge (declared)")
                continue
            cams = _ext(da)[la][:, :3, 3]
            dists, pts = [], []
            for f in range(shp[0]):
                q = pa[f][ok[f]].astype(np.float64)
                if len(q) == 0:
                    continue
                dists.append(float(np.median(np.linalg.norm(q - cams[f], axis=1))))
                pts.append(q[::max(1, len(q) // 20000)])
            if not pts:
                print(f"{LOG} ⚠ seam {a}<->{b}: no valid shared point — not an edge (declared)")
                continue
            cand.append({"a": int(b), "b": int(a), "_sim3": fits_sim3, "_rigid": fits_rigid,
                         "centroid": np.median(np.concatenate(pts), axis=0).tolist(),
                         "lever_m": float(np.median(dists)), "exact": True, "shared_frames": int(len(sh))})
            del pa, pb, P_a, P_b
        cache.clear()
        # the decision on the session's OWN bound (scale_bound): measured over these seams' Sim(3) scales
        sb = scale_bound([c["_sim3"][0] for c in cand if c["_sim3"] is not None])
        self.scale_bound = sb
        print(f"{LOG} seam scale bound: " + (f"|log s − {sb['median']:+.4f}| ≤ {sb['bound']:.4f} (1.1 × the 95 % "
                                              f"interval of the session's σ {sb['sigma']:.4f}, {sb['n']} seams)"
                                              if sb["bound"] is not None else
                                              f"none — {sb['n']} seam scale(s), no spread to measure (declared)"))
        for c in cand:
            a_, b_ = c["b"], c["a"]
            # SEAMS ARE ALWAYS FREE SCALE (USER 2026-11-09: "ajustalo libre" — zaragoza's tight measured bound, σ
            # 0.7 %, sent three seams to the rigid fit at 7.9–14.8 cm): the Sim(3) of the same pixels in two passes
            # is the seam; the rigid fit only when no Sim(3) can be fitted at all. The bound is still measured
            # and logged, and it still judges the closures.
            if c["_sim3"] is not None:
                fit, kind = c["_sim3"], "Sim3"
                if not scale_ok(c["_sim3"][0], sb):
                    print(f"{LOG} seam {a_}<->{b_}: Sim3 scale {c['_sim3'][0]:.4f} outside the session's bound — "
                          f"kept free (declared)")
            elif c["_rigid"] is not None:
                fit, kind = c["_rigid"], "rigid"
                print(f"{LOG} seam {a_}<->{b_}: no Sim3 could be fitted — the rigid fit (declared)")
            else:
                print(f"{LOG} ⚠ seam {a_}<->{b_}: neither a Sim3 nor a rigid fit — not an edge (declared)")
                continue
            s_, R_, t_, res_, n_used = fit
            edges.append({"a": c["a"], "b": c["b"], "s": float(s_), "R": np.asarray(R_, np.float64),
                          "t": np.asarray(t_, np.float64).reshape(3), "centroid": c["centroid"],
                          "lever_m": c["lever_m"], "sigma_m": float(res_), "exact": True, "fit": kind,
                          "shared_frames": c["shared_frames"], "correspondences": int(n_used)})
            print(f"{LOG} seam {a_}<->{b_}: {kind} on {n_used:,} exact correspondences of {c['shared_frames']} "
                  f"shared frame(s), scale {s_:.4f}, median residual {res_ * 100:.1f} cm")
        if any(not (e["sigma_m"] > 0) for e in edges):
            pos = [e["sigma_m"] for e in edges if e["sigma_m"] > 0]
            for e in edges:
                if not (e["sigma_m"] > 0):
                    e["sigma_m"], e["sigma_source"] = (max(pos) if pos else 0.01), "a zero residual: the largest"
        return edges

    # 4. graph ─────────────────────────────────────────────────────────────────────────────────
    def graph(self, edges: List[dict]):
        """ONE Sim(3) graph over the cubes (loop_utils.chunk_graph.solve), started from the spanning tree of the seams
        taken best first (breadth-first from cube 0, each cube's seams by σ, ties by index). chunk_graph.json."""
        from loop_utils import chunk_graph as CG
        K = self.K
        adj = {k: [] for k in range(K)}
        for e in edges:
            adj[e["a"]].append(e)
            adj[e["b"]].append(e)
        tree = {0: CG.identity()}
        queue = [0]
        while queue:
            u = queue.pop(0)
            for e in sorted(adj[u], key=lambda e: (e["sigma_m"], e["a"], e["b"])):
                M = (e["s"], e["R"], e["t"])
                if e["b"] == u:
                    v, P = e["a"], CG.compose(tree[u], M)              # M maps v onto u
                else:
                    v, P = e["b"], CG.compose(tree[u], CG.inverse(M))  # M maps u onto v
                if v in tree:
                    continue
                tree[v] = P
                queue.append(v)
        if len(tree) != K:
            raise RuntimeError(f"{LOG} cube(s) {sorted(set(range(K)) - set(tree))} joined to cube 0 by no fitted seam "
                               f"— the graph cannot place them")
        x0 = [tree[k] for k in range(K)]
        inputs = getattr(self, 'scale_inputs', None) or {}
        sig_a = float(inputs.get("sigma_anchor", 0.08))
        # THE DA3 PER-CUBE SCALES ARE RECORDED, NOT USED (USER 2026-11-07, pccr: "corregí, detené pccr y volvé a
        # correr"). Every seam is an exact fit of the same pixels in two passes: the cubes must coincide there. A DA3
        # prior per cube fought the seams wherever the locks disagree (pccr: neighbours 0.876–1.167 apart) — at
        # σ_anchor/√anchors (1.1–1.5 %) the solve broke two seams by 16 %; at the locks' measured spread (8.8 %) it
        # still bent three by 6–10 cm, and the 10-view support floor then emptied the keyframes at those transitions
        # (#19–25, #55–65, #187–199: 0–2 k points each against a median of 27 k). The metre is scale_align's (every
        # anchor of the session); the scale ALONG the walk is F2's continuous gauge — a smooth curve of the walked
        # distance, so it cannot put a step at a transition. The locks' spread is still measured and recorded.
        ls = np.log([float(e["s"]) for e in edges]) if edges else np.zeros(0)
        sig_lock = float(1.4826 * np.median(np.abs(ls)) / np.sqrt(2.0)) if len(ls) else 0.0
        priors = {}
        for k, s_d in (inputs.get("s_da3") or {}).items():
            s_v = (self.scales_applied or {}).get(int(k))
            if s_d and s_v:
                nk = max(float((inputs.get("n_anchors") or {}).get(k, 1)), 1.0)
                priors[int(k)] = {"log_s": float(np.log(float(s_d) / float(s_v))),
                                  "sigma": max(sig_a / float(np.sqrt(nk)), sig_lock), "anchors": int(nk),
                                  "sigma_anchors": sig_a / float(np.sqrt(nk)), "sigma_seams": sig_lock}
        print(f"{LOG} graph: {K} cube(s), {len(edges)} seam(s) (σ {min(e['sigma_m'] for e in edges) * 100:.1f}–"
              f"{max(e['sigma_m'] for e in edges) * 100:.1f} cm, squared — an exact fit of the same pixels is never "
              f"dropped); the {len(priors)} DA3 per-cube scales recorded, not used (the locks disagree by "
              f"{sig_lock * 100:.1f} % between neighbours; the metre is scale_align's, the scale along the walk F2's)"
              if edges else f"{LOG} graph: one cube")
        # the revisits as SLAM closures (USER 2026-11-08): loop edges under Cauchy — a wrong one is down-weighted,
        # never trusted like a seam; the spanning tree above is the seams' alone
        loops = self.closures() if self.revisits else []
        sol = CG.solve(K, edges, loops, {}, x0=x0, robust_seams=False)
        print(f"{LOG} graph: {len(loops)} revisit closure(s) in the solve")
        if sol["converged"]:
            poses, applied = sol["poses"], "seams"
        else:
            poses, applied = x0, "the seams' spanning tree (the solve did not converge)"
        self.poses = poses

        def _rec(P):
            return [{"s": float(p[0]), "R": np.asarray(p[1]).tolist(), "t": np.asarray(p[2]).tolist()} for p in P]

        def _js(e):
            return {k_: (np.asarray(v_).tolist() if isinstance(v_, np.ndarray) else v_) for k_, v_ in e.items()}
        per_chunk = [{"chunk": k, "scale_vs_tree": float(poses[k][0] / x0[k][0]),
                      "moved_m": float(np.linalg.norm(np.asarray(poses[k][2]) - np.asarray(x0[k][2])))}
                     for k in range(K)]
        seam_err = [{"seam": [e["b"], e["a"]], "sigma_m": e["sigma_m"],
                     "tree_m": CG.edge_error_m(x0[e["a"]], x0[e["b"]], e),
                     "final_m": CG.edge_error_m(poses[e["a"]], poses[e["b"]], e),
                     "final_surface_m": CG.edge_surface_error_m(poses[e["a"]], poses[e["b"]], e)} for e in edges]
        doc = {"version": 1, "mode": "sim3", "layout": "cubes", "layout_sha256": self.digest, "applied": applied,
               "seams": [_js(e) for e in edges], "closures": [_js(e) for e in loops], "priors": {str(k): v for k, v in priors.items()},
               "priors_used": False, "locks_spread_log": sig_lock,
               "solve": {k_: v_ for k_, v_ in sol.items() if k_ != "poses"}, "poses": _rec(poses), "tree": _rec(x0),
               "per_chunk": per_chunk, "seam_errors": seam_err}
        if self.vl._stac_stamp_digest() is not None:
            doc["_stac_fork_stamp"] = self.vl._stac_stamp_digest()
        path = os.path.join(self.vl.output_dir, "chunk_graph.json")
        with open(path + ".tmp", "w") as f:
            json.dump(doc, f, indent=1, default=float)
        os.replace(path + ".tmp", path)
        fe = [s_["final_surface_m"] for s_ in seam_err]
        print(f"{LOG} graph APPLIED ({applied}): cube scale vs the tree {min(c['scale_vs_tree'] for c in per_chunk):.4f}"
              f"–{max(c['scale_vs_tree'] for c in per_chunk):.4f}, moved ≤ "
              f"{max(c['moved_m'] for c in per_chunk) * 100:.1f} cm"
              + (f"; seams after the solve, at their surfaces (scale and rotation read at the lever arm) "
                 f"{np.median(fe) * 100:.1f} cm median, {max(fe) * 100:.1f} max" if fe else "")
              + " → chunk_graph.json")

    def load_graph(self) -> bool:
        """A resume past the apply: the poses chunk_graph.json recorded for THIS layout and stamp."""
        p = os.path.join(self.vl.output_dir, "chunk_graph.json")
        if not os.path.exists(p):
            return False
        try:
            doc = json.load(open(p))
        except ValueError:
            return False
        if doc.get("layout_sha256") != self.digest:
            return False
        st = self.vl._stac_stamp_digest()
        if st is not None and doc.get("_stac_fork_stamp") != st:
            return False
        self.poses = [(float(q["s"]), np.asarray(q["R"], np.float64), np.asarray(q["t"], np.float64))
                      for q in doc["poses"]]
        return len(self.poses) == self.K

    # 5. apply ─────────────────────────────────────────────────────────────────────────────────
    def apply(self):
        """Each cube's points onto the graph's pose (world points by its Sim(3), depth by its scale; the extrinsic stays
        the cube's, composed when read — the chain's convention); chunk_sim3.json records the layout and the poses."""
        todo = [k for k in range(self.K) if not os.path.exists(_path(self.al, k))]
        for k, data in self.vl._stac_iter_chunks(self.un, todo):
            s, R, t = self.poses[k]
            data['world_points'] = self.apply_sim3_direct(_wp(data), s, R, t).astype(np.float32)
            if data.get('depth') is not None:
                data['depth'] = np.asarray(data['depth']) * float(s)
            data['_stac_cube_pose'] = {"s": float(s), "R": np.asarray(R).tolist(), "t": np.asarray(t).tolist()}
            np.save(_path(self.al, k), data)
            try:
                os.remove(_path(self.un, k))
            except OSError:
                pass
            print(f"{LOG} cube {k}: placed (scale {s:.4f})")
            del data
        doc = {"layout": "cubes", "layout_sha256": self.digest,
               "chunk_sets": [F.tolist() for F in self.sets], "owner": self.owner.tolist(),
               "poses": [{"s": float(p[0]), "R": np.asarray(p[1]).tolist(), "t": np.asarray(p[2]).tolist()}
                         for p in self.poses]}
        p = os.path.join(self.vl.output_dir, "chunk_sim3.json")
        with open(p + ".tmp", "w") as f:
            json.dump(doc, f, indent=1)
        os.replace(p + ".tmp", p)

    def pose_of(self, k, ext_row):
        return aligned_pose(self.poses[k], ext_row)

    # 6. two-copy uncertainty ──────────────────────────────────────────────────────────────────
    def uncertainty(self):
        """§4.8 on cubes: a shared frame's owner copy against its backfiller's copy (both aligned) — per pixel, saved as
        uncert/uncert_<frame>.npy, with uncertainty.json (per frame, the session median)."""
        from loop_utils.metric_lock import real_frame_number
        vl = self.vl
        udir = os.path.join(vl.output_dir, "uncert")
        os.makedirs(udir, exist_ok=True)
        rep_path = os.path.join(vl.output_dir, "uncertainty.json")
        if os.path.exists(rep_path):
            try:
                prev = json.load(open(rep_path))
                vl._stac_check_fork_stamp(prev, rep_path)
                if prev.get("layout_sha256") == self.digest:
                    print(f"{LOG} uncertainty: resume ({len(prev.get('frames', {}))} frame(s))")
                    return
            except (ValueError, KeyError) as e:
                print(f"{LOG} uncertainty.json unreadable ({e}) — re-measuring")
        want = {g for g in range(self.n) if self.backfiller[g] >= 0}
        stash, per_frame = {}, {}
        for k, data in vl._stac_iter_chunks(self.al, range(self.K)):
            wp = _wp(data)
            cf = _conf(data, wp.shape[:3])
            for g in self.sets[k]:
                g = int(g)
                if g not in want or k not in (int(self.owner[g]), int(self.backfiller[g])):
                    continue
                i = self.local[k][g]
                row = (wp[i].astype(np.float32), cf[i].astype(np.float32))
                if g not in stash:
                    stash[g] = row
                    continue
                p_prev, c_prev = stash.pop(g)
                ok = (c_prev > 1e-5) & (row[1] > 1e-5)
                d = np.full(wp.shape[1:3], np.nan, np.float32)
                if ok.any():
                    d[ok] = np.linalg.norm(row[0][ok].astype(np.float64) - p_prev[ok].astype(np.float64), axis=1)
                num = real_frame_number(vl.img_list[g])
                np.save(os.path.join(udir, f"uncert_{num}.npy"), d)
                vals = d[np.isfinite(d)]
                per_frame[g] = {"frame": int(num), "median_m": float(np.median(vals)) if len(vals) else None,
                                "p90_m": float(np.percentile(vals, 90)) if len(vals) else None,
                                "n_px": int(len(vals)), "witnesses": 2}
            del data
        meds = [v["median_m"] for v in per_frame.values() if v["median_m"] is not None]
        session_med = float(np.median(meds)) if meds else 0.0
        with open(rep_path, "w") as f:
            json.dump(vl._stac_with_stamp(
                {"version": 1, "layout": "cubes", "layout_sha256": self.digest, "session_median_m": session_med,
                 "n_shared_frames": len(per_frame), "frames": {str(k): v for k, v in per_frame.items()},
                 "note": "a shared frame's owner copy against its backfiller's; frames predicted once carry the "
                         "session median"}), f, indent=1)
        print(f"{LOG} uncertainty: {len(per_frame)} shared frame(s), two-copy disagreement median "
              f"{session_med * 100:.2f} cm → uncert/, uncertainty.json")

    # 7. depth graph (the chain's _stac_depth_graph over the cubes' owned frames) ──────────────
    def depth_graph(self):
        if not self.cfg.get('depth_graph') or self.K < 2:
            return
        import inspect
        from loop_utils.loop_bridges import cfg_req
        from loop_utils.metric_lock import (apply_depth_correction, depth_graph_verdict, depth_pair_samples,
                                            depth_pair_source, pair_depth_relation, pair_relation_error,
                                            solve_depth_graph)
        vl, N = self.vl, self.n
        dg_path = os.path.join(vl.output_dir, "depth_graph.json")
        sol = None
        if os.path.exists(dg_path):
            try:
                prev = json.load(open(dg_path))
            except ValueError as e:
                prev = None
                print(f"[depth-graph] depth_graph.json unreadable ({e}) — remeasuring")
            if prev is not None:
                vl._stac_check_fork_stamp(prev, dg_path)
                if prev.get("layout_sha256") == self.digest:
                    sol = (np.asarray(prev["a"], np.float64), np.asarray(prev["b"], np.float64))
                    print("[depth-graph] resume: solution loaded from depth_graph.json")
        if sol is None:
            cache = {}
            for k, data in vl._stac_iter_chunks(self.al, range(self.K)):
                if data.get('_stac_depth_graph_applied'):
                    raise RuntimeError(f"[depth-graph] cube {k} already depth-corrected but depth_graph.json is "
                                       f"missing/stale — delete _tmp_results_aligned + pcd and re-run")
                wp = _wp(data)
                cf = _conf(data, wp.shape[:3])
                ext, Kin = _ext(data), _K(data)
                for i, g in enumerate(self.sets[k]):
                    g = int(g)
                    if int(self.owner[g]) == k:
                        c2w = self.pose_of(k, ext[i])
                        cache[g] = (wp[i].astype(np.float32), cf[i].astype(np.float32), np.linalg.inv(c2w), Kin[i])
                del data
            fit_offsets, holdout_offsets = (1, 2, 3, 5, 8, 12), (4, 10)
            zref = float(inspect.signature(depth_graph_verdict).parameters["zref"].default)
            meas, held, held_err, before_all = [], [], [], []
            for f in sorted(cache):
                src_f = depth_pair_source(cache[f][0], cache[f][1], seed=f)
                for d in fit_offsets + holdout_offsets:
                    g = f + d
                    if g not in cache:
                        continue
                    zs = depth_pair_samples(cache[f][0], cache[f][1], cache[g][0], cache[g][1], cache[g][2],
                                            cache[g][3], seed=f, src=src_f)
                    if zs is None:
                        continue
                    rel = pair_depth_relation(zs[0], zs[1])
                    if rel is None:
                        continue
                    al, be, before, _n = rel
                    (meas if d in fit_offsets else held).append((f, g, al, be))
                    if d in fit_offsets:
                        before_all.append(before)
                    else:
                        held_err.append(pair_relation_error(zs[0], zs[1], zref))
            del cache
            if len(meas) < N // 4 or len(held) < N // 8:
                print(f"[depth-graph] too thin ({len(meas)} fit / {len(held)} held-out pairs for {N} frames) — "
                      f"SKIPPING (nothing modified)")
                return
            gcfg = vl._stac_graph_cfg()
            conf = float(cfg_req(gcfg, "heldout_confidence", "graph"))
            fac = float(cfg_req(gcfg, "improvement_error_factor", "graph"))
            print(f"[depth-graph] {len(meas)} fit + {len(held)} held-out pairs — disagreement before "
                  f"{np.median(before_all) * 100:.2f}% median")
            ladder = []
            a, b = np.ones(N), np.zeros(N)
            verdict, model = "SKIP", None
            for rung, kw in (("affine", {}), ("scale-only", {"scale_only": True})):
                a_r, b_r = solve_depth_graph(meas, N, **kw)
                v = depth_graph_verdict(a_r, b_r, meas, held, held_err=held_err, error_factor=fac, confidence=conf)
                ladder.append(dict(v, model=rung))
                print(f"[depth-graph] {rung}: held-out {v['med_before'] * 100:.2f}% -> {v['med_after'] * 100:.2f}% | "
                      f"a[{a_r.min():.4f},{a_r.max():.4f}] b[{b_r.min() * 100:.1f},{b_r.max() * 100:.1f}]cm | "
                      f"bounded={v['bounded']} improves={v['improves']} ({v['decision']['reason']})")
                if v["bounded"] and v["improves"]:
                    a, b = a_r, b_r
                    verdict, model = "APPLY", rung
                    break
            print(f"[depth-graph] ✅ {model} model earned the correction" if verdict == "APPLY" else
                  "[depth-graph] ⛔ no rung of the model ladder explains this scan's depth disagreement — geometry "
                  "left UNTOUCHED")
            with open(dg_path, "w") as f_:
                json.dump({**vl._stac_with_stamp({}), "layout": "cubes", "layout_sha256": self.digest,
                           "verdict": verdict, "model": model, "ladder": ladder, "a": a.tolist(), "b": b.tolist(),
                           "n_pairs_fit": len(meas), "n_pairs_holdout": len(held),
                           "pair_disagreement_before_pct": float(np.median(before_all) * 100)}, f_, indent=1,
                          default=float)
            sol = (a, b)
        a, b = sol
        acfg = vl._stac_authority_cfg()
        if acfg:
            fa = float(np.max(np.abs(np.log(np.asarray(a))))) / float(acfg["depth_graph_max_log_a"])
            fb = float(np.max(np.abs(np.asarray(b)))) / float(acfg["depth_graph_max_b_m"])
            fr = max(fa, fb)
            vl._stac_authority_record("depth_graph", fr, fr > float(acfg["saturation_warn"]),
                                      {"max_log_a": float(acfg["depth_graph_max_log_a"]),
                                       "max_b_m": float(acfg["depth_graph_max_b_m"]),
                                       "used_max_log_a": float(np.max(np.abs(np.log(np.asarray(a))))),
                                       "used_max_b_m": float(np.max(np.abs(np.asarray(b))))})
        moved_total = 0
        for k in range(self.K):
            F = self.sets[k]
            if all(abs(a[int(g)] - 1.0) < 1e-9 and abs(b[int(g)]) < 1e-12 for g in F):
                continue
            p = _path(self.al, k)
            data = np.load(p, allow_pickle=True).item()
            if data.get('_stac_depth_graph_applied'):
                print(f"[depth-graph] cube {k}: already applied — skipped")
                continue
            wp = np.asarray(data['world_points'])
            lead = wp.ndim == 5
            if lead:
                wp = wp[0]
            dep, ext = np.asarray(data['depth']), _ext(data)
            moved = 0
            for i, g in enumerate(F):
                g = int(g)
                if abs(a[g] - 1.0) < 1e-9 and abs(b[g]) < 1e-12:
                    continue
                c2w = self.pose_of(k, ext[i])
                wp[i], dep[i] = apply_depth_correction(wp[i], dep[i], c2w[:3, 3], a[g], b[g])
                moved += 1
            data['world_points'] = wp[None] if lead else wp
            data['depth'] = dep
            data['_stac_depth_graph_applied'] = True
            np.save(p, data)
            moved_total += moved
            print(f"[depth-graph] cube {k}: {moved}/{len(F)} frames re-depthed")
        print(f"[depth-graph] {moved_total} frame copy(ies) re-depthed" if moved_total else
              "[depth-graph] no frame needed re-depthing — geometry untouched")

    # 8. outputs ───────────────────────────────────────────────────────────────────────────────
    def outputs(self):
        """PLY + origins per cube: a cube writes the frames it OWNS and, of a frame it is the backfiller of, exactly
        the pixels the owner drops (frozen thresholds — a pixel never enters twice)."""
        from loop_utils.metric_lock import backfill_mask, real_frame_number
        vl = self.vl
        own = bool(self.cfg.get('frame_ownership'))
        if not own:
            raise RuntimeError(f"{LOG} Model.frame_ownership is off — cubes share frames, every shared frame would "
                               f"enter the cloud once per cube")
        bf_on = bool(self.cfg.get('ownership_backfill'))
        thr, owner_conf = {}, {}
        for k, data in vl._stac_iter_chunks(self.al, range(self.K)):
            S = len(self.sets[k])
            cf = np.asarray(data['world_points_conf']).reshape(S, -1).astype(np.float32)
            rows = [i for i, g in enumerate(self.sets[k]) if int(self.owner[int(g)]) == k]
            thr[k] = None if not rows else vl._stac_conf_threshold(cf[rows].reshape(-1))
            for i, g in enumerate(self.sets[k]):
                g = int(g)
                if bf_on and int(self.owner[g]) == k and self.backfiller[g] >= 0:
                    owner_conf[g] = cf[i].copy()
            del data
        for k, data in vl._stac_iter_chunks(self.al, range(self.K)):
            F = self.sets[k]
            S = len(F)
            cf = np.asarray(data['world_points_conf']).reshape(S, -1).astype(np.float32).copy()
            kept, bf_px = 0, 0
            for i, g in enumerate(F):
                g = int(g)
                o = int(self.owner[g])
                if o == k:
                    kept += 1
                elif bf_on and int(self.backfiller[g]) == k and g in owner_conf:
                    m = backfill_mask(owner_conf[g], thr[o])
                    cf[i] = np.where(m, cf[i], 0.0)
                    bf_px += int((cf[i] > 1e-5).sum())
                else:
                    cf[i] = 0.0
            confs = cf.reshape(-1)
            t = thr[k] if thr[k] is not None else vl._stac_conf_threshold(confs)
            ratio = float(self.cfg['Pointcloud_Save']['sample_ratio'])
            if ratio != 1.0:
                raise RuntimeError(f"Model.Pointcloud_Save.sample_ratio is {ratio:g} — the STAC writer needs 1.0")
            pts = _wp(data).reshape(-1, 3)
            cols = self.images_u8(data['images']).transpose(0, 2, 3, 1).reshape(-1, 3)
            self.save_ply(points=pts, colors=cols, confs=confs, output_path=os.path.join(vl.pcd_dir, f"{k}_pcd.ply"),
                          conf_threshold=t, sample_ratio=1.0)
            wp = _wp(data)
            H, W = wp.shape[1:3]
            c32 = confs.astype(np.float32)
            surv = np.flatnonzero((c32 >= t) & (c32 > 1e-5))
            fl, within = surv // (H * W), surv % (H * W)
            real = np.asarray([real_frame_number(vl.img_list[int(g)]) for g in F], np.int32)
            np.savez_compressed(os.path.join(vl.pcd_dir, f"{k}_origins.npz"), frame_global=real[fl],
                                pixel_row=(within // W).astype(np.int16), pixel_col=(within % W).astype(np.int16),
                                confidence=c32[surv].astype(np.float32),
                                scaled_resolution=np.array([H, W], np.int32))
            print(f"{LOG} cube {k}: writes {kept}/{S} owned frame(s)" + (f" + backfills {bf_px:,} px" if bf_px else "")
                  + f" — {len(surv):,} points (PLY + origins 1:1)")
            del data

    # 9. poses ─────────────────────────────────────────────────────────────────────────────────
    def camera_poses(self):
        from loop_utils.camera_files import write_camera_files
        vl = self.vl
        poses = [None] * self.n
        intr = [None] * self.n
        for k, data in vl._stac_iter_chunks(self.al, range(self.K)):
            ext, Kin = _ext(data), _K(data)
            for i, g in enumerate(self.sets[k]):
                g = int(g)
                if int(self.owner[g]) != k:
                    continue
                poses[g] = self.pose_of(k, ext[i])
                if Kin is not None:
                    intr[g] = Kin[i]
            del data
        missing = [g for g in range(self.n) if poses[g] is None]
        if missing:
            raise RuntimeError(f"{LOG} {len(missing)} keyframe(s) got no pose from their owner cube (first {missing[0]})")
        w = write_camera_files(vl.output_dir, poses, intr if all(x is not None for x in intr) else None)
        print(f"{LOG} camera poses → {w['poses']}")
        with open(os.path.join(vl.output_dir, 'camera_poses.ply'), 'w') as f:
            f.write('ply\nformat ascii 1.0\n')
            f.write(f'element vertex {len(poses)}\n')
            f.write('property float x\nproperty float y\nproperty float z\n')
            f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
            for p in poses:
                f.write(f'{p[0, 3]} {p[1, 3]} {p[2, 3]} 255 0 0\n')


def process_cubes(vl, *, images_to_u8, images_u8, save_ply, apply_sim3_direct):
    """The cube run (module docstring), in place of VGGT_Long.process_long_sequence."""
    run = CubeRun(vl, images_to_u8, images_u8, save_ply, apply_sim3_direct)
    sizes = [len(F) + len(c) for F, c in zip(run.sets, vl.chunk_context)]
    print(f"{LOG} {run.K} cube(s) over {run.n} keyframes: {min(sizes)}–{max(sizes)} Omega views each, "
          f"{len(run.pairs)} pair(s) sharing frames; layout {run.digest[:12]}")
    for k in range(run.K):
        print(f"[Progress]: {k}/{run.K - 1}")
        run.infer(k)
        torch.cuda.empty_cache()
        run.witness_pass(k)
        torch.cuda.empty_cache()
    if run.revisits and run.K > 1:
        run.bridges()          # the revisit bridges need the model: before it is released
    del vl.model
    torch.cuda.empty_cache()
    if run.load_graph():
        print(f"{LOG} the graph of this layout and stamp is on disk (chunk_graph.json) — resumed past it")
    else:
        if any(os.path.exists(_path(run.al, k)) for k in range(run.K)):
            raise RuntimeError(f"{LOG} some cubes are placed and some are not, without a graph of this layout — "
                               f"delete _tmp_results_aligned and re-run")
        run.metric_lock()
        run.graph(run.seams() if run.K > 1 else [])
    run.apply()
    if vl._stac_graph_cfg() is not None:
        run.uncertainty()
    run.depth_graph()
    run.outputs()
    run.camera_poses()
    print(f"{LOG} done")
