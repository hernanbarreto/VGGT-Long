# STAC patch — per-chunk METRIC LOCK for the chunked Omega pipeline.
#
# Omega chunks are up-to-scale, each with its OWN arbitrary scale. The old windowed
# pipeline let the Sim(3) overlap alignment negotiate the relative scales — chunks
# disagreed by ±18-50% and the chained scale error produced double surfaces ("onion")
# and metric drift. This module locks EVERY chunk to metric BEFORE alignment, using
# isolated DA3 metric depth on a few anchor frames inside the chunk:
#
#     s_chunk = median over anchors of median(da3_depth / omega_depth)
#
# With all chunks metric, the overlap alignment runs as SE(3) (config
# Model.using_sim3: false) — scale is no longer a degree of freedom — and loop-closure
# constraints come out with relative scale ≈ 1 by construction.
#
# Depth convention: omega depth here is the chunk's per-frame 'depth' (camera-forward
# units, same scale as world_points/poses). The ratio is computed on the near-field
# band (lowest-quartile omega depth) where monocular metric depth is most reliable —
# the same policy as server/reconstruction/scale_align.py, which remains the global
# verifier after alignment.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import re

import cv2
import numpy as np

_FRAME_NUM = re.compile(r"(\d+)")


def real_frame_number(image_path):
    """Numeric stem of a processed frame's filename (the REAL video frame index)."""
    m = _FRAME_NUM.search(os.path.basename(str(image_path)))
    return int(m.group(1)) if m else -1


def anchor_ratio(omega_depth, da3_depth, conf=None, near_frac=0.25, min_px=50):
    """median(da3/omega) over pixels valid in BOTH maps, restricted to the near band
    (omega depth <= its `near_frac` quantile). Returns None when starved."""
    om = np.asarray(omega_depth, np.float32).squeeze()
    da = np.asarray(da3_depth, np.float32).squeeze()
    if om.ndim != 2 or da.ndim != 2 or om.size == 0 or da.size == 0:
        return None
    if da.shape != om.shape:
        da = cv2.resize(da, (om.shape[1], om.shape[0]), interpolation=cv2.INTER_LINEAR)
    m = np.isfinite(om) & np.isfinite(da) & (om > 1e-6) & (da > 1e-6)
    if conf is not None:
        cf = np.asarray(conf, np.float32).squeeze()
        if cf.shape == om.shape:
            m &= cf > 1e-5          # sky is masked by zeroing conf — exclude it
    if m.sum() < min_px:
        return None
    near = om <= np.quantile(om[m], near_frac)
    mm = m & near
    if mm.sum() < min_px:           # near band too thin → all valid pixels
        mm = m
    return float(np.median(da[mm] / om[mm]))


def chunk_scale(chunk_data, frame_numbers, anchor_dir, near_frac=0.25):
    """Metric scale for one chunk from the DA3 anchors that fall inside it.

    chunk_data: the chunk's prediction dict ('depth', 'world_points_conf', ...).
    frame_numbers: REAL frame number of each local frame (len == S).
    anchor_dir: dir holding frame_<num>.npz (keys: depth [, conf]) from isolated DA3.

    Returns (s, n_anchors, ratios) — s is None when no anchor lands in the chunk.
    """
    depth = np.asarray(chunk_data["depth"])
    conf3 = chunk_data.get("world_points_conf")
    ratios = []
    for local, num in enumerate(frame_numbers):
        npz_path = os.path.join(str(anchor_dir), f"frame_{int(num)}.npz")
        if not os.path.exists(npz_path):
            continue
        z = np.load(npz_path)
        if "depth" not in z:
            continue
        conf = None
        if conf3 is not None:
            c = np.asarray(conf3)
            conf = c[local] if local < c.shape[0] else None
        r = anchor_ratio(depth[local], z["depth"], conf=conf, near_frac=near_frac)
        if r is not None and np.isfinite(r) and r > 0:
            ratios.append(r)
    if not ratios:
        return None, 0, []
    return float(np.median(ratios)), len(ratios), ratios


def apply_scale(chunk_data, s):
    """Scale a chunk's whole metric IN PLACE: world_points, per-frame depth and the
    camera translations. Rotations and intrinsics are scale-invariant."""
    s = float(s)
    if chunk_data.get("world_points") is not None:
        chunk_data["world_points"] = np.asarray(chunk_data["world_points"]) * s
    if chunk_data.get("depth") is not None:
        chunk_data["depth"] = np.asarray(chunk_data["depth"]) * s
    ext = chunk_data.get("extrinsic")
    if ext is not None:
        ext = np.asarray(ext).copy()
        ext[..., :3, 3] *= s
        chunk_data["extrinsic"] = ext
    return chunk_data
