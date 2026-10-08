"""The fork's pose and intrinsics files (docs/plan_determinismo.md point 45, audit X-01).

camera_poses.txt (one c2w 4x4 per keyframe, 16 values row-major) and intrinsic.txt
('fx fy cx cy' per keyframe) are re-read by every stage downstream (scale_align, orient, F0-F6,
the TSDF, the certification). The fork wrote them with ``str()`` of float32 values — text that
np.loadtxt reads back as a DIFFERENT float64 — so every stage started from rounded poses. They
are now written as float64 text that reads back EXACTLY (repr of each value, through the
server's repro.write_poses_exact / write_intrinsics_exact — one writer for the whole pipeline),
atomically."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from loop_utils.stac_repro import repro

POSES_NAME = "camera_poses.txt"
INTRINSICS_NAME = "intrinsic.txt"


def write_camera_files(output_dir: str, poses: Sequence[np.ndarray],
                       intrinsics: Optional[Sequence[np.ndarray]] = None) -> dict:
    """camera_poses.txt from ``poses`` (N x (4,4) c2w) and, when given, intrinsic.txt from
    ``intrinsics`` (N x (3,3) K → fx fy cx cy), each value widened to float64 and written as
    its exact repr. A keyframe without a pose (None) FAILS: a file with a hole is not a
    trajectory. Returns the paths written."""
    if any(p is None for p in poses):
        missing = [i for i, p in enumerate(poses) if p is None]
        raise RuntimeError(f"{len(missing)} keyframe(s) without a pose (e.g. {missing[:5]}) — "
                           f"camera_poses.txt would have holes")
    R = repro()
    out = {"poses": str(R.write_poses_exact(
        os.path.join(output_dir, POSES_NAME),
        np.stack([np.asarray(p, np.float64) for p in poses])))}
    if intrinsics is not None:
        if any(K is None for K in intrinsics):
            raise RuntimeError("a keyframe without intrinsics — intrinsic.txt would have holes")
        out["intrinsics"] = str(R.write_intrinsics_exact(
            os.path.join(output_dir, INTRINSICS_NAME),
            np.array([[float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
                      for K in intrinsics], np.float64)))
    return out
