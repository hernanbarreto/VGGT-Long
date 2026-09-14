"""The SALAD loop detector's decision thresholds are config keys (STAC):
``Loop.SALAD.min_gap`` replaces the vendor's hard-coded ``> 10`` — in a
keyframe-pinned image list one index is ~14 video frames, so the gap, the NMS
window and the similarity floor must be set for KEYFRAMES from config.yaml
(loops.salad), never by literals."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _detector(min_gap, similarity_threshold, nms_threshold):
    from LoopModels.LoopModel import LoopDetector
    cfg = {"Weights": {"SALAD": "unused.ckpt"},
           "Loop": {"SALAD": {"image_size": [336, 336], "batch_size": 32,
                              "similarity_threshold": similarity_threshold, "top_k": 5,
                              "use_nms": nms_threshold > 0, "nms_threshold": nms_threshold,
                              "min_gap": min_gap}},
           "Model": {"frame_stride": 1}}
    det = LoopDetector(image_dir="/nonexistent", output="/dev/null", config=cfg)
    # 40 unit descriptors on a circle: frames i and i+20 identical (a revisit),
    # frames 5 apart nearly identical (odometry neighbours)
    n = 40
    ang = np.array([2 * np.pi * (i % 20) / 20 for i in range(n)])
    D = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)
    det.descriptors = torch.from_numpy(D)
    det.image_paths = [Path(f"{i:06d}.jpg") for i in range(n)]
    return det


def test_min_gap_is_read_from_config_and_applied():
    det = _detector(min_gap=15, similarity_threshold=0.99, nms_threshold=0)
    pairs = det.find_loop_closures()
    assert pairs, "identical descriptors 20 apart must be proposed"
    assert all(abs(i - j) >= 15 for i, j, _ in pairs), pairs
    assert all(abs(i - j) == 20 for i, j, _ in pairs), pairs
    det2 = _detector(min_gap=21, similarity_threshold=0.99, nms_threshold=0)
    assert det2.find_loop_closures() == [], "a gap floor above the revisit distance proposes nothing"


def test_top_k_is_taken_among_non_local_frames():
    """pccr 2026-09-13: with the vendor's search the top-5 of every keyframe
    were its odometry neighbours (~0.98), the start↔end revisit at ~0.9 never
    entered the list → 0 proposals. The top-k must be ranked among the
    NON-local frames only."""
    from LoopModels.LoopModel import LoopDetector
    cfg = {"Weights": {"SALAD": "unused.ckpt"},
           "Loop": {"SALAD": {"image_size": [336, 336], "batch_size": 32,
                              "similarity_threshold": 0.8, "top_k": 5, "use_nms": False,
                              "nms_threshold": 0, "min_gap": 11}},
           "Model": {"frame_stride": 1}}
    det = LoopDetector(image_dir="/nonexistent", output="/dev/null", config=cfg)
    # a smooth walk: adjacent descriptors nearly identical (small angular step),
    # and the walk's end comes back near its start (a revisit at ~0.9, i.e. a
    # weaker similarity than any odometry neighbour)
    n = 60
    step = np.deg2rad(2.0)                       # adjacent cosine ≈ 0.9994
    ang = np.arange(n) * step
    ang[-6:] = np.arccos(0.9) + np.arange(6) * step * 0.2     # tail revisits the start at cos ≈ 0.9
    D = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)
    det.descriptors = torch.from_numpy(D)
    det.image_paths = [Path(f"{i:06d}.jpg") for i in range(n)]
    pairs = det.find_loop_closures()
    assert pairs, "the start↔end revisit must be proposed although every odometry neighbour is more similar"
    assert all(abs(i - j) >= 11 for i, j, _ in pairs)
    assert any(j <= 2 and i >= n - 6 for i, j, _ in pairs), pairs      # (later, earlier) ordering


def test_missing_min_gap_key_fails_loudly():
    from LoopModels.LoopModel import LoopDetector
    cfg = {"Weights": {"SALAD": "unused.ckpt"},
           "Loop": {"SALAD": {"image_size": [336, 336], "batch_size": 32, "similarity_threshold": 0.7,
                              "top_k": 5, "use_nms": True, "nms_threshold": 3}},
           "Model": {"frame_stride": 1}}
    try:
        LoopDetector(image_dir="/nonexistent", output="/dev/null", config=cfg)
    except KeyError as e:
        assert "min_gap" in str(e)
    else:
        raise AssertionError("a missing Loop.SALAD.min_gap must not default silently")
