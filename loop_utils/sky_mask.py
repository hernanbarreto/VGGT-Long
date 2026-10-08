"""The sky mask of the fork: a PINNED model and a cache keyed on what makes a mask
(docs/plan_determinismo.md point 20, audit omega-20).

Before: ``skyseg.onnx`` was downloaded from huggingface.co/JianyuanWang/skyseg ``resolve/main``
(no revision: whatever 'main' held that day) when missing, and every frame's mask was cached
as ``sky_masks/<frame name>.png`` — a cache keyed on the frame's NAME alone, kept by the map
worker across new chunk plans, new models and re-extracted frames. The masks zero
``world_points_conf``, which every ``conf > 1e-5`` gate downstream reads.

Now:
- the model is the file vendored next to the fork, verified against :data:`SKYSEG_SHA256` at
  every load; a missing or different file FAILS (nothing is downloaded);
- a mask is cached under ``<cache root>/<model sha>-<segmenter code sha>/<frame>_<image sha>.png``
  — reused only for the same model, the same segmenting code and the same image bytes;
- the segmenter runs on onnxruntime's CPU provider with ONE thread (one kernel order on every
  machine), and the mask is always READ BACK from its lossless PNG — fresh or cached, one path;
- anything that fails STOPS the run: a frame silently left unmasked is another cloud.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from loop_utils.stac_repro import FORK_DIR, repro

# sha256 of the sky segmenter every validated run masked the sky with: the copy vendored at
# vendor/VGGT-Long/skyseg.onnx (176 MB, gitignored), measured 2026-10-07. Its origin is
# huggingface.co/JianyuanWang/skyseg (file skyseg.onnx); a copy fetched from there is accepted
# only when it has exactly these bytes.
SKYSEG_SHA256 = "ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39"
SKYSEG_FILE = "skyseg.onnx"
SKYSEG_ORIGIN = "huggingface.co/JianyuanWang/skyseg (file skyseg.onnx)"
# the segmenter's own threshold is in visual_util.segment_sky (255 = non-sky); a mask pixel
# above this reads as non-sky — the PNG holds exactly 0 or 255, so any value in (0, 255) splits
# them identically (the fork's historical 0.1)
_NONSKY_BAR = 0.1

_SHA_CACHE: Dict[tuple, str] = {}


def _sha256_cached(path: str) -> str:
    """sha256 of a file, cached per (path, mtime, size) within the process (the model is
    hashed for its pin, the run stamp and the cache key — once)."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = repro().sha256_file(path)
    return _SHA_CACHE[key]


def skyseg_path(fork_dir: str = FORK_DIR, expected_sha256: str = SKYSEG_SHA256) -> str:
    """The pinned sky segmenter: ``<fork>/skyseg.onnx`` with exactly ``expected_sha256``.
    Missing or different: FAILS naming the expected digest and the file's origin."""
    p = os.path.join(fork_dir, SKYSEG_FILE)
    if not os.path.isfile(p):
        raise RuntimeError(f"{p} is missing — the sky segmenter is pinned (sha256 "
                           f"{expected_sha256}, from {SKYSEG_ORIGIN}); restore that exact file "
                           f"(plan point 20: nothing is downloaded)")
    got = _sha256_cached(p)
    if got != expected_sha256:
        raise RuntimeError(f"{p} has sha256 {got}, the pinned sky segmenter is {expected_sha256} "
                           f"— another model would mask another sky (plan point 20)")
    return p


def segmenter_code_file() -> str:
    """The module whose code turns the model's output into the mask (resize, threshold)."""
    return os.path.join(FORK_DIR, "loop_utils", "visual_util.py")


def mask_cache_dir(cache_root: str, model_sha256: str, code_sha256: str) -> str:
    """The cache directory of one (model, segmenting code) pair."""
    return os.path.join(cache_root, f"{model_sha256[:16]}-{code_sha256[:16]}")


def mask_path(cache_dir: str, image_path: str) -> str:
    """A frame's cached mask: its name AND the sha256 of its bytes."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(cache_dir, f"{stem}_{repro().sha256_file(image_path)[:16]}.png")


def cpu_session(model_path: str):
    """onnxruntime session of the pinned model: CPU provider, one intra- and one inter-op
    thread (one kernel order on every machine)."""
    import onnxruntime
    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(model_path, sess_options=so,
                                        providers=["CPUExecutionProvider"])


def _prune_stale(cache_root: str, keep_dir: str, log) -> None:
    """Delete what this cache can never reuse (the session's disk is a declared constraint):
    masks of another model or segmenting code (other directories) and the legacy masks keyed
    on the frame name alone (PNG files directly under the root)."""
    import shutil
    keep = os.path.basename(keep_dir)
    gone = 0
    for name in sorted(os.listdir(cache_root)):
        p = os.path.join(cache_root, name)
        if name == keep:
            continue
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p)
            gone += 1
        elif name.lower().endswith(".png"):
            os.remove(p)
            gone += 1
    if gone:
        log(f"[STAC sky] {gone} stale sky-mask cache entr{'y' if gone == 1 else 'ies'} of another "
            f"model / code / naming deleted (never reusable)")


def apply_sky_masks(conf: np.ndarray, image_paths: Sequence[str], cache_root: str, *,
                    model_path: str, segment: Callable[[str, object, str], object],
                    session_factory: Callable[[str], object] = cpu_session,
                    read_mask: Optional[Callable[[str], Optional[np.ndarray]]] = None,
                    log: Callable[[str], object] = print) -> Dict[str, object]:
    """Zero ``conf[i]`` (S, H, W — modified in place) wherever frame ``image_paths[i]`` is sky.
    ``segment(image_path, session, mask_path)`` writes a frame's mask (visual_util.segment_sky);
    ``model_path`` must be the pinned model (:func:`skyseg_path`). Returns what was done: the
    cache directory, how many masks were computed and how many reused."""
    import cv2
    if read_mask is None:
        def read_mask(p):
            return cv2.imread(p, cv2.IMREAD_GRAYSCALE)        # 255 = non-sky, 0 = sky
    S, H, W = conf.shape
    paths = list(image_paths)
    if len(paths) < S:
        raise RuntimeError(f"[STAC sky] {S} predicted frame(s) but {len(paths)} image path(s) — a "
                           f"frame without its image cannot be masked")
    model_sha = _sha256_cached(model_path)
    code_sha = repro().sha256_file(segmenter_code_file())
    cdir = mask_cache_dir(cache_root, model_sha, code_sha)
    os.makedirs(cdir, exist_ok=True)
    _prune_stale(cache_root, cdir, log)
    session = None
    n_new = n_reused = 0
    for i, p in enumerate(paths[:S]):
        mp = mask_path(cdir, p)
        if os.path.exists(mp):
            n_reused += 1
        else:
            if session is None:
                session = session_factory(model_path)
            segment(p, session, mp)
            n_new += 1
        sky = read_mask(mp)
        if sky is None:
            raise RuntimeError(f"[STAC sky] {mp} unreadable — the sky mask of "
                               f"{os.path.basename(p)} cannot be applied")
        if sky.shape[0] != H or sky.shape[1] != W:
            sky = cv2.resize(sky, (W, H), interpolation=cv2.INTER_NEAREST)
        conf[i] *= (sky > _NONSKY_BAR).astype(conf.dtype)     # 1 = keep (non-sky), 0 = sky
    log(f"[STAC sky] masked sky on {S}/{S} chunk frames (model "
        f"{model_sha[:12]}…, {n_new} computed, {n_reused} reused from {cdir})")
    return {"cache_dir": cdir, "model_sha256": model_sha, "code_sha256": code_sha,
            "n_computed": n_new, "n_reused": n_reused}
