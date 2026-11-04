"""STAC (USER 2026-11-04): CONTEXT views of a chunk — frames Omega infers WITH the chunk's
keyframes and whose outputs it drops before anything is saved.

The server (intake/turn_frames.py) chooses the TURN FRAMES between consecutive keyframes and the
map worker passes them per chunk as ``Model.chunk_context`` (one list of file names per
``Model.chunk_ranges`` entry). A chunk's inference then sees its keyframes and its context views
in frame order; the chunk file keeps the keyframes' slots only, so every reader of the chunks —
the seams, the loops, the records, F0–F6 — sees exactly the keyframes it always saw (the loop
bridges' extra frames are the precedent: they never enter the chain either)."""

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np


class ContextViewsError(ValueError):
    pass


def frame_number(path: str) -> int:
    """The video frame number of a frame file: its stem (000123.jpg → 123)."""
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    try:
        return int(stem)
    except ValueError as e:
        raise ContextViewsError(f"{path!r} is not a numbered frame file") from e


def check_chunk_context(chunk_context, chunk_indices: Sequence[Sequence[int]],
                        img_list: Sequence[str], img_dir: str) -> List[List[str]]:
    """``Model.chunk_context`` resolved to paths, per chunk, checked against this run's layout:
    one list per chunk; every file a frame of ``img_dir`` that is NOT a keyframe and lies
    strictly between the chunk's first and last keyframe (a context view never changes the
    chunk's first view, its reference frame); no file twice in a chunk. None → no context."""
    n = len(chunk_indices)
    if chunk_context is None:
        return [[] for _ in range(n)]
    if not isinstance(chunk_context, (list, tuple)) or len(chunk_context) != n:
        raise ContextViewsError(f"Model.chunk_context must hold one list per chunk ({n}), got "
                                f"{type(chunk_context).__name__} of "
                                f"{len(chunk_context) if isinstance(chunk_context, (list, tuple)) else '?'}")
    kf_names = {os.path.basename(p) for p in img_list}
    out: List[List[str]] = []
    for k, ((s, e), files) in enumerate(zip(chunk_indices, chunk_context)):
        files = list(files or [])
        if len(set(files)) != len(files):
            raise ContextViewsError(f"chunk {k}: a context frame is listed twice")
        lo, hi = frame_number(img_list[int(s)]), frame_number(img_list[int(e) - 1])
        paths = []
        for fn in files:
            fn = str(fn)
            if os.path.basename(fn) != fn:
                raise ContextViewsError(f"chunk {k}: context frame {fn!r} must be a file name")
            if fn in kf_names:
                raise ContextViewsError(f"chunk {k}: context frame {fn} is a keyframe")
            if not lo < frame_number(fn) < hi:
                raise ContextViewsError(f"chunk {k}: context frame {fn} is not strictly between "
                                        f"its first and last keyframe ({lo}, {hi})")
            p = os.path.join(img_dir, fn)
            if not os.path.isfile(p):
                raise ContextViewsError(f"chunk {k}: context frame {p} does not exist")
            paths.append(p)
        out.append(sorted(paths, key=frame_number))
    return out


def merge_views(keyframe_paths: Sequence[str], context_paths: Sequence[str]
                ) -> Tuple[List[str], List[int]]:
    """(the views of the inference in frame order, the slots of the keyframes among them)."""
    tagged = [(frame_number(p), 0, i, p) for i, p in enumerate(keyframe_paths)]
    tagged += [(frame_number(p), 1, i, p) for i, p in enumerate(context_paths)]
    nums = [t[0] for t in tagged]
    if len(set(nums)) != len(nums):
        raise ContextViewsError("a context view repeats a frame of the chunk")
    tagged.sort()
    views = [t[3] for t in tagged]
    keep = [slot for slot, t in enumerate(tagged) if t[1] == 0]
    if [tagged[s][2] for s in keep] != list(range(len(keyframe_paths))):
        raise ContextViewsError("the chunk's keyframes are not in frame order")
    if keep and keep[0] != 0:
        raise ContextViewsError("a context view precedes the chunk's first keyframe")
    return views, keep


def keep_keyframe_slots(predictions: Dict, keep: Sequence[int], n_views: int) -> Dict:
    """In place: every per-view array of one inference (first axis = ``n_views``) reduced to the
    keyframes' slots. An array whose first axis is not the view count fails — nothing is
    sliced by guess."""
    idx = np.asarray(list(keep), dtype=np.int64)
    for key, v in list(predictions.items()):
        if not isinstance(v, np.ndarray) or v.ndim == 0:
            continue
        if v.shape[0] != int(n_views):
            raise ContextViewsError(f"prediction {key!r} has {v.shape[0]} entries on its first "
                                    f"axis, the inference had {n_views} views")
        predictions[key] = v[idx]
    return predictions
