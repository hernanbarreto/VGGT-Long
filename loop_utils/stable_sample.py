"""Subsamples chosen by a STABLE KEY of each element (docs/plan_determinismo.md, point 11).

Every robust fit of the fork caps its sample: ``default_rng(seed).choice(len(valid), k)``.
That draw is seeded, so identical inputs gave identical bits — but WHICH elements it picks is
a function of the length of the valid set and of every element's position in it. One pixel
crossing a confidence threshold (a resumed npy, another card's last bit) shifted every
position after it and re-drew the WHOLE sample: millimetres of seam R/t, bridge sigma,
held-out residuals, enough to flip a knife-edge decision downstream (audit omega-11).

Here membership is decided per element, by a hash of ITS OWN key: a pixel (frame, flat
index) is in the sample when its hash ranks among the k smallest of the valid set. A pixel
entering or leaving the valid set moves at most one other element across the boundary; every
other choice stays put. Integer arithmetic only (SplitMix64's finaliser, wrapping uint64), so
the same keys give the same choice on every machine.

Keys:
- :func:`pixel_keys` — (frame key, flat pixel index): the identity of a pixel of one frame.
  The frame key is the caller's namespace (the global keyframe index, a pair id ...).
- :func:`row_keys` — the float64 bits of each row of the given arrays: the identity of a
  correspondence by its own values, for callers that hold no pixel identity (a server caller
  of robust_rigid / robust_sim3). Stable against other rows entering or leaving.
"""

from __future__ import annotations

import numpy as np

_U64 = np.uint64
# SplitMix64 (Steele, Lea & Flood, "Fast splittable pseudorandom number generators", OOPSLA
# 2014; the finaliser of Java's SplittableRandom): the published constants of the mixer,
# not tunables — any odd constants would do, these are the ones with the published avalanche.
_GAMMA = _U64(0x9E3779B97F4A7C15)
_M1 = _U64(0xBF58476D1CE4E5B9)
_M2 = _U64(0x94D049BB133111EB)
# the frame part of a pixel key occupies the high 32 bits: a flat pixel index must fit in the
# low 32 (a 4 Gpx frame), a frame key in the high 32 (asserted, never truncated silently)
_PIXEL_BITS = 32


def mix64(x) -> np.ndarray:
    """SplitMix64 finaliser of uint64 keys (vectorised, wrapping arithmetic)."""
    z = np.asarray(x, dtype=_U64) + _GAMMA
    with np.errstate(over="ignore"):
        z = (z ^ (z >> _U64(30))) * _M1
        z = (z ^ (z >> _U64(27))) * _M2
    return z ^ (z >> _U64(31))


def _salted(keys, salt) -> np.ndarray:
    k = np.asarray(keys, dtype=_U64).ravel()
    if salt is None or int(salt) == 0:
        return mix64(k)
    return mix64(k ^ mix64(np.asarray([int(salt) & 0xFFFFFFFFFFFFFFFF], dtype=_U64))[0])


def pixel_keys(frame_key, flat_idx) -> np.ndarray:
    """uint64 key of each pixel ``flat_idx`` (flat index inside its frame) of frame
    ``frame_key`` (scalar or one per pixel)."""
    f = np.asarray(frame_key, dtype=np.int64)
    p = np.asarray(flat_idx, dtype=np.int64)
    if p.size and (int(p.min()) < 0 or int(p.max()) >= (1 << _PIXEL_BITS)):
        raise ValueError("pixel_keys: a flat pixel index outside [0, 2**32)")
    if f.size and (int(f.min()) < 0 or int(f.max()) >= (1 << (64 - _PIXEL_BITS))):
        raise ValueError("pixel_keys: a frame key outside [0, 2**32)")
    return (f.astype(_U64) << _U64(_PIXEL_BITS)) | p.astype(_U64)


def block_pixel_keys(first_frame_key, flat_idx_in_block, frame_px) -> np.ndarray:
    """Keys of pixels given by their flat index in a block of consecutive frames
    (frame ``first_frame_key + i`` holds indices [i * frame_px, (i + 1) * frame_px))."""
    q = np.asarray(flat_idx_in_block, dtype=np.int64)
    n = int(frame_px)
    return pixel_keys(int(first_frame_key) + q // n, q % n)


def row_keys(*arrays) -> np.ndarray:
    """One uint64 key per row from the float64 bits of the rows of ``arrays`` (same length):
    a row's key depends on its own values only."""
    cols = []
    n = None
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        a = a.reshape(len(a), -1)
        if n is None:
            n = len(a)
        elif len(a) != n:
            raise ValueError("row_keys: arrays of different lengths")
        cols.append(a.view(_U64))
    if not cols:
        raise ValueError("row_keys: nothing to key")
    k = np.zeros(n, dtype=_U64)
    for c in cols:
        for j in range(c.shape[1]):
            k = mix64(k ^ c[:, j])
    return k


def stable_pick(keys, k, salt=0) -> np.ndarray:
    """Indices (ascending — the caller's order is kept) of the ``k`` elements whose salted
    key hash is smallest; all of them when there are no more than ``k``. Ties (equal keys)
    fall to the earlier element."""
    h = _salted(keys, salt)
    n = len(h)
    k = int(k)
    if k >= n:
        return np.arange(n)
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(h, kind="stable")
    return np.sort(order[:k])


def stable_half(keys, salt=0) -> np.ndarray:
    """Boolean membership of the first half of a split-half test, per element, from its own
    salted key hash (a different salt than the pick's, or the half would follow the pick)."""
    return (_salted(keys, salt) & _U64(1)) == _U64(0)
