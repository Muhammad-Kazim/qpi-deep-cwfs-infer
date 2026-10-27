"""Split large images into overlapping patches and blend per-patch results back.

Replaces ``make_crop_array`` / ``grad_stitch_n`` from the original scripts:
- works for non-square images and any number of rows/columns of patches;
- the last patch in each direction is shifted inward so that every patch lies
  fully inside the image (no reflect-padded edge patches);
- images smaller than the patch are padded once, up to the patch size.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def tile_starts(n: int, patch: int, overlap: int) -> List[int]:
    """Start indices along one axis of length ``n`` for patches of size
    ``patch`` overlapping by at least ``overlap`` pixels."""
    if patch <= 0:
        raise ValueError("patch must be positive")
    if not 0 <= overlap < patch:
        raise ValueError("overlap must satisfy 0 <= overlap < patch")
    if n <= patch:
        return [0]
    stride = patch - overlap
    starts = list(range(0, n - patch, stride))
    if starts[-1] != n - patch:
        starts.append(n - patch)
    return starts


def tile_grid(shape: Tuple[int, int], patch: int, overlap: int) -> List[Tuple[int, int]]:
    """List of ``(row, col)`` top-left corners covering an ``(H, W)`` image."""
    H, W = shape
    return [(r, c) for r in tile_starts(H, patch, overlap) for c in tile_starts(W, patch, overlap)]


def blend_weight(h: int, w: int, overlap: int, mode: str = "mean") -> np.ndarray:
    """Per-patch weight. ``"mean"``: uniform (plain average in overlaps, as in
    the original ``grad_stitch_n``). ``"feather"``: linear ramp over the
    overlap width, which hides seams between patches."""
    if mode == "mean" or overlap <= 0:
        return np.ones((h, w))
    if mode != "feather":
        raise ValueError("blend must be 'mean' or 'feather'")

    def ramp(n):
        r = np.ones(n)
        k = min(overlap, n // 2)
        if k > 0:
            ramp_up = (np.arange(k) + 1) / (k + 1)
            r[:k] = ramp_up
            r[n - k :] = ramp_up[::-1]
        return r

    return np.outer(ramp(h), ramp(w))


def stitch(
    patches: List[np.ndarray],
    corners: List[Tuple[int, int]],
    shape: Tuple[int, int],
    overlap: int,
    mode: str = "mean",
) -> np.ndarray:
    """Weighted average of ``(..., h, w)`` patches placed at ``corners`` into
    an array of shape ``(..., H, W)``."""
    lead = patches[0].shape[:-2]
    out = np.zeros(lead + tuple(shape))
    wsum = np.zeros(tuple(shape))
    for p, (r, c) in zip(patches, corners):
        h, w = p.shape[-2:]
        wt = blend_weight(h, w, overlap, mode)
        out[..., r : r + h, c : c + w] += p * wt
        wsum[r : r + h, c : c + w] += wt
    wsum[wsum == 0] = 1.0
    return out / wsum


def pad_to(img: np.ndarray, min_h: int, min_w: int) -> np.ndarray:
    """Pad a 2-D image at the bottom/right up to at least ``(min_h, min_w)`` by
    mirroring (NumPy ``reflect`` also handles pads larger than the image)."""
    H, W = img.shape
    ph, pw = max(0, min_h - H), max(0, min_w - W)
    if ph == 0 and pw == 0:
        return img
    mode = "reflect" if min(H, W) > 1 else "edge"
    return np.pad(img, ((0, ph), (0, pw)), mode=mode)
