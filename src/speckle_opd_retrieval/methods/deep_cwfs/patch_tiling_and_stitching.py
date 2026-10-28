"""Split large images into overlapping patches and blend per-patch results back.

Replaces ``make_crop_array`` / ``grad_stitch_n`` from the original scripts:
- works for non-square images and any number of rows/columns of patches;
- the last patch in each direction is shifted inward so that every patch lies
  fully inside the image (no reflect-padded edge patches);
- images smaller than the patch are padded once, up to the patch size.
Patches, weights and the stitched result are torch tensors on the patches' device.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


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


def feather_ramp(n: int, overlap: int, device=None, dtype=torch.float64) -> torch.Tensor:
    """1-D weight of length ``n``: 1 in the middle, rising linearly over ``overlap`` pixels at both ends."""
    r = torch.ones(n, device=device, dtype=dtype)
    k = min(overlap, n // 2)
    if k > 0:
        ramp_up = (torch.arange(k, device=device, dtype=dtype) + 1) / (k + 1)
        r[:k] = ramp_up
        r[n - k:] = ramp_up.flip(0)
    return r


def blend_weight(h: int, w: int, overlap: int, mode: str = "mean", device=None, dtype=torch.float64) -> torch.Tensor:
    """Per-patch weight. ``"mean"``: uniform (plain average in overlaps, as in
    the original ``grad_stitch_n``). ``"feather"``: linear ramp over the
    overlap width, which hides seams between patches."""
    if mode == "mean" or overlap <= 0:
        return torch.ones((h, w), device=device, dtype=dtype)
    if mode != "feather":
        raise ValueError("blend must be 'mean' or 'feather'")
    return torch.outer(feather_ramp(h, overlap, device, dtype), feather_ramp(w, overlap, device, dtype))


def stitch(
    patches: List[torch.Tensor],
    corners: List[Tuple[int, int]],
    shape: Tuple[int, int],
    overlap: int,
    mode: str = "mean",
) -> torch.Tensor:
    """Weighted average of ``(..., h, w)`` patches placed at ``corners`` into
    an array of shape ``(..., H, W)``."""
    leading_dims = tuple(patches[0].shape[:-2])
    device = patches[0].device
    dtype = patches[0].dtype
    out = torch.zeros(leading_dims + tuple(shape), device=device, dtype=dtype)
    weight_sum = torch.zeros(tuple(shape), device=device, dtype=dtype)
    for patch, (r, c) in zip(patches, corners):
        h, w = patch.shape[-2:]
        weight = blend_weight(h, w, overlap, mode, device, dtype)
        out[..., r:r + h, c:c + w] += patch * weight
        weight_sum[r:r + h, c:c + w] += weight
    weight_sum[weight_sum == 0] = 1.0
    return out / weight_sum


def padding_indices(n: int, m: int, device=None, edge: bool = False) -> torch.Tensor:
    """Indices 0..m-1 into an axis of length n: mirrored without repeating the edge pixel
    (NumPy ``reflect``, also for m > 2n), or repeating the last pixel if ``edge``."""
    i = torch.arange(m, device=device)
    if edge:
        return i.clamp(max=n - 1)
    i = i % (2 * n - 2)
    return torch.where(i >= n, 2 * n - 2 - i, i)


def pad_to(img: torch.Tensor, min_h: int, min_w: int) -> torch.Tensor:
    """Pad a 2-D image at the bottom/right up to at least ``(min_h, min_w)`` by
    mirroring (like NumPy ``reflect``, which also handles pads larger than the image)."""
    H, W = img.shape
    if H >= min_h and W >= min_w:
        return img
    edge = min(H, W) == 1  # a 1-px-thin image cannot be mirrored: repeat it (NumPy "edge")
    rows = padding_indices(H, max(H, min_h), img.device, edge)
    cols = padding_indices(W, max(W, min_w), img.device, edge)
    return img[rows][:, cols]
