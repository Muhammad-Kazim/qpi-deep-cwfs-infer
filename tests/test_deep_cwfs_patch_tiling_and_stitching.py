"""Tiling used by Deep CWFS (torch tensors)."""
import numpy as np
import pytest
import torch

from speckle_opd_retrieval.methods.deep_cwfs.patch_tiling_and_stitching import pad_to, stitch, tile_grid, tile_starts


@pytest.mark.parametrize("n,patch,overlap", [(500, 500, 150), (1000, 500, 150), (1234, 500, 150), (300, 500, 150)])
def test_tiles_cover_axis(n, patch, overlap):
    s = tile_starts(n, patch, overlap)
    assert s[0] == 0 and (n <= patch or s[-1] == n - patch)


@pytest.mark.parametrize("mode", ["mean", "feather"])
def test_stitch_reconstructs(mode, rng):
    img = torch.tensor(rng.standard_normal((2, 640, 1100))); P, ov = 256, 64
    c = tile_grid((640, 1100), P, ov)
    torch.testing.assert_close(stitch([img[:, r:r + P, q:q + P] for r, q in c], c, (640, 1100), ov, mode), img, atol=1e-12, rtol=0)


def test_pad_larger_than_image():
    a = torch.arange(200 * 200, dtype=torch.float64).reshape(200, 200)
    assert pad_to(a, 500, 500).shape == (500, 500)
