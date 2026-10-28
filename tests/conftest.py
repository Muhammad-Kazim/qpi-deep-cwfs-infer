"""Shared test helpers: synthetic speckle, a known phase, the forward warp, and data paths."""
import os

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("SPECKLE_OPD_DATA", os.path.join(PROJECT, "..", "data"))
ARCHIVE = os.path.join(PROJECT, "archive")
# original qpi-deep-cwfs-infer checkout (on a server: a copy of the `main` branch, see gpu_server/)
ORIGINAL_REPO = os.environ.get("SPECKLE_OPD_ORIGINAL_REPO", os.path.join(ARCHIVE, "original_qpi_deep_cwfs_infer_repo"))
DEVICE = os.environ.get("SPECKLE_OPD_DEVICE", "cpu")   # device for the tests ("cuda" on the GPU server)
CKPTS = os.path.join(DATA, "deep_cwfs_model_checkpoints")
SAMPLES = os.path.join(DATA, "2026-04-13_cwfs_speckle_pairs_10_samples")


def speckle(shape, grain=1.5, seed=0):
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal(shape), grain) + 1j * gaussian_filter(rng.standard_normal(shape), grain)
    return np.abs(f) ** 2


def bump(shape, amp=40.0, width=0.12):
    """Smooth phase phi [px²] (gradient = displacement in px)."""
    H, W = shape
    y, x = np.mgrid[0:H, 0:W].astype(float)
    return amp * np.exp(-(((y - H / 2) / (width * H)) ** 2 + ((x - W / 2) / (width * W)) ** 2))


def warp(I0, phi):
    """Specimen for displacement d = grad(phi): specimen(x + d) = reference(x)."""
    gy, gx = np.gradient(phi)
    yy, xx = np.mgrid[0:I0.shape[0], 0:I0.shape[1]].astype(float)
    return map_coordinates(I0, [yy - gy, xx - gx], order=3, mode="reflect")


@pytest.fixture
def rng():
    return np.random.default_rng(1)
