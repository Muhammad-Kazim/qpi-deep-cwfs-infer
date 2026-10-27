"""DIC as an :class:`OPDRetriever`: ZNCC displacement grid → full-resolution field → OPD."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ...core import MethodOutput, OPDRetriever, Setup
from .zncc_displacement_estimation import zncc_displacement_grid


def grid_to_full_resolution(values: np.ndarray, rows: np.ndarray, cols: np.ndarray, shape) -> np.ndarray:
    """Bilinear interpolation of a grid sampled at pixel ``rows`` × ``cols`` to every pixel;
    pixels outside the grid take the nearest grid value (no extrapolation)."""
    f = RegularGridInterpolator((rows, cols), values, method="linear")
    yy = np.clip(np.arange(shape[0], dtype=float), rows[0], rows[-1])
    xx = np.clip(np.arange(shape[1], dtype=float), cols[0], cols[-1])
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    return f(np.stack([Y.ravel(), X.ravel()], 1)).reshape(shape)


class DIC(OPDRetriever):
    """Digital image correlation, robust to amplitude (transmission) changes.

    Args:
        setup: pixel size and mask–sensor distance.
        window: correlation window [px] (default 32).
        step: grid spacing [px] (default 8).
        search: maximum displacement searched, ± px (default 7).
        sigma: Gaussian σ [px] of the local intensity normalisation (default 6; > speckle grain).
        saturation: pixel value treated as saturated and excluded (e.g. 65520 for a 12-bit
            camera stored as 16-bit); None disables the mask.
        integrator: "dct" (default) or "cg".

    ``result.info`` holds the grid results: ``grid_rows``, ``grid_cols``, ``grid_displacement``
    (2, ny, nx), ``grid_correlation_peak`` (ZNCC match quality) and ``grid_outliers``.
    """

    name = "dic"
    default_integrator = "dct"

    def __init__(self, setup: Setup, window: int = 32, step: int = 8, search: int = 7,
                 sigma: float = 6.0, saturation: Optional[float] = None, integrator: Optional[str] = None):
        super().__init__(setup, integrator=integrator, window=int(window), step=int(step), search=int(search),
                         sigma=float(sigma), saturation=saturation)

    def _estimate(self, reference, specimen) -> MethodOutput:
        p = self.params
        H, W = reference.shape
        if min(H, W) < p["window"] + 2 * p["search"] + p["step"]:
            raise ValueError(f"image {H}×{W} too small for window {p['window']} and search ±{p['search']}")
        g = zncc_displacement_grid(reference, specimen, window=p["window"], step=p["step"], search=p["search"],
                                   sigma=p["sigma"], sat=p["saturation"])
        rows, cols = g["cy"][:, 0], g["cx"][0]
        disp = np.stack([grid_to_full_resolution(g["dy"], rows, cols, (H, W)),
                         grid_to_full_resolution(g["dx"], rows, cols, (H, W))])
        info = dict(grid_rows=rows, grid_cols=cols, grid_displacement=np.stack([g["dy"], g["dx"]]),
                    grid_correlation_peak=g["peak"], grid_outliers=g["outliers"],
                    outlier_fraction=float(np.mean(g["outliers"])))
        return MethodOutput(displacement=disp, info=info)
