"""DIC as an :class:`OPDRetriever`: ZNCC displacement grid → full-resolution field → OPD (torch, any device)."""

from __future__ import annotations

from typing import Optional

import torch

from ...core import MethodOutput, OPDRetriever, Setup
from .zncc_displacement_estimation import zncc_displacement_grid


def linear_interpolation_weights(coords: torch.Tensor, n: int):
    """For each pixel 0..n-1: index of the grid point to its left and the weight of the right one.

    ``coords`` are increasing grid positions [px]; pixels outside the grid are clamped to it,
    so they take the nearest grid value.
    """
    pixels = torch.arange(n, device=coords.device, dtype=coords.dtype).clamp(coords[0], coords[-1])
    left = torch.searchsorted(coords, pixels, right=True) - 1
    left = left.clamp(0, len(coords) - 2)
    weight_right = (pixels - coords[left]) / (coords[left + 1] - coords[left])
    return left, weight_right


def grid_to_full_resolution(values: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, shape) -> torch.Tensor:
    """Bilinear interpolation of a grid sampled at pixel ``rows`` × ``cols`` to every pixel;
    pixels outside the grid take the nearest grid value (no extrapolation)."""
    iy, wy = linear_interpolation_weights(rows.to(values.dtype), shape[0])
    ix, wx = linear_interpolation_weights(cols.to(values.dtype), shape[1])
    # interpolate along y (rows), then along x (columns)
    v = values[iy] * (1 - wy)[:, None] + values[iy + 1] * wy[:, None]
    return v[:, ix] * (1 - wx) + v[:, ix + 1] * wx


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
        integrator: ``"poisson_neumann"`` (default), ``"poisson_zero_boundary"``, ``"fourier_mirror"`` or
            ``"fourier_asdi"`` (see ``available_integrators()``).
        device: ``None`` (CUDA if available, else CPU), ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"``.

    ``result.info`` holds the grid results: ``grid_rows``, ``grid_cols``, ``grid_displacement``
    (2, ny, nx), ``grid_correlation_peak`` (ZNCC match quality) and ``grid_outliers``.
    """

    name = "dic"

    def __init__(self, setup: Setup, window: int = 32, step: int = 8, search: int = 7,
                 sigma: float = 6.0, saturation: Optional[float] = None, integrator: Optional[str] = None,
                 device: Optional[str] = None):
        """Store the DIC settings (see the class docstring)."""
        super().__init__(setup, integrator=integrator, device=device, window=int(window), step=int(step),
                         search=int(search), sigma=float(sigma), saturation=saturation)

    def _estimate(self, reference, specimen) -> MethodOutput:
        """ZNCC grid -> bilinear upsampling to full resolution; grid results go into ``info``."""
        p = self.params
        H, W = reference.shape
        if min(H, W) < p["window"] + 2 * p["search"] + p["step"]:
            raise ValueError(f"image {H}×{W} too small for window {p['window']} and search ±{p['search']}")
        g = zncc_displacement_grid(reference, specimen, window=p["window"], step=p["step"], search=p["search"],
                                   sigma=p["sigma"], sat=p["saturation"])
        rows, cols = g["cy"][:, 0].contiguous(), g["cx"][0].contiguous()
        disp = torch.stack([grid_to_full_resolution(g["dy"], rows, cols, (H, W)),
                            grid_to_full_resolution(g["dx"], rows, cols, (H, W))])
        info = dict(grid_rows=rows, grid_cols=cols, grid_displacement=torch.stack([g["dy"], g["dx"]]),
                    grid_correlation_peak=g["peak"], grid_outliers=g["outliers"],
                    outlier_fraction=float(g["outliers"].double().mean()))
        return MethodOutput(displacement=disp, info=info)
