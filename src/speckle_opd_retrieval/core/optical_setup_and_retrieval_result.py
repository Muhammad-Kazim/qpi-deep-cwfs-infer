"""Data types shared by every method: the optical setup and the retrieval result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class Setup:
    """Physical parameters of the coded-wavefront-sensing setup.

    Attributes:
        pixel_size: camera pixel pitch [m], e.g. ``3.45e-6``.
        distance: phase-mask (diffuser) to sensor distance [m], e.g. ``2.48e-3``.
    """

    pixel_size: float
    distance: float

    def __post_init__(self):
        if self.pixel_size <= 0 or self.distance <= 0:
            raise ValueError("pixel_size and distance must be positive (metres)")

    @property
    def opd_scale(self) -> float:
        """Metres of OPD per pixel² of integrated displacement: ``pixel_size² / distance``."""
        return self.pixel_size**2 / self.distance

    @property
    def gradient_scale(self) -> float:
        """OPD gradient (m/m) per pixel of displacement: ``pixel_size / distance``."""
        return self.pixel_size / self.distance


@dataclass
class RetrievalResult:
    """What every method returns. Arrays are NumPy float64 on the CPU at full image size.

    Conventions (identical for all methods):
      - ``displacement`` d [px] satisfies ``specimen(x + d) = reference(x)``;
      - vector fields are ordered (y, x) = (rows, columns);
      - ``opd = setup.opd_scale * integral(d)`` [m], mean-subtracted;
      - ``opd_gradient = setup.gradient_scale * d`` (dimensionless, m/m).

    Attributes:
        opd: (H, W) optical path difference [m].
        opd_gradient: (2, H, W) dOPD/dy, dOPD/dx.
        displacement: (2, H, W) speckle displacement [px].
        amplitude: (H, W) amplitude if the method estimates it, else None.
        method: registry name of the method.
        params: the method parameters used (for provenance).
        setup: the Setup used.
        info: method-specific extras (e.g. DIC grid, correlation quality, timing).
    """

    opd: np.ndarray
    opd_gradient: np.ndarray
    displacement: np.ndarray
    amplitude: Optional[np.ndarray] = None
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    setup: Optional[Setup] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def save(self, out_dir: str, name: str, **kwargs):
        """Save to disk; see :func:`speckle_opd_retrieval.core.image_io_load_crop_save.save_result`."""
        from .image_io_load_crop_save import save_result

        return save_result(self, out_dir, name, **kwargs)
