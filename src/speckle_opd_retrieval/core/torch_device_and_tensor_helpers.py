"""Device choice and NumPy <-> torch conversion shared by core and every method.

All numerics run in torch on one device (``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"``). Default
(``device=None`` or ``"auto"``): CUDA if available, else CPU. Work is done in float64, except on MPS,
which has no float64 (float32 there).
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch

DeviceLike = Union[None, str, torch.device]


def resolve_device(device: DeviceLike = None) -> torch.device:
    """``None`` / ``"auto"`` -> CUDA if available, else CPU; otherwise the given device (checked)."""
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; use device='cpu'.")
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available; use device='cpu'.")
    return dev


def working_dtype(device: torch.device) -> torch.dtype:
    """float64, or float32 on MPS (no float64 support)."""
    return torch.float32 if device.type == "mps" else torch.float64


def as_tensor(x, device: DeviceLike = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """NumPy array / tensor / nested list -> tensor on ``device`` (default: where it is, or CPU)."""
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(np.asarray(x))
    if device is None:
        device = t.device
    if dtype is None:
        dtype = t.dtype
    return t.to(device=device, dtype=dtype)


def to_numpy(x: Any) -> Any:
    """Tensor -> NumPy (float64 for floating tensors); anything else unchanged."""
    if not isinstance(x, torch.Tensor):
        return x
    x = x.detach().cpu()
    if x.is_floating_point():
        x = x.double()
    return x.numpy()
