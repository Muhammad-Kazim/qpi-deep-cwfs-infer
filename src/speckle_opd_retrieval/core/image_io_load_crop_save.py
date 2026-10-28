"""Load speckle images, crop them, and save retrieval results."""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional, Sequence

import numpy as np

UNITS = {"m": 1.0, "mm": 1e3, "um": 1e6, "nm": 1e9}


def load_image(path: str, key: Optional[str] = None, frame: Optional[int] = None) -> np.ndarray:
    """Load a 2-D image as float64: .tif/.tiff, .npy, .npz/.mat (``key``), or .png/.jpg/.bmp.
    ``frame`` selects the first-axis index of a stack."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile
        img = tifffile.imread(path)
    elif ext == ".npy":
        img = np.load(path)
    elif ext == ".npz":
        with np.load(path) as d:
            img = d[key or list(d.keys())[0]]
    elif ext == ".mat":
        from scipy.io import loadmat
        d = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
        if key is None:
            if len(d) != 1:
                raise ValueError(f"{path} holds {sorted(d)}; choose one with key=...")
            key = next(iter(d))
        img = d[key]
    else:
        from PIL import Image
        img = np.asarray(Image.open(path))
        if img.ndim == 3:
            img = img[..., :3].mean(axis=-1)
    img = np.asarray(img)
    if frame is not None:
        img = img[frame]
    img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D image, got {img.shape} (use frame=...)")
    return img.astype(np.float64)


def crop(img: np.ndarray, region: Optional[Sequence] = None) -> np.ndarray:
    """``region = [[row0, col0], [row1, col1]]`` or ``[row0, col0, row1, col1]``; None = no crop."""
    if region is None:
        return img
    r = np.asarray(region).ravel().astype(int)
    return img[r[0]:r[2], r[1]:r[3]]


def save_result(result, out_dir: str, name: str, formats: Iterable[str] = ("tiff", "png"),
                unit: str = "um", save_gradient: bool = False, cmap: str = "gray") -> list:
    """Write ``<name>_opd.<fmt>`` (OPD in ``unit``), optional gradient/amplitude, and ``<name>_meta.json``."""
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {list(UNITS)}")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, name)
    arrays = {"opd": result.opd * UNITS[unit]}
    if save_gradient:
        arrays["opd_gradient_y"], arrays["opd_gradient_x"] = result.opd_gradient
    if result.amplitude is not None:
        arrays["amplitude"] = result.amplitude
    written = []
    for fmt in formats:
        if fmt in ("tiff", "npy"):
            for k, a in arrays.items():
                p = f"{base}_{k}.{fmt}"
                if fmt == "tiff":
                    import tifffile
                    tifffile.imwrite(p, np.asarray(a, np.float32))
                else:
                    np.save(p, a)
                written.append(p)
        elif fmt == "png":
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(arrays["opd"], cmap=cmap)
            fig.colorbar(im, ax=ax, label=f"OPD [{unit}]", shrink=0.85)
            ax.set_title(f"{name} — {result.method}")
            ax.axis("off")
            p = f"{base}_opd.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(p)
        else:
            raise ValueError(f"unknown format '{fmt}' (tiff, npy, png)")
    meta = {"method": result.method, "opd_unit": unit, "params": result.params,
            "setup": None if result.setup is None else {"pixel_size_m": result.setup.pixel_size,
                                                         "distance_m": result.setup.distance},
            "info": {k: v for k, v in result.info.items() if np.isscalar(v) or isinstance(v, (str, list, dict))}}
    p = f"{base}_meta.json"
    with open(p, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    written.append(p)
    return written
