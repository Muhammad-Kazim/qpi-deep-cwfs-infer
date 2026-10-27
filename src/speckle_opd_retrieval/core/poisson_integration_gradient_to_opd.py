"""Integrate a displacement / gradient field into a scalar map (least-squares Poisson solve).

Both solvers take derivatives (d/dy, d/dx) sampled on the pixel grid and return the
mean-subtracted potential phi (same units as the gradient × pixels).
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import convolve


def integrate_dct(grad_y, grad_x) -> np.ndarray:
    """Direct least-squares integration with Neumann boundaries (DCT).

    ``grad_y[i, j] = phi[i+1, j] - phi[i, j]`` and ``grad_x[i, j] = phi[i, j+1] - phi[i, j]``
    (forward differences; the last row / column is ignored). Exact for such input and fast
    on large images.
    """
    gy = np.asarray(grad_y, float); gx = np.asarray(grad_x, float)
    H, W = gx.shape
    gx0 = np.concatenate([gx[:, :-1], np.zeros((H, 1))], 1)
    gy0 = np.concatenate([gy[:-1], np.zeros((1, W))], 0)
    div = gx0 - np.pad(gx0[:, :-1], ((0, 0), (1, 0))) + gy0 - np.pad(gy0[:-1], ((1, 0), (0, 0)))
    y = np.arange(H)[:, None]; x = np.arange(W)[None]
    den = 2 * np.cos(np.pi * x / W) + 2 * np.cos(np.pi * y / H) - 4
    den[0, 0] = 1.0
    ph = dctn(div, norm="ortho") / den
    ph[0, 0] = 0.0
    phi = idctn(ph, norm="ortho")
    return phi - phi.mean()


def integrate_cg(grad_y, grad_x, n_iter: int = 2500, tol: float = 1e-15) -> np.ndarray:
    """Conjugate-gradient Poisson solve with zero boundary values.

    NumPy port of ``integrate_flow_field_torch`` from the original deep-CWFS repository
    (same discretisation: forward-difference divergence with the last difference repeated,
    5-point Laplacian on interior pixels). Used by Deep CWFS by default to reproduce the
    original OPD maps.
    """
    gy = np.asarray(grad_y, float); gx = np.asarray(grad_x, float)
    dx = np.diff(gx, axis=1); dx = np.concatenate([dx, dx[:, -1:]], 1)
    dy = np.diff(gy, axis=0); dy = np.concatenate([dy, dy[-1:]], 0)
    b = -(dx + dy)
    k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], float)
    mask = np.ones_like(b); mask[[0, -1], :] = 0; mask[:, [0, -1]] = 0

    def A(v):
        return convolve(v, k, mode="constant", cval=0.0) * mask

    x = np.zeros_like(b)
    r = (b - A(x)) * mask
    p = r.copy()
    rs_old = float(np.sum(r * r))
    for _ in range(n_iter):
        Ap = A(p)
        alpha = rs_old / (float(np.sum(p * Ap)) + 1e-20)
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(np.sum(r * r))
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / (rs_old + 1e-20)) * p
        rs_old = rs_new
    return x - x.mean()


INTEGRATORS = {"dct": integrate_dct, "cg": integrate_cg}


def integrate(displacement, method: str = "dct", **kwargs) -> np.ndarray:
    """Integrate a (2, H, W) displacement field ordered (y, x). ``method``: ``"dct"`` or ``"cg"``."""
    if method not in INTEGRATORS:
        raise ValueError(f"integrator must be one of {sorted(INTEGRATORS)}")
    d = np.asarray(displacement, float)
    return INTEGRATORS[method](d[0], d[1], **kwargs)
