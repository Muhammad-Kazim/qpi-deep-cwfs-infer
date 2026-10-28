"""Integrate a displacement / gradient field into a scalar map (least-squares Poisson solve), in torch.

Both solvers take derivatives (d/dy, d/dx) sampled on the pixel grid and return the
mean-subtracted potential phi (same units as the gradient × pixels). They run on the device of
their input (or ``device``). NumPy input gives NumPy output (float64), tensor input a tensor.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .torch_device_and_tensor_helpers import DeviceLike, as_tensor, to_numpy, working_dtype


def dct_solve(gy: torch.Tensor, gx: torch.Tensor) -> torch.Tensor:
    """Neumann least-squares integration of two (H, W) tensors (see :func:`integrate_dct`)."""
    H, W = gx.shape

    # divergence of the forward differences (last column / row ignored)
    gx0 = F.pad(gx[:, :-1], (0, 1))
    gy0 = F.pad(gy[:-1], (0, 0, 0, 1))
    div = gx0 - F.pad(gx0[:, :-1], (1, 0)) + gy0 - F.pad(gy0[:-1], (0, 0, 1, 0))

    # The Neumann problem equals the periodic problem on the mirrored image (2H, 2W). Its FFT
    # diagonalises the same 5-point Laplacian as the DCT, with eigenvalues 2cos(πk/N) - 2.
    mirrored = torch.cat([div, div.flip(1)], dim=1)
    mirrored = torch.cat([mirrored, mirrored.flip(0)], dim=0)
    ky = torch.arange(2 * H, device=gx.device, dtype=gx.dtype)[:, None]
    kx = torch.arange(2 * W, device=gx.device, dtype=gx.dtype)[None, :]
    eigenvalues = 2 * torch.cos(math.pi * kx / W) + 2 * torch.cos(math.pi * ky / H) - 4
    eigenvalues[0, 0] = 1.0  # the mean is undetermined; set to 0 below

    spectrum = torch.fft.fft2(mirrored) / eigenvalues
    spectrum[0, 0] = 0.0
    phi = torch.fft.ifft2(spectrum).real[:H, :W]
    return phi - phi.mean()


def cg_solve(gy: torch.Tensor, gx: torch.Tensor, n_iter: int = 2500, tol: float = 1e-15) -> torch.Tensor:
    """Conjugate-gradient Poisson solve of two (H, W) tensors (see :func:`integrate_cg`)."""
    # right-hand side: minus the divergence (forward differences, last difference repeated)
    dx = torch.diff(gx, dim=1)
    dx = torch.cat([dx, dx[:, -1:]], dim=1)
    dy = torch.diff(gy, dim=0)
    dy = torch.cat([dy, dy[-1:]], dim=0)
    b = -(dx + dy)[None, None]

    # 5-point Laplacian on interior pixels; boundary values stay 0
    laplace = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], device=b.device, dtype=b.dtype).view(1, 1, 3, 3)
    interior = torch.ones_like(b)
    interior[..., [0, -1], :] = 0
    interior[..., :, [0, -1]] = 0

    def apply_laplacian(v):
        """The system matrix A applied to v."""
        return F.conv2d(v, laplace, padding=1) * interior

    x = torch.zeros_like(b)
    r = (b - apply_laplacian(x)) * interior
    p = r.clone()
    rs_old = torch.sum(r * r)
    for _ in range(n_iter):
        Ap = apply_laplacian(p)
        alpha = rs_old / (torch.sum(p * Ap) + 1e-20)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r)
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / (rs_old + 1e-20)) * p
        rs_old = rs_new
    x = x[0, 0]
    return x - x.mean()


def integrate_dct(grad_y, grad_x, device: DeviceLike = None):
    """Direct least-squares integration with Neumann boundaries (DCT, computed with FFTs).

    ``grad_y[i, j] = phi[i+1, j] - phi[i, j]`` and ``grad_x[i, j] = phi[i, j+1] - phi[i, j]``
    (forward differences; the last row / column is ignored). Exact for such input and fast
    on large images.
    """
    return _run_solver(dct_solve, grad_y, grad_x, device)


def integrate_cg(grad_y, grad_x, n_iter: int = 2500, tol: float = 1e-15, device: DeviceLike = None):
    """Conjugate-gradient Poisson solve with zero boundary values.

    ``integrate_flow_field_torch`` from the original deep-CWFS repository (same discretisation:
    forward-difference divergence with the last difference repeated, 5-point Laplacian on interior
    pixels), for one field. Used by Deep CWFS by default to reproduce the original OPD maps.
    """
    return _run_solver(cg_solve, grad_y, grad_x, device, n_iter=n_iter, tol=tol)


def _run_solver(solver, grad_y, grad_x, device: DeviceLike, **kwargs):
    """Convert the inputs to tensors, run ``solver`` and return the same type as the input."""
    input_is_tensor = isinstance(grad_y, torch.Tensor)
    gy = as_tensor(grad_y, device)
    gx = as_tensor(grad_x, device)
    dtype = working_dtype(gy.device)  # float64 (as the original solver), float32 on MPS
    phi = solver(gy.to(dtype), gx.to(dtype), **kwargs)
    if input_is_tensor:
        return phi
    return to_numpy(phi)


INTEGRATORS = {"dct": integrate_dct, "cg": integrate_cg}


def integrate(displacement, method: str = "dct", device: DeviceLike = None, **kwargs):
    """Integrate a (2, H, W) displacement field ordered (y, x). ``method``: ``"dct"`` or ``"cg"``."""
    if method not in INTEGRATORS:
        raise ValueError(f"integrator must be one of {sorted(INTEGRATORS)}")
    return INTEGRATORS[method](displacement[0], displacement[1], device=device, **kwargs)
