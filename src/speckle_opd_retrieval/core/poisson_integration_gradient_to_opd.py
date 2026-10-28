"""Integrate a displacement / gradient field into a scalar map (the OPD up to scaling), in torch.

Integrators (name -> function), chosen with ``integrator=`` on every method or ``integrate(..., method=)``:

- ``"poisson_neumann"`` (:func:`integrate_poisson_neumann`, **default**): least-squares Poisson solve with
  Neumann boundaries (no assumption on the values at the image border), solved directly with a DCT.
  Exact for forward differences, fast; keeps tilts and objects cut by the border.
- ``"poisson_zero_boundary"`` (:func:`integrate_poisson_zero_boundary`): Poisson solve with the OPD forced
  to 0 on the border, by conjugate gradients. The original Deep CWFS solver; suppresses ramps but distorts
  objects that touch the border, and is slower.
- ``"fourier_mirror"`` (:func:`integrate_fourier_mirror`): Fourier (spectral) integration after padding the
  gradients by half the image size on every side by mirroring, as if the phase were mirrored to make it
  periodic, but without changing the gradient signs. A tilt (mean gradient) is lost.
- ``"fourier_asdi"`` (:func:`integrate_fourier_asdi`): Fourier integration after the anti-symmetric
  derivative extension (ASDI, Bon et al., Appl. Opt. 51, 5698 (2012)): the mirrored copies of each gradient
  get the sign the derivative of a mirrored phase has, so the extended field is a true periodic gradient.

``notebooks/integration_boundary_conditions_explained.ipynb`` compares them. All integrators take derivatives
(d/dy, d/dx) sampled on the pixel grid and return the mean-subtracted potential phi (same units as the
gradient × pixels). They run on the device of their input (or ``device``). NumPy input gives NumPy output
(float64), tensor input a tensor.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .torch_device_and_tensor_helpers import DeviceLike, as_tensor, to_numpy, working_dtype


def poisson_neumann_solve(gy: torch.Tensor, gx: torch.Tensor) -> torch.Tensor:
    """Neumann least-squares integration of two (H, W) tensors (see :func:`integrate_poisson_neumann`)."""
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


def poisson_zero_boundary_solve(gy: torch.Tensor, gx: torch.Tensor, n_iter: int = 2500, tol: float = 1e-15) -> torch.Tensor:
    """Conjugate-gradient Poisson solve of two (H, W) tensors (see :func:`integrate_poisson_zero_boundary`)."""
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


def integrate_poisson_neumann(grad_y, grad_x, device: DeviceLike = None):
    """Direct least-squares integration with Neumann boundaries (DCT, computed with FFTs).

    ``grad_y[i, j] = phi[i+1, j] - phi[i, j]`` and ``grad_x[i, j] = phi[i, j+1] - phi[i, j]``
    (forward differences; the last row / column is ignored). Exact for such input and fast
    on large images.
    """
    return _run_solver(poisson_neumann_solve, grad_y, grad_x, device)


def integrate_poisson_zero_boundary(grad_y, grad_x, n_iter: int = 2500, tol: float = 1e-15, device: DeviceLike = None):
    """Conjugate-gradient Poisson solve with zero boundary values.

    ``integrate_flow_field_torch`` from the original deep-CWFS repository (same discretisation:
    forward-difference divergence with the last difference repeated, 5-point Laplacian on interior
    pixels), for one field. Used by Deep CWFS by default to reproduce the original OPD maps.
    """
    return _run_solver(poisson_zero_boundary_solve, grad_y, grad_x, device, n_iter=n_iter, tol=tol)


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


# name -> (function, one-line description)
def fourier_integrate(gy: torch.Tensor, gx: torch.Tensor) -> torch.Tensor:
    """Spectral integration of a periodic gradient field: phi_hat = -i (ky gy_hat + kx gx_hat) / |k|²."""
    H, W = gx.shape
    ky = 2 * math.pi * torch.fft.fftfreq(H, device=gx.device, dtype=gx.dtype)[:, None]
    kx = 2 * math.pi * torch.fft.fftfreq(W, device=gx.device, dtype=gx.dtype)[None, :]
    k_squared = kx ** 2 + ky ** 2
    k_squared[0, 0] = 1.0  # the mean is undetermined; set to 0 below
    spectrum = -1j * (ky * torch.fft.fft2(gy) + kx * torch.fft.fft2(gx)) / k_squared
    spectrum[0, 0] = 0.0
    return torch.fft.ifft2(spectrum).real


def mirror_pad(img: torch.Tensor, pad_y: int, pad_x: int) -> torch.Tensor:
    """Pad by mirroring (edge pixel repeated): ``pad_y`` rows above and below, ``pad_x`` columns left and right."""
    img = torch.cat([img[:pad_y].flip(0), img, img[img.shape[0] - pad_y:].flip(0)], dim=0)
    return torch.cat([img[:, :pad_x].flip(1), img, img[:, img.shape[1] - pad_x:].flip(1)], dim=1)


def fourier_mirror_solve(gy: torch.Tensor, gx: torch.Tensor) -> torch.Tensor:
    """Fourier integration of two (H, W) tensors after mirror padding (see :func:`integrate_fourier_mirror`)."""
    H, W = gx.shape
    pad_y = H // 2
    pad_x = W // 2
    phi = fourier_integrate(mirror_pad(gy, pad_y, pad_x), mirror_pad(gx, pad_y, pad_x))
    phi = phi[pad_y:pad_y + H, pad_x:pad_x + W]  # crop the original image
    return phi - phi.mean()


def fourier_asdi_solve(gy: torch.Tensor, gx: torch.Tensor) -> torch.Tensor:
    """Fourier integration of two (H, W) tensors after ASDI extension (see :func:`integrate_fourier_asdi`)."""
    H, W = gx.shape
    # mirroring the phase along x flips the sign of d/dx but not of d/dy, and vice versa
    gx_extended = torch.cat([torch.cat([gx, -gx.flip(1)], dim=1),
                             torch.cat([gx.flip(0), -gx.flip(0).flip(1)], dim=1)], dim=0)
    gy_extended = torch.cat([torch.cat([gy, gy.flip(1)], dim=1),
                             torch.cat([-gy.flip(0), -gy.flip(0).flip(1)], dim=1)], dim=0)
    phi = fourier_integrate(gy_extended, gx_extended)[:H, :W]
    return phi - phi.mean()


def integrate_fourier_mirror(grad_y, grad_x, device: DeviceLike = None):
    """Fourier integration after padding the gradients by half the image size on every side by mirroring.

    The padded field is periodic, so the FFT has no wrap-around jumps, but the mirrored gradients keep
    their sign (a mirrored phase would flip it), so the mean gradient (a tilt) is lost and the
    result is distorted near the border when the phase has a slope there.
    """
    return _run_solver(fourier_mirror_solve, grad_y, grad_x, device)


def integrate_fourier_asdi(grad_y, grad_x, device: DeviceLike = None):
    """Fourier integration with the anti-symmetric derivative extension (ASDI).

    The gradients are extended to (2H, 2W) as the derivatives of the phase mirrored along x and y
    (sign flipped where the mirroring reverses the derivative's direction). The extended field is an
    exact periodic gradient field, so there are no boundary artefacts and tilts are kept.
    Gradients are taken as point samples (unlike the Poisson solvers, which assume forward differences).
    """
    return _run_solver(fourier_asdi_solve, grad_y, grad_x, device)


DEFAULT_INTEGRATOR = "poisson_neumann"

INTEGRATORS = {
    "poisson_neumann": (integrate_poisson_neumann,
                        "least-squares Poisson solve, Neumann boundaries (DCT): exact and fast (default)"),
    "poisson_zero_boundary": (integrate_poisson_zero_boundary,
                              "Poisson solve with OPD = 0 on the border (conjugate gradients): original Deep CWFS solver"),
    "fourier_mirror": (integrate_fourier_mirror,
                       "Fourier integration, gradients mirror-padded by N/2 (signs kept): loses tilts"),
    "fourier_asdi": (integrate_fourier_asdi,
                     "Fourier integration, anti-symmetric derivative extension (ASDI): no boundary artefacts"),
}


def available_integrators():
    """``{name: description}`` of the integrators accepted as ``integrator=`` / ``method=``."""
    return {name: description for name, (_, description) in INTEGRATORS.items()}


def integrate(displacement, method: str = DEFAULT_INTEGRATOR, device: DeviceLike = None, **kwargs):
    """Integrate a (2, H, W) displacement field ordered (y, x) with the integrator named ``method``."""
    if method not in INTEGRATORS:
        raise ValueError(f"unknown integrator '{method}'; available: {sorted(INTEGRATORS)}")
    function = INTEGRATORS[method][0]
    return function(displacement[0], displacement[1], device=device, **kwargs)
