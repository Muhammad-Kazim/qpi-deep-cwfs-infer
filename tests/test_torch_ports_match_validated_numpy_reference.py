"""The torch implementations reproduce the validated NumPy/SciPy ones (frozen copy in
reference_numpy_implementations_validated.py) to rounding, on SPECKLE_OPD_DEVICE (cpu, or cuda on the server)."""
import numpy as np
import pytest
import torch

import reference_numpy_implementations_validated as ref
from speckle_opd_retrieval import (Setup, available_integrators, get_method, integrate, integrate_poisson_neumann,
                                   integrate_poisson_zero_boundary)
from speckle_opd_retrieval.methods.deep_cwfs.patch_tiling_and_stitching import pad_to
from speckle_opd_retrieval.methods.dic import grid_to_full_resolution, zncc_displacement_grid
from conftest import DEVICE, bump, speckle, warp

TOL = 1e-10


def test_integrators_match(rng):
    gy, gx = rng.standard_normal((2, 97, 131))
    np.testing.assert_allclose(integrate_poisson_neumann(gy, gx, device=DEVICE), ref.integrate_dct(gy, gx), atol=TOL)
    np.testing.assert_allclose(integrate_poisson_zero_boundary(gy, gx, device=DEVICE), ref.integrate_cg(gy, gx), atol=TOL)


@pytest.mark.parametrize("name", sorted(available_integrators()))
def test_integrators_keep_tensors_on_device(rng, name):
    g = torch.tensor(rng.standard_normal((2, 40, 50)), device=DEVICE)
    phi = integrate(g, name)
    assert isinstance(phi, torch.Tensor) and phi.device.type == torch.device(DEVICE).type
    assert phi.shape == (40, 50) and phi.dtype == torch.float64


@pytest.mark.parametrize("kw", [dict(search=5), dict(search=7, sat="p99"), dict(window=24, step=4, search=6, sigma=4.0)],
                         ids=["default", "saturation", "small_window"])
def test_zncc_grid_matches(kw):
    sh = (300, 360); I0 = speckle(sh); I1 = warp(I0, bump(sh, amp=40))
    if kw.get("sat") == "p99":
        kw = dict(kw, sat=float(np.percentile(I1, 99)))
    g = zncc_displacement_grid(I0, I1, device=DEVICE, **kw)
    h = ref.zncc_displacement_grid(I0, I1, **kw)
    for k in ("dy", "dx", "peak", "cy", "cx"):
        np.testing.assert_allclose(g[k].cpu().numpy(), h[k], atol=TOL, err_msg=k)
    for k in ("outliers", "border"):
        np.testing.assert_array_equal(g[k].cpu().numpy(), h[k], err_msg=k)


def test_dic_retriever_matches_numpy_pipeline():
    sh = (200, 260); I0 = speckle(sh); I1 = warp(I0, bump(sh, amp=30))
    r = get_method("dic", Setup(1.0, 1.0), search=5, device=DEVICE).retrieve(I0, I1)
    h = ref.zncc_displacement_grid(I0, I1, search=5)
    rows, cols = h["cy"][:, 0], h["cx"][0]
    disp = np.stack([ref.grid_to_full_resolution(h[k], rows, cols, sh) for k in ("dy", "dx")])
    np.testing.assert_allclose(r.displacement, disp, atol=TOL)
    np.testing.assert_allclose(r.opd, ref.integrate_dct(*disp), atol=1e-8)
    assert r.params["device"].startswith(torch.device(DEVICE).type)


def test_grid_interpolation_matches(rng):
    v = rng.standard_normal((9, 12)); rows = 15.5 + 8.0 * np.arange(9); cols = 15.5 + 8.0 * np.arange(12)
    out = grid_to_full_resolution(torch.tensor(v), torch.tensor(rows), torch.tensor(cols), (110, 130))
    np.testing.assert_allclose(out.numpy(), ref.grid_to_full_resolution(v, rows, cols, (110, 130)), atol=1e-14)


@pytest.mark.parametrize("shape", [(200, 200), (500, 120), (1, 7)])
def test_pad_matches_numpy_reflect(shape, rng):
    a = rng.standard_normal(shape)
    mode = "reflect" if min(shape) > 1 else "edge"
    want = np.pad(a, ((0, max(0, 500 - shape[0])), (0, max(0, 500 - shape[1]))), mode=mode)
    np.testing.assert_array_equal(pad_to(torch.tensor(a), 500, 500).numpy(), want)
