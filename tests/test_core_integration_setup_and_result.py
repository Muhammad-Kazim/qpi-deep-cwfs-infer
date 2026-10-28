"""Core: Poisson integrators, Setup scaling, result saving."""
import json

import numpy as np
import pytest

from speckle_opd_retrieval import (DEFAULT_INTEGRATOR, RetrievalResult, Setup, available_integrators, get_method,
                                   integrate, integrate_fourier_asdi, integrate_fourier_mirror,
                                   integrate_poisson_neumann, integrate_poisson_zero_boundary)
from conftest import bump, speckle, warp


def _fwd(phi):
    gy = np.zeros_like(phi); gx = np.zeros_like(phi)
    gy[:-1] = np.diff(phi, axis=0); gx[:, :-1] = np.diff(phi, axis=1)
    return gy, gx


def test_dct_integration_is_exact_for_forward_differences():
    phi = bump((128, 160), amp=5)
    rec = integrate_poisson_neumann(*_fwd(phi))
    assert np.abs(rec - (phi - phi.mean())).max() < 1e-9


def test_cg_integration_matches_shape():
    phi = bump((96, 120), amp=5)
    rec = integrate_poisson_zero_boundary(*_fwd(phi))
    sl = (slice(4, -4), slice(4, -4))
    assert np.corrcoef(rec[sl].ravel(), phi[sl].ravel())[0, 1] > 0.99


def _point_sampled_bump_and_tilt(shape, tilt):
    """Gaussian bump plus a tilt along x, and its exact gradients at the pixel centres."""
    H, W = shape
    y, x = np.mgrid[0:H, 0:W].astype(float)
    width = 0.12 * H
    phi = 5 * np.exp(-(((y - H / 2) / width) ** 2 + ((x - W / 2) / width) ** 2))
    gy = phi * -2 * (y - H / 2) / width ** 2
    gx = phi * -2 * (x - W / 2) / width ** 2 + tilt
    phi = phi + tilt * x
    return phi - phi.mean(), gy, gx


def test_fourier_asdi_is_accurate_and_keeps_tilts():
    phi, gy, gx = _point_sampled_bump_and_tilt((128, 160), tilt=0.02)
    rec = integrate_fourier_asdi(gy, gx)
    assert np.abs(rec - phi).max() < 1e-3 * np.abs(phi).max()


def test_fourier_mirror_loses_tilt_but_not_a_centred_object():
    phi, gy, gx = _point_sampled_bump_and_tilt((128, 160), tilt=0.0)
    assert np.corrcoef(integrate_fourier_mirror(gy, gx).ravel(), phi.ravel())[0, 1] > 0.99
    phi, gy, gx = _point_sampled_bump_and_tilt((128, 160), tilt=0.02)
    rec = integrate_fourier_mirror(gy, gx)
    assert abs(np.mean(np.diff(rec, axis=1))) < 0.25 * 0.02   # most of the tilt is gone


def test_column_ramp_stays_a_column_ramp():
    gx = np.full((64, 96), 0.1); gy = np.zeros((64, 96))
    rec = integrate_poisson_neumann(gy, gx)
    assert np.allclose(np.diff(rec, axis=1)[:, 5:-5], 0.1, atol=1e-6) and np.allclose(np.diff(rec, axis=0), 0, atol=1e-6)


def test_integrators_by_name_and_on_a_method():
    phi = bump((64, 80), amp=5)
    disp = np.stack(_fwd(phi))
    assert set(available_integrators()) == {"poisson_neumann", "poisson_zero_boundary", "fourier_mirror", "fourier_asdi"}
    assert DEFAULT_INTEGRATOR == "poisson_neumann"
    assert np.allclose(integrate(disp), integrate_poisson_neumann(*disp))
    with pytest.raises(ValueError):
        integrate(disp, "unknown")
    with pytest.raises(ValueError):
        get_method("dic", Setup(1.0, 1.0), integrator="unknown")
    I0 = speckle((160, 200))
    spec = warp(I0, bump((160, 200), amp=20))
    neumann = get_method("dic", Setup(1.0, 1.0), search=4, device="cpu").retrieve(I0, spec)
    zero = get_method("dic", Setup(1.0, 1.0), search=4, device="cpu", integrator="poisson_zero_boundary").retrieve(I0, spec)
    assert neumann.params["integrator"] == "poisson_neumann" and zero.params["integrator"] == "poisson_zero_boundary"
    assert np.allclose(zero.opd, integrate_poisson_zero_boundary(*neumann.displacement))


def test_setup_scales_and_validation():
    s = Setup(3.45e-6, 2.48e-3)
    assert np.isclose(s.opd_scale, 3.45e-6**2 / 2.48e-3) and np.isclose(s.gradient_scale, 3.45e-6 / 2.48e-3)
    with pytest.raises(ValueError):
        Setup(-1, 1)


def test_result_save_writes_files(tmp_path):
    r = RetrievalResult(opd=np.ones((8, 9)) * 1e-6, opd_gradient=np.zeros((2, 8, 9)), displacement=np.zeros((2, 8, 9)),
                        method="x", params={"a": 1}, setup=Setup(1e-6, 1e-3), info={"runtime_s": 0.1})
    files = r.save(str(tmp_path), "t", formats=["npy"], unit="um")
    assert np.allclose(np.load(tmp_path / "t_opd.npy"), 1.0)
    meta = json.load(open(tmp_path / "t_meta.json"))
    assert meta["method"] == "x" and meta["opd_unit"] == "um" and meta["params"] == {"a": 1}
