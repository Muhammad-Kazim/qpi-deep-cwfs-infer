"""Core: Poisson integrators, Setup scaling, result saving."""
import json

import numpy as np
import pytest

from speckle_opd_retrieval import RetrievalResult, Setup, integrate_cg, integrate_dct
from conftest import bump


def _fwd(phi):
    gy = np.zeros_like(phi); gx = np.zeros_like(phi)
    gy[:-1] = np.diff(phi, axis=0); gx[:, :-1] = np.diff(phi, axis=1)
    return gy, gx


def test_dct_integration_is_exact_for_forward_differences():
    phi = bump((128, 160), amp=5)
    rec = integrate_dct(*_fwd(phi))
    assert np.abs(rec - (phi - phi.mean())).max() < 1e-9


def test_cg_integration_matches_shape():
    phi = bump((96, 120), amp=5)
    rec = integrate_cg(*_fwd(phi))
    sl = (slice(4, -4), slice(4, -4))
    assert np.corrcoef(rec[sl].ravel(), phi[sl].ravel())[0, 1] > 0.99


def test_column_ramp_stays_a_column_ramp():
    gx = np.full((64, 96), 0.1); gy = np.zeros((64, 96))
    rec = integrate_dct(gy, gx)
    assert np.allclose(np.diff(rec, axis=1)[:, 5:-5], 0.1, atol=1e-6) and np.allclose(np.diff(rec, axis=0), 0, atol=1e-6)


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
