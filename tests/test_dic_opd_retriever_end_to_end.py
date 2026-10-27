"""DIC through the common interface: known phase in, OPD out (sign and scale)."""
import numpy as np
import pytest

from speckle_opd_retrieval import Setup, get_method
from speckle_opd_retrieval.methods.dic import DIC
from conftest import bump, speckle, warp


def test_dic_recovers_phase_with_correct_sign_and_scale():
    sh = (300, 360); I0 = speckle(sh); phi = bump(sh, amp=40)       # max shift ~1.5 px
    setup = Setup(pixel_size=1.0, distance=1.0)                        # opd == phi [px²]
    res = get_method("dic", setup, search=5).retrieve(I0, warp(I0, phi))
    assert res.opd.shape == sh and res.displacement.shape == (2,) + sh and res.method == "dic"
    sl = (slice(30, -30), slice(30, -30))
    a = res.opd[sl] - res.opd[sl].mean(); b = phi[sl] - phi[sl].mean()
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.98            # same sign and shape
    assert abs(np.std(a) / np.std(b) - 1) < 0.15                       # same scale
    assert set(["grid_rows", "grid_correlation_peak", "runtime_s"]) <= set(res.info)


def test_physical_scaling_follows_setup():
    sh = (200, 240); I0 = speckle(sh); phi = bump(sh, amp=30)
    spec = warp(I0, phi)
    r1 = DIC(Setup(1.0, 1.0), search=5).retrieve(I0, spec)
    r2 = DIC(Setup(3.45e-6, 2.48e-3), search=5).retrieve(I0, spec)
    assert np.allclose(r2.opd, r1.opd * 3.45e-6**2 / 2.48e-3)
    assert np.allclose(r2.displacement, r1.displacement)


def test_input_validation():
    m = DIC(Setup(1.0, 1.0))
    with pytest.raises(ValueError):
        m.retrieve(np.zeros((100, 100)), np.zeros((100, 101)))
    with pytest.raises(ValueError):
        m.retrieve(np.zeros((30, 30)), np.zeros((30, 30)))   # too small for the window
