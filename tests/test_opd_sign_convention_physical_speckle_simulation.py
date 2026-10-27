"""OPD sign from physics: a thin object in front of a random diffuser, angular-spectrum propagation
(fields ~ exp(+i(kz - wt)), so a phase lag is exp(+ik·OPD)), then DIC. Textbook convention:
OPD = ∫(n - n_medium) dz, higher RI (lag, converging) -> positive, lower RI (lead, diverging) -> negative."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from speckle_opd_retrieval import Setup, get_method


def _pair(opd_textbook, px, wl, z, seed=0):
    N = opd_textbook.shape[0]
    k = 2 * np.pi / wl
    diffuser = np.exp(1j * 8 * gaussian_filter(np.random.default_rng(seed).standard_normal((N, N)), 2.0))
    f = 2 * np.pi * np.fft.fftfreq(N, px)
    FX, FY = np.meshgrid(f, f)
    H = np.exp(1j * z * np.sqrt(np.maximum(k**2 - FX**2 - FY**2, 0)))
    prop = lambda u: np.fft.ifft2(np.fft.fft2(u) * H)
    return np.abs(prop(diffuser)) ** 2, np.abs(prop(diffuser * np.exp(1j * k * opd_textbook))) ** 2


@pytest.mark.parametrize("ri_sign", [+1, -1], ids=["higher_RI_lag_converging", "lower_RI_lead_diverging"])
def test_dic_opd_sign_follows_convention(ri_sign):
    N, px, wl, z = 256, 1.0e-6, 0.53e-6, 1.0e-3
    y, x = (np.mgrid[0:N, 0:N] - N / 2) * px
    textbook = ri_sign * 0.15e-6 * np.exp(-(x**2 + y**2) / (30e-6) ** 2)     # > 0 for higher RI
    I0, I1 = _pair(textbook, px, wl, z)
    r = get_method("dic", Setup(px, z), window=32, step=4, search=8).retrieve(I0, I1)
    sl = (slice(40, -40), slice(40, -40))
    assert np.corrcoef(r.opd[sl].ravel(), textbook[sl].ravel())[0, 1] > 0.7
