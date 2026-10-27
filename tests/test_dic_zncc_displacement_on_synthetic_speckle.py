"""DIC core (grid level): accuracy on known shifts and insensitivity to amplitude changes."""
import numpy as np
from scipy.ndimage import map_coordinates

from speckle_opd_retrieval.methods.dic import zncc_displacement_grid
from conftest import speckle


def _warp_d(I0, Dy, Dx):
    yy, xx = np.mgrid[0:I0.shape[0], 0:I0.shape[1]].astype(float)
    return map_coordinates(I0, [yy - Dy, xx - Dx], order=3, mode="reflect")


def _err(g, Dy, Dx):
    iy = np.round(g["cy"]).astype(int); ix = np.round(g["cx"]).astype(int)
    return g["dy"] - Dy[iy, ix], g["dx"] - Dx[iy, ix]


def test_recovers_known_displacement():
    sh = (320, 420); I0 = speckle(sh)
    yy, xx = np.mgrid[0:sh[0], 0:sh[1]].astype(float)
    Dy = 1.3 + 1.5 * np.exp(-((yy - 160) ** 2 + (xx - 210) ** 2) / 40**2); Dx = -2.4 + 0.004 * (xx - 210)
    ey, ex = _err(zncc_displacement_grid(I0, _warp_d(I0, Dy, Dx), search=5), Dy, Dx)
    assert np.sqrt(np.mean(ey**2)) < 0.08 and np.sqrt(np.mean(ex**2)) < 0.08


def test_ignores_amplitude_variation():
    sh = (320, 420); I0 = speckle(sh)
    yy, xx = np.mgrid[0:sh[0], 0:sh[1]].astype(float)
    A = 1 + 0.35 * np.sin(xx / 9) * np.cos(yy / 13) - 0.4 * np.exp(-((yy - 160) ** 2 + (xx - 210) ** 2) / 30**2) + 0.3 * xx / sh[1]
    g = zncc_displacement_grid(I0, A * I0 + 0.2 * I0.mean(), search=5)
    assert np.abs(g["dy"]).max() < 0.1 and np.abs(g["dx"]).max() < 0.1
