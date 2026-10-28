"""Frozen NumPy/SciPy reference: the validated DIC (ZNCC grid + grid interpolation) and the Poisson
integrators exactly as they were before the move to torch (commit 798cfba). Tests check that the torch
implementations reproduce them. Do not edit."""
import numpy as np
from scipy.fft import dctn, idctn
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, convolve, gaussian_filter, median_filter

def local_normalise(img, sigma):
    img = img.astype(np.float64)
    m = gaussian_filter(img, sigma, mode="reflect")
    d = img - m
    s = np.sqrt(gaussian_filter(d * d, sigma, mode="reflect"))
    return d / (s + 1e-9 * s.max())


def _boxsum(a, ys, xs, w):
    c = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    c[1:, 1:] = a.cumsum(0).cumsum(1)
    return c[ys + w, xs + w] - c[ys, xs + w] - c[ys + w, xs] + c[ys, xs]


def zncc_displacement_grid(ref, spec, window=32, step=8, search=7, sigma=6.0, sat=None, min_valid=0.5):
    H, W = ref.shape
    a = local_normalise(ref, sigma)
    b = local_normalise(spec, sigma)
    ma = np.ones_like(a); mb = np.ones_like(b)
    if sat is not None:  # zero weight on saturated pixels (+1 px margin)
        ma[binary_dilation(ref >= sat, iterations=1)] = 0
        mb[binary_dilation(spec >= sat, iterations=1)] = 0
    ys = np.arange(search, H - window - search + 1, step)
    xs = np.arange(search, W - window - search + 1, step)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    Y, X = gy.ravel(), gx.ravel()
    lags = np.arange(-search, search + 1)
    L = len(lags)
    score = np.full((L, L, len(Y)), -np.inf)
    wa, wa2 = ma * a, ma * a * a
    for i, sy in enumerate(lags):
        for j, sx in enumerate(lags):
            # b_s[y, x] = b[y + sy, x + sx]
            bs = np.zeros_like(b); ms = np.zeros_like(mb)
            ry = slice(max(0, -sy), H - max(0, sy)); rx = slice(max(0, -sx), W - max(0, sx))
            ry2 = slice(max(0, sy), H + min(0, sy)); rx2 = slice(max(0, sx), W + min(0, sx))
            bs[ry, rx] = b[ry2, rx2]; ms[ry, rx] = mb[ry2, rx2]
            m = ma * ms
            n = _boxsum(m, Y, X, window)
            sa = _boxsum(m * a, Y, X, window); sb = _boxsum(m * bs, Y, X, window)
            saa = _boxsum(m * a * a, Y, X, window); sbb = _boxsum(m * bs * bs, Y, X, window)
            sab = _boxsum(m * a * bs, Y, X, window)
            n_ = np.maximum(n, 1)
            cov = sab - sa * sb / n_
            va = saa - sa * sa / n_; vb = sbb - sb * sb / n_
            z = cov / np.sqrt(np.maximum(va * vb, 1e-12))
            z[n < min_valid * window * window] = -np.inf
            score[i, j] = z
    flat = score.reshape(L * L, -1)
    k = flat.argmax(0)
    pi, pj = np.unravel_index(k, (L, L))
    r = np.arange(len(Y))
    peak = flat[k, r]
    border = (pi == 0) | (pi == L - 1) | (pj == 0) | (pj == L - 1) | ~np.isfinite(peak)
    pic = np.clip(pi, 1, L - 2); pjc = np.clip(pj, 1, L - 2)

    def para(m1, c0, p1):
        den = m1 - 2 * c0 + p1
        return np.where(np.abs(den) > 1e-12, 0.5 * (m1 - p1) / den, 0.0).clip(-0.5, 0.5)

    c0 = score[pic, pjc, r]
    dy = lags[pic] + para(score[pic - 1, pjc, r], c0, score[pic + 1, pjc, r])
    dx = lags[pjc] + para(score[pic, pjc - 1, r], c0, score[pic, pjc + 1, r])
    shape = gy.shape
    dy, dx, peak, border = (v.reshape(shape) for v in (dy, dx, peak, border))
    dy = np.where(border, np.nan, dy); dx = np.where(border, np.nan, dx)

    def median_test(u, thresh=2.0, eps=0.1):
        med = median_filter(np.nan_to_num(u, nan=np.nanmedian(u)), size=3, mode="nearest")
        res = np.abs(u - med)
        rm = median_filter(np.nan_to_num(res, nan=0), size=3, mode="nearest")
        bad = ~(res / (rm + eps) <= thresh)  # NaN counts as bad
        out = np.where(bad, med, u)
        return out, bad

    dyf, by = median_test(dy); dxf, bx = median_test(dx)
    return dict(dy=dyf, dx=dxf, peak=peak, outliers=by | bx, border=border,
                cy=gy + window / 2 - 0.5, cx=gx + window / 2 - 0.5,
                window=window, step=step, sigma=sigma, search=search)


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


def grid_to_full_resolution(values: np.ndarray, rows: np.ndarray, cols: np.ndarray, shape) -> np.ndarray:
    """Bilinear interpolation of a grid sampled at pixel ``rows`` × ``cols`` to every pixel;
    pixels outside the grid take the nearest grid value (no extrapolation)."""
    f = RegularGridInterpolator((rows, cols), values, method="linear")
    yy = np.clip(np.arange(shape[0], dtype=float), rows[0], rows[-1])
    xx = np.clip(np.arange(shape[1], dtype=float), cols[0], cols[-1])
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    return f(np.stack([Y.ravel(), X.ravel()], 1)).reshape(shape)


