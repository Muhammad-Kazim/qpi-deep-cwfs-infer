"""DIC core: speckle displacement on a grid, robust to amplitude variations.

Used by :class:`speckle_opd_retrieval.methods.dic.DIC`; can also be called directly.


1. Local intensity normalisation of each image in its own frame:
       I_n = (I - G_s*I) / sqrt(G_s*(I - G_s*I)^2)
   removes slowly varying background/offset and amplitude (transmission) A(x)
   on scales larger than ~sigma (chosen > speckle grain, < window).
2. Saturated pixels (and their neighbours) get zero weight.
3. Exact weighted zero-mean normalised cross-correlation (ZNCC) recomputed at
   every integer lag over each window (box sums via integral images), so the
   score is invariant to offset and gain of both windows at every shift.
4. Sub-pixel peak by 2-D parabolic (separable 3-point) fit; peaks on the
   search border are flagged invalid. Normalised median test for outliers.
Displacement d: specimen(x + d) = reference(x).
"""
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, median_filter


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
