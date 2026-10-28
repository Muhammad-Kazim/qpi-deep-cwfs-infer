"""DIC core: speckle displacement on a grid, robust to amplitude variations (torch, any device).

Used by :class:`speckle_opd_retrieval.methods.dic.DIC`; can also be called directly.

Algorithm
---------
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

This is a torch port of the validated NumPy/SciPy version and gives the same results to rounding
(checked against the frozen copy in ``tests/reference_numpy_implementations_validated.py``). The
small filters below therefore copy SciPy's behaviour exactly (boundary modes, kernel size).
"""
import math

import torch
import torch.nn.functional as F

from ...core.torch_device_and_tensor_helpers import as_tensor, resolve_device, working_dtype


# -- small image filters (same results as the SciPy functions named in the docstrings) -------------

def _mirror_indices(n, pad, device):
    """Indices for padding an axis of length ``n`` by ``pad`` on both sides by mirroring
    (SciPy mode ``reflect``: d c b a | a b c d | d c b a)."""
    i = torch.arange(-pad, n + pad, device=device) % (2 * n)
    return torch.where(i >= n, 2 * n - 1 - i, i)


def gaussian_filter(img, sigma):
    """Gaussian blur of a 2-D tensor, as ``scipy.ndimage.gaussian_filter(img, sigma, mode="reflect")``."""
    radius = int(4.0 * sigma + 0.5)  # SciPy truncates the kernel at 4 sigma
    x = torch.arange(-radius, radius + 1, device=img.device, dtype=img.dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    H, W = img.shape
    rows = _mirror_indices(H, radius, img.device)
    cols = _mirror_indices(W, radius, img.device)
    padded = img[rows][:, cols]

    # separable: blur along y, then along x
    out = F.conv2d(padded[None, None], kernel.view(1, 1, -1, 1))
    out = F.conv2d(out, kernel.view(1, 1, 1, -1))
    return out[0, 0]


def dilate_cross(mask):
    """Grow a boolean mask by one pixel up/down/left/right,
    as ``scipy.ndimage.binary_dilation(mask, iterations=1)``."""
    out = mask.clone()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def median_filter_3x3(u):
    """3×3 median filter with edge pixels repeated, as ``scipy.ndimage.median_filter(u, 3, mode="nearest")``."""
    padded = F.pad(u[None, None], (1, 1, 1, 1), mode="replicate")
    neighbourhoods = F.unfold(padded, 3)  # (1, 9, H*W)
    return neighbourhoods.median(dim=1).values.view(u.shape)


def nanmedian(u):
    """Median ignoring NaNs, as ``numpy.nanmedian`` (for an even count: mean of the two middle values)."""
    values = u[~torch.isnan(u)].sort().values
    n = values.numel()
    if n == 0:
        return u.new_tensor(float("nan"))
    return 0.5 * (values[(n - 1) // 2] + values[n // 2])


def shift_image(a, sy, sx):
    """Return ``out`` with ``out[y, x] = a[y + sy, x + sx]``; pixels shifted in from outside are 0."""
    H, W = a.shape
    out = torch.zeros_like(a)
    out_rows = slice(max(0, -sy), H - max(0, sy))
    out_cols = slice(max(0, -sx), W - max(0, sx))
    in_rows = slice(max(0, sy), H + min(0, sy))
    in_cols = slice(max(0, sx), W + min(0, sx))
    out[out_rows, out_cols] = a[in_rows, in_cols]
    return out


def local_normalise(img, sigma):
    """Remove local mean and divide by local standard deviation (Gaussian window of width ``sigma``)."""
    mean = gaussian_filter(img, sigma)
    d = img - mean
    std = torch.sqrt(gaussian_filter(d * d, sigma))
    return d / (std + 1e-9 * std.max())


# -- the three steps of the DIC ------------------------------------------------------------------

def window_sums(img, Y, X, window):
    """Sum of ``img`` over each ``window``×``window`` box with top-left corners (Y, X), via an integral image."""
    c = F.pad(img.cumsum(0).cumsum(1), (1, 0, 1, 0))
    return c[Y + window, X + window] - c[Y, X + window] - c[Y + window, X] + c[Y, X]


def zncc_scores(a, b, ma, mb, Y, X, window, search, min_valid):
    """Weighted ZNCC of each reference window with the specimen at every integer lag.

    Args:
        a, b: normalised reference and specimen images.
        ma, mb: pixel weights (0 for saturated pixels).
        Y, X: top-left corners of the windows (1-D).
        window, search: window size and maximum lag [px].
        min_valid: windows with fewer valid pixels than this fraction score -inf.

    Returns:
        (L, L, n_windows) scores, L = 2*search + 1, index [i, j] = lag (i - search, j - search).
    """
    L = 2 * search + 1
    score = torch.full((L, L, len(Y)), -math.inf, device=a.device, dtype=a.dtype)
    for i, sy in enumerate(range(-search, search + 1)):
        for j, sx in enumerate(range(-search, search + 1)):
            bs = shift_image(b, sy, sx)
            m = ma * shift_image(mb, sy, sx)  # pixels valid in both images

            n = window_sums(m, Y, X, window)
            sa = window_sums(m * a, Y, X, window)
            sb = window_sums(m * bs, Y, X, window)
            saa = window_sums(m * a * a, Y, X, window)
            sbb = window_sums(m * bs * bs, Y, X, window)
            sab = window_sums(m * a * bs, Y, X, window)

            n_safe = torch.clamp(n, min=1)
            cov = sab - sa * sb / n_safe
            var_a = saa - sa * sa / n_safe
            var_b = sbb - sb * sb / n_safe
            z = cov / torch.sqrt(torch.clamp(var_a * var_b, min=1e-12))
            z[n < min_valid * window * window] = -math.inf
            score[i, j] = z
    return score


def parabola_offset(minus, centre, plus):
    """Sub-pixel offset of the peak of a parabola through three neighbouring scores, in [-0.5, 0.5]."""
    den = minus - 2 * centre + plus
    offset = torch.where(den.abs() > 1e-12, 0.5 * (minus - plus) / den, torch.zeros_like(den))
    return offset.clamp(-0.5, 0.5)


def subpixel_peaks(score, search):
    """Integer peak of each window's score map plus a parabolic sub-pixel refinement.

    Returns ``dy``, ``dx`` (NaN where invalid), the peak score and the ``border`` flag
    (peak on the edge of the search range or no finite score), each of length n_windows.
    """
    L = 2 * search + 1
    flat = score.reshape(L * L, -1)
    k = flat.argmax(0)
    pi = k // L
    pj = k % L
    r = torch.arange(flat.shape[1], device=score.device)
    peak = flat[k, r]
    border = (pi == 0) | (pi == L - 1) | (pj == 0) | (pj == L - 1) | ~torch.isfinite(peak)

    # neighbours must exist for the fit; border peaks are discarded below anyway
    pi = pi.clamp(1, L - 2)
    pj = pj.clamp(1, L - 2)
    centre = score[pi, pj, r]
    dy = (pi - search) + parabola_offset(score[pi - 1, pj, r], centre, score[pi + 1, pj, r])
    dx = (pj - search) + parabola_offset(score[pi, pj - 1, r], centre, score[pi, pj + 1, r])

    nan = torch.tensor(math.nan, device=score.device, dtype=score.dtype)
    dy = torch.where(border, nan, dy)
    dx = torch.where(border, nan, dx)
    return dy, dx, peak, border


def median_test(u, thresh=2.0, eps=0.1):
    """Normalised median test (Westerweel & Scarano 2005) on a 2-D grid.

    Vectors deviating from their 3×3 median by more than ``thresh`` times the local median residual
    (NaNs included) are replaced by that median. Returns the cleaned grid and the outlier mask.
    """
    filled = torch.where(torch.isnan(u), nanmedian(u), u)
    med = median_filter_3x3(filled)
    residual = (u - med).abs()
    residual_med = median_filter_3x3(torch.nan_to_num(residual, nan=0.0))
    bad = ~(residual / (residual_med + eps) <= thresh)  # NaN counts as bad
    return torch.where(bad, med, u), bad


def zncc_displacement_grid(ref, spec, window=32, step=8, search=7, sigma=6.0, sat=None, min_valid=0.5, device=None):
    """Speckle displacement on a grid of windows.

    Args:
        ref, spec: 2-D reference and specimen images (NumPy arrays or tensors).
        window: correlation window [px].
        step: grid spacing [px].
        search: maximum displacement searched, ± px.
        sigma: Gaussian σ [px] of the local intensity normalisation.
        sat: pixel value treated as saturated (excluded), or None.
        min_valid: minimum fraction of valid (unsaturated) pixels per window.
        device: where to compute; default: the device of ``ref`` if it is a tensor, else CUDA if available.

    Returns:
        dict of tensors: ``dy``, ``dx`` (ny, nx) displacement [px]; ``peak`` (ZNCC match quality);
        ``outliers``, ``border`` (bool); ``cy``, ``cx`` window-centre pixel coordinates; plus the settings.
    """
    if device is None and isinstance(ref, torch.Tensor):
        device = ref.device
    else:
        device = resolve_device(device)
    dtype = working_dtype(device)
    ref = as_tensor(ref, device, dtype)
    spec = as_tensor(spec, device, dtype)
    H, W = ref.shape

    # 1. normalise; 2. zero weight on saturated pixels (+1 px margin)
    a = local_normalise(ref, sigma)
    b = local_normalise(spec, sigma)
    ma = torch.ones_like(a)
    mb = torch.ones_like(b)
    if sat is not None:
        ma[dilate_cross(ref >= sat)] = 0
        mb[dilate_cross(spec >= sat)] = 0

    # window grid (top-left corners)
    ys = torch.arange(search, H - window - search + 1, step, device=device)
    xs = torch.arange(search, W - window - search + 1, step, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    # 3. correlate, 4. sub-pixel peak and outlier test
    score = zncc_scores(a, b, ma, mb, gy.reshape(-1), gx.reshape(-1), window, search, min_valid)
    dy, dx, peak, border = subpixel_peaks(score, search)
    dy, dx, peak, border = (v.reshape(gy.shape) for v in (dy, dx, peak, border))
    dy, outliers_y = median_test(dy)
    dx, outliers_x = median_test(dx)

    return dict(dy=dy, dx=dx, peak=peak, outliers=outliers_y | outliers_x, border=border,
                cy=gy.to(dtype) + window / 2 - 0.5, cx=gx.to(dtype) + window / 2 - 0.5,
                window=window, step=step, sigma=sigma, search=search)
