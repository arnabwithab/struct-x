"""Hot-pixel/refractory filters, ROI crop, B-bin x 2-polarity voxels. See design sec 5-6."""

import numpy as np

DEFAULT_B = 5  # default temporal bins (ablate 5-9)
REFRACTORY_US = 100  # drop same-pixel events <100 us apart
HOT_PIXEL_RATIO = 5.0  # drop pixels with >5x median event count
ROI_FRAC = 0.5  # y-band where mean projection >= 50% of peak
ROI_PAD_PX = 8  # pad on each side
MAX_SIDE = 256  # resize longest side to at most this (size from headers)
EPS = 1e-6  # frozen standardisation guard


def refractory_filter(xs, ys, ts, ps, refractory_us=REFRACTORY_US):
    """Keep mask: drop same-pixel events <refractory_us apart (sorted by time)."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    ts = np.asarray(ts)
    order = np.argsort(ts, kind="stable")
    keep_sorted = np.ones(len(ts), dtype=bool)
    last_t = {}
    for i in order:
        key = (int(xs[i]), int(ys[i]))
        if key in last_t and ts[i] - last_t[key] < refractory_us:
            keep_sorted[i] = False
        else:
            last_t[key] = ts[i]
    return keep_sorted


def hot_pixel_mask(xs, ys, ts, height, width, ratio=HOT_PIXEL_RATIO):
    """Keep mask: drop all events from pixels with >ratio x median count."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    counts = np.zeros((height, width), dtype=np.int64)
    np.add.at(counts, (ys, xs), 1)
    med = float(np.median(counts))
    if med <= 0:
        return np.ones(len(np.asarray(ts)), dtype=bool)
    hot = counts > ratio * med
    return ~hot[ys, xs]


def find_roi(counts_y, frac=ROI_FRAC, pad_px=ROI_PAD_PX):
    """Contiguous y-band (containing the peak) with projection >= frac x peak.

    Returns (y0, y1) exclusive end, padded and clipped to the sensor range.
    """
    counts_y = np.asarray(counts_y, dtype=np.float64)
    peak = counts_y.max()
    if peak <= 0:
        return 0, len(counts_y)
    above = counts_y >= frac * peak
    imax = int(counts_y.argmax())
    y0 = imax
    while y0 - 1 >= 0 and above[y0 - 1]:
        y0 -= 1
    y1 = imax + 1
    while y1 < len(counts_y) and above[y1]:
        y1 += 1
    return max(0, y0 - pad_px), min(len(counts_y), y1 + pad_px)


def compute_resize_shape(h, w, max_side=MAX_SIDE):
    """Resize shape preserving aspect; longest side capped, never upscaled."""
    s = min(1.0, max_side / max(h, w))
    return int(round(h * s)), int(round(w * s))


def events_to_voxel(xs, ys, ts, ps, height, width, n_bins=DEFAULT_B, t0_us=None, t1_us=None):
    """Polarity-split voxel, channels-first (2B, H, W): channel = b*2 + (p>0)."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    ts = np.asarray(ts)
    ps = np.asarray(ps)
    t0 = np.min(ts) if t0_us is None else t0_us
    t1 = np.max(ts) if t1_us is None else t1_us
    if t1 <= t0:
        bins = np.zeros(len(ts), dtype=np.int64)
    else:
        frac = (ts - t0) / (t1 - t0)
        bins = np.clip((frac * n_bins).astype(np.int64), 0, n_bins - 1)
    channels = bins * 2 + (ps > 0).astype(np.int64)
    vox = np.zeros((2 * n_bins, height, width), dtype=np.float32)
    np.add.at(vox, (channels, ys, xs), 1.0)
    return vox


def compute_voxel_train_stats(voxels):
    """Per-channel mean/std over TRAIN voxels; frozen for val/test."""
    stack = np.stack([np.asarray(v, dtype=np.float64) for v in voxels], axis=0)
    return stack.mean(axis=(0, 2, 3)), stack.std(axis=(0, 2, 3))


def standardise_voxel(voxel, mean, std, eps=EPS):
    """Per-channel standardisation with FROZEN train-set stats."""
    mean = np.asarray(mean, dtype=np.float64).reshape(-1, 1, 1)
    std = np.asarray(std, dtype=np.float64).reshape(-1, 1, 1)
    return (np.asarray(voxel, dtype=np.float64) - mean) / (std + eps)
