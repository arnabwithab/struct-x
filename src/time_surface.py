"""Time-surface fallback representation (design sec 5, behind `--repr` flag).

Flow literature shows surfaces win when event rates collapse under bad
illumination; voxels stay the default. Numpy only.
"""

import numpy as np


def events_to_timesurface(xs, ys, ts, ps, height, width, t0_us, t1_us, tau_us=None):
    """Last-timestamp surface per pixel per polarity, decayed to [0,1].

    out[ch, y, x] = exp(-(t1_us - t_last) / tau_us) where t_last is the
    latest event timestamp at that pixel/polarity inside [t0_us, t1_us];
    0 where no event fired. Channel 0 = p=-1, channel 1 = p=+1.
    tau_us defaults to (t1_us - t0_us) / 5. Empty windows return zeros.
    """
    t0_us = float(t0_us)
    t1_us = float(t1_us)
    out = np.zeros((2, height, width), dtype=np.float32)
    if tau_us is None:
        tau_us = (t1_us - t0_us) / 5.0
    tau_us = float(tau_us)
    if not np.isfinite(tau_us) or tau_us <= 0 or not t1_us > t0_us:
        return out
    xs = np.asarray(xs).astype(np.int64).ravel()
    ys = np.asarray(ys).astype(np.int64).ravel()
    ts = np.asarray(ts, dtype=np.float64).ravel()
    ps = np.asarray(ps).ravel()
    if ts.size == 0:
        return out
    valid = (ts >= t0_us) & (ts <= t1_us) & (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if not np.any(valid):
        return out
    xs, ys, ts, ps = xs[valid], ys[valid], ts[valid], ps[valid]
    channels = (ps > 0).astype(np.int64)  # -1 -> 0, +1 -> 1
    last = np.full((2, height, width), -np.inf)
    np.maximum.at(last, (channels, ys, xs), ts)  # latest timestamp wins
    fired = np.isfinite(last)
    out[fired] = np.exp(-(t1_us - last[fired]) / tau_us)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
