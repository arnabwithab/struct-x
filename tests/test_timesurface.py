"""Time-surface fallback (design sec 5, `--repr` flag).

2-channel last-timestamp surface, exponentially decayed, normalised to [0,1].
Channel 0 = polarity -1, channel 1 = polarity +1 (same as voxel polarity split).
"""

import numpy as np

from time_surface import events_to_timesurface


def test_decay_math_exact_three_events():
    # t0=0, t1=100, tau=20 -> surface = exp(-(t1 - t_last) / tau)
    xs = np.array([0, 1, 0])
    ys = np.array([0, 0, 1])
    ts = np.array([100, 80, 60])
    ps = np.array([-1, 1, -1])
    out = events_to_timesurface(xs, ys, ts, ps, 2, 2, 0, 100, 20)
    assert out[0, 0, 0] == np.float32(1.0)  # t_last == t1
    assert out[1, 0, 1] == np.float32(np.exp(-1.0))
    assert out[0, 1, 0] == np.float32(np.exp(-2.0))
    assert out[1, 0, 0] == 0.0  # untouched pixel/polarity stays zero
    assert out[0, 0, 1] == 0.0


def test_empty_window_returns_zeros():
    out = events_to_timesurface(
        np.array([]), np.array([]), np.array([]), np.array([]), 4, 5, 0, 1000, 200
    )
    assert out.shape == (2, 4, 5)
    assert out.dtype == np.float32
    assert np.all(out == 0.0)


def test_shape_dtype_and_unit_range():
    rng = np.random.RandomState(0)
    n = 50
    xs = rng.randint(0, 6, size=n)
    ys = rng.randint(0, 4, size=n)
    ts = rng.randint(0, 1000, size=n)
    ps = rng.choice([-1, 1], size=n)
    out = events_to_timesurface(xs, ys, ts, ps, 4, 6, 0, 1000, 200)
    assert out.shape == (2, 4, 6)
    assert out.dtype == np.float32
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_latest_timestamp_wins_out_of_order():
    # Same pixel+polarity twice, latest t must win regardless of input order.
    xs = np.array([2, 2])
    ys = np.array([1, 1])
    ts = np.array([900, 100])  # out of order: late event arrives first
    ps = np.array([1, 1])
    out = events_to_timesurface(xs, ys, ts, ps, 3, 4, 0, 1000, 200)
    assert out[1, 1, 2] == np.float32(np.exp(-(1000 - 900) / 200))
    assert out[0, 1, 2] == 0.0  # opposite polarity untouched


def test_polarity_mapping_independent_channels():
    xs = np.array([0, 0])
    ys = np.array([0, 0])
    ts = np.array([500, 800])
    ps = np.array([-1, 1])
    out = events_to_timesurface(xs, ys, ts, ps, 2, 2, 0, 1000, 200)
    assert out[0, 0, 0] == np.float32(np.exp(-(1000 - 500) / 200))
    assert out[1, 0, 0] == np.float32(np.exp(-(1000 - 800) / 200))


def test_default_tau_is_window_over_five():
    xs = np.array([0])
    ys = np.array([0])
    ts = np.array([900])
    ps = np.array([1])
    out = events_to_timesurface(xs, ys, ts, ps, 1, 1, 0, 1000)
    assert out[1, 0, 0] == np.float32(np.exp(-(1000 - 900) / 200))
