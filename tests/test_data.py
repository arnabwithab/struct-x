"""F001 data-pipeline tests (design sec 5, sole source of truth).

Pure functions on synthetic data only — no real dataset exists locally.
Covers: LDV velocity->displacement, window-count math, voxel shapes for
B in 5..9, split leakage, val-shift/test-shift separation.
"""

import numpy as np
import pytest

import data_loader
import voxel


# ---------------------------------------------------------------- LDV units
def test_velocity_to_displacement_ramps():
    # Oscillatory velocity -> displacement at the same frequency, detrend
    # harmless on zero-mean signals. (A pure ramp is removed by design.)
    dt = 0.001
    n = 2000
    t = np.arange(n) * dt
    q_vel = np.cos(2 * np.pi * 5 * t)
    q_disp = data_loader.velocity_to_displacement(q_vel, dt)
    assert q_disp.shape == (n,)
    ref = np.sin(2 * np.pi * 5 * t) / (2 * np.pi * 5)
    r = np.corrcoef(q_disp, ref)[0, 1]
    assert r > 0.99


def test_velocity_unit_detection():
    assert data_loader.is_velocity_unit("m/s")
    assert data_loader.is_velocity_unit("mm/s")
    assert not data_loader.is_velocity_unit("m")
    assert not data_loader.is_velocity_unit("mm")


def test_preprocess_ldv_resamples_to_1000hz():
    # Sine displacement at fs_raw=2000 Hz, 2 s -> 2000 samples at 1000 Hz.
    fs_raw = 2000
    dur = 2.0
    t = np.arange(int(fs_raw * dur)) / fs_raw
    q = np.sin(2 * np.pi * 50 * t)
    t_out, q_out = data_loader.preprocess_ldv(t, q, unit="m")
    assert len(t_out) == pytest.approx(fs_raw * dur / 2, rel=0.01)
    assert len(q_out) == len(t_out)
    # 50 Hz content survives the 500 Hz low-pass: correlate with reference.
    ref = np.sin(2 * np.pi * 50 * t_out)
    r = np.corrcoef(q_out, ref)[0, 1]
    assert r > 0.99


def test_preprocess_ldv_velocity_integrates_first():
    # Velocity sine -> displacement sine (lags by 90 deg), same frequency.
    fs_raw = 2000
    dur = 2.0
    t = np.arange(int(fs_raw * dur)) / fs_raw
    v = np.cos(2 * np.pi * 50 * t)  # integral ~ sin(2 pi 50 t) / (2 pi 50)
    _, q_out = data_loader.preprocess_ldv(t, v, unit="m/s")
    # Zero-mean displacement (detrended), dominated by 50 Hz.
    assert abs(q_out.mean()) < 0.05 * abs(q_out).max()
    spec = np.abs(np.fft.rfft(q_out))
    freqs = np.fft.rfftfreq(len(q_out), 1.0 / 1000)
    peak = freqs[np.argmax(spec[1:]) + 1]
    assert peak == pytest.approx(50, abs=2.0)


def test_frozen_zscore_uses_train_stats():
    q = np.array([1.0, 2.0, 3.0])
    z = data_loader.apply_frozen_zscore(q, mean=2.0, std=1.0)
    assert z == pytest.approx([-1.0, 0.0, 1.0])
    stats_mean, stats_std = data_loader.compute_train_stats(
        [np.array([0.0, 2.0]), np.array([4.0, 6.0])]
    )
    assert stats_mean == pytest.approx(3.0)
    assert stats_std == pytest.approx(np.std([0.0, 2.0, 4.0, 6.0]))


# ------------------------------------------------------------------ windows
def test_window_starts_excludes_first_100ms():
    starts = data_loader.window_starts(2.0)
    assert starts == pytest.approx([0.5, 1.0])
    assert all(s >= 0.1 for s in starts)  # window 0 never scored
    starts5 = data_loader.window_starts(5.0)
    assert starts5 == pytest.approx([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])


def test_window_budget_30_trials():
    # 30 trials x 2-5 s -> ~90-270 windows (design sec 5 correction).
    rng = np.random.default_rng(0)
    durations = rng.uniform(2.0, 5.0, size=30)
    total = data_loader.count_windows(durations)
    assert 90 <= total <= 270


# ------------------------------------------------------------------- voxels
@pytest.mark.parametrize("n_bins", [5, 6, 7, 8, 9])
def test_voxel_shapes_B_5_to_9(n_bins):
    # Sensor size comes from (synthetic) headers, never hardcoded in src.
    height, width = 60, 80
    n = 2000
    rng = np.random.default_rng(n_bins)
    xs = rng.integers(0, width, n)
    ys = rng.integers(0, height, n)
    ts = np.sort(rng.integers(0, 1_000_000, n))
    ps = rng.choice([-1, 1], n)
    v = voxel.events_to_voxel(xs, ys, ts, ps, height, width, n_bins=n_bins)
    assert v.shape == (2 * n_bins, height, width)
    assert v.sum() == pytest.approx(n)  # every event lands in exactly one bin


def test_hot_pixel_filter_drops_hot():
    height, width = 20, 20
    rng = np.random.default_rng(1)
    n_bg = 500
    xs = rng.integers(0, width, n_bg)
    ys = rng.integers(0, height, n_bg)
    # Hot pixel (3, 4) fires 100x the per-pixel median.
    hot = np.zeros((height, width), dtype=int)
    hot[3, 4] = 500
    keep = voxel.hot_pixel_mask(xs, ys, np.zeros(n_bg), height, width)
    assert keep.all()  # uniform background: nothing dropped
    xs2 = np.concatenate([xs, np.full(500, 4)])
    ys2 = np.concatenate([ys, np.full(500, 3)])
    keep2 = voxel.hot_pixel_mask(xs2, ys2, np.zeros(len(xs2)), height, width)
    assert not keep2[len(xs2) - 500 :].any()  # hot pixel dropped
    assert keep2[: len(xs2) - 500].mean() > 0.9  # background kept


def test_refractory_filter_drops_close_events():
    # Same pixel 10 us apart -> second dropped; 200 us apart -> kept.
    xs = np.array([5, 5, 5, 6])
    ys = np.array([5, 5, 5, 5])
    ts = np.array([0, 10, 300, 15])  # us
    ps = np.array([1, 1, 1, 1])
    keep = voxel.refractory_filter(xs, ys, ts, ps)
    by_t = dict(zip(ts, keep))
    assert by_t[0] and not by_t[10] and by_t[300] and by_t[15]


def test_roi_crop_band_and_pad():
    height = 100
    proj = np.zeros(height)
    proj[20:41] = 100.0  # active band rows 20..40
    y0, y1 = voxel.find_roi(proj)
    assert y0 <= 20 and y1 >= 41  # band covered ...
    assert y0 >= 20 - 8 and y1 <= 41 + 8  # ... with <=8 px pad, clipped
    assert y1 - y0 <= (41 - 20) + 1 + 16


def test_resize_caps_longest_side():
    assert voxel.compute_resize_shape(260, 346) == (192, 256)
    assert voxel.compute_resize_shape(100, 120) == (100, 120)  # never upscale


def test_voxel_frozen_standardisation():
    v = np.ones((10, 4, 4), dtype=np.float32)
    mean = np.zeros(10)
    std = np.ones(10)
    z = voxel.standardise_voxel(v, mean, std)
    assert z.shape == v.shape
    assert np.allclose(z, 1.0)


# ------------------------------------------------------------------- splits
def _fake_conditions():
    conds = {}
    for i in range(12):
        conds[f"trial_{i:02d}"] = "matched"
    for i in range(12, 21):
        conds[f"trial_{i:02d}"] = "illum_shift"
    for i in range(21, 30):
        conds[f"trial_{i:02d}"] = "bias_shift"
    return conds


def test_loco_folds_equal_conditions():
    folds = data_loader.make_loco_splits(_fake_conditions())
    assert set(folds) == {"matched", "illum_shift", "bias_shift"}  # K folds, never 5


def test_split_no_trial_leakage():
    folds = data_loader.make_loco_splits(_fake_conditions())
    for held, sp in folds.items():
        sets = [
            sp["train"],
            sp["val_matched"],
            sp["test_matched"],
            sp["val_shift"],
            sp["test_shift"],
        ]
        data_loader.check_no_leakage(
            {
                "train": sp["train"],
                "val_matched": sp["val_matched"],
                "test_matched": sp["test_matched"],
                "val_shift": sp["val_shift"],
                "test_shift": sp["test_shift"],
            }
        )
        # Held-out condition trials never appear in matched sets.
        held_trials = {t for t, c in _fake_conditions().items() if c == held}
        assert not (set(sp["train"]) & held_trials)
        assert len(sets) == 5


def test_val_shift_test_shift_disjoint_and_balanced():
    folds = data_loader.make_loco_splits(_fake_conditions())
    for sp in folds.values():
        assert not (set(sp["val_shift"]) & set(sp["test_shift"]))
        assert abs(len(sp["val_shift"]) - len(sp["test_shift"])) <= 1  # 50/50
        assert len(sp["val_shift"]) > 0 and len(sp["test_shift"]) > 0


def test_no_window_shared_across_sets():
    folds = data_loader.make_loco_splits(_fake_conditions())
    sp = next(iter(folds.values()))
    trial_to_set = {}
    for name in ("train", "val_matched", "test_matched", "val_shift", "test_shift"):
        for t in sp[name]:
            trial_to_set[t] = name
    windows = [(t, s) for t in trial_to_set for s in data_loader.window_starts(2.0)]
    by_set = data_loader.assign_windows_to_split(windows, trial_to_set)
    seen = set()
    for name, ws in by_set.items():
        for w in ws:
            assert w not in seen  # no (trial, start) shared across sets
            seen.add(w)
    assert len(seen) == len(windows)


def test_mck_trial_from_train_only():
    folds = data_loader.make_loco_splits(_fake_conditions())
    for sp in folds.values():
        pick = data_loader.pick_mck_fit_trial(sp["train"])
        assert pick in sp["train"]  # TRAIN condition only, never val/test/shift
    sp = next(iter(folds.values()))
    assert data_loader.pick_mck_fit_trial(sp["train"]) == data_loader.pick_mck_fit_trial(
        sp["train"]
    )  # deterministic
    with pytest.raises(ValueError):
        data_loader.pick_mck_fit_trial([])
