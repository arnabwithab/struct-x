"""F013: ridge baseline on pooled voxel features (design sec 5, 9). Tests first.

Leakage: fit sees train-trial windows ONLY (scaler + coef from train).
Determinism: refit on same train trials -> identical predictions.
"""

import numpy as np

import baseline_ridge as br


def _toy_trial(rng, n, shift=0.0):
    vox = rng.normal(loc=shift, scale=1.0, size=(n, 4, 6, 8)).astype(np.float64)
    q = rng.normal(size=(n, 8)).astype(np.float64)
    return vox, q


def test_pooled_features_are_per_channel_means():
    rng = np.random.RandomState(0)
    vox = rng.normal(size=(3, 4, 6, 8))
    got = br.pooled_features(vox)
    assert got.shape == (3, 4)
    assert np.allclose(got, vox.mean(axis=(2, 3)))


def test_fit_uses_train_trials_only():
    rng = np.random.RandomState(1)
    va, qa = _toy_trial(rng, 6, shift=0.0)
    vb, qb = _toy_trial(rng, 6, shift=0.0)
    vs, qs = _toy_trial(rng, 6, shift=100.0)  # shifted trial: must not leak
    data = {"a": (va, qa), "b": (vb, qb), "s": (vs, qs)}
    fit = br.fit_ridge_fold(data, ["a", "b"])
    assert sorted(fit["train_trials"]) == ["a", "b"]
    xa = br.pooled_features(va)
    xb = br.pooled_features(vb)
    expect_mean = np.concatenate([xa, xb]).mean(axis=0)
    assert np.allclose(fit["scaler"].mean_, expect_mean)
    xs = br.pooled_features(vs)
    assert not np.allclose(fit["scaler"].mean_, np.concatenate([xa, xb, xs]).mean(axis=0))


def test_refit_determinism():
    rng = np.random.RandomState(2)
    data = {"a": _toy_trial(rng, 8), "b": _toy_trial(rng, 8)}
    f1 = br.fit_ridge_fold(data, ["a", "b"])
    f2 = br.fit_ridge_fold(data, ["a", "b"])
    assert np.allclose(f1["model"].coef_, f2["model"].coef_)
    rng2 = np.random.RandomState(3)
    x = rng2.normal(size=(5, f1["model"].coef_.shape[1]))
    assert np.allclose(br.predict_ridge(f1, x), br.predict_ridge(f2, x))


def test_predict_shape_and_finite():
    rng = np.random.RandomState(4)
    data = {"a": _toy_trial(rng, 8), "b": _toy_trial(rng, 8)}
    fit = br.fit_ridge_fold(data, ["a", "b"])
    x = br.pooled_features(_toy_trial(rng, 3)[0])
    got = br.predict_ridge(fit, x)
    assert got.shape == (3, 8)
    assert np.all(np.isfinite(got))


def test_window_fft_peak_recovers_sine():
    fs = 1000.0
    t = np.arange(1000) / fs
    q = np.sin(2.0 * np.pi * 50.0 * t)
    assert abs(br.window_fft_peak(q, fs) - 50.0) < 1.0
