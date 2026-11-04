"""F014-3: representational probes + cross-condition latent stability.

Pure-function contract (no checkpoints, no GPU): trial-level probe splits
(window leakage forbidden), linear probes (Ridge/LogisticRegression),
linear CKA + cosine stability, trial-bootstrap CIs, shuffled-label control.
"""

import numpy as np

from f014_layers3 import (
    assign_bins,
    bootstrap_ci,
    centroid,
    cosine_similarity,
    envelope_slope,
    equalize_counts,
    fit_logreg_probe,
    fit_ridge_probe,
    linear_cka,
    majority_baseline,
    permute_labels,
    probe_accuracy,
    probe_r2,
    quantile_bin_edges,
    trial_probe_split,
)


def _toy_2class(seed=0):
    rng = np.random.default_rng(seed)
    xa = rng.normal(-2.0, 0.5, size=(200, 8))
    xb = rng.normal(2.0, 0.5, size=(200, 8))
    x = np.vstack([xa, xb])
    y = np.array([0] * 200 + [1] * 200)
    trials = [f"cA_t{i // 40}" for i in range(200)] + [f"cB_t{i // 40}" for i in range(200)]
    conds = {t: ("cA" if t.startswith("cA") else "cB") for t in set(trials)}
    return x, y, trials, conds


def test_probe_determinism_fixed_seed():
    x, y, trials, conds = _toy_2class()
    split = trial_probe_split(trials, conds, seed=0)
    tr = [i for i, t in enumerate(trials) if t in split["train"]]
    te = [i for i, t in enumerate(trials) if t in split["test"]]
    a1 = probe_accuracy(fit_logreg_probe(x[tr], y[tr], seed=0), x[te], y[te])
    a2 = probe_accuracy(fit_logreg_probe(x[tr], y[tr], seed=0), x[te], y[te])
    r1 = probe_r2(fit_ridge_probe(x[tr], y[tr]), x[te], y[te])
    r2 = probe_r2(fit_ridge_probe(x[tr], y[tr]), x[te], y[te])
    assert a1 == a2 and a1 > 0.9
    assert r1 == r2 and r1 > 0.8


def test_linear_cka_self_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 16))
    assert abs(linear_cka(x, x) - 1.0) < 1e-6
    y = rng.normal(size=(25, 16))
    assert 0.0 <= linear_cka(x, y) <= 1.0


def test_cosine_self_is_one():
    rng = np.random.default_rng(1)
    v = rng.normal(size=32)
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9
    assert abs(cosine_similarity(v, -v) + 1.0) < 1e-9


def test_probe_split_no_window_leakage():
    _, _, trials, conds = _toy_2class()
    s1 = trial_probe_split(trials, conds, seed=0)
    s2 = trial_probe_split(trials, conds, seed=0)
    assert s1 == s2  # deterministic
    assert set(s1["train"]).isdisjoint(s1["test"])  # no shared trial
    assert set(s1["train"]) | set(s1["test"]) == set(trials)  # full coverage
    for c in ("cA", "cB"):  # stratified: every condition on both sides
        assert any(conds[t] == c for t in s1["train"])
        assert any(conds[t] == c for t in s1["test"])


def test_envelope_slope_sign():
    t = np.arange(1000) / 1000.0
    decay = np.sin(2 * np.pi * 30 * t) * np.exp(-3 * t)
    grow = np.sin(2 * np.pi * 30 * t) * np.exp(3 * (t - 1.0))
    assert envelope_slope(decay) < 0.0
    assert envelope_slope(grow) > 0.0


def test_shuffled_control_near_chance():
    x, y, trials, conds = _toy_2class()
    split = trial_probe_split(trials, conds, seed=0)
    tr = [i for i, t in enumerate(trials) if t in split["train"]]
    te = [i for i, t in enumerate(trials) if t in split["test"]]
    assert probe_accuracy(fit_logreg_probe(x[tr], y[tr], seed=0), x[te], y[te]) > 0.9
    ys = permute_labels(y[tr], seed=0)
    assert not np.array_equal(ys, y[tr])
    acc_s = probe_accuracy(fit_logreg_probe(x[tr], ys, seed=0), x[te], y[te])
    assert acc_s <= majority_baseline(y[te]) + 0.15  # chance, not magic


def test_bootstrap_ci_constant_collapses():
    lo, hi, _ = bootstrap_ci(lambda rng: 0.5, seed=0, n_boot=50)
    assert lo == hi == 0.5


def test_quantile_bins_cover_all_labels():
    vals = np.array([6.5, 6.6, 6.7, 6.8, 6.9, 7.0])
    edges = quantile_bin_edges(vals, n_bins=3)
    assert assign_bins(vals, edges).tolist() == [0, 0, 1, 1, 2, 2]


def test_equalize_and_centroid_shapes():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(10, 4))
    y = rng.normal(size=(6, 4))
    xe, ye = equalize_counts(x, y, seed=0)
    assert xe.shape == ye.shape == (6, 4)
    assert centroid(x).shape == (4,)
