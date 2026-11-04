"""F014-3 helpers: representational probes + cross-condition latent stability.

Pure functions only (no checkpoints, no GPU, no IO) so the driver
scripts/f014_layers3.py stays a thin inference wrapper. Probe protocol:
linear Ridge/LogisticRegression on frozen window latents with trial-level
splits (window leakage forbidden); stability: cosine + linear CKA between
matched-trial and shift-trial latent groups with trial-bootstrap CIs.
"""

import numpy as np
from scipy.signal import hilbert
from sklearn.linear_model import LogisticRegression, Ridge


def trial_probe_split(trial_ids, conditions, seed=0, n_test_per_condition=1):
    """Stratified trial split: hold out n trials per condition for probe-test.

    Deterministic given seed. Trial-level only -- windows never cross sides.
    """
    rng = np.random.RandomState(seed)
    by_cond = {}
    for t in sorted(set(trial_ids)):
        by_cond.setdefault(conditions[t], []).append(t)
    train, test = [], []
    for cond in sorted(by_cond):
        group = sorted(by_cond[cond])
        rng.shuffle(group)
        n_test = min(n_test_per_condition, max(1, len(group) - 1)) if len(group) > 1 else 0
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    return {"train": sorted(train), "test": sorted(test)}


def quantile_bin_edges(values, n_bins=3):
    """Interior quantile edges splitting values into n_bins ~equal-count bins."""
    v = np.asarray(values, dtype=float).ravel()
    return np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]).tolist()


def assign_bins(values, edges):
    """Ordinal bin label per value (searchsorted over edges, clipped)."""
    v = np.asarray(values, dtype=float).ravel()
    return np.clip(np.searchsorted(np.asarray(edges, dtype=float), v), 0, len(edges)).astype(int)


def envelope_slope(q, fs=1000.0):
    """Per-second slope of the log Hilbert envelope (free-decay rate sign)."""
    q = np.asarray(q, dtype=float).ravel()
    env = np.abs(hilbert(q))
    env = env[np.isfinite(env)]
    if env.size < 8 or np.all(env <= 0):
        return 0.0
    env = np.clip(env, 1e-12, None)
    t = np.arange(env.size) / float(fs)
    return float(np.polyfit(t, np.log(env), 1)[0])


def centroid(z):
    """Trial/group latent centroid (mean over rows)."""
    return np.asarray(z, dtype=float).mean(axis=0)


def cosine_similarity(a, b):
    """Cosine similarity; 1.0 for identical, -1.0 for opposite, 0.0 if degenerate."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _center_columns(m):
    m = np.asarray(m, dtype=float)
    return m - m.mean(axis=0, keepdims=True)


def linear_cka(x, y, seed=0):
    """Linear CKA in [0,1] (biased HSIC estimator; Kornblith et al. 2019).

    Biased form gives exactly 1.0 for x == y (pinned by test). Unpaired
    groups (different row counts) are equalized to min(n) rows with a fixed
    seed first -- the cross-condition comparison has no pairing, so count
    matching is the honest precondition, made explicit here.
    """
    x, y = equalize_counts(x, y, seed=seed)
    xc, yc = _center_columns(x), _center_columns(y)
    cross = float(np.linalg.norm(yc.T @ xc, ord="fro") ** 2)
    denom = float(np.linalg.norm(xc.T @ xc, ord="fro") * np.linalg.norm(yc.T @ yc, ord="fro"))
    if denom == 0.0:
        return 1.0 if cross == 0.0 else 0.0
    return min(1.0, max(0.0, cross / denom))


def equalize_counts(x, y, seed=0):
    """Subsample the larger group (fixed seed) so both have min(n) rows for CKA."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(x.shape[0], y.shape[0])
    rng = np.random.RandomState(seed)
    if x.shape[0] > n:
        x = x[rng.choice(x.shape[0], n, replace=False)]
    if y.shape[0] > n:
        y = y[rng.choice(y.shape[0], n, replace=False)]
    return x, y


def fit_ridge_probe(x_train, y_train, alpha=1.0):
    """Fixed-alpha Ridge probe (continuous targets: fn value, envelope slope)."""
    return Ridge(alpha=float(alpha)).fit(np.asarray(x_train), np.asarray(y_train))


def fit_logreg_probe(x_train, y_train, seed=0):
    """Fixed-C LogisticRegression probe (categorical: fn bin, slope sign, condition)."""
    return LogisticRegression(C=1.0, max_iter=2000, random_state=int(seed)).fit(
        np.asarray(x_train), np.asarray(y_train)
    )


def probe_accuracy(model, x_test, y_test):
    """Classification accuracy on held-out trials."""
    return float(model.score(np.asarray(x_test), np.asarray(y_test)))


def probe_r2(model, x_test, y_test):
    """R^2 on held-out trials (may be negative: worse than mean)."""
    return float(model.score(np.asarray(x_test), np.asarray(y_test)))


def majority_baseline(y_test):
    """Chance rate: majority-class fraction (the bar a real probe must clear)."""
    y = np.asarray(y_test).ravel()
    if y.size == 0:
        return float("nan")
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / y.size)


def permute_labels(y_train, seed=0):
    """Fixed-seed label permutation for the shuffled-label control probe."""
    y = np.asarray(y_train)
    return y[np.random.RandomState(seed).permutation(y.shape[0])]


def bootstrap_ci(stat_fn, seed=0, n_boot=1000, ci=95.0):
    """Percentile CI over trial-bootstrap replicates; stat_fn(rng) -> float."""
    rng = np.random.RandomState(seed)
    vals = np.array([float(stat_fn(rng)) for _ in range(int(n_boot))])
    lo, hi = np.percentile(vals, [(100.0 - ci) / 2.0, 100.0 - (100.0 - ci) / 2.0])
    return float(lo), float(hi), vals
