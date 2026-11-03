"""F014 layers 1+4: calibration/selection helpers + figure-regeneration determinism.

Layer 1 (behavioral) + Layer 4 (L_phys-vs-error calibration) run inference-only
on existing checkpoints. The determinism test pins figure regeneration:
same predictions JSON + fixed seed -> byte-identical PNG.
"""

import hashlib
import json

import numpy as np

from f014_layers14 import (
    calibration_corr,
    fig_to_png_bytes,
    overlay_fig,
    pick_best_worst,
    scatter_fig,
    spectrum_fig,
)


def test_pick_best_worst_by_trial_rmse():
    trial_rmse = {"t1": 0.5, "t2": 0.1, "t3": 0.9}
    assert pick_best_worst(trial_rmse) == ("t2", "t3")


def test_calibration_corr_perfect_linear():
    err = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    lp = 3.0 * err + 1.0
    out = calibration_corr(lp, err)
    assert abs(out["pearson"] - 1.0) < 1e-9
    assert abs(out["spearman"] - 1.0) < 1e-9


def test_calibration_corr_constant_lphys_is_nan():
    err = np.array([0.1, 0.2, 0.3, 0.4])
    lp = np.full_like(err, 2.0)
    out = calibration_corr(lp, err)
    assert np.isnan(out["pearson"]) and np.isnan(out["spearman"])


def _toy_waveforms():
    rng = np.random.default_rng(0)
    t = np.arange(1000) / 1000.0
    q = np.sin(2 * np.pi * 30 * t) * np.exp(-3 * t)
    qh = q + rng.normal(0, 0.05, size=q.shape)
    return q, qh


def _png_hash(fig):
    return hashlib.sha256(fig_to_png_bytes(fig)).hexdigest()


def test_figure_regeneration_deterministic():
    q, qh = _toy_waveforms()
    lp = np.abs(np.linspace(0.1, 0.9, 20))
    err = np.linspace(0.1, 0.5, 20)
    figs_a = [
        overlay_fig(q, qh, q, qh + 0.5, "t"),
        spectrum_fig(q, qh, 1000.0, 30.0, "t"),
        scatter_fig(lp, err, 0.1, 0.2, "t"),
    ]
    figs_b = [
        overlay_fig(q, qh, q, qh + 0.5, "t"),
        spectrum_fig(q, qh, 1000.0, 30.0, "t"),
        scatter_fig(lp, err, 0.1, 0.2, "t"),
    ]
    assert [_png_hash(f) for f in figs_a] == [_png_hash(f) for f in figs_b]


def test_error_table_json_roundtrip():
    rows = [{"trial": "Off_set1_trail1", "rmse": 0.5, "split": "matched"}]
    assert json.loads(json.dumps(rows)) == rows
