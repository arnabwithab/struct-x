"""F013 LOCO driver: RidgeCV baseline, scored like the deep campaign (sec 9)."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from baseline_ridge import ALPHAS, fit_ridge_fold, pooled_features, predict_ridge, window_fft_peak
from data_loader import (
    apply_frozen_zscore,
    compute_train_stats,
    condition_of,
    make_loco_splits,
    pick_mck_fit_trial,
)
from evaluate import (
    delta_m,
    delta_r,
    l_phys,
    mae,
    pearson_r,
    rel_freq_error,
    rmse,
    rq4_abs,
    rq4_compare,
    se_delta,
    summarize_trials,
)
from mck_fit import fit_mck


def load_vox_dir(vox_dir):
    """trial_id -> (vox (n,C,H,W), q (n,T)) from beam-vox npz files."""
    data = {}
    for name in sorted(os.listdir(vox_dir)):
        if not name.endswith(".npz"):
            continue
        trial = os.path.splitext(name)[0]
        z = np.load(os.path.join(vox_dir, name))
        data[trial] = (np.asarray(z["vox"]), np.asarray(z["q"]))
    if not data:
        raise ValueError("no npz windows in " + vox_dir)
    return data


def stitch_trial(q_windows):
    """Approx full-trial signal from 0.5s-stride windows (train-mck fit only)."""
    q_windows = [np.asarray(w).ravel() for w in q_windows]
    return np.concatenate([q_windows[0]] + [w[500:] for w in q_windows[1:]])


def fold_mck(trial_data, train_trials):
    """m,c,k from ONE train trial only (same policy as the deep campaign)."""
    pick = pick_mck_fit_trial(train_trials)
    try:
        return fit_mck(stitch_trial(trial_data[pick][1]), 1000.0), pick
    except ValueError:
        return fit_mck(np.asarray(trial_data[pick][1][0]).ravel(), 1000.0), pick


def per_trial_scores(trial_data, fitted, trials, f_target, mck):
    """Per-trial means over that trial windows (trial-level, never pooled)."""
    out = {}
    for t in trials:
        vox, q = trial_data[t]
        qh = predict_ridge(fitted, pooled_features(vox))
        per_w = []
        for true, pred in zip(q, qh):
            per_w.append(
                (
                    rmse(pred, true),
                    mae(pred, true),
                    pearson_r(pred, true),
                    rel_freq_error(window_fft_peak(pred), f_target[t]),
                    l_phys(pred, mck["m"], mck["c"], mck["k"]),
                )
            )
        out[t] = tuple(float(np.mean([w[i] for w in per_w])) for i in range(5))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vox-dir", default="beam-vox")
    ap.add_argument("--out", default="results/ridge_verdict.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    trial_data = load_vox_dir(args.vox_dir)
    conditions = {t: condition_of(t) for t in trial_data}
    folds = make_loco_splits(conditions, seed=args.seed)
    f_target = {
        t: float(np.median([window_fft_peak(w) for w in q])) for t, (_, q) in trial_data.items()
    }

    verdict = {"variant": "ridge", "alphas": list(ALPHAS), "seed": args.seed, "folds": {}}
    print("held       n_tr   alpha    d_rmse      se   d_freq    rq4")
    for held in sorted(folds):
        fold = folds[held]
        q_mean, q_std = compute_train_stats([trial_data[t][1] for t in fold["train"]])
        zscored = {
            t: (v, apply_frozen_zscore(q, q_mean, q_std)) for t, (v, q) in trial_data.items()
        }
        fitted = fit_ridge_fold(zscored, fold["train"])
        mck, mck_trial = fold_mck(trial_data, fold["train"])
        alpha = float(np.median(np.atleast_1d(fitted["model"].alpha_)))
        tm = per_trial_scores(zscored, fitted, fold["test_matched"], f_target, mck)
        ts = per_trial_scores(zscored, fitted, fold["test_shift"], f_target, mck)
        m_r = [tm[t][0] for t in sorted(tm)]
        s_r = [ts[t][0] for t in sorted(ts)]
        m_f = [tm[t][3] for t in sorted(tm)]
        s_f = [ts[t][3] for t in sorted(ts)]
        d_rmse, se = delta_m(m_r, s_r), se_delta(m_r, s_r)
        d_freq = delta_m(m_f, s_f)
        rq4 = rq4_compare(d_freq, d_rmse)
        verdict["folds"][held] = {
            "mck_trial": mck_trial,
            "alpha_median": alpha,
            "n_train": len(fold["train"]),
            "q_mean": q_mean,
            "q_std": q_std,
            "rmse_matched": summarize_trials(m_r),
            "rmse_shift": summarize_trials(s_r),
            "freq_matched": summarize_trials(m_f),
            "freq_shift": summarize_trials(s_f),
            "mae_matched": summarize_trials([tm[t][1] for t in sorted(tm)]),
            "mae_shift": summarize_trials([ts[t][1] for t in sorted(ts)]),
            "pearson_matched": summarize_trials([tm[t][2] for t in sorted(tm)]),
            "pearson_shift": summarize_trials([ts[t][2] for t in sorted(ts)]),
            "lphys_matched": summarize_trials([tm[t][4] for t in sorted(tm)]),
            "lphys_shift": summarize_trials([ts[t][4] for t in sorted(ts)]),
            "delta_rmse": d_rmse,
            "se_rmse": se,
            "delta_freq": d_freq,
            "delta_pearson_r": delta_r(
                [tm[t][2] for t in sorted(tm)], [ts[t][2] for t in sorted(ts)]
            ),
            "rq4": rq4,
            "rq4_abs": rq4_abs(d_freq, d_rmse),
            "n_matched": len(m_r),
            "n_shift": len(s_r),
        }
        print(
            held,
            len(fold["train"]),
            round(alpha, 3),
            round(d_rmse, 4),
            round(se, 4),
            round(d_freq, 4),
            rq4,
            flush=True,
        )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=2)
    print("wrote " + args.out)


if __name__ == "__main__":
    main()
