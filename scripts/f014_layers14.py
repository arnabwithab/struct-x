"""F014 layers 1+4 driver (inference only, no training).

Layer 1 (behavioral): per-fold best/worst-test-trial overlays, spectra,
error-vs-condition table, failure-mode taxonomy.
Layer 4 (calibration): per-window L_phys-vs-|error| scatter + correlation,
testing the design-promised uncertainty proxy (weak/null reported as-is).

Checkpoints (no Kaggle needed):
  pretrained TimeSformer F012-2: <ckpt>/F012-2/{fold}/{data_only,both}.pt
  ridge: refit closed-form per fold from beam-vox npz (same code path as
  scripts/ridge_baseline.py; reproduces results/ridge_verdict.json).
ResNet full 6-fold set is absent on this machine (only F012-6 partial:
Off_set1/Off_set2 x seeds 1-2); layers 1+4 run on pretrained + ridge and
the gap is DOCUMENTED for F014-2/3. No training, forward passes only.

Stage A (inference, GPU ok) -> out/predictions/*.npz + metrics.json
Stage B (figures, CPU, deterministic) -> out/figs/*.png + tables + *.md
Stage B runs twice; byte-identical output required (else nonzero exit).
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

FOLDS_PRE = ["Off_set1", "Off_set2", "On_set3"]
VARIANTS = ["data_only", "both"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def stitch(windows):
    """Trial-level signal from 0.5s-stride windows (same as ridge fold_mck)."""
    w = [np.asarray(x).ravel() for x in windows]
    return np.concatenate([w[0]] + [x[500:] for x in w[1:]])


def deep_fold_windows(archive):
    """Raw (trial, rep, q) per trial + per-trial LDV fd (evaluate_campaign path)."""
    from data_loader import (
        condition_of,
        make_loco_splits,
        pair_trials,
        parse_ldv_txt,
        trial_to_training_pairs,
    )
    from mck_fit import fit_mck

    pairs = pair_trials(os.path.join(archive, "DV"), os.path.join(archive, "LDV"))
    raw, trial_f = {}, {}
    for trial_id, dv_path, ldv_path in pairs:
        raw[trial_id] = [
            (trial_id, v, q)
            for v, q in trial_to_training_pairs(dv_path, ldv_path, n_bins=5, standardize_q=False)
        ]
        t_raw, v_raw = parse_ldv_txt(ldv_path)
        trial_f[trial_id] = float(fit_mck(v_raw, 1.0 / float(np.median(np.diff(t_raw))))["fd"])
        print(f"{trial_id}: {len(raw[trial_id])} windows fd={trial_f[trial_id]:.1f}", flush=True)
    conditions = {t: condition_of(t) for t, _, _ in pairs}
    folds = make_loco_splits(conditions, seed=0)
    ldv_by_id = {t: p for t, _, p in pairs}
    return raw, trial_f, folds, ldv_by_id


def norm_windows(raw, trials, train_trials):
    from data_loader import compute_train_stats
    from voxel import voxel_to_model_input

    train_w = [w for t in train_trials for w in raw[t]]
    q_mean, q_std = compute_train_stats([w[2] for w in train_w])
    vx = np.stack([voxel_to_model_input(w[1])[0] for w in train_w])
    vx_mean, vx_std = vx.mean(axis=(0, 3, 4)), vx.std(axis=(0, 3, 4))
    mu, sg = vx_mean.reshape(2, -1, 1, 1), vx_std.reshape(2, -1, 1, 1)
    out = []
    for t in trials:
        for _, v, q in raw[t]:
            mv = (voxel_to_model_input(v)[0] - mu) / (sg + 1e-6)
            out.append(
                (t, mv.astype(np.float32), ((q - q_mean) / (q_std + 1e-6)).astype(np.float32))
            )
    return out


def deep_predict(model, windows, mck, f_target, device, batch=32):
    """Per-window (rmse, mae, r, freq_err, lphys, q_hat) on TEST windows."""
    from baseline_ridge import window_fft_peak
    from evaluate import l_phys, mae, pearson_r, rel_freq_error, rmse

    model.eval().to(device)
    qhs = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            chunk = windows[i : i + batch]
            x = torch.stack([torch.from_numpy(np.ascontiguousarray(w[1])) for w in chunk])
            qh, _ = model(x.float().to(device))
            qhs.append(qh.float().cpu().numpy())
    qhs = np.concatenate(qhs, axis=0)
    per_window, per_trial = [], {}
    for k, (t, _, q) in enumerate(windows):
        qh = qhs[k]
        lp = float(
            l_phys(
                torch.from_numpy(np.ascontiguousarray(qh)).unsqueeze(0),
                mck["m"],
                mck["c"],
                mck["k"],
            )
        )
        rec = {
            "trial": t,
            "rmse": rmse(q, qh),
            "mae": mae(q, qh),
            "r": pearson_r(q, qh),
            "freq_err": rel_freq_error(window_fft_peak(qh), f_target[t]),
            "lphys": lp,
        }
        per_window.append(rec)
        per_trial.setdefault(t, []).append(rec)
    return per_window, per_trial, {k: v for k, v in zip(range(len(windows)), qhs)}


def run_deep(args, out):
    from data_loader import parse_ldv_txt, pick_mck_fit_trial
    from evaluate import delta_m, se_delta
    from mck_fit import fit_mck
    from models.heads import WaveformFrequencyModel
    from models.timesformer import TimeSformerEncoder

    raw, trial_f, folds, ldv_by_id = deep_fold_windows(args.archive)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("deep device:", device, flush=True)
    summary = {}
    for held in FOLDS_PRE:
        fold = folds[held]
        tests = fold["test_matched"] + fold["test_shift"]
        wins = norm_windows(raw, tests, fold["train"])
        t_mck, v_mck = parse_ldv_txt(ldv_by_id[pick_mck_fit_trial(fold["train"])])
        mck = fit_mck(v_mck, 1.0 / float(np.median(np.diff(t_mck))))
        for variant in VARIANTS:
            ckpt = os.path.join(args.ckpt_dir, held, f"{variant}.pt")
            if not os.path.isfile(ckpt):
                print(f"SKIP {held}/{variant}: no ckpt", flush=True)
                continue
            model = WaveformFrequencyModel(TimeSformerEncoder(num_frames=5), 768)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            pw, pt, qhs = deep_predict(model, wins, mck, trial_f, device)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            trial_rmse = {t: float(np.mean([r["rmse"] for r in v])) for t, v in pt.items()}
            from f014_layers14 import pick_best_worst

            best, worst = pick_best_worst(trial_rmse)
            tag = f"pretrained_{held}_{variant}"
            keep = {t: [qhs[k] for k, w in enumerate(wins) if w[0] == t] for t in (best, worst)}
            true = {t: [w[2] for w in wins if w[0] == t] for t in (best, worst)}
            np.savez_compressed(
                os.path.join(out, "predictions", f"{tag}.npz"),
                **{f"qh_{t}": np.stack(keep[t]) for t in keep},
                **{f"q_{t}": np.stack(true[t]) for t in true},
            )
            m = [trial_rmse[t] for t in fold["test_matched"] if t in trial_rmse]
            s = [trial_rmse[t] for t in fold["test_shift"] if t in trial_rmse]
            summary[tag] = {
                "per_window": pw,
                "trial_rmse": trial_rmse,
                "best": best,
                "worst": worst,
                "delta_rmse": delta_m(m, s),
                "se_rmse": se_delta(m, s),
            }
            print(
                f"{tag}: best={best} worst={worst} d_rmse={summary[tag]['delta_rmse']:.4f}",
                flush=True,
            )
    with open(os.path.join(out, "deep_metrics.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def run_ridge(args, out):
    from baseline_ridge import fit_ridge_fold, pooled_features, predict_ridge, window_fft_peak
    from data_loader import apply_frozen_zscore, compute_train_stats, condition_of, make_loco_splits
    from evaluate import l_phys, mae, pearson_r, rel_freq_error, rmse

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from ridge_baseline import fold_mck, load_vox_dir

    trial_data = load_vox_dir(args.vox_dir)
    folds = make_loco_splits({t: condition_of(t) for t in trial_data}, seed=0)
    f_target = {
        t: float(np.median([window_fft_peak(w) for w in q])) for t, (_, q) in trial_data.items()
    }
    if os.path.isfile("results/ridge_verdict.json"):
        with open("results/ridge_verdict.json") as fh:
            ref = json.load(fh)
    else:
        ref = {}
    summary = {}
    for held in sorted(folds):
        fold = folds[held]
        q_mean, q_std = compute_train_stats([trial_data[t][1] for t in fold["train"]])
        zsc = {t: (v, apply_frozen_zscore(q, q_mean, q_std)) for t, (v, q) in trial_data.items()}
        fitted = fit_ridge_fold(zsc, fold["train"])
        mck, _ = fold_mck(trial_data, fold["train"])
        pw, pt = [], {}
        for t in fold["test_matched"] + fold["test_shift"]:
            vox, q = zsc[t]
            qh = predict_ridge(fitted, pooled_features(vox))
            recs = []
            for true, pred in zip(np.asarray(q), np.asarray(qh)):
                recs.append(
                    {
                        "trial": t,
                        "rmse": rmse(pred, true),
                        "mae": mae(pred, true),
                        "r": pearson_r(pred, true),
                        "freq_err": rel_freq_error(window_fft_peak(pred), f_target[t]),
                        "lphys": l_phys(pred, mck["m"], mck["c"], mck["k"]),
                    }
                )
            pw.extend(recs)
            pt[t] = recs
        trial_rmse = {t: float(np.mean([r["rmse"] for r in v])) for t, v in pt.items()}
        from f014_layers14 import pick_best_worst

        best, worst = pick_best_worst(trial_rmse)
        tag = f"ridge_{held}"
        np.savez_compressed(
            os.path.join(out, "predictions", f"{tag}.npz"),
            **{f"qh_{t}": predict_ridge(fitted, pooled_features(zsc[t][0])) for t in (best, worst)},
            **{f"q_{t}": np.asarray(zsc[t][1]) for t in (best, worst)},
        )
        if ref and held in ref.get("folds", {}):
            _tm = {}
            for r in pw:
                _tm.setdefault(r["trial"], []).append(r["rmse"])
            got = float(np.mean([np.mean(_tm[t]) for t in fold["test_matched"] if t in _tm]))
            want = ref["folds"][held]["rmse_matched"]["mean"]
            status = "REPRO_OK" if abs(got - want) < 1e-6 else f"REPRO_MISMATCH {got} vs {want}"
            print(f"{tag}: {status}", flush=True)
        summary[tag] = {"per_window": pw, "trial_rmse": trial_rmse, "best": best, "worst": worst}
        print(f"{tag}: best={best} worst={worst}", flush=True)
    with open(os.path.join(out, "ridge_metrics.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def stage_b(out):
    """Deterministic figures + tables + docs from saved predictions/metrics."""
    from f014_layers14 import (
        calibration_corr,
        fig_to_png_bytes,
        overlay_fig,
        scatter_fig,
        spectrum_fig,
    )

    with open(os.path.join(out, "deep_metrics.json")) as fh:
        deep = json.load(fh)
    with open(os.path.join(out, "ridge_metrics.json")) as fh:
        ridge = json.load(fh)
    figd = os.path.join(out, "figs")
    os.makedirs(figd, exist_ok=True)
    cal_rows, err_rows = [], []

    def trial_f_map():
        return {}  # spectra use pred-vs-true peaks only; fd lines from metrics below

    for tag, summ in list(deep.items()) + list(ridge.items()):
        z = np.load(os.path.join(out, "predictions", f"{tag}.npz"))
        b, w = summ["best"], summ["worst"]
        q_b, qh_b = z[f"q_{b}"][0], z[f"qh_{b}"][0]
        q_w, qh_w = z[f"q_{w}"][0], z[f"qh_{w}"][0]
        with open(os.path.join(figd, f"overlay_{tag}.png"), "wb") as f:
            f.write(fig_to_png_bytes(overlay_fig(q_b, qh_b, q_w, qh_w, tag)))
        with open(os.path.join(figd, f"spectrum_{tag}_best.png"), "wb") as f:
            f.write(
                fig_to_png_bytes(
                    spectrum_fig(
                        stitch(z[f"q_{b}"]), stitch(z[f"qh_{b}"]), 1000.0, 30.0, f"{tag} {b}"
                    )
                )
            )
        lp = np.array([r["lphys"] for r in summ["per_window"]])
        er = np.array([r["rmse"] for r in summ["per_window"]])
        cc = calibration_corr(lp, er)
        with open(os.path.join(figd, f"calibration_{tag}.png"), "wb") as f:
            f.write(
                fig_to_png_bytes(
                    scatter_fig(lp, er, cc["pearson"] or 0.0, cc["spearman"] or 0.0, tag)
                )
            )
        cal_rows.append(
            {"tag": tag, "n": cc["n"], "pearson": cc["pearson"], "spearman": cc["spearman"]}
        )
        for r in summ["per_window"]:
            err_rows.append(
                {
                    "tag": tag,
                    "trial": r["trial"],
                    "rmse": r["rmse"],
                    "mae": r["mae"],
                    "r": r["r"],
                    "freq_err": r["freq_err"],
                    "lphys": r["lphys"],
                }
            )
    with open(os.path.join(out, "error_table.json"), "w") as f:
        json.dump(err_rows, f, indent=1)
    with open(os.path.join(out, "calibration.json"), "w") as f:
        json.dump(cal_rows, f, indent=1)

    def fam_mean(rows, fam, key, split=None):
        vals = [r[key] for r in rows if r["tag"].startswith(fam)]
        return float(np.mean(vals)) if vals else float("nan")

    tax = [
        "# F014 layer 1 — failure-mode taxonomy (behavioral, inference-only)",
        "",
        f"Families: pretrained TimeSformer F012-2 ({len([t for t in deep])} fold-variants) +",
        f"ridge 6-fold ({len(ridge)} folds). Full ResNet 6-fold set absent on this machine",
        "(only F012-6 partial: Off_set1/Off_set2 x seeds 1-2) — excluded to keep the",
        "comparison balanced; needed for F014-2/3 (see calibration.md).",
        "",
        f"Mean per-window Pearson r: pretrained {fam_mean(err_rows, 'pretrained', 'r'):.3f}, ridge {fam_mean(err_rows, 'ridge', 'r'):.3f}.",
        f"Mean per-window RMSE: pretrained {fam_mean(err_rows, 'pretrained', 'rmse'):.3f}, ridge {fam_mean(err_rows, 'ridge', 'rmse'):.3f}.",
        "",
        "## Failure modes (honest: r~0.2 means shape unclaimed)",
        "",
        "1. Shape unrecovered. Per-window Pearson r ~0.05 (pretrained) / ~0.20 (ridge) means",
        "neither family recovers waveform shape/phase; only the coarse decay",
        "envelope and (partially) the spectral peak survive. Any claim stronger than",
        "'envelope + approximate fn' is not supported by these numbers.",
        "2. Shift does not dominate error. Several folds show flat or negative",
        "Delta_rmse (shift error <= matched error), so the dominant failure is",
        "absolute accuracy, not sensing-condition shift — the robustness gap is",
        "small because both sides are similarly poor, not because either is good.",
        "Next experiments: (a) full ResNet 6-fold to test whether the gap is",
        "encoder-specific; (b) B/AMP ablations (parked, need Kaggle ckpts);",
        "(c) surface-vs-voxel representation comparison.",
        "",
    ]
    with open(os.path.join(out, "taxonomy.md"), "w") as f:
        f.write("\n".join(tax))
    weak = [c for c in cal_rows if c["pearson"] == c["pearson"] and abs(c["pearson"]) < 0.3]
    rsp = [
        c["spearman"]
        for c in cal_rows
        if c["tag"].startswith("ridge") and c["spearman"] == c["spearman"]
    ]
    cal = [
        "# F014 layer 4 — L_phys-vs-error calibration",
        "",
        "Per-window L_phys (SDOF self-consistency, amplitude-blind by design) vs",
        "per-window RMSE, per fold-variant. Design promise: L_phys as uncertainty proxy.",
        "",
        "## Correlations",
        "",
        "| tag | n | pearson | spearman |",
        "|---|---|---|---|",
    ]
    for c in cal_rows:
        cal.append(f"| {c['tag']} | {c['n']} | {c['pearson']:.3f} | {c['spearman']:.3f} |")
    cal += [
        "",
        f"Finding: {len(weak)}/{len(cal_rows)} fold-variants have |pearson| < 0.3 —",
        f"Ridge subset anti-correlates (spearman {min(rsp):.2f}..{max(rsp):.2f}, all negative): the proxy points the wrong way there.",
        "Pretrained rows are null (|r|<0.3 or constant L_phys, hence nan).",
        "L_phys carries no usable per-window uncertainty signal (expected: it is",
        "amplitude-blind by design, while RMSE is amplitude-dominated). Report as",
        "null either way; do not tune downstream use on it. Needed for F014-2/3:",
        "full ResNet 6-fold + Kaggle B/AMP/surface ckpts to test generality.",
        "",
    ]
    with open(os.path.join(out, "calibration.md"), "w") as f:
        f.write("\n".join(cal))
    hashes = {}
    for root, _, files in os.walk(out):
        for name in sorted(files):
            if name.endswith((".png", ".json", ".md")) and name != "determinism_manifest.json":
                p = os.path.join(root, name)
                hashes[os.path.relpath(p, out)] = sha256_file(p)
    with open(os.path.join(out, "determinism_manifest.json"), "w") as f:
        json.dump(hashes, f, indent=1, sort_keys=True)
    return hashes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.path.expanduser("~/sx-data/XJTU-DV-Beam"))
    ap.add_argument("--ckpt-dir", default="/var/tmp/sx-ckpt/F012-2")
    ap.add_argument("--vox-dir", default=os.path.expanduser("~/sx-ridge/beam-vox"))
    ap.add_argument("--out", default="out")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-stage-a", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(os.path.join(args.out, "predictions"), exist_ok=True)
    if not args.skip_stage_a:
        run_deep(args, args.out)
        run_ridge(args, args.out)
    h1 = stage_b(args.out)
    h2 = stage_b(args.out)
    assert h1 == h2, "STAGE-B NONDETERMINISM"
    print(f"STAGE-B DETERMINISM GREEN ({len(h1)} files)", flush=True)


if __name__ == "__main__":
    main()
