"""Per-trial campaign eval (design sec 9): proper trial-level Delta/se/O5.

Replicates the kernel's frozen stats + normalization exactly (same code path),
loads ckpts from outputs/v6+v7, and aggregates per TRIAL (not pooled windows).
Writes results/final_verdict.json. CPU-only, no GPU touched.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from data_loader import (
    compute_train_stats,
    condition_of,
    make_loco_splits,
    pair_trials,
    parse_ldv_txt,
    trial_to_training_pairs,
)
from evaluate import (
    delta_m,
    matched_degradation_report,
    rel_freq_error,
    rmse,
    rq4_compare,
    se_delta,
    summarize_trials,
)
from mck_fit import fit_mck
from models.heads import WaveformFrequencyModel
from models.resnet3d import ResNet3DEncoder
from voxel import voxel_to_model_input

VARIANTS = ["data_only", "smoothness_only", "sdof_only", "both"]


def trial_metrics(model, windows, f_target, device="cpu"):
    """Per-trial (rmse, freq_err) means over that trial's windows.

    windows: (trial_id, model-ready (2,5,224,224), q (1000,)) already normalized.
    """
    model.eval()
    per_trial = {}
    with torch.no_grad():
        for trial_id, v, q in windows:
            x = torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(0).to(device)
            qh, fh = model(x)
            qh = qh.cpu().numpy().ravel()
            per_trial.setdefault(trial_id, []).append(
                (rmse(q, qh), rel_freq_error(float(fh.cpu()), f_target[trial_id]))
            )
    return {
        t: (float(np.mean([r for r, _ in v])), float(np.mean([f for _, f in v])))
        for t, v in per_trial.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="beam-dataset/XJTU-DV-Beam")
    ap.add_argument("--ckpts", default="outputs")
    ap.add_argument("--out", default="results/final_verdict.json")
    args = ap.parse_args()

    pairs = pair_trials(os.path.join(args.archive, "DV"), os.path.join(args.archive, "LDV"))
    raw, trial_f = {}, {}
    for trial_id, dv_path, ldv_path in pairs:
        win = trial_to_training_pairs(dv_path, ldv_path, n_bins=5, standardize_q=False)
        t_raw, v_raw = parse_ldv_txt(ldv_path)
        trial_f[trial_id] = float(fit_mck(v_raw, 1 / float(np.median(np.diff(t_raw))))["fd"])
        raw[trial_id] = [(trial_id, v, q) for v, q in win]
        print(f"{trial_id}: {len(win)} windows", flush=True)

    conditions = {t: condition_of(t) for t, _, _ in pairs}
    folds = make_loco_splits(conditions, seed=0)
    verdict = {"folds": {}}
    for held, fold in folds.items():
        train_w = [w for t in fold["train"] for w in raw[t]]
        q_mean, q_std = compute_train_stats([w[2] for w in train_w])
        vx = np.stack([voxel_to_model_input(w[1])[0] for w in train_w])
        vx_mean, vx_std = vx.mean(axis=(0, 3, 4)), vx.std(axis=(0, 3, 4))

        def norm(windows):
            mu = vx_mean.reshape(2, -1, 1, 1)
            sg = vx_std.reshape(2, -1, 1, 1)
            out = []
            for t, v, q in windows:
                mv = (voxel_to_model_input(v)[0] - mu) / (sg + 1e-6)
                mq = (q - q_mean) / (q_std + 1e-6)
                out.append((t, mv.astype(np.float32), mq.astype(np.float32)))
            return out

        fold_res = {"mck_trial": None, "variants": {}}
        for variant in VARIANTS:
            ckpt = None
            for root in [
                os.path.join(args.ckpts, "v6", "ckpts", held),
                os.path.join(args.ckpts, "v7", "ckpts", held),
            ]:
                cand = os.path.join(root, f"{variant}.pt")
                if os.path.isfile(cand):
                    ckpt = cand
            if ckpt is None:
                print(f"SKIP {held}/{variant}: no ckpt", flush=True)
                continue
            model = WaveformFrequencyModel(ResNet3DEncoder(512), 512)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            tm = trial_metrics(
                model, norm([w for t in fold["test_matched"] for w in raw[t]]), trial_f
            )
            ts = trial_metrics(
                model, norm([w for t in fold["test_shift"] for w in raw[t]]), trial_f
            )
            m_r = [tm[t][0] for t in sorted(tm)]
            s_r = [ts[t][0] for t in sorted(ts)]
            m_f = [tm[t][1] for t in sorted(tm)]
            s_f = [ts[t][1] for t in sorted(ts)]
            fold_res["variants"][variant] = {
                "rmse_matched": summarize_trials(m_r),
                "rmse_shift": summarize_trials(s_r),
                "freq_matched": summarize_trials(m_f),
                "freq_shift": summarize_trials(s_f),
                "delta_rmse": delta_m(m_r, s_r),
                "se_rmse": se_delta(m_r, s_r),
                "delta_freq": delta_m(m_f, s_f),
                "n_matched": len(m_r),
                "n_shift": len(s_r),
            }
        v = fold_res["variants"]
        if "data_only" in v and "both" in v:
            b, p = v["data_only"], v["both"]
            fold_res["o5_rmse"] = dict(
                zip(
                    ["R", "sigma_R", "wins", "statement"],
                    [
                        b["delta_rmse"] - p["delta_rmse"],
                        float(np.sqrt(b["se_rmse"] ** 2 + p["se_rmse"] ** 2)),
                        None,
                        "",
                    ],
                )
            )
            r, s = fold_res["o5_rmse"]["R"], fold_res["o5_rmse"]["sigma_R"]
            fold_res["o5_rmse"]["wins"] = bool(r > s)
            fold_res["o5_rmse"]["statement"] = (
                f"physics wins: R={r:.4f} > sigma_R={s:.4f}"
                if r > s
                else f"NON-IDENTIFIABLE: R={r:.4f} <= sigma_R={s:.4f}"
            )
            fold_res["rq4"] = rq4_compare(b["delta_freq"], b["delta_rmse"])
            fold_res["matched_harm"] = matched_degradation_report(
                [b["rmse_matched"]["mean"]], [p["rmse_matched"]["mean"]]
            )
        verdict["folds"][held] = fold_res
        print(
            f"fold {held}: " + (fold_res.get("o5_rmse", {}).get("statement", "incomplete")),
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
