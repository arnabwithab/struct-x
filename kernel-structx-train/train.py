"""Struct-X training campaign kernel (F010). Phase 2: raw store + pilot LOCO fold.

Re-decodes all 30 trials with raw (unscaled) q, computes frozen train stats
per fold, fits m,c,k on ONE train trial, runs all 4 O4 variants on the pilot
fold (held-out On_set1), evaluates TEST trials, saves checkpoints + metrics.
"""

import json
import os
import subprocess
import sys

WORKING_DIR = "/kaggle/working"
PILOT_HELD = "On_set1"


def sh(cmd: str) -> str:
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


def find_dir(name: str) -> str:
    for root, dirs, _files in os.walk("/kaggle/input"):
        if name in dirs:
            return os.path.join(root, name)
    raise FileNotFoundError(f"{name} not found under /kaggle/input")


def main() -> None:
    import numpy as np
    import torch

    print("=== nvidia-smi ===")
    print(sh("nvidia-smi"))
    print("=== clone struct-x ===")
    print(
        sh(f"git clone -q https://github.com/arnabwithab/struct-x.git {WORKING_DIR}/structx-repo")
    )
    sys.path.insert(0, f"{WORKING_DIR}/structx-repo/src")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    from data_loader import (
        compute_train_stats,
        condition_of,
        make_loco_splits,
        pair_trials,
        parse_ldv_txt,
        pick_mck_fit_trial,
        trial_to_training_pairs,
    )
    from evaluate import delta_m, rel_freq_error, rmse
    from mck_fit import fit_mck
    from models.heads import WaveformFrequencyModel
    from models.resnet3d import ResNet3DEncoder
    from train import run_fold
    from voxel import compute_voxel_train_stats, voxel_to_model_input

    archive = find_dir("XJTU-DV-Beam")
    pairs = pair_trials(os.path.join(archive, "DV"), os.path.join(archive, "LDV"))
    print(f"{len(pairs)} paired trials")

    # ---- raw store (redo of phase 1 with unscaled q) ----
    raw, trial_ldv, trial_f = {}, {}, {}
    for trial_id, dv_path, ldv_path in pairs:
        win = trial_to_training_pairs(dv_path, ldv_path, n_bins=5, standardize_q=False)
        t_raw, v_raw = parse_ldv_txt(ldv_path)
        trial_ldv[trial_id] = (t_raw, v_raw)
        trial_f[trial_id] = float(fit_mck(v_raw, 1 / float(np.median(np.diff(t_raw))))["fd"])
        raw[trial_id] = win
        print(f"{trial_id}: {len(win)} windows, f_target={trial_f[trial_id]:.2f} Hz", flush=True)

    # ---- pilot fold ----
    conditions = {t: condition_of(t) for t, _, _ in pairs}
    folds = make_loco_splits(conditions, seed=0)
    fold = folds[PILOT_HELD]
    print("pilot fold:", {k: len(v) for k, v in fold.items()})
    train_ids, vm_ids, tm_ids, vs_ids, ts_ids = (
        fold["train"],
        fold["val_matched"],
        fold["test_matched"],
        fold["val_shift"],
        fold["test_shift"],
    )

    def windows_of(ids):
        out = []
        for t in ids:
            out.extend([(t, v, q) for v, q in raw[t]])
        return out

    train_w = windows_of(train_ids)
    q_mean, q_std = compute_train_stats([q for _, _, q in train_w])
    vx_mean, vx_std = compute_voxel_train_stats([voxel_to_model_input(v)[0] for _, v, _ in train_w])
    print(f"frozen stats: q_mean={q_mean:.4f} q_std={q_std:.4f}", flush=True)

    def batchify(items, micro):
        mu = torch.from_numpy(vx_mean).view(-1, 1, 1).float()
        sigma = torch.from_numpy(vx_std).view(-1, 1, 1).float()
        batches = []
        for i in range(0, len(items), micro):
            chunk = items[i : i + micro]
            v = torch.stack(
                [
                    (torch.from_numpy(voxel_to_model_input(w[1])[0]).float() - mu) / (sigma + 1e-6)
                    for w in chunk
                ]
            )
            q = torch.stack([torch.from_numpy((w[2] - q_mean) / (q_std + 1e-6)) for w in chunk])
            f = torch.tensor([trial_f[w[0]] for w in chunk], dtype=torch.float32)
            batches.append((v, q.float(), f))
        return batches
        return batches

    def make_model():
        return WaveformFrequencyModel(ResNet3DEncoder(512), 512)

    # micro-batch probe for OOM ladder (design sec 10: start 8 on T4)
    micro = 8
    try:
        m = make_model().to(device)
        with torch.no_grad():
            m(torch.randn(micro, 2, 5, 224, 224, device=device))
        del m
        torch.cuda.empty_cache()
    except RuntimeError:
        micro = 4
        torch.cuda.empty_cache()
    print("micro_batch:", micro, flush=True)

    mck_trial = pick_mck_fit_trial(train_ids)
    t_mck, v_mck = trial_ldv[mck_trial]
    mck = fit_mck(v_mck, 1 / float(np.median(np.diff(t_mck))))
    m, c, k = mck["m"], mck["c"], mck["k"]
    print(
        f"mck from {mck_trial}: fd={mck['fd']:.2f} fn={mck['fn']:.2f} m={m} k={k:.1f} c={c:.4f}",
        flush=True,
    )

    train_b = batchify(train_w, micro)
    vm_b = batchify(windows_of(vm_ids), micro)
    vs_b = batchify(windows_of(vs_ids), micro)
    results = {}
    for variant in ["data_only", "smoothness_only", "sdof_only", "both"]:
        ckpt = f"{WORKING_DIR}/ckpts/{PILOT_HELD}/{variant}.pt"
        res = run_fold(
            train_b,
            vm_b,
            vs_b,
            make_model,
            m,
            c,
            k,
            variant=variant,
            micro_batch=micro,
            device=device,
            ckpt_path=ckpt,
        )
        results[variant] = res
        show = {
            k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items() if k != "state"
        }
        print(f"{variant}: {json.dumps(show)}", flush=True)

    # ---- eval on TEST trials ----
    tm_b = batchify(windows_of(tm_ids), micro)
    ts_b = batchify(windows_of(ts_ids), micro)
    metrics = {}
    for variant in results:
        model = make_model().to(device)
        model.load_state_dict(
            torch.load(f"{WORKING_DIR}/ckpts/{PILOT_HELD}/{variant}.pt", map_location=device)
        )
        model.eval()
        for split_name, batches in [("test_matched", tm_b), ("test_shift", ts_b)]:
            rms, fr = [], []
            with torch.no_grad():
                for v, q, f in batches:
                    qh, fh = model(v.to(device))
                    qh = qh.cpu().numpy()
                    for i in range(len(q)):
                        rms.append(rmse(q[i].numpy(), qh[i]))
                        fr.append(rel_freq_error(float(fh[i].cpu()), float(f[i])))
            key = f"{variant}/{split_name}"
            metrics[key] = {
                "rmse_mean": float(np.mean(rms)),
                "rmse_std": float(np.std(rms, ddof=1)),
                "freq_mean": float(np.mean(fr)),
                "freq_std": float(np.std(fr, ddof=1)),
            }
    for variant in results:
        dm = delta_m(
            metrics[f"{variant}/test_matched"]["rmse_mean"],
            metrics[f"{variant}/test_shift"]["rmse_mean"],
        )
        print(
            f"{variant}: Delta_RMSE(test)={dm:.4f} matched={metrics[f'{variant}/test_matched']['rmse_mean']:.4f}",
            flush=True,
        )
    with open(f"{WORKING_DIR}/pilot_results.json", "w") as f:
        json.dump(
            {
                "results": results,
                "metrics": metrics,
                "mck": {
                    k: (float(v) if k in ("k", "c", "fn", "fd") else v) for k, v in mck.items()
                },
                "mck_trial": mck_trial,
                "micro": micro,
            },
            f,
        )
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
