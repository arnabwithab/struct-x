"""Struct-X training campaign kernel (F010). Phase 2: raw store + LOCO folds.

Re-decodes all 30 trials with raw (unscaled) q, then per fold: frozen train
stats, m,c,k from ONE train trial, all 4 O4 variants, TEST-trial eval, O5-ready
metrics. Campaign mode: FOLDS_TO_RUN lists the folds this version trains.
Hyperparams (lr/epochs/patience/grids) come from train.py constants, NOT
base.yaml — keep them in sync when train.py changes.
"""

import json
import os
import subprocess
import sys

WORKING_DIR = "/kaggle/working"
# F012 queue knobs: EDIT PER PUSH (Kaggle script kernels take no CLI args).
# Local runs may export STRUCTX_* instead. Defaults = F011 pilot (1 fold, full grid).
B = int(os.environ.get("STRUCTX_B", "5"))
USE_AMP = os.environ.get("STRUCTX_USE_AMP", "0") == "1"
REPR = os.environ.get("STRUCTX_REPR", "voxel")
FOLDS_TO_RUN = [f for f in os.environ.get("STRUCTX_FOLDS", "On_set1").split(",") if f]
VARIANTS_TO_RUN = [
    v
    for v in os.environ.get("STRUCTX_VARIANTS", "data_only,smoothness_only,sdof_only,both").split(
        ","
    )
    if v
]
USE_PRETRAINED = True
ENCODER = os.environ.get("STRUCTX_ENCODER", "timesformer")  # resnet = control path
assert ENCODER in ("resnet", "timesformer"), f"bad STRUCTX_ENCODER={ENCODER}"
USE_PRETRAINED = ENCODER == "timesformer"
KNOWN_VARIANTS = ("data_only", "smoothness_only", "sdof_only", "both")
B_EFF = 1 if REPR == "timesurface" else B  # surface (2,H,W) -> single-frame (2,1,224,224)


def validate_knobs(b, repr_kind, variants, folds_to_run, folds=None):
    """Fail fast on bad queue knobs (before decode; folds checked once known)."""
    assert b in (3, 5, 7, 9), f"bad B={b}"
    assert repr_kind in ("voxel", "timesurface"), f"bad REPR={repr_kind}"
    assert variants and set(variants) <= set(KNOWN_VARIANTS), f"bad VARIANTS={variants}"
    assert folds_to_run, "empty FOLDS"
    if folds is not None:
        unknown = set(folds_to_run) - set(folds)
        if unknown:
            raise ValueError(f"unknown folds: {sorted(unknown)}")
    return True


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

    validate_knobs(B, REPR, VARIANTS_TO_RUN, FOLDS_TO_RUN)  # before decode
    print("=== nvidia-smi ===")
    print(sh("nvidia-smi"))
    print("=== clone struct-x ===")
    print(
        sh(f"git clone -q https://github.com/arnabwithab/struct-x.git {WORKING_DIR}/structx-repo")
    )
    sys.path.insert(0, f"{WORKING_DIR}/structx-repo/src")
    print(
        f"B={B} frames={B_EFF} repr={REPR} amp={USE_AMP} folds={FOLDS_TO_RUN} variants={VARIANTS_TO_RUN}"
    )
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
    from models.timesformer import KINETICS_SHA256, KINETICS_URL, TimeSformerEncoder
    from train import run_fold
    from voxel import voxel_to_model_input

    archive = find_dir("XJTU-DV-Beam")
    pairs = pair_trials(os.path.join(archive, "DV"), os.path.join(archive, "LDV"))
    print(f"{len(pairs)} paired trials")

    # ---- raw store (unscaled q; frozen stats applied per fold) ----
    raw, trial_ldv, trial_f = {}, {}, {}
    for trial_id, dv_path, ldv_path in pairs:
        win = trial_to_training_pairs(
            dv_path, ldv_path, n_bins=B, repr_kind=REPR, standardize_q=False
        )
        t_raw, v_raw = parse_ldv_txt(ldv_path)
        trial_ldv[trial_id] = (t_raw, v_raw)
        trial_f[trial_id] = float(fit_mck(v_raw, 1 / float(np.median(np.diff(t_raw))))["fd"])
        raw[trial_id] = win
        print(f"{trial_id}: {len(win)} windows, f_target={trial_f[trial_id]:.2f} Hz", flush=True)

    conditions = {t: condition_of(t) for t, _, _ in pairs}
    folds = make_loco_splits(conditions, seed=0)
    validate_knobs(B, REPR, VARIANTS_TO_RUN, FOLDS_TO_RUN, folds)

    if USE_PRETRAINED:
        import urllib.request

        kin_path = f"{WORKING_DIR}/kinetics.bin"
        if not os.path.isfile(kin_path):
            print("downloading kinetics weights (once)...", flush=True)
            urllib.request.urlretrieve(KINETICS_URL, kin_path)
        kin_url = "file://" + kin_path

    def make_model():
        if USE_PRETRAINED:
            enc = TimeSformerEncoder(num_frames=B_EFF)
            enc.load_kinetics_weights(url=kin_url, sha256=KINETICS_SHA256)
            return WaveformFrequencyModel(enc, 768)
        return WaveformFrequencyModel(ResNet3DEncoder(512), 512)

    # micro-batch probe for OOM ladder (design sec 10: start 8 on T4).
    # Must include backward: no-grad forward alone passes while training OOMs.
    micro = 8
    for cand in (8, 4, 2):
        try:
            probe = make_model().to(device)
            probe.train()
            dummy = torch.randn(cand, 2, B_EFF, 224, 224, device=device)
            probe(dummy)[0].pow(2).mean().backward()
            del probe
            torch.cuda.empty_cache()
            micro = cand
            break
        except RuntimeError:
            torch.cuda.empty_cache()
    else:
        raise RuntimeError("OOM even at micro-batch 2; B too large for this GPU")
    print("micro_batch:", micro, flush=True)

    def windows_of(ids):
        out = []
        for t in ids:
            out.extend([(t, v, q) for v, q in raw[t]])
        return out

    for held in FOLDS_TO_RUN:
        fold = folds[held]
        print(f"--- fold {held}: ", {k: len(v) for k, v in fold.items()}, flush=True)
        train_ids = fold["train"]
        train_w = windows_of(train_ids)
        q_mean, q_std = compute_train_stats([q for _, _, q in train_w])
        vx_stack = np.stack([voxel_to_model_input(v)[0] for _, v, _ in train_w])
        vx_mean = vx_stack.mean(axis=(0, 3, 4))  # (N,2,B,H,W) -> per (polarity, bin)
        vx_std = vx_stack.std(axis=(0, 3, 4))
        mu = torch.from_numpy(vx_mean).view(2, -1, 1, 1).float()
        sigma = torch.from_numpy(vx_std).view(2, -1, 1, 1).float()

        def batchify(items, _mu=mu, _sg=sigma, _qm=q_mean, _qs=q_std):
            batches = []
            for i in range(0, len(items), micro):
                chunk = items[i : i + micro]
                v = torch.stack(
                    [
                        (torch.from_numpy(voxel_to_model_input(w[1])[0]).float() - _mu)
                        / (_sg + 1e-6)
                        for w in chunk
                    ]
                )
                q = torch.stack([torch.from_numpy((w[2] - _qm) / (_qs + 1e-6)) for w in chunk])
                f = torch.tensor([trial_f[w[0]] for w in chunk], dtype=torch.float32)
                batches.append((v, q.float(), f))
            return batches

        mck_trial = pick_mck_fit_trial(train_ids)
        t_mck, v_mck = trial_ldv[mck_trial]
        mck = fit_mck(v_mck, 1 / float(np.median(np.diff(t_mck))))
        m, c, k = mck["m"], mck["c"], mck["k"]
        print(
            f"mck from {mck_trial}: fd={mck['fd']:.2f} fn={mck['fn']:.2f} k={k:.1f} c={c:.4f}",
            flush=True,
        )

        train_b = batchify(train_w)
        vm_b = batchify(windows_of(fold["val_matched"]))
        vs_b = batchify(windows_of(fold["val_shift"]))
        results = {}
        for variant in VARIANTS_TO_RUN:
            ckpt = f"{WORKING_DIR}/ckpts/{held}/{variant}.pt"
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
                use_amp=USE_AMP,
                # lr/epochs/patience/grids: train.py defaults (not base.yaml)
            )
            results[variant] = res
            show = {
                kk: (round(vv, 5) if isinstance(vv, float) else vv)
                for kk, vv in res.items()
                if kk != "state"
            }
            print(f"{held}/{variant}: {json.dumps(show)}", flush=True)

        tm_b = batchify(windows_of(fold["test_matched"]))
        ts_b = batchify(windows_of(fold["test_shift"]))
        metrics = {}
        for variant in results:
            model = make_model().to(device)
            model.load_state_dict(
                torch.load(f"{WORKING_DIR}/ckpts/{held}/{variant}.pt", map_location=device)
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
                metrics[f"{variant}/{split_name}"] = {
                    "rmse_mean": float(np.mean(rms)),
                    "rmse_std": float(np.std(rms, ddof=1)),
                    "freq_mean": float(np.mean(fr)),
                    "freq_std": float(np.std(fr, ddof=1)),
                }
        for variant in results:
            dm = delta_m(
                [metrics[f"{variant}/test_matched"]["rmse_mean"]],
                [metrics[f"{variant}/test_shift"]["rmse_mean"]],
            )
            print(
                f"{held}/{variant}: Delta_RMSE(test)={dm:.4f} "
                f"matched={metrics[f'{variant}/test_matched']['rmse_mean']:.4f}",
                flush=True,
            )
        with open(f"{WORKING_DIR}/results_{held}.json", "w") as f:
            json.dump(
                {
                    "held": held,
                    "results": results,
                    "metrics": metrics,
                    "mck": {
                        kk: (float(vv) if kk in ("k", "c", "fn", "fd") else vv)
                        for kk, vv in mck.items()
                    },
                    "mck_trial": mck_trial,
                    "micro": micro,
                    "b": B,
                    "repr": REPR,
                    "use_amp": USE_AMP,
                },
                f,
            )
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
