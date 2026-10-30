"""Remote LOCO campaign driver (F011 option B): pretrained TimeSformer, folds from CLI.

Resume-safe and reboot-safe: every stage checks its outputs first.
  - data zip: re-download only if missing/bad md5
  - decode: per-trial npz cache (raw voxels + raw q + f_target in manifest)
  - kinetics weights: downloaded once
  - folds: skipped when results_{held}.json already has all 4 variants
Run under nohup; re-running after a reboot resumes where it stopped.

Usage:
  PYTHONPATH=src python3 scripts/run_remote_folds.py \
    --data-root /media/snu/8050482150481FF0/structx \
    --folds On_set2,Off_set3
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

ZENODO_URL = "https://zenodo.org/records/17447083/files/XJTU-DV-Beam.zip?download=1"
EXPECTED_MD5 = "d3af5b54a227be70fa932d93323d5920"
VARIANTS = ["data_only", "smoothness_only", "sdof_only", "both"]


def log(msg):
    print(msg, flush=True)


def ensure_zip(data_root):
    import subprocess

    path = os.path.join(data_root, "XJTU-DV-Beam.zip")
    for attempt in range(5):
        if os.path.isfile(path):
            h = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() == EXPECTED_MD5:
                log("zip OK (cached)")
                return path
            log(f"zip md5 mismatch, resuming (attempt {attempt + 1})")
        else:
            log(f"downloading beam zip (~1.6 GB, attempt {attempt + 1})...")
        r = subprocess.run(["curl", "-C", "-", "-L", "-o", path, ZENODO_URL])
        if r.returncode != 0:
            log("curl failed, retrying...")
    raise RuntimeError("beam zip download failed after retries")


def decode_all(archive_dir, cache_dir, pairs):
    from data_loader import parse_ldv_txt, trial_to_training_pairs
    from mck_fit import fit_mck

    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, "manifest.json")
    manifest = {}
    if os.path.isfile(manifest_path):
        manifest = json.load(open(manifest_path))
    for trial_id, dv_path, ldv_path in pairs:
        out = os.path.join(cache_dir, f"{trial_id}.npz")
        if os.path.isfile(out) and trial_id in manifest:
            continue
        win = trial_to_training_pairs(dv_path, ldv_path, n_bins=5, standardize_q=False)
        t_raw, v_raw = parse_ldv_txt(ldv_path)
        fs = 1.0 / float(np.median(np.diff(t_raw)))
        fd = float(fit_mck(v_raw, fs)["fd"])
        np.savez_compressed(out, vox=np.stack([w[0] for w in win]), q=np.stack([w[1] for w in win]))
        manifest[trial_id] = {"n_windows": len(win), "f_target": fd, "fs": fs}
        log(f"decoded {trial_id}: {len(win)} windows, f={fd:.2f} Hz")
    json.dump(manifest, open(manifest_path, "w"))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--folds", default="On_set2,Off_set3")
    ap.add_argument("--micro", type=int, default=4)  # shared 24GB card: 4 fits alongside other jobs
    args = ap.parse_args()
    folds_wanted = args.folds.split(",")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")

    from data_loader import (
        compute_train_stats,
        condition_of,
        make_loco_splits,
        pair_trials,
        parse_ldv_txt,
        pick_mck_fit_trial,
    )
    from evaluate import rel_freq_error, rmse
    from mck_fit import fit_mck
    from models.heads import WaveformFrequencyModel
    from models.timesformer import KINETICS_SHA256, KINETICS_URL, TimeSformerEncoder
    from train import run_fold
    from voxel import voxel_to_model_input

    data_root = args.data_root
    os.makedirs(data_root, exist_ok=True)
    zip_path = ensure_zip(data_root)
    archive = os.path.join(data_root, "XJTU-DV-Beam")
    if not os.path.isdir(os.path.join(archive, "DV")):
        import zipfile

        log("extracting archive...")
        zipfile.ZipFile(zip_path).extractall(data_root)

    pairs = pair_trials(os.path.join(archive, "DV"), os.path.join(archive, "LDV"))
    manifest = decode_all(archive, os.path.join(data_root, "decode"), pairs)
    raw = {}
    for trial_id, _, _ in pairs:
        d = np.load(os.path.join(data_root, "decode", f"{trial_id}.npz"))
        raw[trial_id] = [(trial_id, v, q) for v, q in zip(d["vox"], d["q"])]

    kin_path = os.path.join(data_root, "kinetics.bin")
    if not os.path.isfile(kin_path):
        log("downloading kinetics weights (once)...")
        urllib.request.urlretrieve(KINETICS_URL, kin_path)

    def make_model():
        enc = TimeSformerEncoder(num_frames=5)
        enc.load_kinetics_weights(url="file://" + kin_path, sha256=KINETICS_SHA256)
        return WaveformFrequencyModel(enc, 768)

    micro = args.micro
    try:
        with torch.no_grad():
            make_model().to(device)(torch.randn(micro, 2, 5, 224, 224, device=device))
        torch.cuda.empty_cache()
    except RuntimeError:
        micro = 4
        torch.cuda.empty_cache()
    log(f"micro_batch: {micro}")

    conditions = {t: condition_of(t) for t, _, _ in pairs}
    folds = make_loco_splits(conditions, seed=0)

    def windows_of(ids):
        return [w for t in ids for w in raw[t]]

    for held in folds_wanted:
        res_path = os.path.join(data_root, f"results_{held}.json")
        if os.path.isfile(res_path):
            done = json.load(open(res_path)).get("variants", {})
            if all(v in done for v in VARIANTS):
                log(f"fold {held}: already complete, skipping")
                continue
        fold = folds[held]
        train_w = windows_of(fold["train"])
        q_mean, q_std = compute_train_stats([w[2] for w in train_w])
        vx = np.stack([voxel_to_model_input(w[1])[0] for w in train_w])
        vx_mean, vx_std = vx.mean(axis=(0, 3, 4)), vx.std(axis=(0, 3, 4))
        mu = torch.from_numpy(vx_mean).view(2, -1, 1, 1).float()
        sg = torch.from_numpy(vx_std).view(2, -1, 1, 1).float()

        def batchify(items):
            batches = []
            for i in range(0, len(items), micro):
                chunk = items[i : i + micro]
                v = torch.stack(
                    [
                        (torch.from_numpy(voxel_to_model_input(w[1])[0]).float() - mu) / (sg + 1e-6)
                        for w in chunk
                    ]
                )
                q = torch.stack([torch.from_numpy((w[2] - q_mean) / (q_std + 1e-6)) for w in chunk])
                f = torch.tensor([manifest[w[0]]["f_target"] for w in chunk], dtype=torch.float32)
                batches.append((v, q.float(), f))
            return batches

        t_mck, v_mck = parse_ldv_txt(
            dict((t, p) for t, _, p in pairs)[pick_mck_fit_trial(fold["train"])]
        )
        mck = fit_mck(v_mck, 1 / float(np.median(np.diff(t_mck))))
        m, c, k = mck["m"], mck["c"], mck["k"]
        log(f"fold {held} mck: fd={mck['fd']:.2f} fn={mck['fn']:.2f}")

        train_b = batchify(windows_of(fold["train"]))
        vm_b = batchify(windows_of(fold["val_matched"]))
        vs_b = batchify(windows_of(fold["val_shift"]))
        results, metrics = {}, {}
        for variant in VARIANTS:
            ckpt = os.path.join(data_root, "ckpts", held, f"{variant}.pt")
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
                kk: (round(vv, 5) if isinstance(vv, float) else vv)
                for kk, vv in res.items()
                if kk != "state"
            }
            log(f"{held}/{variant}: {json.dumps(show)}")
        for variant in VARIANTS:
            model = make_model().to(device)
            model.load_state_dict(
                torch.load(
                    os.path.join(data_root, "ckpts", held, f"{variant}.pt"),
                    map_location=device,
                    weights_only=True,
                )
            )
            model.eval()
            for split, batches in [
                ("test_matched", batchify(windows_of(fold["test_matched"]))),
                ("test_shift", batchify(windows_of(fold["test_shift"]))),
            ]:
                rms, fr = [], []
                with torch.no_grad():
                    for v, q, f in batches:
                        qh, fh = model(v.to(device))
                        qh = qh.cpu().numpy()
                        for i in range(len(q)):
                            rms.append(rmse(q[i].numpy(), qh[i]))
                            fr.append(rel_freq_error(float(fh[i].cpu()), float(f[i])))
                metrics[f"{variant}/{split}"] = {"rmse": rms, "freq": fr}
            tm = metrics[f"{variant}/test_matched"]["rmse"]
            tsm = metrics[f"{variant}/test_shift"]["rmse"]
            log(f"{held}/{variant}: Delta_RMSE(test)={np.mean(tsm) - np.mean(tm):+.4f}")
        json.dump(
            {
                "held": held,
                "results": results,
                "metrics": {
                    kk: {
                        "rmse_mean": float(np.mean(vv["rmse"])),
                        "rmse_std": float(np.std(vv["rmse"], ddof=1)),
                        "freq_mean": float(np.mean(vv["freq"])),
                        "freq_std": float(np.std(vv["freq"], ddof=1)),
                    }
                    for kk, vv in metrics.items()
                },
                "mck_trial": pick_mck_fit_trial(fold["train"]),
                "micro": micro,
            },
            open(res_path, "w"),
        )
    log("DONE")


if __name__ == "__main__":
    main()
