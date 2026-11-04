"""F014-3 driver (inference only, zero new training).

Freezes each local `both` encoder (TimeSformer v8/b3/b7/b9/amp/surface +
ResNet v6, all folds present), extracts window latents batch-1 on GPU, then
CPU-only: (a) linear probes (Ridge: fn value, envelope slope; LogisticReg:
fn bin, slope sign, matched-vs-shift) with trial-level splits + shuffled
controls; (b) cross-condition latent stability (cosine + linear CKA between
matched-test and shift-test latents, trial-bootstrap CIs).

B-mismatch notes (documented approximations, no retraining): b3/b7/b9 ckpts
were trained at B=3/7/9 but beam-vox ships B=5 only, so their temporal
pos-embed is linearly interpolated Bk->5 (the same surgery the port uses
for Kinetics 8->B, design sec 6). surface expects 1 frame: B bins are
mean-collapsed per polarity (static count proxy, NOT a true time surface).
v6 ResNet pools time away, so B=5 feeds directly. data_only-vs-both
contrast is skipped (follow-up): this layer asks what `both` latents encode.
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from f014_layers3 import (
    assign_bins,
    bootstrap_ci,
    centroid,
    cosine_similarity,
    envelope_slope,
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

# (tag, ckpt_relpath, held_fold, arch, frames, collapse): arch ts/resnet.
PLAN = [
    ("v8", "v8/ckpts/On_set1/both.pt", "On_set1", "ts", 5, False),
    ("b3", "b3/ckpts/On_set1/both.pt", "On_set1", "ts", 3, False),
    ("b7", "b7/ckpts/On_set1/both.pt", "On_set1", "ts", 7, False),
    ("b9", "b9/ckpts/On_set1/both.pt", "On_set1", "ts", 9, False),
    ("amp", "amp/ckpts/On_set1/both.pt", "On_set1", "ts", 5, False),
    ("surface", "surface/ckpts/On_set1/both.pt", "On_set1", "ts", 1, True),
    ("v6_On_set1", "v6/ckpts/On_set1/both.pt", "On_set1", "resnet", 5, False),
    ("v6_On_set2", "v6/ckpts/On_set2/both.pt", "On_set2", "resnet", 5, False),
    ("v6_Off_set3", "v6/ckpts/Off_set3/both.pt", "Off_set3", "resnet", 5, False),
    ("v6_Off_set1", "v6/ckpts/Off_set1/both.pt", "Off_set1", "resnet", 5, False),
    ("v6_Off_set2", "v6/ckpts/Off_set2/both.pt", "Off_set2", "resnet", 5, False),
]


def gpu_status():
    """One-line nvidia-smi util snapshot (sibling shares the 4060; back off if hot)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "nvidia-smi unavailable"


def load_trial_vox(vox_dir):
    data = {}
    for name in sorted(os.listdir(vox_dir)):
        if name.endswith(".npz"):
            z = np.load(os.path.join(vox_dir, name))
            data[name[: -len(".npz")]] = (np.asarray(z["vox"]), np.asarray(z["q"]))
    if not data:
        raise ValueError("no npz windows in " + vox_dir)
    return data


def stitch(q_windows):
    w = [np.asarray(x).ravel() for x in q_windows]
    return np.concatenate([w[0]] + [x[500:] for x in w[1:]])


def trial_fn(q_windows):
    """Per-trial fn: SDOF fit on stitched windows, median-FFT-peak fallback."""
    from baseline_ridge import window_fft_peak
    from mck_fit import fit_mck

    try:
        return float(fit_mck(stitch(q_windows), 1000.0)["fn"])
    except ValueError:
        return float(np.median([window_fft_peak(w) for w in q_windows]))


def build_model(arch, ckpt_path, ckpt_frames):
    """Frozen encoder + latent dim; Bk->5 temporal pos-embed interp for b3/b7/b9."""
    from models.heads import WaveformFrequencyModel

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if arch == "ts":
        from models.timesformer import TimeSformerEncoder

        target_frames = 1 if ckpt_frames == 1 else 5  # surface ckpt eats 1-frame inputs
        enc = TimeSformerEncoder(num_frames=target_frames)
        key = "encoder.time_pos"
        if state[key].shape[1] != target_frames:  # Bk->5 linear interp (port surgery)
            src = state[key].transpose(1, 2).unsqueeze(-1).float()
            state[key] = (
                torch.nn.functional.interpolate(
                    src, size=(target_frames, 1), mode="bilinear", align_corners=False
                )
                .squeeze(-1)
                .transpose(1, 2)
                .to(state[key].dtype)
            )
        model = WaveformFrequencyModel(enc, 768)
    else:
        from models.resnet3d import ResNet3DEncoder

        model = WaveformFrequencyModel(ResNet3DEncoder(512), 512)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    latent_dim = 768 if arch == "ts" else 512
    return model, latent_dim


def model_inputs(trial_data, train_trials, collapse):
    """Fold-frozen (mu, sg) input stats (train.py batchify replica) + per-trial inputs."""
    from voxel import voxel_to_model_input

    def convert(vox):
        vox = np.asarray(vox, dtype=np.float32)
        if collapse:  # static polarity-count proxy for the 1-frame surface ckpt
            b = vox.shape[0] // 2
            vox = vox.reshape(b, 2, *vox.shape[1:]).mean(axis=0)
        return voxel_to_model_input(vox)[0]

    train_x = np.stack([convert(w) for t in train_trials for w in trial_data[t][0]])
    mu, sg = train_x.mean(axis=(0, 3, 4)), train_x.std(axis=(0, 3, 4))
    mu = mu.reshape(2, -1, 1, 1)
    sg = sg.reshape(2, -1, 1, 1)
    per_trial = {}
    for t, (vox, _) in trial_data.items():
        per_trial[t] = np.stack([(convert(w) - mu) / (sg + 1e-6) for w in vox])
    return per_trial


@torch.no_grad()
def extract_latents(model, per_trial, device):
    """Batch-1 frozen-encoder forward; returns trial -> (n, d) latents + peak MiB."""
    model.to(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    latents = {}
    for t in sorted(per_trial):
        zs = []
        for w in per_trial[t]:
            x = torch.from_numpy(np.ascontiguousarray(w)).unsqueeze(0).float().to(device)
            zs.append(model.encoder(x).float().cpu().numpy())
        latents[t] = np.concatenate(zs, axis=0)
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else 0.0
    return latents, peak


def run_probes(latents, labels, trials, conds, seed=0):
    """Trial-split linear probes + shuffled controls. Returns (rows, edges)."""
    split = trial_probe_split(trials, conds, seed=seed)
    tr = [i for i, t in enumerate(trials) if t in split["train"]]
    te = [i for i, t in enumerate(trials) if t in split["test"]]
    x_tr, x_te = latents[tr], latents[te]
    out = []

    def clf_row(name, y, model_fn):
        y_tr, y_te = y[tr], y[te]
        m = model_fn(x_tr, y_tr)
        acc = probe_accuracy(m, x_te, y_te)
        ms = model_fn(x_tr, permute_labels(y_tr, seed=seed))
        out.append(
            {
                "probe": name,
                "kind": "accuracy",
                "score": acc,
                "chance": majority_baseline(y_te),
                "shuffled": probe_accuracy(ms, x_te, y_te),
                "n_train": len(tr),
                "n_test": len(te),
            }
        )

    def reg_row(name, y):
        y_tr, y_te = y[tr], y[te]
        m = fit_ridge_probe(x_tr, y_tr)
        ms = fit_ridge_probe(x_tr, permute_labels(y_tr, seed=seed))
        out.append(
            {
                "probe": name,
                "kind": "r2",
                "score": probe_r2(m, x_te, y_te),
                "chance": 0.0,
                "shuffled": probe_r2(ms, x_te, y_te),
                "n_train": len(tr),
                "n_test": len(te),
            }
        )

    reg_row("fn_value", labels["fn"])
    clf_row("fn_bin", labels["fn_bin"], lambda a, b: fit_logreg_probe(a, b, seed=seed))
    reg_row("slope", labels["slope"])
    clf_row("slope_sign", labels["slope_sign"], lambda a, b: fit_logreg_probe(a, b, seed=seed))
    clf_row("shift", labels["shift"], lambda a, b: fit_logreg_probe(a, b, seed=seed))
    return out, split


def run_stability(latents, trials, test_matched, test_shift, seed=0, n_boot=1000):
    """Cosine + linear CKA between matched-test and shift-test latents + trial CIs."""
    tm = sorted(set(test_matched) & set(trials))
    ts = sorted(set(test_shift) & set(trials))
    xm = np.concatenate([latents[t] for t in tm], axis=0)
    xs = np.concatenate([latents[t] for t in ts], axis=0)
    cos = cosine_similarity(centroid(xm), centroid(xs))
    cka = linear_cka(xm, xs, seed=seed)

    def boot_cos(rng):
        bm = rng.choice(len(tm), len(tm), replace=True)
        bs = rng.choice(len(ts), len(ts), replace=True)
        a = centroid(np.concatenate([latents[tm[i]] for i in bm], axis=0))
        b = centroid(np.concatenate([latents[ts[i]] for i in bs], axis=0))
        return cosine_similarity(a, b)

    def boot_cka(rng):
        bm = rng.choice(len(tm), len(tm), replace=True)
        bs = rng.choice(len(ts), len(ts), replace=True)
        a = np.concatenate([latents[tm[i]] for i in bm], axis=0)
        b = np.concatenate([latents[ts[i]] for i in bs], axis=0)
        return linear_cka(a, b, seed=int(rng.randint(0, 2**31 - 1)))

    cos_lo, cos_hi, _ = bootstrap_ci(boot_cos, seed=seed, n_boot=n_boot)
    cka_lo, cka_hi, _ = bootstrap_ci(boot_cka, seed=seed, n_boot=n_boot)
    return {
        "cosine": cos,
        "cosine_ci95": [cos_lo, cos_hi],
        "cka": cka,
        "cka_ci95": [cka_lo, cka_hi],
        "n_matched_trials": len(tm),
        "n_shift_trials": len(ts),
        "n_matched_windows": int(xm.shape[0]),
        "n_shift_windows": int(xs.shape[0]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vox-dir", default="beam-vox")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out", default="results/f014_layers3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--only", default="", help="comma list of tags to run")
    args = ap.parse_args(argv)

    from data_loader import condition_of, make_loco_splits

    print("gpu:", gpu_status(), flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    trial_data = load_trial_vox(args.vox_dir)
    trials_all = sorted(trial_data)
    conds = {t: condition_of(t) for t in trials_all}
    folds = make_loco_splits(conds, seed=args.seed)

    fn_by_trial = {t: trial_fn(trial_data[t][1]) for t in trials_all}
    edges = quantile_bin_edges(list(fn_by_trial.values()), n_bins=3)
    lo, hi = min(fn_by_trial.values()), max(fn_by_trial.values())
    print(f"fn range: {lo:.2f}-{hi:.2f} edges={[round(e, 3) for e in edges]}", flush=True)

    want = {s for s in args.only.split(",") if s} or None
    os.makedirs(os.path.join(args.out, "latents"), exist_ok=True)
    probes_all, stab_all, peaks = {}, {}, {}
    for tag, rel, held, arch, frames, collapse in PLAN:
        if want is not None and tag not in want:
            continue
        ckpt = os.path.join(args.outputs, rel)
        fold = folds[held]
        per_trial = model_inputs(trial_data, fold["train"], collapse)
        model, _ = build_model(arch, ckpt, frames)
        lat, peak = extract_latents(model, per_trial, device)
        peaks[tag] = round(float(peak), 1)
        np.savez_compressed(
            os.path.join(args.out, "latents", f"{tag}.npz"),
            **{t: lat[t] for t in lat},
            trials=np.array(sorted(lat)),
        )
        order, z = sorted(lat), np.concatenate([lat[t] for t in sorted(lat)], axis=0)
        rows = []
        for t in order:
            rows += [t] * lat[t].shape[0]
        labels = {
            "fn": np.array([fn_by_trial[t] for t in rows]),
            "fn_bin": assign_bins([fn_by_trial[t] for t in rows], edges),
            "slope": np.array([s for t in order for s in map(envelope_slope, trial_data[t][1])]),
            "slope_sign": None,
            "shift": np.array([1 if conds[t] == held else 0 for t in rows]),
        }
        labels["slope_sign"] = (labels["slope"] < 0).astype(int)
        probe_rows, _ = run_probes(z, labels, rows, {t: conds[t] for t in order}, seed=args.seed)
        probes_all[tag] = {"held": held, "variant": "both", "arch": arch, "rows": probe_rows}
        trial_lat = {t: lat[t] for t in order}
        stab_all[tag] = {
            "held": held,
            "variant": "both",
            "arch": arch,
            **run_stability(
                trial_lat,
                order,
                fold["test_matched"],
                fold["test_shift"],
                seed=args.seed,
                n_boot=args.n_boot,
            ),
        }
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(
            f"{tag}: peak={peaks[tag]}MiB "
            + " ".join(f"{r['probe']}={r['score']:.3f}" for r in probe_rows),
            flush=True,
        )

    with open(os.path.join(args.out, "probes.json"), "w") as f:
        json.dump(probes_all, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out, "stability.json"), "w") as f:
        json.dump(stab_all, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out, "peaks.json"), "w") as f:
        json.dump(
            {
                "device": device,
                "peak_mib_per_tag": peaks,
                "max_mib": max(peaks.values()) if peaks else 0.0,
            },
            f,
            indent=2,
        )
    print("wrote", args.out, "max_peak_MiB=", max(peaks.values()) if peaks else 0.0, flush=True)


if __name__ == "__main__":
    main()
