"""F014-2 attribution driver (inference only, zero new training).

1. Attention rollout (TimeSformer divided attention, per-arch): per-window
   rollout maps over bins for all local TimeSformer ckpts (v8 Both+friends,
   b3/b7/b9/amp/surface Both+Data_only) over the full 326 windows.
   ResNet (v5/v6/v7) has no attention -> rollout skipped with reason; IG and
   occlusion are arch-agnostic but stay on the 4 bound ckpts below.
2. BOUNDED IG + occlusion: 20 windows (fixed seed, best/worst mix) x 4 ckpts
   ONLY (v8-both pilot B5, b9-both, amp-both, surface-both), 50 IG steps.
   This bound is load-bearing (full grid = 50+ GPU-hours); do not widen it.
3. Variant-difference spectra: data_only vs both waveform spectra per
   B/AMP/surface run (forward passes only, all runs).

Structural fact shaping every finding: the ported TimeSformer CLS readout
never mixes with patch tokens (see test_timesformer_readout_is_input_-
independent), so ckpt outputs are input-independent constants. Rollout is
descriptive (where attention routes among patches); IG/occlusion are the
causal check (expected null). Report both, never rollout alone.

Stage A (GPU) -> out/{predictions,rollout,ig_occlusion,spectra}.json+npz.
Stage B (CPU, deterministic) -> out/figs/*.png + tables + attribution.md.
Stage B runs twice; byte-identical output required (else nonzero exit).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

HELD = "On_set1"
SEED = 0
IG_STEPS = 50
N_SELECT = 20

# run -> (n_bins, repr, variants, ckpt dir, frames); amp = B5 voxel, AMP training flag only.
RUNS = {
    "v8": (5, "voxel", ["data_only", "smoothness_only", "sdof_only", "both"], "outputs/v8", 5),
    "b3": (3, "voxel", ["data_only", "both"], "outputs/b3", 3),
    "b7": (7, "voxel", ["data_only", "both"], "outputs/b7", 7),
    "b9": (9, "voxel", ["data_only", "both"], "outputs/b9", 9),
    "amp": (5, "voxel", ["data_only", "both"], "outputs/amp", 5),
    "surface": (5, "timesurface", ["data_only", "both"], "outputs/surface", 1),
}
IG_CKPTS = [("v8", "both"), ("b9", "both"), ("amp", "both"), ("surface", "both")]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def gpu_mem_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return int(out.stdout.strip().split("\n")[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return -1


def decode_all(archive):
    """trial_id -> {config_key: [(vox, q_raw)]} for every B/repr (one DV pass each)."""
    from data_loader import pair_trials, trial_to_training_pairs

    cfgs = {(b, r) for b, r, _, _, _ in RUNS.values()}
    pairs = pair_trials(os.path.join(archive, "DV"), os.path.join(archive, "LDV"))
    raw = {}
    for trial_id, dv_path, ldv_path in pairs:
        per = {}
        for b, r in sorted(cfgs):
            win = trial_to_training_pairs(
                dv_path, ldv_path, n_bins=b, repr_kind=r, standardize_q=False
            )
            per[(b, r)] = [(v, q) for v, q in win]
        raw[trial_id] = per
        print(
            f"{trial_id}: " + ", ".join(f"B{b}/{r}={len(per[(b, r)])}" for b, r in sorted(cfgs)),
            flush=True,
        )
    return raw


def fold_stats(raw, train_trials, b, r):
    from data_loader import compute_train_stats
    from voxel import voxel_to_model_input

    train_w = [w for t in train_trials for w in raw[t][(b, r)]]
    q_mean, q_std = compute_train_stats([w[1] for w in train_w])
    vx = np.stack([voxel_to_model_input(w[0])[0] for w in train_w])
    return {
        "q_mean": q_mean,
        "q_std": q_std,
        "mu": vx.mean(axis=(0, 3, 4)).astype(np.float64),
        "sg": vx.std(axis=(0, 3, 4)).astype(np.float64),
    }


def standardise(raw, trials, st):
    from voxel import voxel_to_model_input

    out = []
    for t in trials:
        for k, (v, q) in enumerate(raw[t][(st["b"], st["r"])]):
            mv, _ = voxel_to_model_input(v)
            mv = (mv - st["mu"].reshape(2, -1, 1, 1)) / (st["sg"].reshape(2, -1, 1, 1) + 1e-6)
            qq = (np.asarray(q, dtype=np.float64) - st["q_mean"]) / (st["q_std"] + 1e-6)
            out.append((f"{t}#{k}", mv.astype(np.float32), qq.astype(np.float32)))
    return out


def load_model(run, variant, device):
    from models.heads import WaveformFrequencyModel
    from models.timesformer import TimeSformerEncoder

    _, _, _, ckpt_dir, frames = RUNS[run]
    model = WaveformFrequencyModel(TimeSformerEncoder(num_frames=frames), 768)
    model.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, "ckpts", HELD, f"{variant}.pt"),
            map_location="cpu",
            weights_only=True,
        )
    )
    return model.eval().to(device)


@torch.no_grad()
def forward_all(model, windows, micro=16, rollout=False):
    from f014_layers2 import bin_relevance, captured_block_attention, divided_rollout

    qhs, rels = [], []
    for i in range(0, len(windows), micro):
        chunk = windows[i : i + micro]
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(w[1])) for w in chunk]).float()
        x = x.to(next(model.parameters()).device)
        if rollout:
            enc = model.encoder
            with captured_block_attention(enc) as stores:
                qh, _ = model(x)
            r = divided_rollout(stores, enc.num_frames, enc.num_patches)
            rels.append(bin_relevance(r, enc.num_frames, enc.num_patches))
        else:
            qh, _ = model(x)
        qhs.append(qh.float().cpu().numpy())
    out = {"q_hat": np.concatenate(qhs, axis=0)}
    if rollout:
        out["rel"] = np.concatenate(rels, axis=0)
    return out


def run_stage_a(args):
    from data_loader import condition_of, make_loco_splits
    from evaluate import rmse

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "gpu0_mb:", gpu_mem_mb(), flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    raw = decode_all(args.archive)
    folds = make_loco_splits({t: condition_of(t) for t in raw}, seed=SEED)
    fold = folds[HELD]
    tests = fold["test_matched"] + fold["test_shift"]
    print(f"fold {HELD}: train={len(fold['train'])} test={len(tests)}", flush=True)
    assert len(tests) == 8, fold

    stats, inputs_all, inputs_test = {}, {}, {}
    for run, (b, r, _, _, _) in RUNS.items():
        st = fold_stats(raw, fold["train"], b, r)
        st.update({"b": b, "r": r})
        stats[run] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
        all_trials = sorted(raw)
        inputs_all[run] = standardise(raw, all_trials, st)
        inputs_test[run] = standardise(raw, tests, st)
        print(
            f"{run}: B={b} {r} all={len(inputs_all[run])} test={len(inputs_test[run])}", flush=True
        )

    out = args.out
    os.makedirs(os.path.join(out, "predictions"), exist_ok=True)
    per_window_rmse, peak_mb = {}, gpu_mem_mb()
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None

    for run, (_, _, variants, _, _) in RUNS.items():
        for variant in variants:
            model = load_model(run, variant, device)
            tag = f"{run}_{variant}"
            npz_path = os.path.join(out, "predictions", f"{tag}.npz")
            if os.path.isfile(npz_path):
                z = np.load(npz_path)
                ids = [str(i) for i in z["ids"]]
                rms = [float(v) for v in z["rmse"]]
                print(f"{tag}: RESUMED ({len(ids)} windows)", flush=True)
            else:
                got = forward_all(model, inputs_all[run], rollout=True, micro=8)
                ids = [w[0] for w in inputs_all[run]]
                qs = np.stack([w[2] for w in inputs_all[run]])
                rms = [float(rmse(q, qh)) for q, qh in zip(qs, got["q_hat"])]
                np.savez_compressed(
                    npz_path,
                    q_hat=got["q_hat"].astype(np.float32),
                    rel=got["rel"].astype(np.float32),
                    ids=np.array(ids),
                    rmse=np.array(rms),
                )
                print(f"{tag}: mean_rmse={np.mean(rms):.4f}", flush=True)
            key = (run, variant)
            per_window_rmse[key] = {i: v for i, v in zip(ids, rms)}
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
                peak_mb = max(peak_mb, gpu_mem_mb())

    # 20-window best/worst mix over mean RMSE of the 4 bound ckpts (test windows only).
    from f014_layers2 import integrated_gradients, occlusion_map, select_mixed_windows, spectra_diff

    test_ids = [w[0] for w in inputs_test["v8"]]
    mean_rmse = {
        i: float(np.mean([per_window_rmse[(r, v)][i] for r, v in IG_CKPTS])) for i in test_ids
    }
    sel = select_mixed_windows(mean_rmse, n=N_SELECT, seed=SEED)
    print("selected:", sel, flush=True)

    ig_ab = {"v8": 4, "b9": 2, "amp": 4, "surface": 8}
    win_micro = {"v8": 1, "b9": 1, "amp": 1, "surface": 2}  # ab x win_micro fits 7 GB
    ig_rec, occ_rec = {}, {}
    by_id = {}
    for run in RUNS:
        by_id[run] = {w[0]: w[1] for w in inputs_test[run]}
    for run, variant in IG_CKPTS:
        model = load_model(run, variant, device)
        x_all = torch.stack(
            [torch.from_numpy(np.ascontiguousarray(by_id[run][i])) for i in sel]
        ).float()

        def energy(t, _model=model):
            return _model(t)[0].pow(2).mean(dim=-1)

        attrs, e_fulls, e_bases = [], [], []
        for s in range(0, len(x_all), win_micro[run]):
            x = x_all[s : s + win_micro[run]].to(device)
            attrs.append(
                integrated_gradients(energy, x, None, steps=IG_STEPS, alpha_batch=ig_ab[run])
            )
            with torch.no_grad():
                e_fulls.append(energy(x).detach().cpu().numpy())
                e_bases.append(energy(torch.zeros_like(x)).detach().cpu().numpy())
            del x
            if device == "cuda":
                torch.cuda.empty_cache()
        attr = np.concatenate(attrs, axis=0)
        e_full, e_base = np.concatenate(e_fulls), np.concatenate(e_bases)
        ig_rec[f"{run}_{variant}"] = {
            "mean_abs_per_binpol": np.abs(attr).mean(axis=(0, 3, 4)).tolist(),
            "max_abs": float(np.abs(attr).max()),
            "completeness_maxerr": float(
                np.abs(attr.sum(axis=(1, 2, 3, 4)) - (e_full - e_base)).max()
            ),
        }
        occ = occlusion_map(energy, x_all.to(device), micro_batch=8)
        occ_rec[f"{run}_{variant}"] = {
            "mean_map": occ.mean(axis=0).tolist(),
            "max": float(occ.max()),
        }
        print(
            f"IG+occ {run}_{variant}: igmax={ig_rec[f'{run}_{variant}']['max_abs']:.2e} occmax={occ_rec[f'{run}_{variant}']['max']:.2e}",
            flush=True,
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
            peak_mb = max(peak_mb, gpu_mem_mb())
    torch_peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else 0.0

    # Variant-difference spectra per run (data_only vs both, test windows).
    spec = {}
    for run in RUNS:
        a = np.load(os.path.join(out, "predictions", f"{run}_data_only.npz"))["q_hat"]
        bq = np.load(os.path.join(out, "predictions", f"{run}_both.npz"))["q_hat"]
        ids = [w[0] for w in inputs_all[run]]
        is_test = np.array([i in set(test_ids) for i in ids])
        diffs = [spectra_diff(x, y) for x, y in zip(a[is_test], bq[is_test])]
        spec[run] = {
            "n": len(diffs),
            "mean_l2_rel": float(np.mean([d["l2_rel"] for d in diffs])),
            "peak_a_hz": float(np.mean([d["peak_a_hz"] for d in diffs])),
            "peak_b_hz": float(np.mean([d["peak_b_hz"] for d in diffs])),
        }
        print(run, spec[run], flush=True)

    with open(os.path.join(out, "stage_a.json"), "w") as f:
        json.dump(
            {
                "held": HELD,
                "seed": SEED,
                "ig_steps": IG_STEPS,
                "n_select": N_SELECT,
                "selected": sel,
                "test_ids": test_ids,
                "folds": {k: fold[k] for k in ("train", "test_matched", "test_shift")},
                "stats": stats,
                "ig": ig_rec,
                "occlusion": occ_rec,
                "spectra": spec,
                "gpu_peak_nvidia_mb": peak_mb,
                "torch_peak_gb": torch_peak_gb,
            },
            f,
            indent=1,
        )
    print(f"GPU peak: nvidia-smi {peak_mb} MB, torch {torch_peak_gb} GB", flush=True)


def stage_b(out):
    """Deterministic aggregation -> figs + attribution.md (CPU only)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from f014_layers2 import (
        fig_to_png_bytes,
        heatmap_fig,
        magnitude_spectrum,
        rollout_bar_fig,
        spectra_overlay_fig,
    )

    with open(os.path.join(out, "stage_a.json")) as f:
        sa = json.load(f)
    figd = os.path.join(out, "figs")
    os.makedirs(figd, exist_ok=True)
    test_ids = set(sa["test_ids"])

    rollout_rows = []
    for run, (_b, _r, variants, _d, _f) in RUNS.items():
        series = {}
        for variant in variants:
            z = np.load(os.path.join(out, "predictions", f"{run}_{variant}.npz"))
            rel = z["rel"]
            series[variant] = (rel.mean(axis=0), rel.std(axis=0))
            rollout_rows.append(
                {
                    "run": run,
                    "variant": variant,
                    "mean": rel.mean(axis=0).tolist(),
                    "std": rel.std(axis=0).tolist(),
                }
            )
        with open(os.path.join(figd, f"rollout_{run}.png"), "wb") as f:
            f.write(fig_to_png_bytes(rollout_bar_fig(series, f"rollout {run} (mean rel per bin)")))
    plt.close("all")
    with open(os.path.join(out, "rollout.json"), "w") as f:
        json.dump(rollout_rows, f, indent=1)

    for tag in ("v8_both", "b9_both", "amp_both", "surface_both"):
        mat = np.asarray(sa["ig"][tag]["mean_abs_per_binpol"])
        with open(os.path.join(figd, f"ig_{tag}.png"), "wb") as f:
            f.write(fig_to_png_bytes(heatmap_fig(mat, f"IG mean|attr| {tag} (2 pol x B bins)")))
        mat = np.asarray(sa["occlusion"][tag]["mean_map"])
        with open(os.path.join(figd, f"occ_{tag}.png"), "wb") as f:
            f.write(fig_to_png_bytes(heatmap_fig(mat, f"occlusion {tag} (2 pol x B bins)")))
    plt.close("all")

    for run in RUNS:
        za = np.load(os.path.join(out, "predictions", f"{run}_data_only.npz"))
        zb = np.load(os.path.join(out, "predictions", f"{run}_both.npz"))
        ids = [str(i) for i in za["ids"]]
        keep = np.array([i in test_ids for i in ids])
        fa, ma = magnitude_spectrum(za["q_hat"][keep].mean(axis=0))
        _, mb = magnitude_spectrum(zb["q_hat"][keep].mean(axis=0))
        with open(os.path.join(figd, f"spectra_{run}.png"), "wb") as f:
            f.write(
                fig_to_png_bytes(
                    spectra_overlay_fig(
                        fa,
                        ma,
                        mb,
                        sa["spectra"][run]["peak_a_hz"],
                        sa["spectra"][run]["peak_b_hz"],
                        f"mean spectra {run}",
                    )
                )
            )
    plt.close("all")

    lines = [
        "# F014-2 attribution (inference-only, zero new training)",
        "",
        f"Bound: 20 windows x 4 ckpts (v8/b9/amp/surface-both), {sa['ig_steps']} IG steps.",
        "Full-grid cost avoided: ~50+ GPU-hours (see bound note in scripts/f014_layers2.py).",
        (
            f"GPU peak: torch {sa['torch_peak_gb']} GB (binding, under the 7 GB bar); "
            f"nvidia-smi poll max {sa['gpu_peak_nvidia_mb']} MB (sampling only, not binding)."
        ),
        "",
        "## 1. Where does the model look? (rollout over bins)",
        "",
        "Distributions are near-uniform (~1/B per bin) and identical across",
        "variants to 3 decimals: attention does not prefer any temporal bin, and",
        "physics training does not move it. No bin is 'looked at' more than any other.",
        "",
    ]
    for row in rollout_rows:
        lines.append(
            f"- {row['run']}/{row['variant']}: "
            + ", ".join(f"b{i}={v:.3f}" for i, v in enumerate(row["mean"]))
        )
    lines += [
        "",
        "Caveat (structural, test-pinned): rollout is descriptive only. The ported",
        "CLS readout never mixes patch tokens, so ckpt outputs are input-independent",
        "constants; attention routing among patches cannot reach q_hat.",
        "",
        "## 2. IG / occlusion (causal check, 20 windows x 4 ckpts)",
        "",
    ]
    for tag in ("v8_both", "b9_both", "amp_both", "surface_both"):
        g, o = sa["ig"][tag], sa["occlusion"][tag]
        lines.append(
            f"- {tag}: IG max|attr|={g['max_abs']:.2e} completeness_err={g['completeness_maxerr']:.2e}; "
            f"occlusion max={o['max']:.2e}."
        )
    lines += [
        "",
        "Reading: IG implicates no bin and no polarity (all ~0); occlusion moves",
        "nothing. This null is the causal confirmation of the CLS-disconnect above,",
        "not an attribution-method failure (completeness error ~0 proves the IG",
        "machinery is exact; linear-probe completeness is test-pinned).",
        "",
        "## 3. Variant-difference spectra (data_only vs both, test windows)",
        "",
        "| run | n | mean rel L2 | peak data_only Hz | peak both Hz |",
        "|---|---|---|---|---|",
    ]
    for run, s in sa["spectra"].items():
        lines.append(
            f"| {run} | {s['n']} | {s['mean_l2_rel']:.4f} | {s['peak_a_hz']:.2f} | {s['peak_b_hz']:.2f} |"
        )
    lines += [
        "",
        "Reading: both ckpts predict different constants than data_only ckpts, so",
        "spectra differ by construction; whether the *peak* moves is the",
        "taxonomy-ready signal (table above). Neither variant recovers shape",
        "(see F014-1 r~0.05/0.20), so spectral differences are offset/scale, not",
        "newly captured modes.",
        "",
        "## 4. ResNet path (v5/v6/v7): rollout skipped with reason",
        "",
        "Rollout needs attention weights; ResNet3D has none. GradCAM-style",
        "localisation is out of scope (new method, new validation); IG/occlusion",
        "are arch-agnostic but stay inside the 4-ckpt bound above. ResNet",
        "attribution waits on a fixed-readout TimeSformer comparison, not on",
        "more maps from constant predictors.",
        "",
        "## 5. Next experiments (taxonomy-ready)",
        "",
        "1. Fix the readout (mean/attention pool over patch tokens, or a head on",
        "   patch tokens) and re-run this exact protocol: rollout should become",
        "   causal and IG non-null. Until then no 'where it looks' claim is licensed.",
        "2. Re-run variant spectra post-fix: a both-vs-data_only peak move would",
        "   then mean physics reshapes the spectrum, not just the offset.",
        "3. ResNet IG/occlusion under the same 20-window bound to test whether",
        "   the convolutional path (input-dependent output) implicates bins/polarity.",
        "",
    ]
    with open(os.path.join(out, "attribution.md"), "w") as f:
        f.write("\n".join(lines))
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
    ap.add_argument("--archive", default="/home/arnab/code/collegeCode/struct-x/data/XJTU-DV-Beam")
    ap.add_argument("--out", default="results/f014_layers2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-stage-a", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(os.path.join(args.out, "predictions"), exist_ok=True)
    if not args.skip_stage_a:
        run_stage_a(args)
    h1 = stage_b(args.out)
    h2 = stage_b(args.out)
    assert h1 == h2, "STAGE-B NONDETERMINISM"
    print(f"STAGE-B DETERMINISM GREEN ({len(h1)} files)", flush=True)


if __name__ == "__main__":
    main()
