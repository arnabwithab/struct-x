"""v2e threshold/illumination sweep on the beam video (F018-1). Thin CLI over src/v2e_synth."""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from v2e_synth import (
    DEFAULT_SEED,
    aggregate_vox_dir,
    apply_gain,
    event_stats,
    load_clip,
    run_emulator,
    within_2x,
)

VIDEO = "/home/arnab/code/collegeCode/struct-x/data/XJTU-DV-Beam/Supplementary materials/Video of the acquisition process.mp4"  # read-only source, never copied

# Contrast-threshold + illumination/noise sweep (>=3 settings, design sec 12).
SETTINGS = [
    {"name": "C0.15", "pos_thr": 0.15, "neg_thr": 0.15, "sigma_thres": 0.03, "gain": 1.0},
    {"name": "C0.30", "pos_thr": 0.30, "neg_thr": 0.30, "sigma_thres": 0.03, "gain": 1.0},
    {"name": "C0.50", "pos_thr": 0.50, "neg_thr": 0.50, "sigma_thres": 0.03, "gain": 1.0},
    {"name": "dim", "pos_thr": 0.30, "neg_thr": 0.30, "sigma_thres": 0.03, "gain": 0.6},
    {"name": "bright-noisy", "pos_thr": 0.30, "neg_thr": 0.30, "sigma_thres": 0.06, "gain": 1.4},
]

METRICS = ["mean_rate_px_s", "polarity_ratio", "refr_viol_frac"]


def nvidia_snapshot():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "nvidia-smi unavailable"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=VIDEO)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--max-side", type=int, default=480)
    parser.add_argument("--vox-dir", default="beam-vox")
    parser.add_argument("--out", default="results/v2e_sweep.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    gpu_before = nvidia_snapshot()
    t0 = time.time()
    frames, fps = load_clip(args.video, n_frames=args.frames, max_side=args.max_side)
    real = aggregate_vox_dir(args.vox_dir)

    rows = []
    for cfg in SETTINGS:
        stim = apply_gain(frames, cfg["gain"])
        ev = run_emulator(
            stim,
            fps,
            pos_thr=cfg["pos_thr"],
            neg_thr=cfg["neg_thr"],
            sigma_thres=cfg["sigma_thres"],
            seed=args.seed,
        )
        synth = event_stats(
            ev["xs"], ev["ys"], ev["ts_us"], ev["ps"], ev["height"], ev["width"], ev["duration_s"]
        )
        gates = {m: within_2x(synth[m], real[m]) for m in METRICS}
        rows.append(
            {
                "setting": cfg["name"],
                "params": {k: v for k, v in cfg.items() if k != "name"},
                "synth": synth,
                "gate": {m: {"pass": p, "ratio": r} for m, (p, r) in gates.items()},
                "pass_all": all(p for p, _ in gates.values()),
            }
        )

    verdict = all(r["pass_all"] for r in rows)
    result = {
        "video": args.video,
        "n_frames": len(frames),
        "frame_shape": list(frames.shape[1:]),
        "fps": fps,
        "seed": args.seed,
        "device": "cpu",
        "real": real,
        "rows": rows,
        "verdict": "PASS" if verdict else "FAIL",
        "elapsed_s": round(time.time() - t0, 1),
        "gpu_before": gpu_before,
        "gpu_after": nvidia_snapshot(),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"real: rate={real['mean_rate_px_s']:.4f} pol={real['polarity_ratio']:.3f} "
        f"refr={real['refr_viol_frac']:.2e} ({real['n_files']} files)"
    )
    for r in rows:
        g = r["gate"]
        print(
            f"{r['setting']:>12}: rate={r['synth']['mean_rate_px_s']:.4f} "
            f"({'ok' if g['mean_rate_px_s']['pass'] else 'XX'} "
            f"{g['mean_rate_px_s']['ratio']:.2f}x) "
            f"pol={r['synth']['polarity_ratio']:.3f} "
            f"({'ok' if g['polarity_ratio']['pass'] else 'XX'} "
            f"{g['polarity_ratio']['ratio']:.2f}x) "
            f"refr={r['synth']['refr_viol_frac']:.2e} "
            f"({'ok' if g['refr_viol_frac']['pass'] else 'XX'})"
        )
    print("verdict:", result["verdict"])
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
