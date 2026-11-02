"""CPU miniature end-to-end smoke: preprocess -> train -> eval (F015-4).

Synthetic damped-sine trials, one LOCO fold, few epochs, minutes on CPU.
No GPU touched (forced device=cpu: shared lab box), no dataset download.
Exercises the real paths: make_loco_splits, train.run_fold, evaluate
Delta/se/O5, checkpoint write + checkpoint_index round-trip. Exits nonzero
on any failed self-check.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import checkpoint_index
import numpy as np
import torch

from data_loader import make_loco_splits
from evaluate import delta_m, o5_verdict, rmse, se_delta
from models.heads import WaveformFrequencyModel
from models.resnet3d import ResNet3DEncoder
from train import run_fold, seed_everything

T = 1000  # real window length (design sec 5); tiny only in trials/epochs
FD = 6.7  # synthetic dominant frequency, mirrors real folds
ZETA = 0.05


def synth_wave(rng, phase):
    """Damped free decay + noise: what the LDV would have recorded."""
    t = np.arange(T, dtype=np.float64) / 1000.0
    q = np.exp(-ZETA * 2 * math.pi * FD * t) * np.sin(2 * math.pi * FD * t + phase)
    return (q + 0.05 * rng.standard_normal(T)).astype(np.float32)


def main(argv=None):
    """Build synthetic store, train one fold, score Delta/O5, index ckpts."""
    p = argparse.ArgumentParser(description="miniature CPU e2e smoke (synthetic)")
    p.add_argument("--out", default="outputs-mini")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    seed_everything(args.seed)
    rng = np.random.RandomState(args.seed)

    # --- preprocess: 12 synthetic trials x 3 fake conditions, 6 windows each ---
    trials = {f"t{i}": ("CondA", "CondB", "CondC")[i % 3] for i in range(12)}
    raw = {}
    for i, trial_id in enumerate(sorted(trials)):
        phase = rng.uniform(0, 2 * math.pi)
        raw[trial_id] = [
            (
                torch.from_numpy(rng.standard_normal((2, 5, 32, 32)).astype(np.float32)),
                torch.from_numpy(synth_wave(rng, phase)),
                torch.tensor(FD, dtype=torch.float32),
            )
            for _ in range(6)
        ]
    folds = make_loco_splits(trials, seed=args.seed)  # n>=2 per TEST group
    held = min(folds)
    fold = folds[held]

    def batchify(ids):
        items = [w for t in ids for w in raw[t]]
        return [
            (
                torch.stack([w[0] for w in items[i : i + 4]]),
                torch.stack([w[1] for w in items[i : i + 4]]),
                torch.stack([w[2] for w in items[i : i + 4]]),
            )
            for i in range(0, len(items), 4)
        ]

    # --- train: one fold, both variant, 1x1 grid, few epochs, CPU only ---
    k = (2 * math.pi * FD) ** 2
    c = 2 * ZETA * math.sqrt(k)

    def make_model():
        return WaveformFrequencyModel(ResNet3DEncoder(latent_dim=32), 32)

    ckpt = os.path.join(args.out, "ckpts", held, "both.pt")
    res = run_fold(
        batchify(fold["train"]),
        batchify(fold["val_matched"]),
        batchify(fold["val_shift"]),
        make_model,
        1.0,
        c,
        k,
        variant="both",
        lp_grid=[0.01],
        ls_grid=[0.001],
        max_epochs=args.epochs,
        patience=args.epochs + 1,
        seed=args.seed,
        device="cpu",  # never touch GPU on the shared box
        ckpt_path=ckpt,
    )
    assert math.isfinite(res["best_val_delta"]), "non-finite val delta"
    assert os.path.isfile(ckpt), "checkpoint not written"

    # --- eval: per-trial RMSE on TEST trials, Delta/se/O5 ---
    model = make_model()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        trial_rmse = {}
        for t in fold["test_matched"] + fold["test_shift"]:
            errs = [
                rmse(model(w[0].unsqueeze(0))[0].squeeze(0).numpy(), w[1].numpy()) for w in raw[t]
            ]
            trial_rmse[t] = float(np.mean(errs))
    matched = [trial_rmse[t] for t in fold["test_matched"]]
    shift = [trial_rmse[t] for t in fold["test_shift"]]
    assert all(math.isfinite(v) for v in matched + shift), "non-finite RMSE"
    d = delta_m(matched, shift)
    se = se_delta(matched, shift)
    verdict = {
        "held": held,
        "matched_rmse": matched,
        "shift_rmse": shift,
        "delta": d,
        "se": se,
        "o5": o5_verdict(d, 0.0, se, 0.0),
    }
    verdict.update({k: v for k, v in res.items() if k != "state"})
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "mini_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, sort_keys=True)

    # --- manifest: every ckpt indexed exactly once ---
    index = checkpoint_index.write_index(args.out, os.path.join(args.out, "checkpoint_index.json"))
    errors = checkpoint_index.validate_index(args.out, index)
    assert errors == [], errors
    print(f"mini-e2e OK: fold={held} delta={d:.4f} se={se:.4f} o5={verdict['o5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
