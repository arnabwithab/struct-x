"""Build voxels + window index (trial-level, no leakage). Thin CLI over src modules."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import condition_of, pair_trials, trial_to_training_pairs
from train import VALID_B


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", help="optional base.yaml for --bins default")
    parser.add_argument("--archive", default="data/XJTU-DV-Beam")
    parser.add_argument("--out", default="data/voxels")
    parser.add_argument("--trials", default="", help="comma-separated trial ids, empty = all")
    parser.add_argument("--b", "--bins", dest="bins", type=int, default=None, choices=list(VALID_B))
    parser.add_argument("--repr", choices=["voxel", "timesurface"], default="voxel")
    return parser


def resolve_bins(args):
    """CLI --b wins; else base.yaml representation.B; else 5."""
    if args.bins is not None:
        return args.bins
    if args.config:
        import yaml

        with open(args.config) as f:
            return int((yaml.safe_load(f) or {}).get("representation", {}).get("B", 5))
    return 5


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    n_bins = resolve_bins(args)

    pairs = pair_trials(os.path.join(args.archive, "DV"), os.path.join(args.archive, "LDV"))
    want = set(args.trials.split(",")) if args.trials else None
    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for trial_id, dv_path, ldv_path in pairs:
        if want is not None and trial_id not in want:
            continue
        out_path = os.path.join(args.out, f"{trial_id}.npz")
        if os.path.isfile(out_path):
            print(f"skip {trial_id} (cached)")
        else:
            import numpy as np

            win = trial_to_training_pairs(dv_path, ldv_path, n_bins=n_bins, repr_kind=args.repr)
            vox = np.stack([w[0] for w in win]).astype(np.float32)
            q = np.stack([w[1] for w in win]).astype(np.float32)
            np.savez_compressed(out_path, vox=vox, q=q)
            print(f"{trial_id}: {len(win)} windows -> {out_path}")
        manifest.append(
            {"trial_id": trial_id, "condition": condition_of(trial_id), "path": out_path}
        )
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    print(f"wrote {len(manifest)} trials")


if __name__ == "__main__":
    main()
