"""Struct-X training campaign kernel (F010). Phase 1: voxel store for all 30 trials.

Writes /kaggle/working/vox/{trial}.npz + manifest.json (exfil via kernels output,
then versioned to a dataset for resume). Training phases follow in v2+.
"""

import json
import os
import subprocess
import sys

WORKING_DIR = "/kaggle/working"


def sh(cmd: str) -> str:
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


def find_archive_dir() -> str:
    for root, dirs, _files in os.walk("/kaggle/input"):
        if "XJTU-DV-Beam" in dirs:
            return os.path.join(root, "XJTU-DV-Beam")
    raise FileNotFoundError("XJTU-DV-Beam not found under /kaggle/input")


def main() -> None:
    print("=== nvidia-smi ===")
    print(sh("nvidia-smi"))
    print("=== clone struct-x ===")
    print(
        sh(f"git clone -q https://github.com/arnabwithab/struct-x.git {WORKING_DIR}/structx-repo")
    )
    sys.path.insert(0, f"{WORKING_DIR}/structx-repo/src")
    archive = find_archive_dir()
    print("archive:", archive)
    vox_dir = f"{WORKING_DIR}/vox"

    from data_loader import condition_of, pair_trials, trial_to_training_pairs

    pairs = pair_trials(os.path.join(archive, "DV"), os.path.join(archive, "LDV"))
    print(f"{len(pairs)} paired trials")
    os.makedirs(vox_dir, exist_ok=True)
    manifest = []
    for trial_id, dv_path, ldv_path in pairs:
        import numpy as np

        win = trial_to_training_pairs(dv_path, ldv_path, n_bins=5)
        vox = np.stack([w[0] for w in win]).astype(np.float32)
        q = np.stack([w[1] for w in win]).astype(np.float32)
        out_path = os.path.join(vox_dir, f"{trial_id}.npz")
        np.savez_compressed(out_path, vox=vox, q=q)
        manifest.append({"trial_id": trial_id, "n_windows": len(win), "path": out_path})
        print(f"{trial_id}: {len(win)} windows", flush=True)
    by_cond: dict = {}
    for m in manifest:
        by_cond.setdefault(condition_of(m["trial_id"]), []).append(m["n_windows"])
    total = sum(m["n_windows"] for m in manifest)
    print("windows by condition:", {k: (len(v), sum(v)) for k, v in sorted(by_cond.items())})
    print("TOTAL WINDOWS:", total)
    with open(os.path.join(vox_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
