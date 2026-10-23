"""Struct-X Kaggle entry point: env log + dataset inventory + shape smoke.

Expects dataset mount at /kaggle/input/eros483/struct-x-beam/.
Writes only to /kaggle/working (persisted to output bundle).
"""

import os
import subprocess
import sys

DATASET_TOP = "XJTU-DV-Beam"
WORKING_DIR = "/kaggle/working"


def find_dataset_dir() -> str:
    """Discover the dataset mount at runtime (slug layout varies; Kaggle
    extracts the uploaded zip, so look for the top folder, not the zip)."""
    base = "/kaggle/input"
    for d in sorted(os.listdir(base)):
        if os.path.isdir(os.path.join(base, d, DATASET_TOP)):
            return os.path.join(base, d, DATASET_TOP)
        if d == DATASET_TOP:
            return os.path.join(base, d)
    raise FileNotFoundError(f"{DATASET_TOP} not found under {base}: {os.listdir(base)}")


def sh(cmd: str) -> str:
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


def main() -> None:
    print("=== nvidia-smi ===")
    print(sh("nvidia-smi"))
    print("=== pip freeze (torch/scipy/numpy) ===")
    freeze = sh("pip freeze")
    for line in freeze.splitlines():
        if line.lower().startswith(("torch", "numpy", "scipy", "pyyaml")):
            print(line)
    print("=== dataset mount ===")
    dataset_dir = find_dataset_dir()
    print("mount:", dataset_dir)
    print(os.listdir(dataset_dir))
    for root, _dirs, files in os.walk(dataset_dir):
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, dataset_dir)
            size = os.path.getsize(full)
            print(f"{rel}  {size / 1e6:.1f} MB")
    print("=== trial inventory (LDV dir) ===")
    ldv_dir = os.path.join(dataset_dir, "LDV")
    print(sorted(os.listdir(ldv_dir)))
    print("=== Code dir (author loaders) ===")
    print(sorted(os.listdir(os.path.join(dataset_dir, "Code"))))
    print("=== struct-x smoke ===")
    sh("git clone -q https://github.com/arnabwithab/struct-x.git /kaggle/working/structx-repo")
    sys.path.insert(0, "/kaggle/working/structx-repo/src")
    import torch

    from models.heads import FrequencyHead, WaveformHead
    from models.resnet3d import ResNet3DEncoder

    torch.manual_seed(0)
    enc, wave, freq = ResNet3DEncoder(64), WaveformHead(64), FrequencyHead(64)
    if torch.cuda.is_available():
        enc, wave, freq = enc.cuda(), wave.cuda(), freq.cuda()
    x = torch.randn(1, 2, 5, 224, 224)
    if torch.cuda.is_available():
        x = x.cuda()
    z = enc(x)
    print("wave:", tuple(wave(z).shape), "freq:", tuple(freq(z).shape))
    if torch.cuda.is_available():
        print("max_memory_GB:", torch.cuda.max_memory_allocated() / 1e9)
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
