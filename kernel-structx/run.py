"""Struct-X Kaggle entry point: env log + dataset inventory + shape smoke.

Expects dataset mount at /kaggle/input/eros483/struct-x-beam/.
Writes only to /kaggle/working (persisted to output bundle).
"""

import hashlib
import os
import subprocess
import sys

DATASET_DIR = "/kaggle/input/eros483/struct-x-beam"
WORKING_DIR = "/kaggle/working"
EXPECTED_MD5 = "d3af5b54a227be70fa932d93323d5920"


def sh(cmd: str) -> str:
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    print("=== nvidia-smi ===")
    print(sh("nvidia-smi"))
    print("=== pip freeze (torch/scipy/numpy) ===")
    freeze = sh("pip freeze")
    for line in freeze.splitlines():
        if line.lower().startswith(("torch", "numpy", "scipy", "pyyaml")):
            print(line)
    print("=== dataset mount ===")
    print(os.listdir(DATASET_DIR))
    for root, _dirs, files in os.walk(DATASET_DIR):
        for name in files:
            full = os.path.join(root, name)
            size = os.path.getsize(full)
            print(f"{os.path.relpath(full, DATASET_DIR)}  {size / 1e9:.2f} GB")
            if name.endswith(".zip") and "Beam" in name:
                print("md5:", md5_of(full), "match:", md5_of(full) == EXPECTED_MD5)
    print("=== structx smoke ===")
    import torch

    from structx.models.heads import FrequencyHead, WaveformHead
    from structx.models.resnet3d import ResNet3DEncoder

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
