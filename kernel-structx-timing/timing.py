"""15a micro-benchmark (design sec 15a): per-step cost of the SDOF physics term.

Times 20 train steps / batch config on identical hardware/seeds for O4 pairs
(c)vs(a) and (d)vs(b); reports ms/window train-step overhead. FLOP prior:
~6T multiply-adds/window vs ~1e10 encoder (measurement wins if they disagree).
"""

import subprocess
import sys
import time


def sh(cmd: str) -> str:
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


def main() -> None:
    import torch

    print(sh("nvidia-smi --query-gpu=name --format=csv,noheader"))
    print(sh("git clone -q https://github.com/arnabwithab/struct-x.git /kaggle/working/structx-repo"))
    sys.path.insert(0, "/kaggle/working/structx-repo/src")

    from models.heads import WaveformFrequencyModel
    from models.resnet3d import ResNet3DEncoder
    from physics import total_loss

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    micro, steps = 8, 20
    v = torch.randn(micro, 2, 5, 224, 224, device=device)
    q = torch.randn(micro, 1000, device=device)
    f = torch.full((micro,), 6.75, device=device)
    m, c, k = 1.0, 2.2302, 1802.6

    def bench(lp, ls, label):
        model = WaveformFrequencyModel(ResNet3DEncoder(512), 512).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=2e-4)
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(steps):
            opt.zero_grad()
            qh, fh = model(v)
            loss = total_loss(qh, q, fh, f, m, c, k, lp, ls)
            loss.backward()
            opt.step()
        torch.cuda.synchronize() if device == "cuda" else None
        ms = (time.perf_counter() - t0) / steps / micro * 1000
        print(f"{label}: {ms:.2f} ms/window/step", flush=True)
        return ms

    a = bench(0.0, 0.0, "(a) data-only     ")
    c = bench(0.01, 0.0, "(c) +SDOF        ")
    b = bench(0.0, 0.001, "(b) +smooth      ")
    d = bench(0.01, 0.001, "(d) both         ")
    print(f"overhead (c-a)/a: {(c - a) / a * 100:.1f}%")
    print(f"overhead (d-b)/b: {(d - b) / b * 100:.1f}%")
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
