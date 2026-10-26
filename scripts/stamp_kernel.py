"""Stamp kernel/kernel-metadata.json for one mode, then `make kernel-*` pushes it.

One dir holds all kernel scripts (Kaggle executes code_file standalone, so
scripts stay self-contained — no shared imports). Modes:
  smoke  -> eros483/struct-x-run    (env log + inventory + shape smoke)
  train  -> eros483/struct-x-train  (LOCO campaign)
  timing -> eros483/struct-x-timing (15a ms/window micro-benchmark)
"""

import json
import sys

MODES = {
    "smoke": ("eros483/struct-x-run", "Struct-X Run", "smoke.py"),
    "train": ("eros483/struct-x-train", "Struct-X Train", "train.py"),
    "timing": ("eros483/struct-x-timing", "Struct-X Timing", "timing.py"),
}


def main() -> None:
    mode = sys.argv[1]
    kid, title, code = MODES[mode]
    path = "kernel/kernel-metadata.json"
    with open(path) as f:
        meta = json.load(f)
    meta.update({"id": kid, "title": title, "code_file": code})
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"stamped {mode}: {kid} / {code}")


if __name__ == "__main__":
    main()
