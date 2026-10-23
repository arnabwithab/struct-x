"""Build voxels + window index (trial-level, no leakage). Thin CLI over structx modules."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    print(f"preprocess with {args.config}: not yet implemented (F001)")


if __name__ == "__main__":
    main()
