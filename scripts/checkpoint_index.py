"""Checkpoint index manifest: checkpoint -> fold/variant/seed (design sec 10).

Covers an outputs/ tree: every ``*.pt`` under it must be listed exactly once.
Sidecars (``train_results.json``, ``results_<held>.json``) supply
seed/lambdas/epochs; a checkpoint with no sidecar entry is indexed with null
metadata (visible, never silent).
"""

import argparse
import json
import os

KNOWN_VARIANTS = ("data_only", "smoothness_only", "sdof_only", "both")


def _load_sidecars(outputs_dir):
    """(fold, variant) -> metadata from train_results.json / results_*.json."""
    meta, sidecars = {}, []
    for root, _, files in os.walk(outputs_dir):
        for name in sorted(files):
            path = os.path.join(root, name)
            try:
                if name == "train_results.json":
                    with open(path) as f:
                        doc = json.load(f)
                    seed = (doc.get("config") or {}).get("seed")
                    for held, variants in (doc.get("folds") or {}).items():
                        for variant, res in (variants or {}).items():
                            meta[(held, variant)] = {
                                "seed": seed,
                                "best_lambda_p": (res or {}).get("best_lambda_p"),
                                "best_lambda_s": (res or {}).get("best_lambda_s"),
                                "epochs_run": (res or {}).get("epochs_run"),
                            }
                    sidecars.append(os.path.relpath(path, outputs_dir))
                elif name.startswith("results_") and name.endswith(".json"):
                    with open(path) as f:
                        doc = json.load(f)
                    if isinstance(doc.get("variants"), dict):
                        held = name[len("results_") : -len(".json")]
                        seed = (doc.get("config") or {}).get("seed")
                        for variant, res in doc["variants"].items():
                            meta.setdefault(
                                (held, variant),
                                {
                                    "seed": seed,
                                    "best_lambda_p": (res or {}).get("best_lambda_p"),
                                    "best_lambda_s": (res or {}).get("best_lambda_s"),
                                    "epochs_run": (res or {}).get("epochs_run"),
                                },
                            )
                        sidecars.append(os.path.relpath(path, outputs_dir))
            except (OSError, ValueError):
                continue
    return meta, sorted(sidecars)


def _split_ckpt(relpath):
    """ckpts/<held>/<variant>.pt (or deeper roots like v6/ckpts/...) -> (fold, variant)."""
    parts = relpath.replace(os.sep, "/").split("/")
    if len(parts) >= 3 and parts[-3] == "ckpts" and parts[-1].endswith(".pt"):
        return parts[-2], parts[-1][: -len(".pt")]
    return None, None


def build_index(outputs_dir):
    """Walk outputs_dir for *.pt; attach sidecar metadata where available."""
    meta, sidecars = _load_sidecars(outputs_dir)
    checkpoints = []
    for root, _, files in os.walk(outputs_dir):
        for name in sorted(files):
            if not name.endswith(".pt"):
                continue
            rel = os.path.relpath(os.path.join(root, name), outputs_dir)
            fold, variant = _split_ckpt(rel)
            entry = {"path": rel, "fold": fold, "variant": variant}
            entry.update(meta.get((fold, variant), {"seed": None}))
            entry.setdefault("seed", None)
            checkpoints.append(entry)
    checkpoints.sort(key=lambda e: e["path"])
    return {"checkpoints": checkpoints, "sidecars": sidecars}


def write_index(outputs_dir, out_path):
    """Write the manifest JSON; return it."""
    index = build_index(outputs_dir)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    return index


def validate_index(outputs_dir, index):
    """Errors for: ckpt on disk missing from index, indexed path gone, dupes, unknown variant."""
    errors = []
    on_disk = set()
    for root, _, files in os.walk(outputs_dir):
        for name in sorted(files):
            if name.endswith(".pt"):
                on_disk.add(os.path.relpath(os.path.join(root, name), outputs_dir))
    indexed = [e["path"] for e in index.get("checkpoints", [])]
    for rel in sorted(on_disk - set(indexed)):
        errors.append(f"unindexed checkpoint on disk: {rel}")
    for rel in sorted(set(indexed) - on_disk):
        errors.append(f"indexed checkpoint missing from disk: {rel}")
    seen = set()
    for e in index.get("checkpoints", []):
        key = (e.get("fold"), e.get("variant"))
        if key in seen:
            errors.append(f"duplicate fold/variant entry: {key[0]}/{key[1]}")
        seen.add(key)
        if e.get("variant") not in KNOWN_VARIANTS:
            errors.append(f"unknown variant for {e.get('path')}: {e.get('variant')}")
    return sorted(errors)


def main(argv=None):
    """CLI: write outputs/checkpoint_index.json, then validate (exit 1 on error)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--out", default="", help="default: <outputs>/checkpoint_index.json")
    p.add_argument("--check", action="store_true", help="validate only, do not write")
    args = p.parse_args(argv)
    out = args.out or os.path.join(args.outputs, "checkpoint_index.json")
    if args.check:
        with open(out) as f:
            errors = validate_index(args.outputs, json.load(f))
    else:
        errors = validate_index(args.outputs, write_index(args.outputs, out))
        print(f"wrote {out}")
    for e in errors:
        print(f"ERROR: {e}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
