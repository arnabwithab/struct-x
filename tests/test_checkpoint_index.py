"""F015-3: checkpoint index manifest (design sec 10). Tmp outputs trees only."""

import importlib.util
import json
import os

INDEX_PY = os.path.join(os.path.dirname(__file__), "..", "scripts", "checkpoint_index.py")


def _load():
    spec = importlib.util.spec_from_file_location("structx_checkpoint_index", INDEX_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path, data=b"fake-ckpt"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _tree(root):
    _write(root / "ckpts" / "CondA" / "data_only.pt")
    _write(root / "ckpts" / "CondA" / "both.pt")
    with open(root / "train_results.json", "w") as f:
        json.dump(
            {
                "config": {"seed": 0},
                "folds": {
                    "CondA": {
                        "data_only": {
                            "best_lambda_p": 0.0,
                            "best_lambda_s": 0.0,
                            "epochs_run": 3,
                        },
                        "both": {
                            "best_lambda_p": 0.01,
                            "best_lambda_s": 0.001,
                            "epochs_run": 3,
                        },
                    }
                },
            },
            f,
        )


def test_build_index_maps_ckpt_to_fold_variant_seed(tmp_path):
    mod = _load()
    _tree(tmp_path)
    entries = {e["path"]: e for e in mod.build_index(tmp_path)["checkpoints"]}
    assert set(entries) == {"ckpts/CondA/both.pt", "ckpts/CondA/data_only.pt"}
    assert entries["ckpts/CondA/data_only.pt"]["fold"] == "CondA"
    assert entries["ckpts/CondA/data_only.pt"]["variant"] == "data_only"
    assert entries["ckpts/CondA/data_only.pt"]["seed"] == 0
    assert entries["ckpts/CondA/both.pt"]["best_lambda_p"] == 0.01


def test_build_index_null_metadata_without_sidecar(tmp_path):
    mod = _load()
    _write(tmp_path / "ckpts" / "CondB" / "sdof_only.pt", b"x")
    (entry,) = mod.build_index(tmp_path)["checkpoints"]
    assert (entry["fold"], entry["variant"], entry["seed"]) == ("CondB", "sdof_only", None)


def test_validate_ok_on_roundtrip(tmp_path):
    mod = _load()
    _tree(tmp_path)
    index = mod.write_index(tmp_path, tmp_path / "checkpoint_index.json")
    assert mod.validate_index(tmp_path, index) == []
    reread = json.loads((tmp_path / "checkpoint_index.json").read_text())
    assert mod.validate_index(tmp_path, reread) == []


def test_validate_flags_missing_and_extra(tmp_path):
    mod = _load()
    _tree(tmp_path)
    index = mod.build_index(tmp_path)
    (tmp_path / "ckpts" / "CondA" / "both.pt").unlink()
    missing = mod.validate_index(tmp_path, index)
    assert any("ckpts/CondA/both.pt" in e and "missing" in e for e in missing)
    _write(tmp_path / "ckpts" / "CondA" / "smoothness_only.pt", b"x")
    extra = mod.validate_index(tmp_path, index)
    assert any("smoothness_only" in e and "unindexed" in e for e in extra)


def test_validate_rejects_unknown_variant(tmp_path):
    mod = _load()
    _write(tmp_path / "ckpts" / "CondA" / "bogus.pt", b"x")
    errors = mod.validate_index(tmp_path, mod.build_index(tmp_path))
    assert any("bogus" in e and "unknown variant" in e for e in errors)
