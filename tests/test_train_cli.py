"""F012 queue prep: --b / --use-amp / --repr plumbing (design sec 5,7,8,10).

Covers the three Kaggle-ready flag paths end-to-end (preprocess + train):
B override with temporal pos-embed re-interp, AMP flag with fp32 carve-out,
repr flag voxel|timesurface. No training runs; parser/plumbing only.
"""

import importlib.util
import os

import pytest

import train

BASE_YAML = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")
PREPROCESS_PY = os.path.join(os.path.dirname(__file__), "..", "scripts", "preprocess.py")
KERNEL_TRAIN_PY = os.path.join(os.path.dirname(__file__), "..", "kernel", "train.py")


def _parse(*argv):
    return train.build_arg_parser().parse_args(list(argv))


# --- train CLI defaults ---
def test_train_parser_defaults():
    args = _parse()
    assert args.b == 5
    assert args.use_amp is False
    assert args.repr == "voxel"
    assert args.folds == "" and args.variants == ""


# --- (1) B override grid ---
@pytest.mark.parametrize("b", [3, 7, 9])
def test_train_parser_accepts_b_grid(b):
    assert _parse("--b", str(b)).b == b


def test_train_parser_rejects_bad_b():
    with pytest.raises(SystemExit):
        _parse("--b", "4")


def test_effective_num_frames_voxel_passthrough():
    assert train.effective_num_frames(7, "voxel") == 7


def test_effective_num_frames_timesurface_is_single_frame():
    # (2,H,W) surface -> (2,1,224,224); n_bins ignored, no pos-embed interp needed
    assert train.effective_num_frames(5, "timesurface") == 1


# --- (2) AMP flag ---
def test_train_parser_use_amp_opt_in():
    assert _parse("--use-amp").use_amp is True


# --- (3) repr flag ---
def test_train_parser_repr_timesurface():
    assert _parse("--repr", "timesurface").repr == "timesurface"


def test_train_parser_rejects_bad_repr():
    with pytest.raises(SystemExit):
        _parse("--repr", "frames")


# --- config plumbing: base.yaml supplies lr/micro/patience, CLI overrides B ---
def test_resolve_config_reads_base_yaml():
    cfg = train.resolve_config(_parse(), BASE_YAML)
    assert cfg["lr"] == pytest.approx(2e-4)
    assert cfg["max_epochs"] == 50 and cfg["patience"] == 10
    assert cfg["b"] == 5 and cfg["use_amp"] is False and cfg["repr"] == "voxel"


def test_resolve_config_cli_overrides_yaml():
    cfg = train.resolve_config(_parse("--b", "7", "--use-amp"), BASE_YAML)
    assert cfg["b"] == 7 and cfg["use_amp"] is True


def _load_preprocess():
    spec = importlib.util.spec_from_file_location("structx_preprocess", PREPROCESS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_preprocess_b_alias():
    mod = _load_preprocess()
    assert mod.build_arg_parser().parse_args(["--b", "7"]).bins == 7
    assert mod.build_arg_parser().parse_args(["--bins", "3"]).bins == 3


def test_preprocess_config_supplies_bins_default():
    mod = _load_preprocess()
    args = mod.build_arg_parser().parse_args(["--config", BASE_YAML])
    assert mod.resolve_bins(args) == 5  # representation.B in base.yaml


def test_kernel_train_has_queue_knobs():
    with open(KERNEL_TRAIN_PY) as f:
        src = f.read()
    for knob in (
        "STRUCTX_B",
        "STRUCTX_USE_AMP",
        "STRUCTX_REPR",
        "STRUCTX_FOLDS",
        "STRUCTX_VARIANTS",
    ):
        assert knob in src
