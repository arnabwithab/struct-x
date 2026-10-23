"""m,c,k identification tests (design sec 7, feature F002).

Fit is on ONE global TRAIN-condition trial only; this module takes an LDV
array + fs and never touches val/test/shift trials (caller enforces that).
Synthetic data only — no real dataset needed.
"""

import json

import numpy as np
import pytest

from mck_fit import fit_mck, load_mck, save_mck

FS = 1000.0


def make_free_decay(zeta, fn, seconds=4.0, amp=1.0, fs=FS):
    """Synthetic hammer free decay; returns (q, true damped freq fd)."""
    fd = fn * np.sqrt(1.0 - zeta**2)
    t = np.arange(int(fs * seconds)) / fs
    q = amp * np.exp(-zeta * 2.0 * np.pi * fn * t) * np.cos(2.0 * np.pi * fd * t)
    return q, fd


@pytest.mark.parametrize(
    ("zeta", "fn"),
    [(0.01, 60.0), (0.02, 80.0), (0.05, 150.0)],
)
def test_damped_sine_recovers_zeta_fd_k_c(zeta, fn):
    q, fd = make_free_decay(zeta, fn)
    params = fit_mck(q, FS)
    assert abs(params["zeta"] - zeta) < 0.002
    assert abs(params["fd"] - fd) / fd < 0.01
    k_true = (2.0 * np.pi * fn) ** 2
    c_true = 2.0 * zeta * np.sqrt(k_true)
    assert abs(params["k"] - k_true) / k_true < 0.05
    assert abs(params["c"] - c_true) / c_true < 0.05


def test_undamped_gives_zero_damping():
    q, fd = make_free_decay(0.0, 80.0)
    params = fit_mck(q, FS)
    assert params["zeta"] < 0.002
    assert abs(params["fn"] - params["fd"]) / params["fd"] < 0.01
    assert params["c"] < 0.05 * np.sqrt(params["k"])  # ~zero damping


def test_output_contract_m1_and_json_serialisable():
    q, _ = make_free_decay(0.02, 80.0)
    params = fit_mck(q, FS)
    assert params["m"] == 1.0
    assert set(params) >= {"m", "c", "k", "zeta", "fd", "fn"}
    assert all(isinstance(v, float) for v in params.values())
    json.dumps(params)  # must not raise
    assert params["fn"] >= params["fd"]  # undamped >= damped by construction


def test_save_load_roundtrip(tmp_path):
    q, _ = make_free_decay(0.02, 80.0)
    params = fit_mck(q, FS)
    path = tmp_path / "mck.json"
    save_mck(params, path)
    reloaded = load_mck(path)
    assert reloaded == params


def test_flat_signal_raises():
    with pytest.raises(ValueError):
        fit_mck(np.zeros(int(FS * 4)), FS)


def test_too_short_signal_raises():
    with pytest.raises(ValueError):
        fit_mck(np.ones(10), FS)


def test_scale_convention_documented():
    import mck_fit as mck_fit

    assert "unidentifiable" in mck_fit.__doc__
