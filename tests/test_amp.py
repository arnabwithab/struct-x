"""AMP fp16 with fp32 physics carve-out (design sec 7, 10). Tests first.

CPU-only: AMP-on-CPU may warn/fall back; tests assert flag plumbing,
fp32 carve-out via as_fp32, and that the use_amp=True path executes.
"""

import inspect
from unittest import mock

import pytest
import torch
import torch.nn as nn

import physics
import train
from physics import as_fp32


# --- as_fp32 helper ---
def test_as_fp32_upcasts_fp16():
    x = torch.ones(4, dtype=torch.float16)
    y = as_fp32(x)
    assert y.dtype == torch.float32
    assert torch.allclose(y, torch.ones(4, dtype=torch.float32))


def test_as_fp32_fp32_unchanged():
    x = torch.randn(3, dtype=torch.float32)
    y = as_fp32(x)
    assert y.dtype == torch.float32
    assert torch.equal(y, x)


def test_as_fp32_downcasts_float64():
    x = torch.ones(2, dtype=torch.float64)
    assert as_fp32(x).dtype == torch.float32


def test_as_fp32_non_tensor_passthrough():
    assert as_fp32(1e-6) == 1e-6
    assert as_fp32(None) is None
    assert as_fp32("eps") == "eps"


def test_as_fp32_int_tensor_noop():
    x = torch.ones(3, dtype=torch.int64)
    y = as_fp32(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, x)


def test_as_fp32_bfloat16_upcasts():
    x = torch.ones(2, dtype=torch.bfloat16)
    assert as_fp32(x).dtype == torch.float32


# --- signature plumbing ---
def test_train_single_config_has_use_amp_default_off():
    sig = inspect.signature(train._train_single_config)
    assert "use_amp" in sig.parameters
    assert sig.parameters["use_amp"].default is False


def test_run_fold_has_use_amp_default_off():
    sig = inspect.signature(train.run_fold)
    assert "use_amp" in sig.parameters
    assert sig.parameters["use_amp"].default is False


# --- tiny model + batches (mirrors test_train.py) ---
T = 16


class TinyModel(nn.Module):
    def __init__(self, n_feat=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Linear(2, n_feat)
        self.wave = nn.Linear(n_feat, T)
        self.freq = nn.Linear(n_feat, 1)
        self.softplus = nn.Softplus()

    def forward(self, v):
        z = self.proj(self.pool(v).flatten(1))
        return self.wave(z), self.softplus(self.freq(z)).squeeze(-1)


def _batches(n=2, batch=2, seed=0):
    torch.manual_seed(seed)
    out = []
    for _ in range(n):
        t = torch.arange(T, dtype=torch.float32) / T
        q = torch.sin(2 * torch.pi * 3 * t).repeat(batch, 1)
        q = q + 0.05 * torch.randn(batch, T)
        out.append((torch.randn(batch, 2, 2, 8, 8), q, torch.full((batch,), 30.0)))
    return out


def _fold_kw(**kw):
    args = dict(
        train_batches=_batches(),
        val_matched_batches=_batches(seed=1),
        val_shift_batches=_batches(seed=2),
        make_model=TinyModel,
        m=1.0,
        c=0.5,
        k=100.0,
        variant="data_only",
        max_epochs=1,
        seed=0,
    )
    args.update(kw)
    return args


# --- use_amp=False unchanged ---
def test_use_amp_false_runs_cpu():
    res = train.run_fold(**_fold_kw(use_amp=False))
    assert torch.isfinite(torch.tensor(res["best_val_delta"]))


# --- use_amp=True executes on CPU ---
def test_use_amp_true_runs_cpu():
    res = train.run_fold(**_fold_kw(use_amp=True, device="cpu"))
    assert torch.isfinite(torch.tensor(res["best_val_delta"]))
    assert 1 <= res["epochs_to_target"] <= res["epochs_run"]


def test_use_amp_true_single_config_keys():
    out = train._train_single_config(
        TinyModel(),
        _batches(),
        _batches(seed=1),
        _batches(seed=2),
        m=1.0,
        c=0.5,
        k=100.0,
        lambda_p=1e-3,
        lambda_s=1e-4,
        max_epochs=1,
        patience=10,
        device="cpu",
        use_amp=True,
    )
    assert set(out) == {"best_val_delta", "epochs_to_target", "epochs_run", "state"}
    assert torch.isfinite(torch.tensor(out["best_val_delta"]))


def test_use_amp_true_engages_autocast():
    real_autocast = torch.autocast
    with mock.patch.object(train.torch, "autocast", wraps=real_autocast) as spy:
        train.run_fold(**_fold_kw(use_amp=True, device="cpu"))
    assert spy.called, "use_amp=True must wrap forward in torch.autocast"


def test_use_amp_false_skips_autocast():
    real_autocast = torch.autocast
    with mock.patch.object(train.torch, "autocast", wraps=real_autocast) as spy:
        train.run_fold(**_fold_kw(use_amp=False, device="cpu"))
    assert not spy.called, "use_amp=False must be byte-identical (no autocast)"


def test_use_amp_true_uses_scaler():
    real_scaler = torch.amp.GradScaler
    with mock.patch.object(train.torch.amp, "GradScaler", wraps=real_scaler) as spy:
        train.run_fold(**_fold_kw(use_amp=True, device="cpu"))
    assert spy.called, "use_amp=True must use GradScaler"


def test_physics_carve_out_fp32_under_amp():
    # fp16 q_hat (as autocast might produce) still yields finite fp32 loss
    q_hat = torch.sin(torch.arange(64, dtype=torch.float16)).unsqueeze(0)
    lp = physics.physics_loss(as_fp32(q_hat), 1.0, 0.5, 100.0)
    assert lp.dtype == torch.float32
    assert torch.isfinite(lp).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_use_amp_true_runs_cuda():
    res = train.run_fold(**_fold_kw(use_amp=True, device="cuda", max_epochs=1))
    assert torch.isfinite(torch.tensor(res["best_val_delta"]))
