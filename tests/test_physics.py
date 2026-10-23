"""F004: fp32 physics loss + collapse guard (design sec 7). Tests first."""

import math

import torch

from physics import (
    first_derivative,
    frequency_loss,
    ldv_loss,
    physics_loss,
    second_derivative,
    smoothness_loss,
    total_loss,
)

T = 1000
DT = 1e-3
FN = 50.0
M = 1.0
K = (2.0 * math.pi * FN) ** 2
ZETA = 0.02
C = 2.0 * ZETA * math.sqrt(K)


def _sine(phase=0.0):
    t = torch.arange(T, dtype=torch.float32) * DT
    return torch.sin(2.0 * math.pi * FN * t + phase).unsqueeze(0)


def test_zero_prediction_collapse_guard():
    """q_hat=0 must exercise the eps denominator path: finite, and total > 0."""
    torch.manual_seed(0)
    q_hat = torch.zeros(2, T)
    q = _sine().repeat(2, 1)
    lp = physics_loss(q_hat, M, C, K, DT)
    assert torch.isfinite(lp).all(), f"L_phys(0) not finite: {lp}"
    tot = total_loss(
        q_hat,
        q,
        torch.tensor([FN, FN]),
        torch.tensor([FN, FN]),
        M,
        C,
        K,
        lambda_p=1e-2,
        lambda_s=1e-3,
        dt=DT,
    )
    assert torch.isfinite(tot).all(), f"total(0) not finite: {tot}"
    assert tot.item() > 0, f"collapse to zero must still pay L_LDV, got {tot.item()}"


def test_physics_loss_amplitude_blind():
    """G5 consequence: q -> alpha*q cancels in L_phys (binding)."""
    q = _sine()
    base = physics_loss(q, M, C, K, DT)
    scaled = physics_loss(2.5 * q, M, C, K, DT)
    assert torch.isfinite(base) and torch.isfinite(scaled)
    assert torch.allclose(base, scaled, rtol=1e-4, atol=1e-6), f"{base} vs {scaled}"


def test_sine_near_fn_beats_white_noise():
    """A pure sine near fn has lower L_phys than white noise at the same RMS."""
    torch.manual_seed(42)
    sine = _sine()
    noise = torch.randn(1, T)
    noise = noise * (sine.pow(2).mean().sqrt() / noise.pow(2).mean().sqrt())
    lp_sine = physics_loss(sine, M, C, K, DT)
    lp_noise = physics_loss(noise, M, C, K, DT)
    assert torch.isfinite(lp_sine) and torch.isfinite(lp_noise)
    assert lp_sine < lp_noise, f"sine {lp_sine} not below noise {lp_noise}"


def test_stencil_exact_on_quadratics():
    """2nd deriv of t^2 is 2 (all samples); 1st deriv of t is 1 (interior)."""
    t = (
        torch.arange(T, dtype=torch.float32) - T / 2
    ) * DT  # centred: same d2/dt2, less fp32 cancellation
    qdd = second_derivative((t**2).unsqueeze(0), DT)
    assert torch.allclose(
        qdd, torch.full_like(qdd, 2.0), rtol=1e-2, atol=0.1
    ), f"max err {(qdd - 2.0).abs().max()}"
    qd = first_derivative(t.unsqueeze(0), DT)
    interior = qd[0, 1:-1]
    assert torch.allclose(
        interior, torch.ones_like(interior), rtol=1e-5, atol=1e-5
    ), f"max err {(interior - 1.0).abs().max()}"


def test_fp16_input_still_fp32_finite():
    """fp16 inputs get cast to fp32 before differencing: finite fp32 result."""
    q_hat = _sine().half()
    q = _sine(phase=0.5).half()
    tot = total_loss(
        q_hat,
        q,
        torch.tensor([FN], dtype=torch.float16),
        torch.tensor([FN], dtype=torch.float16),
        M,
        C,
        K,
        lambda_p=1e-2,
        lambda_s=1e-3,
        dt=DT,
    )
    assert tot.dtype == torch.float32, f"expected fp32, got {tot.dtype}"
    assert torch.isfinite(tot).all(), f"fp16 input gave non-finite loss: {tot}"
    lp = physics_loss(q_hat, M, C, K, DT)
    assert lp.dtype == torch.float32 and torch.isfinite(lp).all()


def test_batch_mean_reduction_shape():
    """All losses reduce (B, T=1000) -> scalar (mean_T, then mean over batch)."""
    torch.manual_seed(1)
    q_hat = torch.randn(4, T)
    q = torch.randn(4, T)
    f_hat = torch.full((4,), FN) * 1.01
    f_tgt = torch.full((4,), FN)
    for loss in (
        ldv_loss(q_hat, q),
        smoothness_loss(q_hat, DT),
        physics_loss(q_hat, M, C, K, DT),
        frequency_loss(f_hat, f_tgt),
        total_loss(q_hat, q, f_hat, f_tgt, M, C, K, 1e-2, 1e-3, DT),
    ):
        assert loss.shape == torch.Size([]), f"expected scalar, got {loss.shape}"
        assert torch.isfinite(loss).all()
    assert frequency_loss(f_tgt, f_tgt).item() == 0.0
