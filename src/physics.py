"""Mean-reduced LDV/smoothness/SDOF-physics/frequency losses (design sec 7).

Total: ``L = L_LDV + lp*L_phys + ls*L_smooth + 0.1*L_freq`` (``lambdaf=0.1``
fixed, never tuned). ``mean_T`` = 1/T sum over T=1000 samples at dt=1ms,
then mean over batch. No pre-filtering of ``q_hat``.

G5 COLLAPSE GUARD (binding): ``L_phys`` carries the ``mean_T||q_hat||^2``
denominator -- NEVER remove it. Without it ``q_hat -> 0`` trivially
minimises the residual. Consequence: ``L_phys`` is amplitude-blind by design
(``q -> alpha*q`` cancels); amplitude is anchored by ``L_LDV`` alone.

PRECISION BINDING: ``L_phys``, all finite differences, and ``eps`` are
computed in fp32 -- every entry point casts inputs via ``.to(torch.float32)``
before differencing (fp16 min normal ~6e-5 would underflow ``eps=1e-6`` and
NaN the second differences). Elementwise/sub/mul/mean ops used here are not
autocast-promoted, so the cast also makes the internals AMP-autocast-safe.
"""

import torch

EPS = 1e-6  # placeholder, relative to the z-scored unit-variance scale (sec 5)
LAMBDA_F = 0.1  # fixed, never tuned
DT = 1e-3  # 1 ms at Fs = 1000 Hz


def first_derivative(q, dt=DT):
    """d/dt via 2nd-order central interior, 1st-order one-sided edges. fp32."""
    q = q.to(dtype=torch.float32)
    dt = float(dt)
    qd = torch.empty_like(q)
    qd[..., 1:-1] = (q[..., 2:] - q[..., :-2]) / (2.0 * dt)
    qd[..., 0] = (q[..., 1] - q[..., 0]) / dt
    qd[..., -1] = (q[..., -1] - q[..., -2]) / dt
    return qd


def second_derivative(q, dt=DT):
    """d2/dt2 via 2nd-order central interior, 1st-order one-sided edges. fp32."""
    q = q.to(dtype=torch.float32)
    dt = float(dt)
    qdd = torch.empty_like(q)
    qdd[..., 1:-1] = (q[..., 2:] - 2.0 * q[..., 1:-1] + q[..., :-2]) / dt**2
    qdd[..., 0] = (q[..., 0] - 2.0 * q[..., 1] + q[..., 2]) / dt**2
    qdd[..., -1] = (q[..., -1] - 2.0 * q[..., -2] + q[..., -3]) / dt**2
    return qdd


def ldv_loss(q_hat, q):
    """mean_T ||q_hat - q||^2, then mean over batch."""
    q_hat = q_hat.to(dtype=torch.float32)
    q = q.to(dtype=torch.float32)
    return (q_hat - q).pow(2).mean(dim=-1).mean()


def smoothness_loss(q_hat, dt=DT):
    """mean_T ||q_hat_dot||^2, then mean over batch."""
    return first_derivative(q_hat, dt).pow(2).mean(dim=-1).mean()


def physics_loss(q_hat, m, c, k, dt=DT, eps=EPS):
    """Normalised SDOF residual (G5 guard). Amplitude-blind; fp32 throughout."""
    q_hat = q_hat.to(dtype=torch.float32)
    m = torch.as_tensor(m, dtype=torch.float32)
    c = torch.as_tensor(c, dtype=torch.float32)
    k = torch.as_tensor(k, dtype=torch.float32)
    eps = torch.as_tensor(eps, dtype=torch.float32)
    residual = m * second_derivative(q_hat, dt) + c * first_derivative(q_hat, dt) + k * q_hat
    return (residual.pow(2).mean(dim=-1) / (q_hat.pow(2).mean(dim=-1) + eps)).mean()


def frequency_loss(f_hat, f_target):
    """(f_hat - f_target)^2 / f_target^2, mean over batch."""
    f_hat = f_hat.to(dtype=torch.float32)
    f_target = f_target.to(dtype=torch.float32)
    return ((f_hat - f_target).pow(2) / f_target.pow(2)).mean()


def total_loss(q_hat, q, f_hat, f_target, m, c, k, lambda_p, lambda_s, dt=DT):
    """L = L_LDV + lp*L_phys + ls*L_smooth + 0.1*L_freq (scalar)."""
    return (
        ldv_loss(q_hat, q)
        + lambda_p * physics_loss(q_hat, m, c, k, dt)
        + lambda_s * smoothness_loss(q_hat, dt)
        + LAMBDA_F * frequency_loss(f_hat, f_target)
    )
