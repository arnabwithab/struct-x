"""LOCO folds, O4 ablations, gradient-ratio controller, checkpoints. See design sec 7-8, 10.

Loss scale (binding, sec 7): ``L = L_LDV + lp*L_phys + ls*L_smooth + 0.1*L_freq``
(mean-reduced, fp32 physics via :mod:`structx.physics`). Tuning and early-stop
use **val Delta_RMSE only**, never matched RMSE. Each O4 variant gets the same
fixed budget (same grids x same epoch cap); best-by-val-Delta_RMSE.

Checkpoints: best-per-fold ONLY (one file, never best+last accumulated).

Runtime env (binding, sec 8/10): seeds, trial IDs, m/c/k + fd/fn, lambdas,
pretrain checkpoint URL+SHA, GPU type + ``nvidia-smi`` + ``pip freeze`` hash
are logged by the caller at runtime -- never pinned here. Micro-batch comes
from the config (4060: 4, T4: 8); effective batch (target 16) via accumulation.
"""

import math
import random
from pathlib import Path

import numpy as np
import torch

from structx import physics
from structx.data_loader import make_loco_splits

LP_GRID = [1e-3, 1e-2, 1e-1]
LS_GRID = [1e-4, 1e-3, 1e-2]
LP_FLOOR = 1e-4  # controller lower bound (sec 7)
LP_CAP = 0.1  # controller upper bound (sec 7)
WARMUP_EPOCHS = 5  # first 5 epochs run at WARMUP_SCALE x lp
WARMUP_SCALE = 0.1
MAX_EPOCHS = 50
PATIENCE = 10  # early-stop patience on val Delta_RMSE (sec 7)


def seed_everything(seed):
    """Seed stdlib/numpy/torch (logged per run; see module docstring)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def o4_variant_configs(lp_grid=None, ls_grid=None):
    """The 4 O4 ablations over the SAME fixed grids (sec 8, non-gameable)."""
    lp_grid = list(LP_GRID if lp_grid is None else lp_grid)
    ls_grid = list(LS_GRID if ls_grid is None else ls_grid)
    return {
        "data_only": [(0.0, 0.0)],
        "smoothness_only": [(0.0, ls) for ls in ls_grid],
        "sdof_only": [(lp, 0.0) for lp in lp_grid],
        "both": [(lp, ls) for lp in lp_grid for ls in ls_grid],
    }


def effective_lambda_p(base_lp, epoch):
    """Loss weight actually used: 0.1x base during the first 5 epochs."""
    return base_lp * WARMUP_SCALE if epoch < WARMUP_EPOCHS else base_lp


def update_lambda_p(r_history, lp, floor=LP_FLOOR, cap=LP_CAP):
    """One gradient-ratio controller step (pure; sec 7).

    r = ||grad L_phys|| / ||grad L_LDV|| tracked per epoch. Applies the last-3
    rule whenever 3+ epochs are recorded: halve lp if all ran hot (r > 1.0,
    floored), double if all ran cold (r < 0.01, capped). Warmup gating (no
    adaptation during the first 5 epochs; 0.1x scale via effective_lambda_p)
    is the caller's job -- see _train_single_config. lp <= 0 (ablations a/b)
    is fixed, never adapted.
    """
    if lp <= 0:
        return lp
    tail = list(r_history)[-3:]
    if len(tail) < 3:
        return lp
    if all(r > 1.0 for r in tail):
        return max(lp / 2.0, floor)
    if all(r < 0.01 for r in tail):
        return min(lp * 2.0, cap)
    return lp


def should_stop(history, patience=PATIENCE):
    """Early-stop on val Delta_RMSE (lower-better): True after `patience`
    epochs with no new best (pure; sec 7)."""
    h = list(history)
    if len(h) <= patience:
        return False
    best = min(h[:-patience])
    return all(v >= best for v in h[-patience:])


def _grad_norm(model, loss):
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = sum(float(g.detach().double().pow(2).sum()) for g in grads if g is not None)
    return math.sqrt(total)


def _grad_ratio(model, v, q, m, c, k):
    """r = ||grad L_phys|| / ||grad L_LDV|| on one batch (fake-able in tests)."""
    q_hat, _ = model(v)
    l_ph = physics.physics_loss(q_hat, m, c, k)
    l_lv = physics.ldv_loss(q_hat, q)
    return _grad_norm(model, l_ph) / (_grad_norm(model, l_lv) + 1e-12)


def _window_rmse(model, batches, device):
    tot, cnt = 0.0, 0
    model.eval()
    with torch.no_grad():
        for v, q, _ in batches:
            v, q = v.to(device), q.to(device)
            q_hat, _ = model(v)
            mse = (q_hat - q).pow(2).mean(dim=-1)
            tot += float(mse.sum())
            cnt += mse.numel()
    if cnt == 0:
        raise ValueError("no validation windows")
    return math.sqrt(tot / cnt)


def _val_delta_rmse(model, val_matched, val_shift, device):
    """Delta_RMSE on VAL trials only (tuning signal; TEST shift never touched)."""
    return _window_rmse(model, val_shift, device) - _window_rmse(model, val_matched, device)


def _train_single_config(
    model,
    train_batches,
    val_matched_batches,
    val_shift_batches,
    m,
    c,
    k,
    lambda_p,
    lambda_s,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
    lr=2e-4,
    accum_steps=1,
    seed=0,
    device="cpu",
    ckpt_path=None,  # accepted for call symmetry; saving is run_fold's job
):
    """Train one (lp, ls) config; return best-val-Delta run summary + state."""
    seed_everything(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    base_lp, r_hist, val_hist = lambda_p, [], []
    best_state, best_delta = None, math.inf
    for epoch in range(max_epochs):
        lp_eff = effective_lambda_p(base_lp, epoch)
        model.train()
        opt.zero_grad()
        pending = 0
        for v, q, f in train_batches:
            v, q, f = v.to(device), q.to(device), f.to(device)
            q_hat, f_hat = model(v)
            loss = physics.total_loss(q_hat, q, f_hat, f, m, c, k, lp_eff, lambda_s)
            (loss / accum_steps).backward()
            pending += 1
            if pending % accum_steps == 0:
                opt.step()
                opt.zero_grad()
                pending = 0
        if pending:
            opt.step()
            opt.zero_grad()
        v0, q0, _ = train_batches[0]
        r_hist.append(_grad_ratio(model, v0.to(device), q0.to(device), m, c, k))
        if base_lp > 0 and len(r_hist) >= WARMUP_EPOCHS:
            base_lp = update_lambda_p(r_hist, base_lp)
        val_hist.append(_val_delta_rmse(model, val_matched_batches, val_shift_batches, device))
        if val_hist[-1] < best_delta:
            best_delta = val_hist[-1]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if should_stop(val_hist, patience):
            break
    tol = 0.05 * abs(best_delta) if best_delta != 0 else 0.0
    target = next(i + 1 for i, v in enumerate(val_hist) if v <= best_delta + tol)
    return {
        "best_val_delta": best_delta,
        "epochs_to_target": target,  # first epoch within 5% of this run's best (sec 15a)
        "epochs_run": len(val_hist),
        "state": best_state,
    }


def run_fold(
    train_batches,
    val_matched_batches,
    val_shift_batches,
    make_model,
    m,
    c,
    k,
    variant="both",
    lp_grid=None,
    ls_grid=None,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
    lr=2e-4,
    micro_batch=4,
    eff_batch=16,
    seed=0,
    device="cpu",
    ckpt_path=None,
):
    """One LOCO fold x one O4 variant: grid-search, best-by-val-Delta_RMSE.

    Saves the single best state_dict to ckpt_path (best-per-fold ONLY).
    Returns exactly: best_lambda_p, best_lambda_s, best_val_delta,
    epochs_to_target, epochs_run, micro_batch, eff_batch.
    """
    configs = o4_variant_configs(lp_grid, ls_grid)[variant]
    accum = max(1, math.ceil(eff_batch / micro_batch))
    best, best_cfg = None, None
    for i, (lp, ls) in enumerate(configs):
        seed_everything(seed + i)  # identical init per config for a given seed
        out = _train_single_config(
            make_model(),
            train_batches,
            val_matched_batches,
            val_shift_batches,
            m,
            c,
            k,
            lambda_p=lp,
            lambda_s=ls,
            max_epochs=max_epochs,
            patience=patience,
            lr=lr,
            accum_steps=accum,
            seed=seed + i,  # distinct init per config, reproducible per seed
            device=device,
        )
        if best is None or out["best_val_delta"] < best["best_val_delta"]:
            best, best_cfg = out, (lp, ls)
    if ckpt_path is not None and best["state"] is not None:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best["state"], ckpt_path)
    return {
        "best_lambda_p": best_cfg[0],
        "best_lambda_s": best_cfg[1],
        "best_val_delta": best["best_val_delta"],
        "epochs_to_target": best["epochs_to_target"],
        "epochs_run": best["epochs_run"],
        "micro_batch": micro_batch,
        "eff_batch": eff_batch,
    }


def run_loco(trial_conditions, build_fold_data, make_model, m, c, k, variant="both", seed=0, **kw):
    """LOCO loop over the conditions found (folds = conditions, never fixed 5)."""
    out = {}
    splits = make_loco_splits(dict(trial_conditions), seed=seed)
    for held in sorted(splits):
        tr, vm, vs = build_fold_data(held, splits[held])
        out[held] = run_fold(tr, vm, vs, make_model, m, c, k, variant=variant, seed=seed, **kw)
    return out
