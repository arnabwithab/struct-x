"""LOCO folds, O4 ablations, gradient-ratio controller, checkpoints. See design sec 7-8, 10.

Loss scale (binding, sec 7): ``L = L_LDV + lp*L_phys + ls*L_smooth + 0.1*L_freq``
(mean-reduced, fp32 physics via :mod:`physics`). Tuning and early-stop
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

import physics
from data_loader import make_loco_splits

VALID_B = (3, 5, 7, 9)  # B=5 default; {3,7,9} = F012-3 ablation grid (sec 8)
REPR_CHOICES = ("voxel", "timesurface")

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
    use_amp=False,
):
    """Train one (lp, ls) config; return best-val-Delta run summary + state."""
    seed_everything(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    scaler = torch.amp.GradScaler(device_type) if use_amp else None
    base_lp, r_hist, val_hist = lambda_p, [], []
    best_state, best_delta = None, math.inf
    for epoch in range(max_epochs):
        lp_eff = effective_lambda_p(base_lp, epoch)
        model.train()
        opt.zero_grad()
        pending = 0
        for v, q, f in train_batches:
            v, q, f = v.to(device), q.to(device), f.to(device)
            if not use_amp:
                q_hat, f_hat = model(v)
                loss = physics.total_loss(q_hat, q, f_hat, f, m, c, k, lp_eff, lambda_s)
                (loss / accum_steps).backward()
            else:
                with torch.autocast(device_type=device_type, enabled=True):
                    q_hat, f_hat = model(v)
                    l_ldv = physics.ldv_loss(q_hat, q)
                    l_smooth = physics.smoothness_loss(q_hat)
                    l_freq = physics.frequency_loss(f_hat, f)
                l_phys = physics.physics_loss(physics.as_fp32(q_hat), m, c, k)
                loss = l_ldv + lp_eff * l_phys + lambda_s * l_smooth + physics.LAMBDA_F * l_freq
                scaler.scale(loss / accum_steps).backward()
            pending += 1
            if pending % accum_steps == 0:
                if scaler is None:
                    opt.step()
                else:
                    scaler.step(opt)
                    scaler.update()
                opt.zero_grad()
                pending = 0
        if pending:
            if scaler is None:
                opt.step()
            else:
                scaler.step(opt)
                scaler.update()
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


def effective_num_frames(b, repr_kind):
    """Model frames: voxels run T=B; a (2,H,W) time-surface is a single frame.

    voxel_to_model_input already maps (2,H,W) -> (2,1,224,224), so the surface
    path needs no pos-embed re-interp; B only matters for voxels.
    """
    return 1 if repr_kind == "timesurface" else int(b)


def build_arg_parser():
    """CLI for `python -m train` / run_folds.sh (F012 queue flags).

    Queue flags are the source of truth for B/AMP/repr; --config supplies the
    training hyperparams (lr, epochs, patience, micro/eff batch, grids, seed).
    """
    import argparse

    p = argparse.ArgumentParser(description="LOCO training, O4 ablations (design sec 7-8).")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--b", type=int, default=None, choices=list(VALID_B))
    p.add_argument("--use-amp", action="store_true", help="AMP fp16 + fp32 physics carve-out")
    p.add_argument("--repr", default=None, choices=list(REPR_CHOICES))
    p.add_argument("--encoder", default="resnet", choices=["resnet", "timesformer"])
    p.add_argument("--folds", default="", help="comma-separated held conditions, empty = all")
    p.add_argument(
        "--variants", default="", help="subset of data_only,smoothness_only,sdof_only,both"
    )
    p.add_argument("--archive", default="data/XJTU-DV-Beam")
    p.add_argument("--out", default="outputs/train_results.json")
    p.add_argument("--ckpt-dir", default="outputs/ckpts")
    p.add_argument("--micro", type=int, default=None, help="override config micro-batch")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda/cpu; default auto")
    return p


def resolve_config(args, config_path=None):
    """Merge base.yaml hyperparams with the CLI queue flags (no training runs)."""
    import yaml

    path = config_path if config_path is not None else getattr(args, "config", "")
    with open(path) as f:
        base = yaml.safe_load(f) or {}
    rep = base.get("representation", {}) or {}
    opt = base.get("optim", {}) or {}
    loss = base.get("loss", {}) or {}
    data = base.get("data", {}) or {}
    return {
        "b": args.b if args.b is not None else int(rep.get("B", 5)),
        "use_amp": args.use_amp,
        "repr": args.repr or rep.get("repr", "voxel"),
        "encoder": args.encoder,
        "lr": float(opt.get("lr", 2e-4)),
        "max_epochs": args.max_epochs or int(opt.get("max_epochs", MAX_EPOCHS)),
        "patience": int(opt.get("patience", PATIENCE)),
        "micro": args.micro,
        "micro_4060": int(opt.get("micro_batch_4060", 4)),
        "micro_t4": int(opt.get("micro_batch_t4", 8)),
        "eff_batch": int(opt.get("effective_batch", 16)),
        "lp_grid": list(loss.get("lambda_p_grid", LP_GRID)),
        "ls_grid": list(loss.get("lambda_s_grid", LS_GRID)),
        "seed": args.seed if args.seed is not None else int(data.get("seed", 0)),
        "folds": args.folds,
        "variants": args.variants,
        "archive": args.archive,
        "out": args.out,
        "ckpt_dir": args.ckpt_dir,
        "device": args.device,
    }


def resolve_wants(folds, folds_arg, variants_arg):
    """Parse + validate --folds/--variants against LOCO folds (early ValueError)."""
    want_folds = [f for f in folds_arg.split(",") if f] or sorted(folds)
    want_v = [v for v in variants_arg.split(",") if v] or sorted(o4_variant_configs())
    unknown_v = set(want_v) - set(o4_variant_configs())
    if unknown_v:
        raise ValueError(f"unknown variants: {sorted(unknown_v)}")
    unknown_f = set(want_folds) - set(folds)
    if unknown_f:
        raise ValueError(f"unknown folds: {sorted(unknown_f)}")
    return want_folds, want_v


def main(argv=None):
    """Offline LOCO driver: decode -> frozen stats -> folds x variants -> ckpts."""
    import json
    import os

    args = build_arg_parser().parse_args(argv)
    cfg = resolve_config(args)
    b_eff = effective_num_frames(cfg["b"], cfg["repr"])

    from data_loader import (
        compute_train_stats,
        condition_of,
        pair_trials,
        parse_ldv_txt,
        pick_mck_fit_trial,
        trial_to_training_pairs,
    )
    from mck_fit import fit_mck
    from voxel import voxel_to_model_input

    device = cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    micro = cfg["micro"] or (cfg["micro_t4"] if device.startswith("cuda") else cfg["micro_4060"])
    print(
        f"B={cfg['b']} frames={b_eff} repr={cfg['repr']} amp={cfg['use_amp']} "
        f"encoder={cfg['encoder']} device={device} micro={micro}",
        flush=True,
    )

    pairs = pair_trials(os.path.join(cfg["archive"], "DV"), os.path.join(cfg["archive"], "LDV"))
    raw, trial_f = {}, {}
    for trial_id, dv_path, ldv_path in pairs:
        win = trial_to_training_pairs(
            dv_path, ldv_path, n_bins=cfg["b"], repr_kind=cfg["repr"], standardize_q=False
        )
        t_raw, v_raw = parse_ldv_txt(ldv_path)
        trial_f[trial_id] = float(fit_mck(v_raw, 1 / float(np.median(np.diff(t_raw))))["fd"])
        raw[trial_id] = [(trial_id, v, q) for v, q in win]
        print(f"{trial_id}: {len(win)} windows", flush=True)

    def make_model():
        from models.heads import WaveformFrequencyModel

        if cfg["encoder"] == "timesformer":
            from models.timesformer import TimeSformerEncoder

            enc = TimeSformerEncoder(num_frames=b_eff)
            try:
                enc.load_kinetics_weights()
            except (OSError, ValueError, RuntimeError) as e:
                raise SystemExit(
                    "timesformer needs Kinetics-pretrained init (design sec 5); "
                    f"refusing to train Transformer from scratch: {e}"
                )
            return WaveformFrequencyModel(enc, 768)
        from models.resnet3d import ResNet3DEncoder

        return WaveformFrequencyModel(ResNet3DEncoder(512), 512)

    conditions = {t: condition_of(t) for t, _, _ in pairs}
    folds = make_loco_splits(conditions, seed=cfg["seed"])
    want_folds, want_v = resolve_wants(folds, cfg["folds"], cfg["variants"])

    out = {"config": {k: v for k, v in cfg.items()}, "folds": {}}
    for held in want_folds:
        fold = folds[held]
        train_w = [w for t in fold["train"] for w in raw[t]]
        q_mean, q_std = compute_train_stats([w[2] for w in train_w])
        vx = np.stack([voxel_to_model_input(w[1])[0] for w in train_w])
        mu = torch.from_numpy(vx.mean(axis=(0, 3, 4))).view(2, -1, 1, 1).float()
        sg = torch.from_numpy(vx.std(axis=(0, 3, 4))).view(2, -1, 1, 1).float()

        def batchify(items, _mu=mu, _sg=sg, _qm=q_mean, _qs=q_std, _tf=trial_f):
            batches = []
            for i in range(0, len(items), micro):
                chunk = items[i : i + micro]
                v = torch.stack(
                    [
                        (torch.from_numpy(voxel_to_model_input(w[1])[0]).float() - _mu)
                        / (_sg + 1e-6)
                        for w in chunk
                    ]
                )
                q = torch.stack([torch.from_numpy((w[2] - _qm) / (_qs + 1e-6)) for w in chunk])
                f = torch.tensor([_tf[w[0]] for w in chunk], dtype=torch.float32)
                batches.append((v, q.float(), f))
            return batches

        t_mck, v_mck = parse_ldv_txt({t: p for t, _, p in pairs}[pick_mck_fit_trial(fold["train"])])
        mck = fit_mck(v_mck, 1 / float(np.median(np.diff(t_mck))))
        m, c, k = mck["m"], mck["c"], mck["k"]
        train_b = batchify(train_w)
        vm_b = batchify([w for t in fold["val_matched"] for w in raw[t]])
        vs_b = batchify([w for t in fold["val_shift"] for w in raw[t]])
        fold_res = {}
        for variant in want_v:
            ckpt = os.path.join(cfg["ckpt_dir"], held, f"{variant}.pt")
            res = run_fold(
                train_b,
                vm_b,
                vs_b,
                make_model,
                m,
                c,
                k,
                variant=variant,
                micro_batch=micro,
                eff_batch=cfg["eff_batch"],
                max_epochs=cfg["max_epochs"],
                patience=cfg["patience"],
                lr=cfg["lr"],
                seed=cfg["seed"],
                device=device,
                ckpt_path=ckpt,
                use_amp=cfg["use_amp"],
                lp_grid=cfg["lp_grid"],
                ls_grid=cfg["ls_grid"],
            )
            fold_res[variant] = res
            print(f"{held}/{variant}: best_val_delta={res['best_val_delta']:.4f}", flush=True)
        out["folds"][held] = fold_res
    os.makedirs(os.path.dirname(cfg["out"]) or ".", exist_ok=True)
    with open(cfg["out"], "w") as f:
        json.dump(out, f)
    print(f"wrote {cfg['out']}")
    return out


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
    use_amp=False,
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
            use_amp=use_amp,
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


if __name__ == "__main__":
    main()
