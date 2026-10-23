"""F005: LOCO loop, O4 ablations, gradient-ratio controller, checkpoints (design sec 7-8).

Synthetic data only (tiny linear model, CPU-fast). Controller/early-stop/grid
tests use fake histories, not real training.
"""

import itertools

import torch
import torch.nn as nn

from structx.train import (
    LP_GRID,
    LS_GRID,
    effective_lambda_p,
    o4_variant_configs,
    run_fold,
    run_loco,
    should_stop,
    update_lambda_p,
)

T = 16  # tiny waveform length (physics losses work for any T)


class TinyModel(nn.Module):
    """Minimal V -> (q_hat, f_hat) stand-in for encoder+heads."""

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


# --- O4 variant configs ---


def test_o4_variant_configs_keys_and_grids():
    cfgs = o4_variant_configs()
    assert set(cfgs) == {"data_only", "smoothness_only", "sdof_only", "both"}
    assert cfgs["data_only"] == [(0.0, 0.0)]
    assert cfgs["smoothness_only"] == [(0.0, ls) for ls in LS_GRID]
    assert cfgs["sdof_only"] == [(lp, 0.0) for lp in LP_GRID]
    # both: jointly-tuned pair from the SAME grid, never hand-picked
    assert sorted(cfgs["both"]) == sorted(itertools.product(LP_GRID, LS_GRID))
    assert LP_GRID == [1e-3, 1e-2, 1e-1] and LS_GRID == [1e-4, 1e-3, 1e-2]


# --- gradient-ratio controller (fake grad norms -> r history) ---


def test_update_lambda_p_halves_on_three_high():
    assert update_lambda_p([2.0, 1.5, 3.0], 0.01) == 0.005


def test_update_lambda_p_doubles_on_three_low():
    assert update_lambda_p([0.001, 0.005, 0.009], 0.01) == 0.02


def test_update_lambda_p_holds_on_mixed():
    assert update_lambda_p([2.0, 0.5, 0.001], 0.01) == 0.01


def test_update_lambda_p_floor_and_cap():
    assert update_lambda_p([5.0, 5.0, 5.0], 1e-4) == 1e-4  # floor, never below
    assert update_lambda_p([1e-3, 1e-3, 1e-3], 0.1) == 0.1  # cap, never above


def test_update_lambda_p_warmup_holds_base():
    # fewer than 3 epochs recorded: no adaptation; warmup scale via effective_lambda_p
    assert update_lambda_p([2.0, 2.0], 0.01) == 0.01
    assert update_lambda_p([], 0.01) == 0.01
    assert effective_lambda_p(0.01, epoch=0) == 0.001  # 0.1x warmup scale
    assert effective_lambda_p(0.01, epoch=4) == 0.001
    assert effective_lambda_p(0.01, epoch=5) == 0.01


# --- early stop on val Delta_RMSE ---


def test_should_stop_improving_is_false():
    assert should_stop([0.5, 0.4, 0.3, 0.2], patience=3) is False


def test_should_stop_flat_triggers():
    assert should_stop([0.5, 0.3, 0.3, 0.3, 0.3], patience=3) is True


def test_should_stop_patience_boundary():
    base = [0.5, 0.3, 0.3, 0.3]  # only 2 flat epochs after best -> keep going
    assert should_stop(base, patience=3) is False
    assert should_stop(base + [0.3], patience=3) is True  # 3rd flat epoch -> stop


def test_should_stop_short_history_never():
    assert should_stop([0.5], patience=10) is False
    assert should_stop([], patience=3) is False


# --- run_fold smoke + grid best-selection ---


def _run(**kw):
    args = dict(
        train_batches=_batches(),
        val_matched_batches=_batches(seed=1),
        val_shift_batches=_batches(seed=2),
        make_model=TinyModel,
        m=1.0,
        c=0.5,
        k=100.0,
        variant="data_only",
        max_epochs=2,
        seed=0,
    )
    args.update(kw)
    return run_fold(**args)


def test_run_fold_result_keys():
    res = _run()
    assert set(res) == {
        "best_lambda_p",
        "best_lambda_s",
        "best_val_delta",
        "epochs_to_target",
        "epochs_run",
        "micro_batch",
        "eff_batch",
    }
    assert torch.isfinite(torch.tensor(res["best_val_delta"]))
    assert 1 <= res["epochs_to_target"] <= res["epochs_run"] <= 2
    assert res["micro_batch"] == 4 and res["eff_batch"] == 16
    assert (res["best_lambda_p"], res["best_lambda_s"]) == (0.0, 0.0)


def test_run_fold_grid_picks_min_val_delta(tmp_path):
    from structx.train import _train_single_config, seed_everything

    lp_grid, ls_grid = [1e-3, 1e-1], [1e-4, 1e-2]
    per_cfg = {}
    for i, (lp, ls) in enumerate(itertools.product(lp_grid, ls_grid)):
        seed_everything(i)  # same init run_fold uses for config i (seed=0)
        per_cfg[(lp, ls)] = _train_single_config(
            TinyModel(),
            _batches(),
            _batches(seed=1),
            _batches(seed=2),
            m=1.0,
            c=0.5,
            k=100.0,
            lambda_p=lp,
            lambda_s=ls,
            max_epochs=1,
            patience=10,
            lr=2e-4,
            accum_steps=1,
            seed=i,  # run_fold seeds config i with seed+i
            device="cpu",
            ckpt_path=None,
        )["best_val_delta"]
    res = _run(variant="both", lp_grid=lp_grid, ls_grid=ls_grid, max_epochs=1, eff_batch=4)
    expect = min(per_cfg, key=per_cfg.get)
    assert (res["best_lambda_p"], res["best_lambda_s"]) == expect
    assert res["best_val_delta"] == per_cfg[expect]


def test_run_fold_deterministic_and_checkpoint(tmp_path):
    ckpt = tmp_path / "best.pt"
    a = _run(ckpt_path=str(ckpt), max_epochs=1)
    b = _run(ckpt_path=str(ckpt), max_epochs=1)
    assert a == b
    assert ckpt.exists()  # best-per-fold ONLY: one file, never best+last


def test_run_loco_folds_equal_conditions():
    conds = {"t%02d" % i: ("illum" if i < 2 else "thresh") for i in range(4)}

    def build(held, split):
        return _batches(), _batches(seed=1), _batches(seed=2)

    out = run_loco(
        conds,
        build,
        make_model=TinyModel,
        m=1.0,
        c=0.5,
        k=100.0,
        variant="data_only",
        max_epochs=1,
        seed=0,
    )
    assert set(out) == {"illum", "thresh"}  # folds = conditions found
    for res in out.values():
        assert res["epochs_run"] >= 1
