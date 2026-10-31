"""Per-trial campaign eval (design sec 9): proper trial-level Delta/se/O5.

Replicates the kernel's frozen stats + normalization exactly (same code path),
loads ckpts from outputs/v6+v7, and aggregates per TRIAL (not pooled windows).
Writes results/final_verdict.json. CPU-only, no GPU touched.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from data_loader import (
    compute_train_stats,
    condition_of,
    make_loco_splits,
    pair_trials,
    parse_ldv_txt,
    pick_mck_fit_trial,
    trial_to_training_pairs,
)
from evaluate import (
    delta_m,
    delta_r,
    matched_degradation_report,
    rel_freq_error,
    rmse,
    rq4_compare,
    se_delta,
    summarize_trials,
)
from mck_fit import fit_mck
from models.heads import WaveformFrequencyModel
from voxel import voxel_to_model_input

VARIANTS = ["data_only", "smoothness_only", "sdof_only", "both"]


def trial_metrics(model, windows, f_target, mck, device="cpu", batch=64):
    """Per-trial means over that trial's windows: rmse, mae, pearson r, freq err, L_phys.

    windows: (trial_id, model-ready (2,5,224,224), q (1000,)) already normalized.
    mck: dict with m, c, k for the self-consistency L_phys.
    Forwards run batched on `device`; metrics per window after.
    """
    from evaluate import l_phys, mae, pearson_r

    model.eval().to(device)
    qhs, fhs, order = [], [], []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            chunk = windows[i : i + batch]
            x = torch.stack([torch.from_numpy(np.ascontiguousarray(w[1])) for w in chunk]).to(
                device
            )
            qh, fh = model(x.float())
            qhs.append(qh.cpu().float())
            fhs.append(fh.cpu().float())
            order.extend(w[0] for w in chunk)
    qhs = torch.cat(qhs).numpy()
    fhs = torch.cat(fhs).numpy().ravel()
    per_trial = {}
    for (t, _, q), qh, fh in zip(
        windows, [qhs[i] for i in range(len(windows))], [fhs[i] for i in range(len(windows))]
    ):
        lp = float(
            l_phys(
                torch.from_numpy(np.ascontiguousarray(qh)).unsqueeze(0),
                mck["m"],
                mck["c"],
                mck["k"],
            )
        )
        per_trial.setdefault(t, []).append(
            (rmse(q, qh), mae(q, qh), pearson_r(q, qh), rel_freq_error(float(fh), f_target[t]), lp)
        )
    return {
        t: tuple(float(np.mean([m[i] for m in vv])) for i in range(5))
        for t, vv in per_trial.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="data/XJTU-DV-Beam")
    ap.add_argument("--ckpts", default="outputs")
    ap.add_argument("--encoder", default="resnet", choices=["resnet", "timesformer"])
    ap.add_argument(
        "--remote-ckpts",
        default="",
        help="dir with {held}/{variant}.pt layout (e.g. downloaded remote ckpts)",
    )
    ap.add_argument("--out", default="results/final_verdict.json")
    ap.add_argument("--folds", default="", help="comma-separated held conditions, empty = all")
    ap.add_argument("--device", default=None, help="cuda/cpu; default auto")
    ap.add_argument("--batch", type=int, default=64, help="eval forward batch size")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    want = set(args.folds.split(",")) if args.folds else None

    pairs = pair_trials(os.path.join(args.archive, "DV"), os.path.join(args.archive, "LDV"))
    raw, trial_f = {}, {}
    for trial_id, dv_path, ldv_path in pairs:
        win = trial_to_training_pairs(dv_path, ldv_path, n_bins=5, standardize_q=False)
        t_raw, v_raw = parse_ldv_txt(ldv_path)
        trial_f[trial_id] = float(fit_mck(v_raw, 1 / float(np.median(np.diff(t_raw))))["fd"])
        raw[trial_id] = [(trial_id, v, q) for v, q in win]
        print(f"{trial_id}: {len(win)} windows", flush=True)

    conditions = {t: condition_of(t) for t, _, _ in pairs}
    folds = make_loco_splits(conditions, seed=0)
    verdict = {"folds": {}}
    for held, fold in folds.items():
        if want is not None and held not in want:
            continue
        train_w = [w for t in fold["train"] for w in raw[t]]
        q_mean, q_std = compute_train_stats([w[2] for w in train_w])
        vx = np.stack([voxel_to_model_input(w[1])[0] for w in train_w])
        vx_mean, vx_std = vx.mean(axis=(0, 3, 4)), vx.std(axis=(0, 3, 4))

        def norm(windows):
            mu = vx_mean.reshape(2, -1, 1, 1)
            sg = vx_std.reshape(2, -1, 1, 1)
            out = []
            for t, v, q in windows:
                mv = (voxel_to_model_input(v)[0] - mu) / (sg + 1e-6)
                mq = (q - q_mean) / (q_std + 1e-6)
                out.append((t, mv.astype(np.float32), mq.astype(np.float32)))
            return out

        fold_res = {"mck_trial": None, "variants": {}}
        ldv_by_id = {t: p for t, _, p in pairs}
        for variant in VARIANTS:
            ckpt = None
            roots = [
                os.path.join(args.ckpts, "v6", "ckpts", held),
                os.path.join(args.ckpts, "v7", "ckpts", held),
            ]
            if args.remote_ckpts:
                roots.insert(0, os.path.join(args.remote_ckpts, held))
            for root in roots:
                cand = os.path.join(root, f"{variant}.pt")
                if os.path.isfile(cand):
                    ckpt = cand
            if ckpt is None:
                print(f"SKIP {held}/{variant}: no ckpt", flush=True)
                continue
            if args.encoder == "timesformer":
                from models.timesformer import TimeSformerEncoder

                model = WaveformFrequencyModel(TimeSformerEncoder(num_frames=5), 768)
            else:
                from models.resnet3d import ResNet3DEncoder

                model = WaveformFrequencyModel(ResNet3DEncoder(512), 512)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            model.to(device)
            import time as _time

            model.eval()
            _sample = torch.randn(1, 2, 5, 224, 224).to(device)
            with torch.no_grad():
                model(_sample)
                _t0 = _time.perf_counter()
                for _ in range(20):
                    model(_sample)
                if device == "cuda":
                    torch.cuda.synchronize()
                _ms = (_time.perf_counter() - _t0) / 20 * 1000
            fold_res.setdefault("latency_ms_per_window", {})[variant] = round(float(_ms), 2)
            mck_trial = pick_mck_fit_trial(fold["train"])
            t_mck, v_mck = parse_ldv_txt(ldv_by_id[mck_trial])
            mck = fit_mck(v_mck, 1.0 / float(np.median(np.diff(t_mck))))
            fold_res["mck_trial"] = mck_trial
            tm = trial_metrics(
                model,
                norm([w for t in fold["test_matched"] for w in raw[t]]),
                trial_f,
                mck,
                device=device,
                batch=args.batch,
            )
            ts = trial_metrics(
                model,
                norm([w for t in fold["test_shift"] for w in raw[t]]),
                trial_f,
                mck,
                device=device,
                batch=args.batch,
            )
            m_r = [tm[t][0] for t in sorted(tm)]
            s_r = [ts[t][0] for t in sorted(ts)]
            m_f = [tm[t][3] for t in sorted(tm)]
            s_f = [ts[t][3] for t in sorted(ts)]
            m_a = [tm[t][1] for t in sorted(tm)]
            s_a = [ts[t][1] for t in sorted(ts)]
            m_p = [tm[t][2] for t in sorted(tm)]
            s_p = [ts[t][2] for t in sorted(ts)]
            m_l = [tm[t][4] for t in sorted(tm)]
            s_l = [ts[t][4] for t in sorted(ts)]
            fold_res["variants"][variant] = {
                "rmse_matched": summarize_trials(m_r),
                "rmse_shift": summarize_trials(s_r),
                "freq_matched": summarize_trials(m_f),
                "freq_shift": summarize_trials(s_f),
                "mae_matched": summarize_trials(m_a),
                "mae_shift": summarize_trials(s_a),
                "pearson_matched": summarize_trials(m_p),
                "pearson_shift": summarize_trials(s_p),
                "lphys_matched": summarize_trials(m_l),
                "lphys_shift": summarize_trials(s_l),
                "delta_rmse": delta_m(m_r, s_r),
                "se_rmse": se_delta(m_r, s_r),
                "delta_freq": delta_m(m_f, s_f),
                "delta_pearson_r": delta_r(m_p, s_p),
                "n_matched": len(m_r),
                "n_shift": len(s_r),
            }
        v = fold_res["variants"]
        if "data_only" in v and "both" in v:
            b, p = v["data_only"], v["both"]
            fold_res["o5_rmse"] = dict(
                zip(
                    ["R", "sigma_R", "wins", "statement"],
                    [
                        b["delta_rmse"] - p["delta_rmse"],
                        float(np.sqrt(b["se_rmse"] ** 2 + p["se_rmse"] ** 2)),
                        None,
                        "",
                    ],
                )
            )
            r, s = fold_res["o5_rmse"]["R"], fold_res["o5_rmse"]["sigma_R"]
            fold_res["o5_rmse"]["wins"] = bool(r > s)
            fold_res["o5_rmse"]["statement"] = (
                f"physics wins: R={r:.4f} > sigma_R={s:.4f}"
                if r > s
                else f"NON-IDENTIFIABLE: R={r:.4f} <= sigma_R={s:.4f}"
            )
            fold_res["rq4"] = rq4_compare(b["delta_freq"], b["delta_rmse"])
            fold_res["matched_harm"] = matched_degradation_report(
                [b["rmse_matched"]["mean"]], [p["rmse_matched"]["mean"]]
            )
        verdict["folds"][held] = fold_res
        print(
            f"fold {held}: " + (fold_res.get("o5_rmse", {}).get("statement", "incomplete")),
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
