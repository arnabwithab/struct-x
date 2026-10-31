"""Figures for results/report.md (Agg backend).

Run: uv run --with matplotlib python scripts/make_figures.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "figs")
os.makedirs(OUT, exist_ok=True)

FOLDS = ["On_set1", "Off_set1", "Off_set2", "Off_set3", "On_set2", "On_set3"]


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ctrl = load("results/final_verdict.json")["folds"]
    pre = load("results/final_verdict_pretrained.json")["folds"]

    # 1. O5 forest plot: R +- sigma_R per fold (control, data_only vs both)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, held in enumerate(FOLDS):
        v = ctrl[held]["variants"]
        b, p = v["data_only"], v["both"]
        r = b["delta_rmse"] - p["delta_rmse"]
        s = float(np.hypot(b["se_rmse"], p["se_rmse"]))
        ax.errorbar(r, i, xerr=s, fmt="o", capsize=4,
                    label="R +/- sigma_R" if i == 0 else "")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(range(len(FOLDS)))
    ax.set_yticklabels(FOLDS)
    ax.set_xlabel("R = Delta_base - Delta_phys (RMSE)")
    ax.set_title("O5 falsification: no fold clears the noise bar")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/o5_forest.png", dpi=120)

    # 2. Matched RMSE by fold x variant (control)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(FOLDS))
    for j, var in enumerate(["data_only", "sdof_only", "both"]):
        means = [ctrl[h]["variants"][var]["rmse_matched"]["mean"] for h in FOLDS]
        stds = [ctrl[h]["variants"][var]["rmse_matched"]["std"] for h in FOLDS]
        ax.bar(x + (j - 1) * 0.25, means, 0.25, yerr=stds, capsize=3, label=var)
    ax.set_xticks(x)
    ax.set_xticklabels(FOLDS, rotation=20)
    ax.set_ylabel("matched test RMSE (z-scored units)")
    ax.set_title("Matched RMSE varies more across folds than variants move it")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/matched_rmse.png", dpi=120)

    # 3. Pretrained vs control: matched RMSE + epochs to target
    pfolds = ["On_set1", "On_set2", "Off_set3"]
    pilot = {"data_only": 1.957, "both": 2.598}  # v8 kernel log
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for j, var in enumerate(["data_only", "both"]):
        cm, pm = [], []
        for h in pfolds:
            cm.append(ctrl[h]["variants"][var]["rmse_matched"]["mean"])
            pm.append(pilot[var] if h == "On_set1"
                      else pre[h]["variants"][var]["rmse_matched"]["mean"])
        xi = [i + (j - 0.5) * 0.35 for i in range(3)]
        axes[0].bar(xi, cm, 0.35, label=f"control {var}")
        axes[0].bar(xi, pm, 0.35, alpha=0.55, hatch="//" if j == 0 else "\\\\",
                    label=f"pretrained {var}")
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(pfolds)
    axes[0].set_ylabel("matched test RMSE")
    axes[0].set_title("Pretrained init: same endpoint")
    axes[0].legend(fontsize=8)
    ctrl_ep = {"On_set1": 8, "On_set2": 39, "Off_set3": 1}
    pre_ep = {"On_set1": 11, "On_set2": 1, "Off_set3": 1}
    x = np.arange(3)
    axes[1].bar(x - 0.2, [ctrl_ep[h] for h in pfolds], 0.4, label="control")
    axes[1].bar(x + 0.2, [pre_ep[h] for h in pfolds], 0.4, label="pretrained")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pfolds)
    axes[1].set_ylabel("epochs to 5%-of-best (data_only)")
    axes[1].set_title("Convergence: init helps speed, not outcome")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/pretrained_vs_control.png", dpi=120)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
