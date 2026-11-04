"""ASR (mean ± std over seeds) per attack, one bar per fusion method.

    python3 scripts/plot/plot_asr_bars.py --dataset Fakeddit
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configuration_files.configuration import ATTACK_CLASSES

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
ATTACKS = ("PGD", "TREPAT", "Union", "Joint")
METHODS = (("min", "min"), ("mean", "mean"), ("max", "max"), ("svm-rbf", "SVM-RBF"),
           ("linear", "linear"), ("feature-fusion", "Feature fusion"))
COLORS = ("#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#2a78d6")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="Fakeddit")
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(ROOT, "results", args.dataset, "200", "aggregate", ATTACK_CLASSES, "metrics_adv.csv"))
    seeds = df["seeds"].iloc[0].split()
    df = df.set_index(["method", "scenario"])

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    x = np.arange(len(ATTACKS))
    w = 0.8 / len(METHODS)
    for k, ((m, label), c) in enumerate(zip(METHODS, COLORS)):
        mean = [100 * df.loc[(m, a), "ASR_mean"] for a in ATTACKS]
        std = [100 * df.loc[(m, a), "ASR_std"] for a in ATTACKS]
        ax.bar(x + (k - (len(METHODS) - 1) / 2) * w, mean, w, yerr=std, color=c, label=label,
               edgecolor="white", linewidth=1, error_kw=dict(lw=1, capsize=2, ecolor="#333333"))
    ax.set_xticks(x, ATTACKS)
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(METHODS), frameon=False,
              fontsize=9, handlelength=1.2, handletextpad=0.4, columnspacing=0.9)
    fig.tight_layout()

    out = os.path.join(ROOT, "figures", "asr", args.dataset)
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f"{args.dataset}_seeds{seeds[0]}-{seeds[-1]}_{ATTACK_CLASSES}_asr_bars")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
