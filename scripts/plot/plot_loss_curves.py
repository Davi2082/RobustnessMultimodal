"""ASR as a function of the attack budget, per attack and method.

For each attack (PGD, TREPAT, joint) and each method, the curve is the share of
the clean-correct samples already misclassified after a given fraction of the
attack budget; at the full budget it equals the ASR of the run. The union
attack is not shown: its two channels are each optimised against the clean
other modality, so no combined state exists before the end.

Steps are read from loss_log.csv, which stores the attacked method's score:
* PGD: one entry per step, logged before the update, so entry k is the state
  after k steps; the stored final score is the state after the last step.
* TREPAT: one entry per queried variant, the score of that variant; the sample
  is flipped from the first variant that crosses the threshold.
* joint: max(PGD steps, TREPAT variants) rounds; round r runs TREPAT variant r
  and, while the PGD budget lasts, PGD step r (5 PGD steps: all in the first
  rounds). Time is the round at which the sample first crosses, as a fraction
  of the rounds.
A flipped sample with no misclassified logged state flipped at the last,
unlogged update: its time is the fraction of the budget the attack had used.
The dashed black line is the unimodal classifier (image-only under PGD,
text-only under TREPAT).

Usage:
    python3 scripts/plot/plot_loss_curves.py --dataset Fakeddit
Writes figures/loss_curves/<DS>_loss_curves.{png,pdf}
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

from configuration_files.configuration import RAND_SEED, method_threshold
from configuration_files.paths import dataset_result_root

ATTACKS = [("PGD", "pgd", "image"), ("TREPAT", "trepat", "text"), ("Joint", "interleaved", None)]
METHODS = [("feature-fusion", "Feature fusion", "#7f7f7f"), ("min", "min", "#1f77b4"),
           ("mean", "mean", "#2ca02c"), ("max", "max", "#9467bd"),
           ("svm-rbf", "SVM-RBF", "#d62728"), ("linear", "linear", "#ff7f0e")]
OUT_DIR = "figures/loss_curves"
GRID = np.linspace(0, 1, 201)


def flip_times(root, attack, method, dataset):
    """Budget fraction at which each clean-correct sample is first misclassified
    (np.inf if it is not misclassified at the end)."""
    run = os.path.join(root, "perturbed", attack, method)
    c = pd.read_csv(os.path.join(root, "clean", method, "results.csv")).set_index("index")
    p = pd.read_csv(os.path.join(run, "perturbed_results.csv")).set_index("index")
    log = pd.read_csv(os.path.join(run, "loss_log.csv"))
    t = method_threshold(method, dataset)
    ok = c.index[c["pred"] == c["label"]]
    budget = log.groupby(["channel", "index"]).size().groupby("channel").max().to_dict()
    if attack == "interleaved":
        # Joint runs max(PGD steps, TREPAT variants) rounds; round r holds PGD
        # step r (while PGD budget lasts) and TREPAT variant r, so both channels
        # share the round count as their time axis.
        rounds = max(budget.values())
        budget = {ch: rounds for ch in budget}

    times = []
    for idx in ok:
        label = int(c.at[idx, "label"])
        wrong = lambda s: (np.asarray(s, dtype=float) > t).astype(int) != label
        if not wrong([p.at[idx, "score"]])[0]:
            times.append(np.inf)  # not misclassified after the attack
            continue
        # Default when no logged state is misclassified: the flip happened at
        # the last, unlogged update, i.e. where the attack stopped (the full
        # budget, or earlier for an attack that stops at the first crossing).
        sample_log = log[log["index"] == idx]
        first = min(1.0, max([len(g) / budget[ch] for ch, g in sample_log.groupby("channel")],
                             default=1.0))
        for ch, g in sample_log.groupby("channel"):
            hit = np.flatnonzero(wrong(g.sort_values("iteration")["score"]))
            if ch == "pgd":  # entry k = state after k steps
                hit = hit[hit > 0]
                if hit.size:
                    first = min(first, hit[0] / budget[ch])
            elif hit.size:  # trepat: variant k (0-based) is query k + 1
                first = min(first, (hit[0] + 1) / budget[ch])
        times.append(first)
    return np.array(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="Fakeddit")
    args = ap.parse_args()
    root = dataset_result_root(args.dataset)

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, len(ATTACKS), figsize=(11, 3.4), sharey=True)
    for ax, (aname, attack, uni) in zip(axes, ATTACKS):
        rows = ([(uni, "unimodal", "k")] if uni else []) + METHODS
        for method, label, color in rows:
            if not os.path.isfile(os.path.join(root, "perturbed", attack, method, "loss_log.csv")):
                continue
            times = flip_times(root, attack, method, args.dataset)
            asr = [(times <= x).mean() * 100 for x in GRID]
            ax.plot(GRID, asr, color=color, lw=1.6, ls="--" if color == "k" else "-", label=label)
        ax.set_title(aname)
        ax.set_xlabel("fraction of the attack budget")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ASR (%)")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0),
               frameon=False, fontsize=10)
    fig.suptitle(f"{args.dataset} (seed {RAND_SEED}): ASR vs attack budget", fontsize=13, y=1.12)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, f"{args.dataset}_seed{RAND_SEED}_loss_curves")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
