"""ROC curves: one figure per scenario, one panel per fusion method.

Four figures (clean, PGD, TREPAT, PGD+TREPAT), each a 2x3 grid over the five
late-fusion rules and feature fusion, so a reader compares fusion methods
within a scenario at a glance. Fake is the positive class, matching the rest of
the evaluation.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from configuration_files.paths import ROC_PLOTS_DIR
from fusion_scores import METHODS, SCENARIOS, ScoreSource

TITLES = {
    "clean": "No attack",
    "PGD": "PGD (image)",
    "TREPAT": "TREPAT (text)",
    "PGD+TREPAT": "PGD + TREPAT (both)",
}
LABELS = {"svm-rbf": "SVM-RBF", "feature-fusion": "Feature fusion"}


def panel(ax, source, method, scenario, clean_curve):
    data = source.scores(method, scenario)
    title = LABELS.get(method, method)

    if data is None:
        ax.text(0.5, 0.5, "not available", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        ax.set_title(title)
        return None

    label, score = data[0], data[1]
    # Fake is the positive class, so labels and scores are both inverted.
    fpr, tpr, _ = roc_curve(1 - np.asarray(label), 1 - np.asarray(score))
    area = auc(fpr, tpr)

    if clean_curve is not None and scenario != "clean":
        ax.plot(*clean_curve[:2], color="0.7", lw=1.2, ls="--",
                label=f"clean (AUC {clean_curve[2]:.3f})")
    ax.plot(fpr, tpr, color="#1f77b4", lw=1.8, label=f"{TITLES[scenario]} (AUC {area:.3f})")
    ax.plot([0, 1], [0, 1], color="0.85", lw=1, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    return fpr, tpr, area


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=ROC_PLOTS_DIR)
    parser.add_argument("--paper-dir", default=None,
                        help="Also copy the PDFs into this directory.")
    args = parser.parse_args()

    source = ScoreSource()
    os.makedirs(args.out_dir, exist_ok=True)

    clean_curves = {}
    for method in METHODS:
        data = source.scores(method, "clean")
        if data is None:
            continue
        fpr, tpr, _ = roc_curve(1 - np.asarray(data[0]), 1 - np.asarray(data[1]))
        clean_curves[method] = (fpr, tpr, auc(fpr, tpr))

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})

    for scenario in SCENARIOS:
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4), sharex=True, sharey=True)
        for ax, method in zip(axes.ravel(), METHODS):
            panel(ax, source, method, scenario, clean_curves.get(method))
        fig.supxlabel("False positive rate")
        fig.supylabel("True positive rate")
        fig.suptitle(f"ROC by fusion method -- {TITLES[scenario]}", fontsize=15)
        fig.tight_layout()

        stem = f"roc_grid_{scenario.replace('+', '_')}"
        for ext in ("png", "pdf"):
            path = os.path.join(args.out_dir, f"{stem}.{ext}")
            fig.savefig(path, dpi=200, bbox_inches="tight")
        if args.paper_dir:
            os.makedirs(args.paper_dir, exist_ok=True)
            fig.savefig(os.path.join(args.paper_dir, f"{stem}.pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {os.path.join(args.out_dir, stem)}.{{png,pdf}}")


if __name__ == "__main__":
    main()
