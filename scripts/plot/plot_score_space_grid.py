"""Score-space scatters: one figure per scenario, one panel per fusion method.

Each panel places every test sample in the plane spanned by the two uni-modal
logits and shades the region the fusion rule accepts, so the effect of an
attack is visible as movement relative to a fixed boundary. Feature fusion has
no such plane -- its decision is not a function of the two uni-modal scores --
so its panel uses the joint model's own image-only and text-only logits and
carries no decision region.
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from configuration_files.configuration import THRESHOLD
from fusion_scores import LATE_RULES, METHODS, SCENARIOS, ScoreSource, rule_scores

TITLES = {
    "clean": "No attack",
    "PGD": "PGD (image)",
    "TREPAT": "TREPAT (text)",
    "PGD+TREPAT": "PGD + TREPAT (both)",
}
LABELS = {"svm-rbf": "SVM-RBF", "feature-fusion": "feature fusion"}
FIG_DIR = "figures/classification_results/scatter"
GRID = 300


def decision_region(ax, rule, heads, xlim, ylim):
    gx, gy = np.meshgrid(np.linspace(*xlim, GRID), np.linspace(*ylim, GRID))
    grid = rule_scores(rule, gx.ravel(), gy.ravel(), heads).reshape(gx.shape)
    ax.contourf(gx, gy, grid, levels=[-1, THRESHOLD, 2],
                colors=["#f6d5d5", "#d9e8f5"], alpha=0.7)
    ax.contour(gx, gy, grid, levels=[THRESHOLD], colors="k", linewidths=1.3)


def panel(ax, source, method, scenario, limits):
    data = source.scores(method, scenario)
    name = LABELS.get(method, method)

    if data is None:
        ax.text(0.5, 0.5, "not available", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        ax.set_title(name)
        return

    label, score, image_logit, text_logit = data
    if image_logit is None or text_logit is None:
        ax.text(0.5, 0.5, "axes unavailable\n(no ablation run)", ha="center",
                va="center", transform=ax.transAxes, color="0.5")
        ax.set_title(name)
        return

    if method in LATE_RULES:
        xlim, ylim = limits
        decision_region(ax, method, source.heads, xlim, ylim)
    else:
        xlim = (image_logit.min() - 0.5, image_logit.max() + 0.5)
        ylim = (text_logit.min() - 0.5, text_logit.max() + 0.5)

    real, fake = np.asarray(label) == 1, np.asarray(label) == 0
    ax.scatter(image_logit[real], text_logit[real], s=15, c="#1f77b4",
               label="Real", alpha=0.75, edgecolors="none")
    ax.scatter(image_logit[fake], text_logit[fake], s=20, c="#d62728",
               label="Fake", alpha=0.9, marker="^", edgecolors="none")

    # How many samples the rule gets wrong in this scenario, for context.
    wrong = int((source.predictions(score) != np.asarray(label)).sum())
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(f"{name}\n{wrong} of {len(label)} misclassified", fontsize=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=FIG_DIR)
    parser.add_argument("--paper-dir", default=None)
    args = parser.parse_args()

    source = ScoreSource()
    os.makedirs(args.out_dir, exist_ok=True)

    # Shared axes across the late-fusion panels of every scenario, so movement
    # between scenarios is comparable rather than rescaled away.
    xs, ys = [source.image_logit], [source.text_logit]
    for method in LATE_RULES:
        for scenario in SCENARIOS:
            data = source.scores(method, scenario)
            if data and data[2] is not None:
                xs.append(data[2])
                ys.append(data[3])
    all_x, all_y = np.concatenate(xs), np.concatenate(ys)
    limits = ((all_x.min() - 1, all_x.max() + 1), (all_y.min() - 1, all_y.max() + 1))

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})

    for scenario in SCENARIOS:
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
        for ax, method in zip(axes.ravel(), METHODS):
            panel(ax, source, method, scenario, limits)
        # Attach the legend to a panel that actually drew points: with a
        # missing run the first panel can be empty.
        for ax in axes.ravel():
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="lower left", frameon=True, fontsize=10)
                break
        fig.supxlabel("image logit")
        fig.supylabel("text logit")
        fig.suptitle(f"Score space by fusion method -- {TITLES[scenario]}", fontsize=15)
        fig.tight_layout()

        stem = f"score_space_{scenario.replace('+', '_')}"
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(args.out_dir, f"{stem}.{ext}"),
                        dpi=200, bbox_inches="tight")
        if args.paper_dir:
            os.makedirs(args.paper_dir, exist_ok=True)
            fig.savefig(os.path.join(args.paper_dir, f"{stem}.pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {os.path.join(args.out_dir, stem)}.{{png,pdf}}")


if __name__ == "__main__":
    main()
