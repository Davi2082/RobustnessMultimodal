"""Score-space scatters: one figure per scenario, one panel per fusion method.

Each panel places every test sample in the plane spanned by the two uni-modal
logits and shades the region the fusion rule accepts, so the effect of an
attack is visible as movement relative to a fixed boundary. Feature fusion has
no such plane -- its decision is not a function of the two uni-modal scores --
so its panel uses the joint model's own image-only and text-only logits and
carries no decision region.

Usage:
    python3 scripts/plot/plot_score_space_grid.py [--dataset Recovery] [--out-dir ...]
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

from configuration_files.configuration import THRESHOLD, DATASET
from configuration_files.paths import dataset_result_root, dataset_weights_dir

LATE_RULES = ("min", "mean", "max", "svm-rbf", "linear")
METHODS = list(LATE_RULES) + ["feature-fusion"]

ATTACK_DIRS = {
    "PGD": "PGD",
    "TREPAT": "TREPAT",
    "PGD+TREPAT": "sum",
    "Interleaved": "interleaved",
}

TITLES = {
    "clean": "No attack",
    "PGD": "PGD (image)",
    "TREPAT": "TREPAT (text)",
    "PGD+TREPAT": "PGD + TREPAT (sum)",
    "Interleaved": "Interleaved",
}

LABELS = {"svm-rbf": "SVM-RBF", "feature-fusion": "feature fusion"}
FIG_DIR = "figures/classification_results/scatter"
GRID = 300

HEAD_FILES = {"svm-rbf": "svm_rbf_head.pkl", "linear": "linear_head.pkl"}


def load_sklearn_heads(dataset=None):
    """Load fitted sklearn heads (SVM-RBF, linear) from checkpoints."""
    import joblib
    heads = {}
    weights_dir = dataset_weights_dir(dataset)
    for rule, fname in HEAD_FILES.items():
        path = os.path.join(weights_dir, fname)
        if os.path.exists(path):
            try:
                heads[rule] = joblib.load(path)
            except Exception as e:
                print(f"Warning: could not load {rule} head from {path}: {e}")
    return heads if heads else None


def rule_scores(name, img_logit, txt_logit, heads=None):
    pi = 1 / (1 + np.exp(-img_logit))
    pt = 1 / (1 + np.exp(-txt_logit))
    if name == "min":
        return np.minimum(pi, pt)
    if name == "mean":
        return (pi + pt) / 2
    if name == "max":
        return np.maximum(pi, pt)
    if name in ("svm-rbf", "linear"):
        if heads is None or name not in heads:
            return None
        head = getattr(heads[name], "best_estimator_", heads[name])
        return head.predict_proba(np.column_stack([img_logit, txt_logit]))[:, 1]
    raise ValueError(f"Unknown rule: {name}")


def load_clean(result_root, method):
    path = os.path.join(result_root, "clean", method, "results.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_perturbed(result_root, attack_dir, method):
    path = os.path.join(result_root, "perturbed", attack_dir, method, "perturbed_results.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_ff_ablation(result_root):
    path = os.path.join(result_root, "ablation", "feature-fusion", "modality_ablation.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def get_axes(df, method, ff_ablation=None):
    """Extract (label, image_logit, text_logit) arrays from a results CSV."""
    if method == "feature-fusion":
        if ff_ablation is not None:
            return (
                ff_ablation["label"].values,
                ff_ablation["drop_logit_img_only"].values,
                ff_ablation["drop_logit_txt_only"].values,
            )
        return None
    if "logit_image" in df.columns and "logit_text" in df.columns:
        return df["label"].values, df["logit_image"].values, df["logit_text"].values
    return None


def decision_region(ax, rule, heads, xlim, ylim):
    gx, gy = np.meshgrid(np.linspace(*xlim, GRID), np.linspace(*ylim, GRID))
    grid = rule_scores(rule, gx.ravel(), gy.ravel(), heads)
    if grid is None:
        return
    grid = grid.reshape(gx.shape)
    ax.contourf(gx, gy, grid, levels=[-1, THRESHOLD, 2],
                colors=["#f6d5d5", "#d9e8f5"], alpha=0.7)
    ax.contour(gx, gy, grid, levels=[THRESHOLD], colors="k", linewidths=1.3)


def panel(ax, label, img_logit, txt_logit, method, heads, limits):
    name = LABELS.get(method, method)

    if method in LATE_RULES:
        xlim, ylim = limits
        decision_region(ax, method, heads, xlim, ylim)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    else:
        ax.set_xlim(img_logit.min() - 0.5, img_logit.max() + 0.5)
        ax.set_ylim(txt_logit.min() - 0.5, txt_logit.max() + 0.5)

    real = label == 1
    fake = label == 0
    ax.scatter(img_logit[real], txt_logit[real], s=15, c="#1f77b4",
               label="Real", alpha=0.75, edgecolors="none")
    ax.scatter(img_logit[fake], txt_logit[fake], s=20, c="#d62728",
               label="Fake", alpha=0.9, marker="^", edgecolors="none")

    scores = rule_scores(method, img_logit, txt_logit, heads) if method in LATE_RULES else None
    if scores is None:
        scores = 1 / (1 + np.exp(-img_logit))
    pred = (scores > THRESHOLD).astype(int)
    wrong = int((pred != label).sum())
    ax.set_title(f"{name}\n{wrong} of {len(label)} misclassified", fontsize=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--result-root", default=None,
                        help="Override result root (default: derived from config)")
    parser.add_argument("--out-dir", default=FIG_DIR)
    parser.add_argument("--paper-dir", default=None)
    args = parser.parse_args()

    result_root = args.result_root or dataset_result_root(args.dataset)
    if not os.path.isdir(result_root):
        print(f"No results at {result_root}")
        return

    heads = load_sklearn_heads(args.dataset)
    ff_ablation = load_ff_ablation(result_root)

    # Discover which scenarios have results
    scenarios = ["clean"]
    perturbed_root = os.path.join(result_root, "perturbed")
    for scenario_name, attack_dir in ATTACK_DIRS.items():
        attack_path = os.path.join(perturbed_root, attack_dir)
        if os.path.isdir(attack_path):
            scenarios.append(scenario_name)

    if not scenarios:
        print("No scenarios found")
        return

    # Collect all logit values for shared axis limits across late-fusion panels
    all_img, all_txt = [], []
    for scenario in scenarios:
        for method in LATE_RULES:
            if scenario == "clean":
                df = load_clean(result_root, method)
            else:
                df = load_perturbed(result_root, ATTACK_DIRS[scenario], method)
            if df is not None:
                axes = get_axes(df, method)
                if axes is not None:
                    all_img.append(axes[1])
                    all_txt.append(axes[2])

    if not all_img:
        # Fall back to clean unimodal logits
        for m in ("text", "image"):
            df = load_clean(result_root, m)
            if df is not None and "logit" in df.columns:
                vals = df["logit"].values
                all_img.append(vals)
                all_txt.append(vals)

    if all_img:
        cat_img = np.concatenate(all_img)
        cat_txt = np.concatenate(all_txt)
        limits = ((cat_img.min() - 1, cat_img.max() + 1),
                  (cat_txt.min() - 1, cat_txt.max() + 1))
    else:
        limits = ((-5, 10), (-5, 10))

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
    os.makedirs(args.out_dir, exist_ok=True)

    for scenario in scenarios:
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))

        for ax, method in zip(axes.ravel(), METHODS):
            if scenario == "clean":
                df = load_clean(result_root, method)
            else:
                df = load_perturbed(result_root, ATTACK_DIRS[scenario], method)

            data = get_axes(df, method, ff_ablation) if df is not None else None

            if data is None:
                ax.text(0.5, 0.5, "not available", ha="center", va="center",
                        transform=ax.transAxes, color="0.5")
                ax.set_title(LABELS.get(method, method))
                continue

            label, img_logit, txt_logit = data
            panel(ax, label, img_logit, txt_logit, method, heads, limits)

        for ax in axes.ravel():
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="lower left", frameon=True, fontsize=10)
                break

        fig.supxlabel("image logit")
        fig.supylabel("text logit")
        title = TITLES.get(scenario, scenario)
        fig.suptitle(f"Score space by fusion method -- {title}", fontsize=15)
        fig.tight_layout()

        stem = f"score_space_{scenario.replace('+', '_').lower()}"
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(args.out_dir, f"{stem}.{ext}"),
                        dpi=200, bbox_inches="tight")
        if args.paper_dir:
            os.makedirs(args.paper_dir, exist_ok=True)
            fig.savefig(os.path.join(args.paper_dir, f"{stem}.pdf"),
                        bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {os.path.join(args.out_dir, stem)}.{{png,pdf}}")


if __name__ == "__main__":
    main()
