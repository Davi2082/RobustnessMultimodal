"""Score-space scatters: one figure per scenario, one panel per fusion method.

Each panel places every test sample in the plane spanned by the two uni-modal
scores (sigmoid outputs) and shades the region the fusion rule accepts, so the
effect of an attack is visible as movement relative to a fixed boundary.
Feature fusion has no such plane -- its decision is not a function of the two
uni-modal scores -- so its panel plots the joint model's own scores when fed
the image only and the text only (drop ablation), with no decision region.

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
import matplotlib.ticker
import numpy as np
import pandas as pd

from configuration_files.configuration import THRESHOLD, DATASET
from configuration_files.paths import dataset_result_root, dataset_weights_dir

LATE_RULES = ("min", "mean", "max", "svm-rbf", "linear")
METHODS = list(LATE_RULES) + ["feature-fusion"]

ATTACK_DIRS = {
    "PGD": "pgd",
    "TREPAT": "trepat",
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
# Scores saturate near 0 and 1, so axes use a logit scale (ticks stay in score
# units). EPS clips exact 0/1 so every point lands on the axes.
EPS = 1e-6
LIMITS = (EPS / 2, 1 - EPS / 2)
TICKS = [1e-4, 1e-2, 0.5, 1 - 1e-2, 1 - 1e-4]
TICK_LABELS = ["1e-4", "0.01", "0.5", "0.99", "0.9999"]


# (image column, text column) per source, for each plotting space.
COLUMNS = {
    "scores": {"clean": ("score_image", "score_text"),
               "perturbed": ("image_score", "text_score"),
               "ablation": ("drop_score_img_only", "drop_score_txt_only")},
    "logits": {"clean": ("logit_image", "logit_text"),
               "perturbed": ("image_logit", "text_logit"),
               "ablation": ("drop_logit_img_only", "drop_logit_txt_only")},
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def data_limits(values, pad=0.5):
    """Limits hugging the data, padded by `pad` in logit units."""
    lo, hi = (np.log(v / (1 - v)) for v in (values.min(), values.max()))
    return tuple(1 / (1 + np.exp(-x)) for x in (lo - pad, hi + pad))


def score_grid(lim):
    """GRID points evenly spaced on the logit scale between the limits."""
    lo, hi = (np.log(v / (1 - v)) for v in lim)
    return 1 / (1 + np.exp(-np.linspace(lo, hi, GRID)))

# The linear head is a torch state_dict, not a joblib pickle; it is recovered
# from its results CSV instead (see recover_linear_head).
HEAD_FILES = {"svm-rbf": "svm_rbf_head.pkl"}


def head_input_space(dataset=None):
    """Feature space the learned heads were fitted on ("scores" or "logits")."""
    from models.fusion import head_input_space as _space
    return _space(dataset)


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


class LinearHead:
    """sigmoid(w_text * s_text + w_image * s_image + b), sklearn-style."""

    def __init__(self, coef, bias):
        self.coef, self.bias = np.asarray(coef, dtype=float), float(bias)

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(X @ self.coef + self.bias)))
        return np.column_stack([1 - p, p])


def recover_linear_head(result_root, tol=1e-4):
    """Back the linear head out of its clean results CSV.

    The head is sigmoid(affine(text score, image score)), so logit(score) is
    exactly affine in the two unimodal scores and least squares recovers it.
    This is the head that produced the results, whatever is (or is not) on
    disk. Returns None if the fit is not exact.
    """
    df = load_clean(result_root, "linear")
    if df is None:
        return None
    X = np.column_stack([df["score_text"], df["score_image"], np.ones(len(df))])
    s = np.clip(df["score"].values, 1e-12, 1 - 1e-12)
    y = np.log(s / (1 - s))
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = np.abs(X @ w - y).max()
    if resid > tol:
        print(f"Warning: linear head not recoverable from CSV (max residual {resid:.2e})")
        return None
    print(f"linear head recovered from CSV: text={w[0]:.4f} image={w[1]:.4f} "
          f"bias={w[2]:.4f} (max residual {resid:.1e})")
    return LinearHead(w[:2], w[2])


def rule_scores(name, pi, pt, heads=None, input_space="scores"):
    """Fused score for image score pi and text score pt."""
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
        # Heads are fitted on (text, image) columns -- see run_clean.py.
        feats = np.column_stack([pt, pi])
        if input_space == "logits":
            feats = np.clip(feats, 1e-7, 1 - 1e-7)
            feats = np.log(feats / (1 - feats))
        return head.predict_proba(feats)[:, 1]
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


def get_axes(df, method, ff_ablation=None, space="scores"):
    """Extract (label, image value, text value) arrays in `space`.

    Clean CSVs name the columns score_image/logit_image, perturbed CSVs
    image_score/image_logit. Feature fusion uses the clean modality ablation
    when given, otherwise the per-modality components of an attacked run.
    """
    cols = COLUMNS[space]
    if method == "feature-fusion":
        if ff_ablation is not None:
            img_col, txt_col = cols["ablation"]
            return ff_ablation["label"].values, ff_ablation[img_col].values, ff_ablation[txt_col].values
        # Runs before the FeatureFusionClassifier fix copied the joint score
        # into the component columns; treat those as missing.
        if df is None or "image_score" not in df.columns or (
            np.allclose(df["image_score"], df["score"])
            and np.allclose(df["text_score"], df["score"])
        ):
            return None
    for img_col, txt_col in (cols["clean"], cols["perturbed"]):
        if img_col in df.columns and txt_col in df.columns:
            return df["label"].values, df[img_col].values, df[txt_col].values
    return None


def decision_region(ax, rule, heads, xlim, ylim, input_space, space):
    if space == "scores":
        gx, gy = np.meshgrid(score_grid(xlim), score_grid(ylim))
        pi, pt = gx.ravel(), gy.ravel()
    else:
        gx, gy = np.meshgrid(np.linspace(*xlim, GRID), np.linspace(*ylim, GRID))
        pi, pt = sigmoid(gx.ravel()), sigmoid(gy.ravel())
    grid = rule_scores(rule, pi, pt, heads, input_space)
    if grid is None:
        return
    grid = grid.reshape(gx.shape)
    ax.contourf(gx, gy, grid, levels=[-1, THRESHOLD, 2],
                colors=["#f6d5d5", "#d9e8f5"], alpha=0.7)
    ax.contour(gx, gy, grid, levels=[THRESHOLD], colors="k", linewidths=1.3)


def panel(ax, label, img_val, txt_val, method, heads, pred, input_space,
          space, logit_limits):
    name = LABELS.get(method, method)

    if space == "scores":
        img_val = np.clip(img_val, EPS, 1 - EPS)
        txt_val = np.clip(txt_val, EPS, 1 - EPS)
        ax.set_xscale("logit")
        ax.set_yscale("logit")
        ax.minorticks_off()
        if method in LATE_RULES:
            xlim = ylim = LIMITS
            ax.set_xticks(TICKS, TICK_LABELS)
            ax.set_yticks(TICKS, TICK_LABELS)
        else:
            # Feature fusion's per-modality scores occupy a narrow band.
            xlim, ylim = data_limits(img_val), data_limits(txt_val)
            plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.3g}")
            ax.xaxis.set_major_formatter(plain)
            ax.yaxis.set_major_formatter(plain)
    elif method in LATE_RULES:
        xlim, ylim = logit_limits
    else:
        xlim = (img_val.min() - 0.5, img_val.max() + 0.5)
        ylim = (txt_val.min() - 0.5, txt_val.max() + 0.5)

    if method in LATE_RULES:
        decision_region(ax, method, heads, xlim, ylim, input_space, space)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    real = label == 1
    fake = label == 0
    ax.scatter(img_val[real], txt_val[real], s=15, c="#1f77b4",
               label="Real", alpha=0.75, edgecolors="none")
    ax.scatter(img_val[fake], txt_val[fake], s=20, c="#d62728",
               label="Fake", alpha=0.9, marker="^", edgecolors="none")

    # Count errors from the pipeline's own predictions, not a re-derivation.
    wrong = int((pred != label).sum())
    ax.set_title(f"{name}\n{wrong} of {len(label)} misclassified", fontsize=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--result-root", default=None,
                        help="Override result root (default: derived from config)")
    parser.add_argument("--out-dir", default=FIG_DIR)
    parser.add_argument("--paper-dir", default=None)
    parser.add_argument("--space", choices=("scores", "logits"), default="scores",
                        help="Plot uni-modal scores (logit-scaled axes) or raw logits")
    args = parser.parse_args()

    result_root = args.result_root or dataset_result_root(args.dataset)
    if not os.path.isdir(result_root):
        print(f"No results at {result_root}")
        return

    heads = load_sklearn_heads(args.dataset) or {}
    input_space = head_input_space(args.dataset)
    ff_ablation = load_ff_ablation(result_root)

    linear = recover_linear_head(result_root)
    if linear is not None:
        heads["linear"] = linear

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

    # Shared axis limits for the late-fusion panels in logit space.
    all_img, all_txt = [], []
    for scenario in scenarios:
        for method in LATE_RULES:
            df = (load_clean(result_root, method) if scenario == "clean"
                  else load_perturbed(result_root, ATTACK_DIRS[scenario], method))
            data = get_axes(df, method, space="logits") if df is not None else None
            if data is not None:
                all_img.append(data[1])
                all_txt.append(data[2])
    if all_img:
        img, txt = np.concatenate(all_img), np.concatenate(all_txt)
        logit_limits = ((img.min() - 1, img.max() + 1), (txt.min() - 1, txt.max() + 1))
    else:
        logit_limits = ((-10, 10), (-10, 10))

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
    os.makedirs(args.out_dir, exist_ok=True)

    for scenario in scenarios:
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))

        for ax, method in zip(axes.ravel(), METHODS):
            if scenario == "clean":
                df = load_clean(result_root, method)
            else:
                df = load_perturbed(result_root, ATTACK_DIRS[scenario], method)

            # The modality ablation was run on clean inputs only.
            ablation = ff_ablation if scenario == "clean" else None
            data = get_axes(df, method, ablation, args.space) if df is not None else None

            if data is None:
                msg = "not available"
                title = LABELS.get(method, method)
                if df is not None and "pred" in df.columns:
                    wrong = int((df["pred"] != df["label"]).sum())
                    title += f"\n{wrong} of {len(df)} misclassified"
                    msg = "no per-modality outputs\n(re-run attack)"
                ax.text(0.5, 0.5, msg, ha="center", va="center",
                        transform=ax.transAxes, color="0.5")
                ax.set_title(title, fontsize=11)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            label, img_val, txt_val = data
            pred = df["pred"].values
            if method == "feature-fusion" and ablation is not None:
                pred = ablation.set_index("index").index.map(
                    df.set_index("index")["pred"]).values
            panel(ax, label, img_val, txt_val, method, heads, pred, input_space,
                  args.space, logit_limits)

        for ax in axes.ravel():
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="lower left", frameon=True, fontsize=10)
                break

        unit = args.space[:-1]
        fig.supxlabel(f"image {unit}")
        fig.supylabel(f"text {unit}")
        title = TITLES.get(scenario, scenario)
        fig.suptitle(f"Score space by fusion method -- {title}", fontsize=15)
        fig.tight_layout()

        stem = f"score_space_{scenario.replace('+', '_').lower()}"
        if args.space == "logits":
            stem += "_logits"
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
