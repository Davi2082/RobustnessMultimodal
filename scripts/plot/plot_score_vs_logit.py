"""Unimodal outputs in score space vs logit space, no decision boundaries.

Motivates fitting the learned late-fusion heads on logits: the unimodal
scores (sigmoid outputs) pile up at 0 and 1, so in score space most samples
collapse onto the edges and corners of the unit square, while in logit space
the same samples keep their spread. Held-out splits only (validation, test);
train-split outputs are saturated by fitting.

Left: scores on log axes. Right: logits on linear axes. One split.

Usage:
    python3 scripts/plot/plot_score_vs_logit.py --dataset Fakeddit [--split validation]
Writes figures/classification_results/scatter/score_vs_logit/<DS>_score_vs_logit_<split>.{png,pdf}
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from configuration_files.paths import dataset_result_root

OUT_DIR = "figures/classification_results/scatter/score_vs_logit"
SPLITS = (("val", "validation"), ("clean", "test"))


def load_split(root, split):
    """Text and image results of one split merged by sample (score_/logit_ + _text/_image)."""
    paths = {m: os.path.join(root, split, m, "results.csv") for m in ("text", "image")}
    if not all(os.path.isfile(p) for p in paths.values()):
        return None
    t, i = (pd.read_csv(paths[m]) for m in ("text", "image"))
    df = t.merge(i, on=["index", "label"], suffixes=("_text", "_image"))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", choices=[n for _, n in SPLITS], default="test")
    args = ap.parse_args()
    split = dict((n, d) for d, n in SPLITS)[args.split]
    df = load_split(dataset_result_root(args.dataset), split)
    if df is None:
        sys.exit(f"no {args.split} unimodal results for {args.dataset}")

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(10, 4.6))
    real, fake = df["label"] == 1, df["label"] == 0
    for ax, col, title in ((ax_s, "score", "Scores"), (ax_l, "logit", "Logits")):
        for mask, color, marker, lab in ((real, "#1f77b4", "o", "Real"),
                                         (fake, "#d62728", "^", "Fake")):
            ax.scatter(df.loc[mask, f"{col}_image"], df.loc[mask, f"{col}_text"],
                       s=14, c=color, marker=marker, alpha=0.7, linewidths=0, label=lab)
        ax.set_xlabel(f"image {col}")
        ax.set_ylabel(f"text {col}")
        ax.set_title(title)
    # Log axes: the scores span several decades towards 0.
    ax_s.set_xscale("log")
    ax_s.set_yscale("log")
    for axis, col in (("x", "score_image"), ("y", "score_text")):
        getattr(ax_s, f"set_{axis}lim")(df[col].min() / 2, 2)
    fig.tight_layout()
    fig.legend(*ax_s.get_legend_handles_labels(), loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=11)

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, f"{args.dataset}_score_vs_logit_{args.split}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
