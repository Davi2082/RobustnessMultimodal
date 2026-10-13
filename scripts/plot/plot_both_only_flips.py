"""Samples flipped only by attacking both modalities, in logit space.

Same panels, axes and decision regions as the clean single-column figure
(plot_score_space_grid.py --column), restricted to the clean-correct samples
that the union or the joint attack flips but neither PGD nor TREPAT does.
Each such sample is drawn at its clean position, with an arrow to its
position after the union attack (dashed) and after the joint attack (solid);
an arrow is drawn only for the attack(s) that flip it. All other test samples
are drawn faintly at their position after the joint attack, without arrows.

Usage:
    python3 scripts/plot/plot_both_only_flips.py --dataset Fakeddit
Writes figures/classification_results/scatter/unimodal_scores/<DS>/<DS>_both_only_flips.{png,pdf}
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import plot_score_space_grid as G
from configuration_files.configuration import RAND_SEED, method_threshold
from configuration_files.paths import dataset_result_root

UNIMODAL = ("pgd", "trepat")
BOTH = (("union", "sum", "--"), ("joint", "interleaved", "-"))
COLORS = {1: "#1f77b4", 0: "#d62728"}
MARKERS = {1: "o", 0: "^"}


def flips(root, attack, method, clean):
    """Clean-correct samples the attack misclassifies, and its per-modality scores."""
    p = pd.read_csv(os.path.join(root, "perturbed", attack, method, "perturbed_results.csv"))
    p = p.set_index("index").reindex(clean.index)
    flipped = (clean["pred"] == clean["label"]) & (p["pred"] != clean["label"])
    return flipped, p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="Fakeddit")
    ap.add_argument("--out-dir", default=G.FIG_DIR)
    args = ap.parse_args()
    root = dataset_result_root(args.dataset)

    # Same look as the clean --column figure.
    G.LOGIT_TICKS[:] = [-10, 0, 10]
    ticks = list(G.sigmoid(np.array(G.LOGIT_TICKS, dtype=float)))
    plt.rcParams.update({"font.size": 13})
    from models.fusion import fusion_head_path
    heads = {r: G.TorchHead(r, args.dataset) for r in ("svm-rbf", "linear")
             if os.path.isfile(fusion_head_path(r, args.dataset))}
    space = G.head_input_space(args.dataset)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), sharex=True, sharey=True)
    for k, (ax, method) in enumerate(zip(axes.ravel(), G.METHODS)):
        clean = pd.read_csv(os.path.join(root, "clean", method, "results.csv")).set_index("index")
        uni = np.zeros(len(clean), dtype=bool)
        for a in UNIMODAL:
            uni |= flips(root, a, method, clean)[0].values
        both = {name: flips(root, d, method, clean) for name, d, _ in BOTH}
        keep = (both["union"][0].values | both["joint"][0].values) & ~uni

        ax.set_xscale("logit")
        ax.set_yscale("logit")
        ax.minorticks_off()
        ax.set_xlim(*G.LIMITS)
        ax.set_ylim(*G.LIMITS)
        ax.set_xticks(ticks, [f"{t:g}" for t in G.LOGIT_TICKS])
        ax.set_yticks(ticks, [f"{t:g}" for t in G.LOGIT_TICKS])
        if method in G.LATE_RULES:
            G.decision_region(ax, method, heads, G.LIMITS, G.LIMITS, space,
                              threshold=method_threshold(method, args.dataset))
            x0, y0 = clean["score_image"].values, clean["score_text"].values
            xcol, ycol = "image_score", "text_score"
        else:
            # Feature fusion: its own drop-ablation axes (no boundary).
            ab = G.load_ff_ablation(root).set_index("index").reindex(clean.index)
            x0, y0 = ab["drop_score_img_only"].values, ab["drop_score_txt_only"].values
            xcol, ycol = "image_score", "text_score"
        clip = lambda v: np.clip(np.asarray(v, dtype=float), G.EPS, 1 - G.EPS)
        x0, y0 = clip(x0), clip(y0)
        labels = clean["label"].values

        # Every other sample, faint, where the joint attack left it.
        _, pj = both["joint"]
        xo, yo = clip(pj[xcol].values), clip(pj[ycol].values)
        for lab in (1, 0):
            sel = ~keep & (labels == lab)
            ax.scatter(xo[sel], yo[sel], s=9, c=COLORS[lab], marker=MARKERS[lab],
                       alpha=0.25, linewidths=0, zorder=2)

        for name, _, style in BOTH:
            flipped, p = both[name]
            sel = keep & flipped.values
            x1, y1 = clip(p[xcol].values), clip(p[ycol].values)
            for i in np.where(sel)[0]:
                ax.annotate("", xy=(x1[i], y1[i]), xytext=(x0[i], y0[i]),
                            arrowprops=dict(arrowstyle="-|>", linestyle=style, lw=1.0,
                                            color=COLORS[labels[i]], alpha=0.85,
                                            shrinkA=0, shrinkB=0, mutation_scale=8))
        for lab in (1, 0):
            sel = keep & (labels == lab)
            ax.scatter(x0[sel], y0[sel], s=22, c=COLORS[lab], marker=MARKERS[lab],
                       edgecolors="k", linewidths=0.4, zorder=3)
        ax.tick_params(bottom=ax.get_subplotspec().is_last_row(),
                       left=ax.get_subplotspec().is_first_col())
        # Counts per method are in the both-only table (scripts/results_table.py).
        ax.text(0.04, 0.04, f"({'abcdef'[k]})", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=12, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))

    handles = [Line2D([], [], marker="o", color=COLORS[1], ls="", label="Real"),
               Line2D([], [], marker="^", color=COLORS[0], ls="", label="Fake"),
               Line2D([], [], color="0.3", ls="--", label="union"),
               Line2D([], [], color="0.3", ls="-", label="joint")]
    fig.supxlabel("image logit", fontsize=15)
    fig.supylabel("text logit", fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.97),
               frameon=False, fontsize=11, handletextpad=0.3, columnspacing=1.0)
    out = os.path.join(args.out_dir, "unimodal_scores", args.dataset)
    os.makedirs(out, exist_ok=True)
    fig.suptitle(f"{args.dataset} (seed {RAND_SEED})", fontsize=13, y=1.06)
    stem = os.path.join(out, f"{args.dataset}_seed{RAND_SEED}_both_only_flips")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
