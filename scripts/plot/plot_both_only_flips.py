"""Samples flipped by union/joint but by neither PGD nor TREPAT, in logit space.

Arrows: union or joint (farthest past the boundary), PGD, TREPAT. Other
samples: faint, at their largest-shift attacked position.

    python3 scripts/plot/plot_both_only_flips.py --dataset Fakeddit
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
UNI_COLORS = {"pgd": "#2a78d6", "trepat": "#e8730c"}  # Fig. 1 colours
BOTH_COLOR = "#6b3fa0"
MARKERS = {1: "o", 0: "^"}


def flips(root, attack, method, clean):
    """(flipped clean-correct mask, attacked results)."""
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
        uni_runs = {}
        for a in UNIMODAL:
            uni_runs[a] = flips(root, a, method, clean)
            uni |= uni_runs[a][0].values
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
            ab = G.load_ff_ablation(root).set_index("index").reindex(clean.index)
            x0, y0 = ab["drop_score_img_only"].values, ab["drop_score_txt_only"].values
            xcol, ycol = "image_score", "text_score"
        clip = lambda v: np.clip(np.asarray(v, dtype=float), G.EPS, 1 - G.EPS)
        x0, y0 = clip(x0), clip(y0)
        labels = clean["label"].values

        # Background: largest-shift attacked position.
        logit = lambda v: np.log(v / (1 - v))
        ends = [(clip(both[n][1][xcol].values), clip(both[n][1][ycol].values)) for n in ("union", "joint")]
        dist = np.stack([np.hypot(logit(ex) - logit(x0), logit(ey) - logit(y0)) for ex, ey in ends])
        pick = np.nan_to_num(dist, nan=-1).argmax(axis=0)  # 0 = union, 1 = joint
        xo = np.where(pick == 0, ends[0][0], ends[1][0])
        yo = np.where(pick == 0, ends[0][1], ends[1][1])
        for lab in (1, 0):
            sel = ~keep & (labels == lab)
            ax.scatter(xo[sel], yo[sel], s=9, c=COLORS[lab], marker=MARKERS[lab],
                       alpha=0.25, linewidths=0, zorder=2)

        # Highlighted: union/joint landing farthest past the boundary, plus PGD and TREPAT.
        t = method_threshold(method, args.dataset)
        lt = np.log(t / (1 - t))
        margin = np.stack([np.abs(logit(clip(both[n][1]["score"].values)) - lt) for n in ("union", "joint")])
        flip_mask = np.stack([both[n][0].values for n in ("union", "joint")])
        fpick = np.where(flip_mask, np.nan_to_num(margin, nan=-1), -np.inf).argmax(axis=0)
        for i in np.where(keep)[0]:
            for a in UNIMODAL:
                pu = uni_runs[a][1]
                ax.annotate("", xy=(clip(pu[xcol].values)[i], clip(pu[ycol].values)[i]), xytext=(x0[i], y0[i]),
                            arrowprops=dict(arrowstyle="-|>", lw=0.7, color=UNI_COLORS[a], alpha=0.9,
                                            shrinkA=0, shrinkB=0, mutation_scale=6))
            j = fpick[i]
            ax.annotate("", xy=(ends[j][0][i], ends[j][1][i]), xytext=(x0[i], y0[i]),
                        arrowprops=dict(arrowstyle="-|>", linestyle=BOTH[j][2], lw=1.0,
                                        color=BOTH_COLOR, alpha=0.9,
                                        shrinkA=0, shrinkB=0, mutation_scale=8))
        for lab in (1, 0):
            sel = keep & (labels == lab)
            ax.scatter(x0[sel], y0[sel], s=22, c=COLORS[lab], marker=MARKERS[lab],
                       edgecolors="k", linewidths=0.4, zorder=3)
        ax.tick_params(bottom=ax.get_subplotspec().is_last_row(),
                       left=ax.get_subplotspec().is_first_col())
        ax.text(0.04, 0.04, f"({'abcdef'[k]})", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=12, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))

    handles = [Line2D([], [], marker="o", color=COLORS[1], ls="", label="Real"),
               Line2D([], [], marker="^", color=COLORS[0], ls="", label="Fake"),
               Line2D([], [], color=UNI_COLORS["pgd"], lw=0.9, label="PGD"),
               Line2D([], [], color=UNI_COLORS["trepat"], lw=0.9, label="TREPAT"),
               Line2D([], [], color=BOTH_COLOR, ls="--", label="union"),
               Line2D([], [], color=BOTH_COLOR, ls="-", label="joint")]
    fig.supxlabel("image logit", fontsize=15)
    fig.supylabel("text logit", fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.97),
               frameon=False, fontsize=10, handletextpad=0.3, columnspacing=0.8, handlelength=1.5)
    out = os.path.join(args.out_dir, "unimodal_scores", args.dataset)
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f"{args.dataset}_seed{RAND_SEED}_both_only_flips")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
