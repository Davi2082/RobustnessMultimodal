"""Two-column figure: clean score space (left) next to union/joint-only flips (right).

    RM_RAND_SEED=42 python3 scripts/plot/plot_score_space_merged.py --dataset Fakeddit
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

import plot_both_only_flips as B
import plot_score_space_grid as G
from configuration_files.configuration import ATTACK_CLASSES, RAND_SEED
from configuration_files.paths import dataset_result_root


def draw_clean(fig, dataset):
    """Clean panels as in ``plot_score_space_grid.py --column``, without labels or legend."""
    root = dataset_result_root(dataset)
    G.LOGIT_TICKS[:] = [-10, 0, 10]
    G.TICKS = list(G.sigmoid(np.array(G.LOGIT_TICKS, dtype=float)))
    G.TICK_LABELS = [f"{k:g}" for k in G.LOGIT_TICKS]
    G.MARKER_SCALE = 0.45
    from models.fusion import fusion_head_path
    heads = {r: G.TorchHead(r, dataset) for r in ("svm-rbf", "linear")
             if os.path.isfile(fusion_head_path(r, dataset))}
    space = G.head_input_space(dataset)
    ablation = G.load_ff_ablation(root)

    axes = fig.subplots(2, 3, sharex=True, sharey=True)
    for k, (ax, method) in enumerate(zip(axes.ravel(), G.METHODS)):
        df = G.load_clean(root, method)
        label, img_val, txt_val = G.get_axes(df, method, ablation)
        pred = df["pred"].values
        if method == "feature-fusion":
            pred = ablation.set_index("index").index.map(df.set_index("index")["pred"]).values
        G.panel(ax, label, img_val, txt_val, method, heads, pred, space, dataset=dataset)
        ax.set_title("")
        ax.text(0.04, 0.96, f"({'abcdef'[k]})", transform=ax.transAxes, ha="left", va="top",
                fontsize=13, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
        ax.tick_params(bottom=ax.get_subplotspec().is_last_row(),
                       left=ax.get_subplotspec().is_first_col())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="Fakeddit")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 13})

    fig = plt.figure(figsize=(14.4, 4.6))
    left, right = fig.subfigures(1, 2, wspace=0)
    draw_clean(left, args.dataset)
    handles = B.draw(right, args.dataset)
    left.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.99, wspace=0.05, hspace=0.05)
    right.subplots_adjust(left=0.035, right=0.95, bottom=0.08, top=0.99, wspace=0.05, hspace=0.05)
    for ax in right.axes:
        ax.tick_params(labelleft=False)
    fig.supxlabel("image logit", fontsize=15, y=-0.01, va="top")
    fig.supylabel("text logit", fontsize=15, x=0.0, ha="right")
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, -0.07),
               frameon=False, fontsize=12, handletextpad=0.3, columnspacing=1.2, handlelength=1.5)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"{args.dataset}_seed{RAND_SEED}_{ATTACK_CLASSES}_score_space_merged")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
