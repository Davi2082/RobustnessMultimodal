"""ASR against the image budget epsilon, one line per fusion method.

Reads every run of one attack: the main-config run in perturbed/<attack>/ and
the sweep runs in perturbed/<attack>_eps<k>/ (see run_multimodal_attacks
--epsilons). Epsilon is taken from each run's parameters.json, so the x values
are what was actually used. ASR = samples flipped / clean-correct samples, the
untargeted definition used everywhere else.

Usage:
    python3 scripts/plot/plot_asr_vs_epsilon.py --datasets Fakeddit Recovery --eps 1 2 3 [--attack pgd]
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from configuration_files.configuration import DATASET
from configuration_files.paths import dataset_result_root
from scripts.plot.plot_score_space_grid import LABELS, METHODS, attack_asr

FIG_DIR = "figures/classification_results/asr_vs_epsilon"

# Reference categorical palette, fixed order (slot 1..6), plus one marker shape
# per method so identity never rests on colour alone.
STYLE = {
    "min": ("#2a78d6", "o"),
    "mean": ("#eb6834", "s"),
    "max": ("#1baf7a", "^"),
    "svm-rbf": ("#eda100", "D"),
    "linear": ("#e87ba4", "v"),
    "feature-fusion": ("#4a3aa7", "P"),
}


def run_epsilon(run_dir):
    """Epsilon recorded by any fusion's parameters.json in this attack dir."""
    for params in glob.glob(os.path.join(run_dir, "*", "parameters.json")):
        with open(params) as f:
            eps = (json.load(f).get("Attack") or {}).get("Epsilon")
        if eps is not None:
            return float(eps)
    return None


def collect(result_root, attack):
    """One row per (run, method): epsilon, flipped, clean-correct, ASR."""
    perturbed = os.path.join(result_root, "perturbed")
    runs = [attack] + sorted(
        os.path.basename(d) for d in glob.glob(os.path.join(perturbed, f"{attack}_eps*")))
    rows = []
    for run in runs:
        run_dir = os.path.join(perturbed, run)
        eps = run_epsilon(run_dir) if os.path.isdir(run_dir) else None
        if eps is None:
            continue
        for method in METHODS:
            asr = attack_asr(result_root, method, run)
            if asr is None:
                continue
            flipped, correct = asr
            rows.append({"run": run, "epsilon": eps, "eps_x255": round(eps * 255, 4),
                         "method": method, "flipped": flipped, "clean_correct": correct,
                         "asr": flipped / correct if correct else float("nan")})
    # The main run may share an epsilon with a sweep run; keep one per point.
    return (pd.DataFrame(rows).drop_duplicates(["eps_x255", "method"])
            .sort_values(["method", "epsilon"]) if rows else pd.DataFrame(rows))


def draw(ax, df, title):
    """One panel: an ASR line per fusion method across epsilon."""
    for method in METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        color, marker = STYLE[method]
        ax.plot(sub["eps_x255"], sub["asr"] * 100, color=color, marker=marker,
                linewidth=2, markersize=7, markeredgecolor="white", markeredgewidth=1.5,
                label=LABELS.get(method, method))
    eps_values = sorted(df["eps_x255"].unique())
    ax.set_xticks(eps_values, [f"{e:g}/255" for e in eps_values])
    ax.minorticks_off()
    ax.set_ylim(-3, 103)
    ax.set_xlabel("image budget ε")
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=13)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=[DATASET],
                        help="One panel per dataset, side by side with a shared y-axis")
    parser.add_argument("--attack", default="pgd",
                        help="Attack whose epsilon runs to plot (pgd, sum, interleaved)")
    parser.add_argument("--eps", nargs="+", type=float, default=None, metavar="K",
                        help="Keep only these budgets, in units of 1/255 (e.g. 1 2 3)")
    parser.add_argument("--out-dir", default=FIG_DIR)
    args = parser.parse_args()

    frames = {}
    for ds in args.datasets:
        df = collect(dataset_result_root(ds), args.attack)
        if not df.empty and args.eps is not None:
            df = df[df["eps_x255"].round(4).isin([round(e, 4) for e in args.eps])]
        if df.empty:
            print(f"No {args.attack} runs with a recorded epsilon for {ds}")
            continue
        frames[ds] = df
    if not frames:
        return

    name = "_".join(frames) if len(frames) > 1 else next(iter(frames))
    out_dir = os.path.join(args.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"asr_vs_epsilon_{args.attack}")
    pd.concat([df.assign(dataset=ds) for ds, df in frames.items()]).to_csv(f"{stem}.csv", index=False)

    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, len(frames), figsize=(4.6 * len(frames) + 1.6, 4.2),
                             sharey=True, squeeze=False)
    for ax, (ds, df) in zip(axes[0], frames.items()):
        draw(ax, df, ds)
    axes[0][0].set_ylabel("attack success rate (%)")
    axes[0][-1].legend(frameon=False, fontsize=10, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"{args.attack.upper()} attack success rate vs ε", fontsize=14)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.{{png,pdf,csv}}")
    for ds, df in frames.items():
        print(f"\n{ds}")
        print(df.pivot(index="method", columns="eps_x255", values="asr").round(3).to_string())


if __name__ == "__main__":
    main()
