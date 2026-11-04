"""Mean ± std of the per-seed results over the test-subset seeds.

Reads results/<ds>/<subset>/<seed>/<attack_classes>/metrics_{clean,ablation,adv,attack_split}.csv
(written by scripts/results_table.py) for every seed folder that has them, and writes CSV (mean, std, n_seeds) and Markdown (mean ± std)
into results/<ds>/<subset>/aggregate/<attack_classes>/. Every file and title names the dataset
and the seeds. No LaTeX, nothing outside results/.

Usage:
    python3 scripts/aggregate_seeds.py --dataset Fakeddit
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configuration_files.configuration import ATTACK_CLASSES, dataset_subset_size  # noqa: E402

METHODS = [("Text-only", "text"), ("Image-only", "image"), ("Feature fusion", "feature-fusion"),
           ("min", "min"), ("mean", "mean"), ("max", "max"), ("SVM-RBF", "svm-rbf"), ("linear", "linear")]
FUSIONS = [m for m in METHODS if m[1] not in ("text", "image")]
SCENARIOS = ["PGD", "TREPAT", "Union", "Joint"]
TABLES = {  # name: (grouping keys, reported (header, column) pairs)
    "clean": (["method"], [("AUC", "AUC"), ("Acc", "accuracy"), ("TPR", "TPR"), ("FPR", "FPR")]),
    "ablation": (["method", "condition"], [("AUC", "AUC"), ("Acc", "accuracy")]),
    "adv": (["method", "scenario"], [("TPR", "TPR"), ("FPR", "FPR"), ("ASR", "ASR")]),
    "attack_split": (["method"], [("Total", "total"), ("Unimodal only", "unimodal_only"),
                                  ("Union or joint only", "multimodal_only"),
                                  ("Union only", "union_only"), ("Joint only", "joint_only")]),
}


def pm(mean, std, pct=False):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "—"
    if pct:
        return f"{mean:.1f} ± {std:.1f}"
    return f"{mean:.3f} ± {std:.3f}"


def write_md(path, title, header, rows):
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    with open(path, "w") as f:
        f.write(title + "\n\n" + "\n".join(md) + "\n")
    print(title + "\n" + "\n".join(md) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    base = os.path.join("results", args.dataset, str(dataset_subset_size(args.dataset)))
    out = os.path.join(base, "aggregate", ATTACK_CLASSES)
    os.makedirs(out, exist_ok=True)

    # Every finished seed folder (not only those in config rand_seed), so runs
    # done in several batches are aggregated together.
    seeds = sorted(int(d) for d in os.listdir(base)
                   if d.isdigit() and os.path.isfile(os.path.join(base, d, ATTACK_CLASSES, "metrics_adv.csv")))
    if not seeds:
        sys.exit(f"no finished seeds under {base}")
    tag = f"{args.dataset} (seeds {', '.join(map(str, seeds))}; attacked classes: {ATTACK_CLASSES}; mean ± std)"

    agg = {}
    for name, (keys, cols) in TABLES.items():
        frames = [pd.read_csv(os.path.join(base, str(s), ATTACK_CLASSES, f"metrics_{name}.csv")) for s in seeds
                  if os.path.isfile(os.path.join(base, str(s), ATTACK_CLASSES, f"metrics_{name}.csv"))]
        if not frames:
            continue
        df = pd.concat(frames)
        deltas = {"adv": ("dTPR", "dFPR"), "ablation": ("dAUC", "daccuracy")}.get(name, ())
        values = [c for _, c in cols] + [c for c in deltas if c in df]
        g = df.groupby(keys)[values]
        res = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
        res["n_seeds"] = g.size()
        res = res.reset_index()
        res.insert(0, "seeds", " ".join(map(str, seeds)))
        res.insert(0, "attack_classes", ATTACK_CLASSES)
        res.insert(0, "dataset", args.dataset)
        res.to_csv(os.path.join(out, f"metrics_{name}.csv"), index=False)
        agg[name] = res.set_index(keys)

    def cell(name, key, col, pct=False, delta=False):
        """mean ± std, followed by the mean ± std of its delta column when asked."""
        t = agg.get(name)
        if t is None or key not in t.index:
            return "—"
        r = t.loc[key]
        scale = 100 if (pct and name == "adv") else 1
        out = pm(r[f"{col}_mean"] * scale, r[f"{col}_std"] * scale, pct)
        d = f"d{col}"
        if delta and f"{d}_mean" in r and not np.isnan(r[f"{d}_mean"]):
            sign = "−" if r[f"{d}_mean"] < -5e-4 else "+"
            out += f" ({sign}{abs(r[f'{d}_mean']):.3f} ± {r[f'{d}_std']:.3f})"
        return out

    write_md(os.path.join(out, "results_clean.md"), f"{tag}, clean test set",
             ["Method", "AUC", "Acc", "TPR", "FPR"],
             [[n] + [cell("clean", m, c) for c in ("AUC", "accuracy", "TPR", "FPR")] for n, m in METHODS])
    write_md(os.path.join(out, "results_ablation.md"),
             f"{tag}, missing-modality ablation (in parentheses: change from both modalities)",
             ["Method", "Image only AUC (Δ)", "Image only Acc (Δ)", "Text only AUC (Δ)", "Text only Acc (Δ)"],
             [[n] + [cell("ablation", (m, cond), c, delta=True) for cond in ("image_only", "text_only")
                     for c in ("AUC", "accuracy")] for n, m in FUSIONS])
    write_md(os.path.join(out, "results_adv.md"), f"{tag}, ASR (%) per attack",
             ["Method"] + SCENARIOS,
             [[n] + [cell("adv", (m, sc), "ASR", pct=True) for sc in SCENARIOS] for n, m in METHODS])
    write_md(os.path.join(out, "results_adv_tpr_fpr.md"),
             f"{tag}, TPR / FPR under attack (in parentheses: change from clean)",
             ["Method"] + [f"{sc} {h} (Δ)" for sc in SCENARIOS for h in ("TPR", "FPR")],
             [[n] + [cell("adv", (m, sc), c, delta=True) for sc in SCENARIOS for c in ("TPR", "FPR")]
              for n, m in METHODS])
    write_md(os.path.join(out, "results_attack_split.md"), f"{tag}, ASR (%) split by the attacks that succeed",
             ["Method"] + [h for h, _ in TABLES["attack_split"][1]],
             [[n] + [cell("attack_split", m, c, pct=True) for _, c in TABLES["attack_split"][1]]
              for n, m in FUSIONS])
    print(f"wrote {out}/metrics_*.csv and results_*.md")


if __name__ == "__main__":
    main()
