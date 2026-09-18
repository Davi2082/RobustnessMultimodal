#!/usr/bin/env python3
"""Print result tables to stdout (plain text, formatted for the terminal).

Tables
------
  clean       – unimodal + fusion clean performance
  adversarial – fusion methods under PGD / TREPAT / PGD+TREPAT  (delta + ASR)
  ablation    – missing-modality probing

Usage:
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate multimodal
  python3 -m scripts.make_tables --dataset Recovery
  python3 -m scripts.make_tables --dataset Recovery --table adversarial
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configuration_files.configuration import DATASET, SOURCE_LABEL, TARGET_LABEL
from configuration_files.paths import (
    model_clean_dir,
    model_perturbed_dir,
    model_ablation_dir,
)


UNIMODAL = ["text", "image"]
LATE_FUSION_MODES = ["min", "mean", "max", "linear", "svm-rbf"]
FUSIONS = LATE_FUSION_MODES + ["feature-fusion"]

ATTACK_LABELS = {
    "pgd": "PGD",
    "trepat": "TREPAT",
    "sum": "PGD+TREPAT",
}
ATTACKS = list(ATTACK_LABELS.keys())


DISPLAY_NAMES = {
    "text": "Text-only",
    "image": "Image-only",
    "min": "LF min",
    "mean": "LF mean",
    "max": "LF max",
    "linear": "LF linear",
    "svm-rbf": "LF SVM-RBF",
    "feature-fusion": "Feature fusion",
}


# ─────────────────────────── helpers ─────────────────────────────────────────

def load_metrics(path):
    p = Path(path) / "metrics.json"
    if not p.is_file():
        return None
    with open(p) as f:
        m = json.load(f)
    if all(v == 0.0 for v in m.values()):
        return None
    return m


def load_csv(path, filename="results.csv"):
    p = Path(path) / filename
    if not p.is_file():
        return None
    return pd.read_csv(p)


def compute_asr(clean_df, pert_df):
    if len(clean_df) == len(pert_df):
        y_true = np.asarray(clean_df["label"])
        y_clean = np.asarray(clean_df["pred"])
        y_pert = np.asarray(pert_df["pred"])
    else:
        merged = clean_df.merge(pert_df, on="index", suffixes=("_clean", "_pert"))
        y_true = np.asarray(merged["label_clean"])
        y_clean = np.asarray(merged["pred_clean"])
        y_pert = np.asarray(merged["pred_pert"])
    correct = y_true == y_clean
    total = np.sum(correct)
    if total == 0:
        return None
    flipped = np.sum(correct & (y_pert != y_true))
    return flipped / total


def compute_metrics_from_csv(df):
    from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
    y_true = np.asarray(df["label"])
    y_pred = np.asarray(df["pred"])
    scores = np.asarray(df["score"])
    fpr, tpr, _ = roc_curve(y_true, scores)
    return {"auc": auc(fpr, tpr), "f1": f1_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred)}


def metrics_from_csv(df):
    """Compute AUC, F1 (Real-positive), Acc, TPR, TNR from a CSV.
    Original labels: 0=Fake, 1=Real. Real is the positive class."""
    from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
    label = np.asarray(df["label"])
    pred = np.asarray(df["pred"])
    score = np.asarray(df["score"])
    fpr, tpr_curve, _ = roc_curve(label, score)
    auc_val = auc(fpr, tpr_curve)
    f1_val = f1_score(label, pred)
    acc_val = accuracy_score(label, pred)
    fake_mask = label == 0
    real_mask = label == 1
    tpr = np.mean(pred[real_mask] == 1) if real_mask.any() else None
    tnr = np.mean(pred[fake_mask] == 0) if fake_mask.any() else None
    return {"auc": auc_val, "f1": f1_val, "accuracy": acc_val, "tpr": tpr, "tnr": tnr}


def f(val, w=7):
    if val is None:
        return "---".center(w)
    return f"{val:.3f}".rjust(w)


def separator(width):
    return "-" * width


# ─────────────────────────── Table: clean ────────────────────────────────────

def _fmt_row(name, m, indent=2):
    prefix = " " * indent + f"{name:<{20 - indent + 2}s}"
    if m is None:
        return f"{prefix} {'---':>7s} {'---':>7s} {'---':>7s} {'---':>7s} {'---':>7s}"
    return f"{prefix} {f(m['auc'])} {f(m['f1'])} {f(m['accuracy'])} {f(m.get('tpr'))} {f(m.get('tnr'))}"


def print_clean_table(dataset):
    all_m = {}
    for model in UNIMODAL + FUSIONS:
        csv_df = load_csv(model_clean_dir(model, dataset), "results.csv")
        all_m[model] = metrics_from_csv(csv_df) if csv_df is not None else None

    hdr = f"{'Method':<20s} {'AUC':>7s} {'F1':>7s} {'Acc':>7s} {'TPR':>7s} {'TNR':>7s}"
    w = len(hdr)
    print(f"\n  Clean performance — {dataset}  (F1: Real-positive)")
    print(f"  {separator(w)}")
    print(f"  {hdr}")
    print(f"  {separator(w)}")

    for model in UNIMODAL:
        print(f"  {_fmt_row(DISPLAY_NAMES[model], all_m[model])}")

    print(f"  {separator(w)}")
    print(f"  {'Late fusion':<20s}")

    for mode in LATE_FUSION_MODES:
        print(f"  {_fmt_row(DISPLAY_NAMES[mode], all_m[mode], indent=4)}")

    ff = "feature-fusion"
    print(f"  {_fmt_row(DISPLAY_NAMES[ff], all_m[ff])}")
    print(f"  {separator(w)}")
    print()


# ─────────────────────── Table: adversarial robustness ───────────────────────

def print_adversarial_table(dataset):
    clean_csvs = {}
    for model in FUSIONS:
        clean_csvs[model] = load_csv(model_clean_dir(model, dataset), "results.csv")

    hdr = (f"{'Attack':<12s} {'Method':<18s} {'AUC':>7s} {'F1':>7s} "
           f"{'Acc':>7s} {'TPR':>7s} {'TNR':>7s} {'ASR':>7s}")
    w = len(hdr)
    print(f"\n  Adversarial robustness — {dataset}  (F1: Real-positive)")
    print(f"  {separator(w)}")
    print(f"  {hdr}")
    print(f"  {separator(w)}")

    for attack in ATTACKS:
        label = ATTACK_LABELS[attack]
        first = True

        for model in FUSIONS:
            pert_dir = model_perturbed_dir(model, attack, dataset)
            pert_csv = load_csv(pert_dir, "perturbed_results.csv")
            clean_csv = clean_csvs[model]

            col = label if first else ""
            first = False
            name = DISPLAY_NAMES[model]

            if pert_csv is None or len(pert_csv) <= 2:
                print(f"  {col:<12s} {name:<18s} {'---':>7s} {'---':>7s} {'---':>7s} {'---':>7s} {'---':>7s} {'---':>7s}")
                continue

            m = metrics_from_csv(pert_csv)

            asr = None
            if clean_csv is not None:
                asr = compute_asr(clean_csv, pert_csv)

            print(f"  {col:<12s} {name:<18s} {f(m['auc'])} {f(m['f1'])} "
                  f"{f(m['accuracy'])} {f(m['tpr'])} {f(m['tnr'])} {f(asr)}")

        print(f"  {separator(w)}")
    print()


# ──────────────────── Table: missing-modality ablation ───────────────────────

def print_ablation_table(dataset):
    ff_csv = Path(model_ablation_dir("feature-fusion", dataset)) / "modality_ablation_metrics.csv"
    lf_csv = Path(model_ablation_dir("late-fusion", dataset)) / "modality_ablation.csv"

    ff_ablation = pd.read_csv(ff_csv) if ff_csv.is_file() else None
    lf_ablation = pd.read_csv(lf_csv) if lf_csv.is_file() else None

    hdr = f"{'Modalities':<14s} {'Method':<18s} {'AUC':>7s} {'F1':>7s} {'Acc':>7s} {'TPR':>7s} {'TNR':>7s}"
    w = len(hdr)
    print(f"\n  Missing-modality ablation — {dataset}  (F1: Real-positive)")
    print(f"  {separator(w)}")
    print(f"  {hdr}")
    print(f"  {separator(w)}")

    for scenario_label, scenario_key in [("Image only", "image_only"), ("Text only", "text_only")]:
        first = True
        for model in FUSIONS:
            name = DISPLAY_NAMES[model]
            sc_col = scenario_label if first else ""
            first = False

            abl_m = None
            if model == "feature-fusion" and ff_ablation is not None:
                row = ff_ablation[ff_ablation["condition"] == scenario_key]
                if not row.empty:
                    abl_m = row.iloc[0].to_dict()
            elif model != "feature-fusion" and lf_ablation is not None:
                row = lf_ablation[(lf_ablation["method"] == model) &
                                  (lf_ablation["condition"] == scenario_key)]
                if not row.empty:
                    abl_m = row.iloc[0].to_dict()

            if abl_m is None:
                print(f"  {sc_col:<14s} {name:<18s} {'---':>7s} {'---':>7s} {'---':>7s} {'---':>7s} {'---':>7s}")
                continue

            auc_val = abl_m.get("auc") or abl_m.get("AUC")
            f1_val = abl_m.get("f1") or abl_m.get("F1")
            acc_val = abl_m.get("accuracy") or abl_m.get("Acc") or abl_m.get("acc")
            # Ablation CSVs don't have per-sample predictions, so no per-class acc
            print(f"  {sc_col:<14s} {name:<18s} {f(auc_val)} {f(f1_val)} {f(acc_val)} {'---':>7s} {'---':>7s}")

        print(f"  {separator(w)}")
    print()


# ──────────────────────────────── CLI ────────────────────────────────────────

TABLE_PRINTERS = {
    "clean": print_clean_table,
    "adversarial": print_adversarial_table,
    "ablation": print_ablation_table,
}


def parse_args():
    p = argparse.ArgumentParser(description="Print result tables to stdout.")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--table", default="all",
                   choices=list(TABLE_PRINTERS.keys()) + ["all"])
    return p.parse_args()


def main():
    args = parse_args()
    tables = list(TABLE_PRINTERS.keys()) if args.table == "all" else [args.table]
    for name in tables:
        TABLE_PRINTERS[name](args.dataset)


if __name__ == "__main__":
    main()
