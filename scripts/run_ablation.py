"""Missing-modality ablation for all fusion methods.

Measures what each fusion method predicts when one input modality is removed:

  1. Feature-fusion ablation   (scripts/modality_ablation.py — blank + drop)
  2. Late-fusion ablation      (post-hoc: set the missing modality's score to 0.5
                                and apply min / mean / max / svm-rbf / linear)

Usage:
    python3 -m scripts.run_ablation --dataset Recovery --device cuda:0
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from configuration_files.configuration import DATASET, DEVICE_EVAL, THRESHOLD
from configuration_files.paths import DATASET_WEIGHTS_DIR

LATE_FUSION_MODES = ("min", "mean", "max", "svm-rbf", "linear")
DEFAULT_SCORE = 0.5


def run(cmd):
    print(f"\n{'='*70}")
    print(f">>> {' '.join(cmd)}")
    print(f"{'='*70}")
    subprocess.run(cmd, check=True)


def fuse_scores(s_txt, s_img, mode, head_dir):
    """Apply a fusion rule to text and image score arrays."""
    if mode == "mean":
        return (s_txt + s_img) / 2
    elif mode == "min":
        return np.minimum(s_txt, s_img)
    elif mode == "max":
        return np.maximum(s_txt, s_img)
    elif mode in ("svm-rbf", "linear"):
        slug = "svm_rbf" if mode == "svm-rbf" else "linear"
        head_path = os.path.join(head_dir, f"{slug}_head.pkl")
        if not os.path.isfile(head_path):
            return None
        head = joblib.load(head_path)
        X = np.column_stack([s_txt, s_img])
        if hasattr(head, "predict_proba"):
            return head.predict_proba(X)[:, 1]
        return head.decision_function(X)
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")


def compute_metrics(y_true, scores, threshold=THRESHOLD):
    """AUC, F1, Acc with Fake (label=0) as positive class."""
    y_inv = 1 - y_true
    s_inv = 1 - scores
    y_pred = (s_inv > threshold).astype(int)
    auc = roc_auc_score(y_inv, s_inv)
    f1 = f1_score(y_inv, y_pred)
    acc = accuracy_score(y_inv, y_pred)
    return auc, f1, acc


def late_fusion_ablation(text_csv, image_csv, output_dir, head_dir):
    """Compute late-fusion metrics under missing-modality conditions."""
    df_txt = pd.read_csv(text_csv)
    df_img = pd.read_csv(image_csv)
    labels = df_txt["label"].values
    s_txt = df_txt["score"].values
    s_img = df_img["score"].values

    rows = []
    for mode in LATE_FUSION_MODES:
        both = fuse_scores(s_txt, s_img, mode, head_dir)
        img_only = fuse_scores(np.full_like(s_txt, DEFAULT_SCORE), s_img, mode, head_dir)
        txt_only = fuse_scores(s_txt, np.full_like(s_img, DEFAULT_SCORE), mode, head_dir)

        if both is None or img_only is None or txt_only is None:
            print(f"  [SKIP] {mode} — fitted head not found")
            continue

        auc_b, f1_b, acc_b = compute_metrics(labels, both)
        auc_i, f1_i, acc_i = compute_metrics(labels, img_only)
        auc_t, f1_t, acc_t = compute_metrics(labels, txt_only)

        rows.append({
            "method": mode, "condition": "both",
            "AUC": round(auc_b, 3), "F1": round(f1_b, 3), "Acc": round(acc_b, 3),
        })
        rows.append({
            "method": mode, "condition": "image_only",
            "AUC": round(auc_i, 3), "F1": round(f1_i, 3), "Acc": round(acc_i, 3),
            "dAUC": round(auc_i - auc_b, 3), "dF1": round(f1_i - f1_b, 3), "dAcc": round(acc_i - acc_b, 3),
        })
        rows.append({
            "method": mode, "condition": "text_only",
            "AUC": round(auc_t, 3), "F1": round(f1_t, 3), "Acc": round(acc_t, 3),
            "dAUC": round(auc_t - auc_b, 3), "dF1": round(f1_t - f1_b, 3), "dAcc": round(acc_t - acc_b, 3),
        })

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "late_fusion_modality_ablation.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved to {csv_path}")
    print(df.to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--device", default=DEVICE_EVAL)
    args = parser.parse_args()

    result_path = f"results/{args.dataset}/classification_results"
    clean_text_csv = os.path.join(result_path, "clean", "text", "results.csv")
    clean_image_csv = os.path.join(result_path, "clean", "image", "results.csv")
    head_dir = DATASET_WEIGHTS_DIR

    print("=" * 70)
    print(f"MISSING-MODALITY ABLATION — {args.dataset}")
    print(f"  Device: {args.device}")
    print("=" * 70)

    # ── 1. Feature-fusion ablation (blank + drop) ──
    print("\n[1/2] Feature-fusion modality ablation")
    run([
        sys.executable, "-m", "scripts.modality_ablation",
        "--modality", "feature-fusion",
        "--dataset", args.dataset,
    ])

    # ── 2. Late-fusion ablation ──
    print("\n[2/2] Late-fusion modality ablation")
    if not os.path.isfile(clean_text_csv):
        print(f"ERROR: Text CSV not found: {clean_text_csv}")
        print("       Run scripts.run_clean first.")
        sys.exit(1)
    if not os.path.isfile(clean_image_csv):
        print(f"ERROR: Image CSV not found: {clean_image_csv}")
        print("       Run scripts.run_clean first.")
        sys.exit(1)

    late_fusion_ablation(clean_text_csv, clean_image_csv, head_dir, head_dir)

    print("\n" + "=" * 70)
    print("ABLATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
