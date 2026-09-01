"""Missing-modality ablation for all fusion methods on Recovery.

Measures what each fusion method predicts when one input modality is removed:

  1. Feature-fusion ablation   (scripts/modality_ablation.py — blank + drop)
  2. Late-fusion ablation      (post-hoc: set the missing modality's score to 0.5
                                and apply min / mean / max / svm-rbf / linear)

The paper table (themis_missing_modality.tex) reports AUC, F1, Acc and
deltas versus the both-modalities baseline for each method.

Usage:
    python3 -m scripts.run_ablation
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from configuration_files.configuration import THRESHOLD
from configuration_files.paths import CLEAN_BASE, CLEAN_IMAGE_CSV, CLEAN_TEXT_CSV, RESULT_PATH

FUSION_HEAD_DIR = os.path.join(RESULT_PATH, "fusion_analysis")
SVM_HEAD_PATH = os.path.join(FUSION_HEAD_DIR, "svm_rbf_head.pkl")
LINEAR_HEAD_PATH = os.path.join(FUSION_HEAD_DIR, "linear_head.pkl")

LATE_FUSION_MODES = ("min", "mean", "max", "svm-rbf", "linear")
DEFAULT_SCORE = 0.5


def run(cmd):
    print(f"\n{'='*70}")
    print(f">>> {' '.join(cmd)}")
    print(f"{'='*70}")
    subprocess.run(cmd, check=True)


def fuse_scores(s_txt, s_img, mode):
    """Apply a fusion rule to text and image score arrays."""
    if mode == "mean":
        return (s_txt + s_img) / 2
    elif mode == "min":
        return np.minimum(s_txt, s_img)
    elif mode == "max":
        return np.maximum(s_txt, s_img)
    elif mode in ("svm-rbf", "linear"):
        head_path = SVM_HEAD_PATH if mode == "svm-rbf" else LINEAR_HEAD_PATH
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


def late_fusion_ablation(text_csv, image_csv, output_dir):
    """Compute late-fusion metrics under missing-modality conditions."""
    df_txt = pd.read_csv(text_csv)
    df_img = pd.read_csv(image_csv)
    labels = df_txt["label"].values
    s_txt = df_txt["score"].values
    s_img = df_img["score"].values

    rows = []
    for mode in LATE_FUSION_MODES:
        both = fuse_scores(s_txt, s_img, mode)
        img_only = fuse_scores(np.full_like(s_txt, DEFAULT_SCORE), s_img, mode)
        txt_only = fuse_scores(s_txt, np.full_like(s_img, DEFAULT_SCORE), mode)

        if both is None or img_only is None or txt_only is None:
            print(f"  [SKIP] {mode} — fitted head not found")
            continue

        auc_b, f1_b, acc_b = compute_metrics(labels, both)
        auc_i, f1_i, acc_i = compute_metrics(labels, img_only)
        auc_t, f1_t, acc_t = compute_metrics(labels, txt_only)

        rows.append({
            "method": mode,
            "condition": "both",
            "AUC": round(auc_b, 3), "F1": round(f1_b, 3), "Acc": round(acc_b, 3),
        })
        rows.append({
            "method": mode,
            "condition": "image_only",
            "AUC": round(auc_i, 3), "F1": round(f1_i, 3), "Acc": round(acc_i, 3),
            "dAUC": round(auc_i - auc_b, 3), "dF1": round(f1_i - f1_b, 3), "dAcc": round(acc_i - acc_b, 3),
        })
        rows.append({
            "method": mode,
            "condition": "text_only",
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
    print("=" * 70)
    print("MISSING-MODALITY ABLATION — Recovery dataset")
    print("=" * 70)

    # ── 1. Feature-fusion ablation (blank + drop) ──
    print("\n[1/2] Feature-fusion modality ablation")
    run([
        sys.executable, "-m", "scripts.modality_ablation",
        "--modality", "feature-fusion",
    ])

    # ── 2. Late-fusion ablation ──
    print("\n[2/2] Late-fusion modality ablation")
    if not os.path.isfile(CLEAN_TEXT_CSV):
        print(f"ERROR: Text CSV not found: {CLEAN_TEXT_CSV}")
        print("       Run run_clean.py first.")
        sys.exit(1)
    if not os.path.isfile(CLEAN_IMAGE_CSV):
        print(f"ERROR: Image CSV not found: {CLEAN_IMAGE_CSV}")
        print("       Run run_clean.py first.")
        sys.exit(1)

    output_dir = os.path.join(RESULT_PATH, "fusion_analysis")
    late_fusion_ablation(CLEAN_TEXT_CSV, CLEAN_IMAGE_CSV, output_dir)

    print("\n" + "=" * 70)
    print("ABLATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
