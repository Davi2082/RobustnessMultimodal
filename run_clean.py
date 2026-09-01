"""Clean evaluation for all modalities on Recovery.

Runs:
  1. Text-only          (L14 checkpoint — CLIP unused in text forward pass)
  2. Image-only         (B32 checkpoint)
  3. Feature-fusion     (B32 joint checkpoint)
  4. Late-fusion        min / mean / max / svm-rbf / linear
                        (computed post-hoc from text + image CSVs to avoid the
                         mixed-encoder issue in eval.py)

Usage:
    python3 run_clean.py
"""

import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

from configuration_files.configuration import (
    B32_IMAGE_WEIGHTS_PATH,
    FF_NAME_IMG_EMBED,
    THRESHOLD,
)
from configuration_files.paths import (
    CLEAN_BASE,
    CLEAN_IMAGE_CSV,
    CLEAN_TEXT_CSV,
    RESULT_PATH,
)

FUSION_HEAD_DIR = os.path.join(RESULT_PATH, "fusion_analysis")
SVM_HEAD_PATH = os.path.join(FUSION_HEAD_DIR, "svm_rbf_head.pkl")
LINEAR_HEAD_PATH = os.path.join(FUSION_HEAD_DIR, "linear_head.pkl")


def run(cmd):
    print(f"\n{'='*70}")
    print(f">>> {' '.join(cmd)}")
    print(f"{'='*70}")
    subprocess.run(cmd, check=True)


def late_fusion_from_csvs(text_csv, image_csv, mode, output_dir, threshold=THRESHOLD):
    """Compute late-fusion scores from separate text and image CSVs."""
    df_txt = pd.read_csv(text_csv)
    df_img = pd.read_csv(image_csv)

    assert len(df_txt) == len(df_img), "Text and image CSVs have different lengths"
    assert (df_txt["index"].values == df_img["index"].values).all()

    s_txt = df_txt["score"].values
    s_img = df_img["score"].values

    if mode == "mean":
        scores = (s_txt + s_img) / 2
    elif mode == "min":
        scores = np.minimum(s_txt, s_img)
    elif mode == "max":
        scores = np.maximum(s_txt, s_img)
    elif mode in ("svm-rbf", "linear"):
        head_path = SVM_HEAD_PATH if mode == "svm-rbf" else LINEAR_HEAD_PATH
        if not os.path.isfile(head_path):
            print(f"  [SKIP] Fitted head not found: {head_path}")
            print(f"         Run scripts/fit_fusion_heads.py first.")
            return
        head = joblib.load(head_path)
        X = np.column_stack([s_txt, s_img])
        if hasattr(head, "predict_proba"):
            scores = head.predict_proba(X)[:, 1]
        else:
            scores = head.decision_function(X)
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")

    preds = (scores >= threshold).astype(int)

    result_df = pd.DataFrame({
        "index": df_txt["index"],
        "label": df_txt["label"],
        "score": scores,
        "pred": preds,
        "score_text": s_txt,
        "score_image": s_img,
        "logit_text": df_txt["logit"].values,
        "logit_image": df_img["logit"].values,
    })

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "results.csv")
    result_df.to_csv(csv_path, index=False)

    params = {
        "Modality": "late-fusion",
        "Fusion Mode": mode,
        "Threshold": threshold,
        "Text CSV": text_csv,
        "Image CSV": image_csv,
    }
    if mode in ("svm-rbf", "linear"):
        params["Head Path"] = head_path

    with open(os.path.join(output_dir, "parameters.json"), "w") as f:
        json.dump(params, f, indent=4)

    print(f"  Saved {len(result_df)} samples to {csv_path}")


def main():
    print("=" * 70)
    print("CLEAN EVALUATION — Recovery dataset")
    print("=" * 70)

    # ── 1. Text-only ──
    print("\n[1/3] Text-only evaluation")
    run([
        sys.executable, "scripts/eval.py",
        "--modality", "text",
    ])

    # ── 2. Image-only (B32) ──
    print("\n[2/3] Image-only evaluation (B32)")
    run([
        sys.executable, "scripts/eval.py",
        "--modality", "image",
        "--name_img_embed", FF_NAME_IMG_EMBED,
        "--model_path", B32_IMAGE_WEIGHTS_PATH,
    ])

    # ── 3. Feature-fusion ──
    print("\n[3/3] Feature-fusion evaluation")
    run([
        sys.executable, "scripts/eval.py",
        "--modality", "feature-fusion",
    ])

    # ── 4. Late-fusion (post-hoc from text + image CSVs) ──
    print("\n" + "=" * 70)
    print("LATE-FUSION (post-hoc from text + image CSVs)")
    print("=" * 70)

    if not os.path.isfile(CLEAN_TEXT_CSV):
        print(f"ERROR: Text CSV not found: {CLEAN_TEXT_CSV}")
        sys.exit(1)
    if not os.path.isfile(CLEAN_IMAGE_CSV):
        print(f"ERROR: Image CSV not found: {CLEAN_IMAGE_CSV}")
        sys.exit(1)

    for mode in ("min", "mean", "max", "svm-rbf", "linear"):
        output_dir = os.path.join(CLEAN_BASE, "late-fusion", mode)
        print(f"\n  Late-fusion: {mode}")
        late_fusion_from_csvs(CLEAN_TEXT_CSV, CLEAN_IMAGE_CSV, mode, output_dir)

    print("\n" + "=" * 70)
    print("ALL CLEAN EVALUATIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
