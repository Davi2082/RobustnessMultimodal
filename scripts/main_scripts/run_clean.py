"""Clean evaluation for all modalities.

Runs:
  1. Text-only          (L14 checkpoint)
  2. Image-only         (B32 checkpoint)
  3. Feature-fusion     (B32 joint checkpoint)
  4. Late-fusion        min / mean / max / svm-rbf / linear
                        (computed post-hoc from text + image CSVs)

Usage:
    python3 -m scripts.main_scripts.run_clean --dataset Recovery --devices cuda:0
    python3 -m scripts.main_scripts.run_clean --devices all
"""

import argparse, json, os, subprocess, sys, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from configuration_files.configuration import (
    IMAGE_WEIGHTS_PATH, DATASET, DEVICES, NAME_IMG_EMBED, THRESHOLD,
)
from configuration_files.paths import DATASET_WEIGHTS_DIR, RESULT_PATH
from scripts.utils.devices import resolve_devices


def late_fusion_from_csvs(text_csv, image_csv, mode, output_dir, head_dir, threshold=THRESHOLD):
    df_txt = pd.read_csv(text_csv)
    df_img = pd.read_csv(image_csv)
    assert len(df_txt) == len(df_img), "Text and image CSVs have different lengths"
    assert (df_txt["index"].values == df_img["index"].values).all()
    s_txt, s_img = df_txt["score"].values, df_img["score"].values

    if mode == "mean":
        scores = (s_txt + s_img) / 2
    elif mode == "min":
        scores = np.minimum(s_txt, s_img)
    elif mode == "max":
        scores = np.maximum(s_txt, s_img)
    elif mode in ("svm-rbf", "linear"):
        from models.fusion import pytorch_head_score, fusion_head_path
        if not os.path.isfile(fusion_head_path(mode)):
            print(f"  [SKIP] Fitted head not found: {fusion_head_path(mode)}")
            return
        scores = pytorch_head_score(mode, np.column_stack([s_txt, s_img]))
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")

    preds = (scores >= threshold).astype(int)
    result_df = pd.DataFrame({
        "index": df_txt["index"], "label": df_txt["label"],
        "score": scores, "pred": preds,
        "score_text": s_txt, "score_image": s_img,
        "logit_text": df_txt["logit"].values, "logit_image": df_img["logit"].values,
    })
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "results.csv")
    result_df.to_csv(csv_path, index=False)
    with open(os.path.join(output_dir, "parameters.json"), "w") as f:
        json.dump({"Modality": "late-fusion", "Fusion Mode": mode,
                    "Threshold": threshold, "Text CSV": text_csv, "Image CSV": image_csv}, f, indent=4)
    print(f"  Saved {len(result_df)} samples to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--devices", nargs="+", default=DEVICES, metavar="DEV",
                        help="GPU device(s): cuda:0 cuda:1 ... or 'all'.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results already exist.")
    args = parser.parse_args()

    args.devices = resolve_devices(args.devices)

    result_path = f"results/{args.dataset}/classification_results"
    clean_base = os.path.join(result_path, "clean")
    head_dir = DATASET_WEIGHTS_DIR

    dev_str = ", ".join(args.devices)
    print("=" * 70)
    print(f"CLEAN EVALUATION — {args.dataset}")
    print(f"  Device(s): {dev_str}")
    print("=" * 70)

    t0 = time.time()

    # GPU eval jobs (text, image, feature-fusion) — sequential, each uses DataParallel internally
    jobs = [
        ("text",           os.path.join(clean_base, "text", "results.csv"),
         [sys.executable, "-m", "scripts.utils.eval",
          "--modality", "text", "--dataset", args.dataset] + ["--devices"] + args.devices),
        ("image",          os.path.join(clean_base, "image", "results.csv"),
         [sys.executable, "-m", "scripts.utils.eval",
          "--modality", "image", "--dataset", args.dataset,
          "--name_img_embed", NAME_IMG_EMBED, "--model_path", IMAGE_WEIGHTS_PATH] + ["--devices"] + args.devices),
        ("feature-fusion", os.path.join(clean_base, "feature-fusion", "results.csv"),
         [sys.executable, "-m", "scripts.utils.eval",
          "--modality", "feature-fusion", "--dataset", args.dataset] + ["--devices"] + args.devices),
    ]

    for i, (label, csv_path, cmd) in enumerate(jobs):
        if not args.force and os.path.isfile(csv_path):
            print(f"\n[{i+1}/3] {label} — SKIP")
            continue
        print(f"\n[{i+1}/3] {label}")
        print(f">>> {' '.join(cmd)}")
        sys.stdout.flush()
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"  FAILED (exit {rc})")

    # Late-fusion (CPU, always sequential)
    clean_text_csv = os.path.join(clean_base, "text", "results.csv")
    clean_image_csv = os.path.join(clean_base, "image", "results.csv")

    print("\n" + "=" * 70)
    print("LATE-FUSION (post-hoc from text + image CSVs)")
    print("=" * 70)

    if not os.path.isfile(clean_text_csv):
        print(f"ERROR: {clean_text_csv} not found — run text eval first.")
        sys.exit(1)
    if not os.path.isfile(clean_image_csv):
        print(f"ERROR: {clean_image_csv} not found — run image eval first.")
        sys.exit(1)

    for mode in ("min", "mean", "max", "svm-rbf", "linear"):
        output_dir = os.path.join(clean_base, "late-fusion", mode)
        lf_csv = os.path.join(output_dir, "results.csv")
        if not args.force and os.path.isfile(lf_csv):
            print(f"  {mode} — SKIP")
            continue
        print(f"  {mode}")
        late_fusion_from_csvs(clean_text_csv, clean_image_csv, mode, output_dir, head_dir)

    elapsed = time.time() - t0
    print(f"\nCLEAN EVALUATION COMPLETE ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
