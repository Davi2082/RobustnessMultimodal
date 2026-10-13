"""Tune one decision threshold per method on the validation split.

Real (label 1) is the positive class and the score is P(Real): a higher score
means Real. Predictions everywhere are ``score > t`` -> Real. The threshold is
the ROC operating point on the validation split (both classes): among
thresholds with FPR = P(predicted Real | Fake) <= ``threshold_fpr``
(config.yaml), the one with the highest TPR = P(predicted Real | Real).

Steps:
  1. GPU: score the full non-augmented validation split with the text-only,
     image-only and feature-fusion models (scripts.utils.eval --split val).
  2. CPU: fit the learned heads (svm-rbf, linear) on the validation logits
     (scripts/train_scripts/fusion_heads.py).
  3. CPU: validation scores per method: min/mean/max on the unimodal scores;
     svm-rbf/linear out-of-fold (cv_folds-fold cross-fitting), so their
     thresholds are not tuned on scores of the samples they were fitted on.
  4. Write checkpoints/<dataset>/thresholds.json and the out-of-fold scores
     (results/.../val/<head>/oof_scores.csv).

Usage:
    python3 -m scripts.main_scripts.compute_thresholds --dataset Fakeddit --devices cuda:1
    python3 -m scripts.main_scripts.compute_thresholds --dataset Fakeddit --skip-eval   # reuse val scores
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from configuration_files.configuration import (
    DATASET, HEAD_CV_FOLDS, NAME_IMG_EMBED, PIPELINE_LATE_FUSION_MODES, RAND_SEED, TRAIN_SEED,
    THRESHOLD_FPR, dataset_devices, image_weights_path, thresholds_path,
)
from configuration_files.paths import dataset_result_root
from scripts.utils.devices import resolve_devices

NEURAL = ("text", "image", "feature-fusion")


def val_csv(dataset, method):
    return os.path.join(dataset_result_root(dataset), "val", method, "results.csv")


def fpr_threshold(scores, labels, fpr):
    """ROC operating point on both classes, Real (label 1) = positive.

    Predict Real iff score > t. Candidates are all validation scores of both
    classes; among those with FPR = P(score > t | Fake) <= fpr, pick the one
    with the highest TPR = P(score > t | Real) (ties: lower FPR, then larger t).
    """
    real, fake = np.sort(scores[labels == 1]), np.sort(scores[labels == 0])
    cands = np.unique(np.concatenate([scores, [np.nextafter(scores.min(), -np.inf)]]))
    tpr_c = 1 - np.searchsorted(real, cands, side="right") / len(real)
    fpr_c = 1 - np.searchsorted(fake, cands, side="right") / len(fake)
    ok = np.where(fpr_c <= fpr)[0]
    best = max(ok, key=lambda k: (tpr_c[k], -fpr_c[k], cands[k]))
    if tpr_c[best] == 0:
        print(f"  WARNING: no threshold reaches FPR <= {fpr} with any Real predicted (saturated scores)")
    return float(cands[best])


def static_rule_scores(mode, s_txt, s_img):
    if mode == "mean":
        return (s_txt + s_img) / 2
    if mode == "min":
        return np.minimum(s_txt, s_img)
    if mode == "max":
        return np.maximum(s_txt, s_img)
    raise ValueError(f"not a static rule: {mode}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--devices", nargs="+", default=None)
    p.add_argument("--fpr", type=float, default=THRESHOLD_FPR)
    p.add_argument("--skip-eval", action="store_true", help="reuse existing validation scores")
    p.add_argument("--seed", type=int, default=TRAIN_SEED)
    args = p.parse_args()

    devices = resolve_devices(args.devices or dataset_devices(args.dataset))
    if not args.skip_eval:
        for modality in NEURAL:
            cmd = [sys.executable, "-m", "scripts.utils.eval", "--modality", modality,
                   "--dataset", args.dataset, "--split", "val", "--devices", devices[0]]
            if modality == "image":
                cmd += ["--name_img_embed", NAME_IMG_EMBED, "--model_path", image_weights_path(args.dataset)]
            print(">>>", " ".join(cmd), flush=True)
            if subprocess.run(cmd).returncode != 0:
                sys.exit(f"validation scoring failed for {modality}")

    frames = {m: pd.read_csv(val_csv(args.dataset, m)) for m in NEURAL}
    t, i = frames["text"], frames["image"]
    assert (t["index"].values == i["index"].values).all(), "text/image validation rows differ"
    labels = t["label"].values
    scores = {m: frames[m]["score"].values for m in NEURAL}
    assert (frames["feature-fusion"]["label"].values == labels).all()
    from scripts.train_scripts.fusion_heads import (
        LEARNED_HEADS, fit_and_save, head_features, out_of_fold_scores)
    X = head_features(t, i)
    for mode in PIPELINE_LATE_FUSION_MODES:
        if mode in LEARNED_HEADS:
            fit_and_save(mode, args.dataset, args.seed)
            scores[mode] = out_of_fold_scores(mode, X, labels, args.seed)
            out_dir = os.path.join(dataset_result_root(args.dataset), "val", mode)
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame({"index": t["index"], "label": labels, "oof_score": scores[mode]}).to_csv(
                os.path.join(out_dir, "oof_scores.csv"), index=False)
        else:
            scores[mode] = static_rule_scores(mode, scores["text"], scores["image"])

    table = {}
    print(f"\n{'method':15s} {'threshold':>12s} {'val FPR':>8s} {'val TPR':>8s}   (Real = positive)")
    for method, s in scores.items():
        thr = fpr_threshold(s, labels, args.fpr)
        pred_real = s > thr
        fpr = float(pred_real[labels == 0].mean())
        tpr = float(pred_real[labels == 1].mean())
        table[method] = {"threshold": thr, "val_fpr": round(fpr, 4), "val_tpr": round(tpr, 4)}
        print(f"{method:15s} {thr:12.6g} {fpr:8.3f} {tpr:8.3f}")

    out = thresholds_path(args.dataset)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"dataset": args.dataset, "split": "val (non-augmented)", "positive_class": "Real",
                   "target_fpr": args.fpr, "n_real": int((labels == 1).sum()),
                   "n_fake": int((labels == 0).sum()),
                   "rule": "predict Real iff P(Real) > threshold",
                   "learned_heads": f"thresholds on {HEAD_CV_FOLDS}-fold out-of-fold validation scores",
                   "thresholds": table}, f, indent=4)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
