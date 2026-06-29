import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from paths import RESULT_PATH

# Custom imports
from utils import compute_metrics, plot_confusion_matrix, build_curve_name, update_roc_cache, regenerate_plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, choices=["clean", "perturbed"])
    parser.add_argument("--modality", type=str, required=True, choices=["feature-fusion", "intermediate-fusion", "late-fusion", "text", "image"])
    parser.add_argument("--mode",  type=str, choices=["mean", "min", "max"])
    parser.add_argument("--perturbation_type",  type=str, choices=["biperturbed", "image-perturbed", "text-perturbed"])
    parser.add_argument("--roc-set", type=str, help="Name of ROC comparison group")
    args = parser.parse_args()

    if args.type == "perturbed" and args.modality == "feature-fusion" and args.perturbation_type is None:
        parser.error("--perturbation_type is required for feature-fusion when --type is perturbed")
    elif args.type == "clean":
        args.perturbation_type = ""
    elif args.perturbation_type is None:
        args.perturbation_type = ""
        
    if args.modality == "late-fusion" and args.mode is None:
            parser.error("--mode is required when --modality is late-fusion")
    elif args.modality != "late-fusion":
        args.mode = ""

    # Resolve path and filename based on actual output structure of each attack script
    if args.modality == "feature-fusion" and args.type == "perturbed":
        # multimodal_attack.py writes all variants into one flat directory with distinct filenames
        base = os.path.join(RESULT_PATH, "perturbed", "feature-fusion")
        fname_map = {
            "biperturbed":     "perturbed_results.csv",
            "text-perturbed":  "txts_perturbed_results.csv",
            "image-perturbed": "imgs_perturbed_results.csv",
        }
        results_file = fname_map[args.perturbation_type]
    elif args.modality == "late-fusion" and args.type == "perturbed" and args.perturbation_type == "biperturbed":
        # late_fusion_perturbation.py writes biperturbed directly under the mode directory
        base = os.path.join(RESULT_PATH, "perturbed", "late-fusion", args.mode)
        results_file = "perturbed_results.csv"
    else:
        results_file = f"{'perturbed_' if args.type == 'perturbed' else ''}results.csv"
        base = os.path.join(RESULT_PATH, args.type, args.modality, args.mode, args.perturbation_type)
    df = pd.read_csv(os.path.join(base, results_file))

    y_true = df["label"]
    y_pred = df["pred"]
    scores = df["score"]

    y_true = 1-np.asarray(y_true)
    y_pred = 1-np.asarray(y_pred)
    scores = 1-np.asarray(scores)

    if os.path.exists(os.path.join(base, "metrics.json")):
        fpr, tpr, thr = roc_curve(y_true, scores)
        auc_score = auc(fpr, tpr)
    else:
        metrics, fpr, tpr, cm = compute_metrics(y_true, y_pred, scores)
        auc_score = metrics["auc"]
        plot_confusion_matrix(cm, range(cm.shape[0]), os.path.join(base, "confusion_matrix.png"))
        with open(os.path.join(base, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
    
    if args.roc_set is not None:
        curve_name = build_curve_name(args)
        roc_cache = update_roc_cache(args.roc_set, curve_name, auc_score, fpr, tpr)
        regenerate_plot(roc_cache, args.roc_set)