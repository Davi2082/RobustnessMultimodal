import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from configuration_files.configuration import ATTACK_SCOPE, DATASET, PIPELINE_ATTACKS
from configuration_files.paths import model_clean_dir, model_perturbed_dir
from scripts.utils.utils import compute_metrics, plot_confusion_matrix, build_curve_name, update_roc_cache, regenerate_plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, choices=["clean", "perturbed"])
    parser.add_argument("--modality", type=str, required=True,
                        choices=["feature-fusion", "intermediate-fusion", "late-fusion", "text", "image"])
    parser.add_argument("--mode", type=str, choices=["mean", "min", "max", "svm-rbf", "linear", "feature-fusion"],
                        help="Fusion method (required for late-fusion modality).")
    parser.add_argument("--perturbation_type", type=str,
                        help="Only affects the ROC curve name.")
    parser.add_argument("--attack", type=str, choices=list(ATTACK_SCOPE.keys()), default="sum",
                        help="Attack type — determines which perturbed directory to read from.")
    parser.add_argument("--roc-set", type=str, help="Name of ROC comparison group")
    parser.add_argument("--dataset", type=str, default=DATASET)
    args = parser.parse_args()

    if args.type == "clean":
        args.perturbation_type = ""

    if args.modality == "late-fusion" and args.mode is None:
        parser.error("--mode is required when --modality is late-fusion")
    elif args.modality != "late-fusion":
        args.mode = ""

    # Resolve the model/fusion name used as the directory
    if args.modality in ("late-fusion", "feature-fusion") and args.type == "perturbed":
        fusion_name = args.mode or "feature-fusion"
        base = model_perturbed_dir(fusion_name, args.attack, args.dataset)
        results_file = "perturbed_results.csv"
    elif args.modality in ("text", "image") and args.type == "perturbed":
        base = model_perturbed_dir(args.modality, args.attack, args.dataset)
        results_file = "perturbed_results.csv"
    elif args.type == "clean":
        if args.modality == "late-fusion":
            base = model_clean_dir(args.mode, args.dataset)
        else:
            base = model_clean_dir(args.modality, args.dataset)
        results_file = "results.csv"
    else:
        parser.error(f"Unsupported combination: --type {args.type} --modality {args.modality}")

    csv_path = os.path.join(base, results_file)
    if not os.path.isfile(csv_path):
        parser.error(f"Results file not found: {csv_path}")
    df = pd.read_csv(csv_path)

    y_true = 1 - np.asarray(df["label"])
    y_pred = 1 - np.asarray(df["pred"])
    scores = 1 - np.asarray(df["score"])

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
