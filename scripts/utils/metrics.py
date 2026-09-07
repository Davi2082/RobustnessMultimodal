import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from configuration_files.paths import RESULT_PATH
from scripts.utils.utils import compute_metrics, plot_confusion_matrix, build_curve_name, update_roc_cache, regenerate_plot

ATTACK_DIRS = {
    "pgd": "late-fusion-pgd", "trepat": "late-fusion-trepat",
    "sum": "late-fusion", "interleaved": "late-fusion-interleaved",
    "joint": "late-fusion-joint",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, choices=["clean", "perturbed"])
    parser.add_argument("--modality", type=str, required=True,
                        choices=["feature-fusion", "intermediate-fusion", "late-fusion", "text", "image"])
    parser.add_argument("--mode", type=str, choices=["mean", "min", "max", "svm-rbf", "linear", "feature-fusion"],
                        help="Fusion method (required for late-fusion modality).")
    parser.add_argument("--perturbation_type", type=str, choices=["biperturbed", "image-perturbed", "text-perturbed"])
    parser.add_argument("--attack", type=str, choices=list(ATTACK_DIRS.keys()), default="sum",
                        help="Attack type — determines which output directory to read from.")
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

    if args.modality == "feature-fusion" and args.type == "perturbed":
        base = os.path.join(RESULT_PATH, "perturbed", "feature-fusion")
        fname_map = {"biperturbed": "perturbed_results.csv",
                     "text-perturbed": "txts_perturbed_results.csv",
                     "image-perturbed": "imgs_perturbed_results.csv"}
        results_file = fname_map[args.perturbation_type]
    elif args.modality == "late-fusion" and args.type == "perturbed":
        base = os.path.join(RESULT_PATH, "perturbed", ATTACK_DIRS[args.attack], args.mode)
        if args.perturbation_type in ("image-perturbed", "text-perturbed"):
            base = os.path.join(base, args.perturbation_type)
        results_file = "perturbed_results.csv"
    else:
        results_file = f"{'perturbed_' if args.type == 'perturbed' else ''}results.csv"
        base = os.path.join(RESULT_PATH, args.type, args.modality, args.mode, args.perturbation_type)

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
