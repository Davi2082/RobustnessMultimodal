"""Clean score-space figure: unimodal logits with late-fusion decision regions.

Produces one panel per fusion rule (min / mean / max / RBF-SVM) in the space
spanned by the two dedicated unimodal Themis classifiers, plus a panel for the
feature-fusion model in its OWN ablation space (logit when fed image only vs.
logit when fed text only) -- the feature-fusion decision is not a function of
the unimodal classifiers' logits, so it cannot share their axes.

The RBF-SVM is fitted on clean TRAIN-split predictions from the same two
unimodal checkpoints, with C and gamma selected by stratified cross-validated
ROC-AUC.

Usage:
    python3 plot_clean_score_space.py
"""

import os
import glob
import json
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from data_loading import my_datasets
from utils import load_available_datasets, load_model
from configuration_files.configuration import (
    NAME_LLM, NAME_IMG_EMBED, TEXT_WEIGHTS_PATH, IMAGE_WEIGHTS_PATH,
    BATCH_SIZE, N_TOKENS, THRESHOLD, DEVICE_EVAL, LATE_FUSION_SVM_SEED,
)
from configuration_files.paths import RESULT_PATH
from models.fusion import load_fitted_heads

TRAIN_CSV = "data/Recovery/train.csv"
TRAIN_LOGITS = os.path.join(RESULT_PATH, "fusion_analysis", "train_unimodal_logits_clean.csv")
FIG_DIR = "figures/classification_results/scatter"


def unimodal_train_logits(args, device, dataset_classes, load_functions):
    """Clean train-split logits from the text-only and image-only checkpoints."""
    if os.path.exists(TRAIN_LOGITS) and not args.force:
        return pd.read_csv(TRAIN_LOGITS)

    out = {}
    for modality, weights, encoder in (("text", TEXT_WEIGHTS_PATH, NAME_IMG_EMBED),
                                      ("image", IMAGE_WEIGHTS_PATH, NAME_IMG_EMBED)):
        args.modality, args.model_path, args.name_img_embed = modality, weights, encoder
        model, tokenizer, processor = load_model(device, args)
        model.eval()
        ds = my_datasets.get_dataset(
            dataset_classes[args.dataset], load_functions[args.dataset],
            args.n_tokens, processor, tokenizer, TRAIN_CSV, f"data/{args.dataset}/images",
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        logits, labels = [], []
        with torch.no_grad():
            for images, y, texts, _, _ in tqdm(loader, desc=f"train/{modality}"):
                images, texts = images.to(device), texts.to(device)
                if modality == "text":
                    _, lg = model(images=None, texts=texts)
                else:
                    _, lg = model(images=images, texts=None)
                logits.append(lg.detach().cpu().flatten())
                labels.append(y)
        out[modality] = torch.cat(logits).numpy()
        out["label"] = torch.cat(labels).numpy()
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame({"label": out["label"], "text_logit": out["text"], "image_logit": out["image"]})
    os.makedirs(os.path.dirname(TRAIN_LOGITS), exist_ok=True)
    df.to_csv(TRAIN_LOGITS, index=False)
    return df


def fit_heads(train_df, seed):
    """Fit the two learned fusion heads on the clean train-split unimodal logits.

    ``svm-rbf``: RBF-SVM, C and gamma by cross-validated ROC-AUC.
    ``linear``:  a single linear layer trained with cross-entropy (logistic
                 regression), inverse regularisation strength C by the same CV.
    """
    X = train_df[["image_logit", "text_logit"]].values
    y = train_df["label"].values  # 1 = Real (positive)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    Cs = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    svm_pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=seed))
    svm_pipe.steps[1] = ("svc", svm_pipe.steps[1][1])
    svm = GridSearchCV(svm_pipe, {"svc__C": Cs, "svc__gamma": ["scale", 0.01, 0.1, 1.0]},
                       scoring="roc_auc", cv=cv, n_jobs=-1).fit(X, y)

    lin_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed))
    lin_pipe.steps[1] = ("lr", lin_pipe.steps[1][1])
    lin = GridSearchCV(lin_pipe, {"lr__C": Cs}, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X, y)

    return {"svm-rbf": svm, "linear": lin}


def rule_scores(name, img, txt, heads=None):
    """Fused P(Real) for each late-fusion rule. sigmoid of the unimodal logits."""
    pi, pt = 1 / (1 + np.exp(-img)), 1 / (1 + np.exp(-txt))
    if name == "min":
        return np.minimum(pi, pt)
    if name == "mean":
        return (pi + pt) / 2
    if name == "max":
        return np.maximum(pi, pt)
    if name in ("svm-rbf", "linear"):
        head = getattr(heads[name], "best_estimator_", heads[name])
        return head.predict_proba(np.column_stack([img, txt]))[:, 1]
    raise ValueError(name)



ROW_LABELS = {
    "min": r"\quad min",
    "mean": r"\quad mean",
    "max": r"\quad max",
    "linear": r"\quad linear",
    "svm-rbf": r"\quad SVM-RBF",
}
ORDER = ("min", "mean", "max", "linear", "svm-rbf")


def paper_tables_dir():
    projects = [p for p in sorted(glob.glob(os.path.join("paper", "overleaf", "*")))
                if os.path.isdir(p)]
    newest = max(projects, key=os.path.getmtime)
    return os.path.join(newest, "tables")


def unimodal_row(name):
    frame = pd.read_csv(os.path.join(RESULT_PATH, "clean", name, "results.csv"))
    label, score = frame["label"].values, frame["score"].values
    pred = (score >= THRESHOLD).astype(int)
    # Real is the positive class in this table, matching its caption.
    return {
        "AUC": roc_auc_score(label, score),
        "F1": f1_score(label, pred),
        "Acc": accuracy_score(label, pred),
    }


def write_clean_table(metrics):
    """Emit the clean-performance table beside the figure."""
    metrics = metrics.set_index("method")

    text, image = unimodal_row("text"), unimodal_row("image")

    # Bold the best value of each metric, compared at the precision shown so
    # that two rows printing the same number are emphasised the same way.
    best = {m: round(metrics[m].max(), 3) for m in ("AUC", "F1", "Acc")}

    def cell(value, metric):
        text_value = f"{value:.3f}"
        return rf"\textbf{{{text_value}}}" if round(value, 3) >= best[metric] else text_value

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Clean performance of the uni-modal classifiers and the six fusion",
        r"methods on the ReCoVery test set. Real is the positive class. The two learned",
        r"rules are fitted on a held-out validation split. Best result per metric in bold.}",
        r"\label{tab:unimodal-clean}",
        r"\begin{tabular}{l c c c}",
        r"\hline",
        r"\textbf{Uni-modal method} & \textbf{AUC} & \textbf{F1} & \textbf{Acc} \\",
        r"\hline",
        rf"Text-only classifier   & {text['AUC']:.3f} & {text['F1']:.3f} & {text['Acc']:.3f} \\",
        rf"Image-only classifier  & {image['AUC']:.3f} & {image['F1']:.3f} & {image['Acc']:.3f} \\",
        r"\hline",
        r"\textbf{Multi-modal fusion method} & \textbf{AUC} & \textbf{F1} & \textbf{Acc} \\",
        r"\hline",
        r"\multicolumn{4}{l}{\textit{Late fusion}} \\",
    ]
    for rule in ORDER:
        row = metrics.loc[rule]
        lines.append(
            f"{ROW_LABELS[rule]:<14} & {cell(row.AUC, 'AUC')} & "
            f"{cell(row.F1, 'F1')} & {cell(row.Acc, 'Acc')} \\\\"
        )
    ff = metrics.loc["feature-fusion"]
    lines += [
        rf"\textit{{Feature fusion}} & {cell(ff.AUC, 'AUC')} & {cell(ff.F1, 'F1')} & {cell(ff.Acc, 'Acc')} \\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    out = os.path.join(paper_tables_dir(), "themis_unimodal_clean.tex")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"wrote {out}")

def main():
    dataset_classes, load_functions = load_available_datasets()
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Recovery")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--n_tokens", type=int, default=N_TOKENS)
    p.add_argument("--name_llm", default=NAME_LLM)
    p.add_argument("--name_img_embed", default=NAME_IMG_EMBED)
    p.add_argument("--merge_tokens", type=int, default=0)
    p.add_argument("--lora_alpha", type=int)
    p.add_argument("--lora_r", type=int)
    p.add_argument("--lora_dropout", type=float)
    p.add_argument("--use_lora", type=bool)
    p.add_argument("--set_params", type=bool, default=True)
    p.add_argument("--force", action="store_true", help="recompute train logits")
    args = p.parse_args()

    device = torch.device(DEVICE_EVAL)
    train_df = unimodal_train_logits(args, device, dataset_classes, load_functions)

    # Use the heads persisted by scripts/fit_fusion_heads.py when present, so
    # the clean table describes the same rules the attacks target.
    heads = load_fitted_heads() or fit_heads(train_df, LATE_FUSION_SVM_SEED)
    for name, h in heads.items():
        if hasattr(h, "best_params_"):
            print(f"{name}: best={h.best_params_}  CV ROC-AUC={h.best_score_:.4f}")
        else:
            print(f"{name}: loaded from disk (see fusion_heads.json for its CV record)")

    # ---- test-set unimodal logits ----
    B = os.path.join(RESULT_PATH, "clean")
    t = pd.read_csv(os.path.join(B, "text", "results.csv"))
    i = pd.read_csv(os.path.join(B, "image", "results.csv"))
    te = t.merge(i, on="index", suffixes=("_txt", "_img"))
    assert (te["label_txt"] == te["label_img"]).all()
    y = te["label_txt"].values
    txt, img = te["logit_txt"].values, te["logit_img"].values

    rows = []
    for rule in ("min", "mean", "max", "svm-rbf", "linear"):
        s = rule_scores(rule, img, txt, heads)
        yp = (s > THRESHOLD).astype(int)
        rows.append({"method": rule, "AUC": roc_auc_score(y, s),
                     "F1": f1_score(y, yp), "Acc": accuracy_score(y, yp)})

    # ---- feature fusion, in its own ablation space ----
    ff = pd.read_csv(os.path.join(RESULT_PATH, "fusion_analysis", "feature-fusion_modality_ablation.csv"))
    ff_s = 1 / (1 + np.exp(-ff["drop_logit_both"].values))
    rows.append({"method": "feature-fusion", "AUC": roc_auc_score(ff["label"], ff_s),
                 "F1": f1_score(ff["label"], (ff_s > THRESHOLD).astype(int)),
                 "Acc": accuracy_score(ff["label"], (ff_s > THRESHOLD).astype(int))})
    summary = pd.DataFrame(rows)

    # ---- figure ----
    # Five late-fusion rules plus feature fusion. The feature-fusion panel uses
    # its OWN axes (logit when fed image only vs. logit when fed text only): its
    # decision is not a function of the unimodal classifiers' logits, so no
    # decision region is drawn there.
    panels = [("min", img, txt, y), ("mean", img, txt, y), ("max", img, txt, y),
              ("svm-rbf", img, txt, y), ("linear", img, txt, y),
              ("feature-fusion", ff["drop_logit_img_only"].values,
               ff["drop_logit_txt_only"].values, ff["label"].values)]

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    })

    # Shared limits across the five late-fusion panels so their axis labels and
    # tick labels can be factored out to the left column and bottom row.
    lo_x, hi_x = img.min() - 1, img.max() + 1
    lo_y, hi_y = txt.min() - 1, txt.max() + 1

    fig, axes2d = plt.subplots(2, 3, figsize=(13.5, 8.6))
    axes = axes2d.ravel()
    for k, (ax, (name, xv, yv, lab)) in enumerate(zip(axes, panels)):
        is_ff = name == "feature-fusion"
        if is_ff:
            xlo, xhi = xv.min() - 0.5, xv.max() + 0.5
            ylo, yhi = yv.min() - 0.2, yv.max() + 0.2
        else:
            xlo, xhi, ylo, yhi = lo_x, hi_x, lo_y, hi_y
            gx, gy = np.meshgrid(np.linspace(xlo, xhi, 300), np.linspace(ylo, yhi, 300))
            gs = rule_scores(name, gx.ravel(), gy.ravel(), heads).reshape(gx.shape)
            ax.contourf(gx, gy, gs, levels=[-1, THRESHOLD, 2], colors=["#f6d5d5", "#d9e8f5"], alpha=0.7)
            ax.contour(gx, gy, gs, levels=[THRESHOLD], colors="k", linewidths=1.3)
        ax.scatter(xv[lab == 1], yv[lab == 1], s=15, c="#1f77b4", label="Real", alpha=0.75, edgecolors="none")
        ax.scatter(xv[lab == 0], yv[lab == 0], s=20, c="#d62728", label="Fake", alpha=0.9, marker="^", edgecolors="none")
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)

        # A single shared label per side (set below via supxlabel/supylabel);
        # only tick labels on the outer edge. The feature-fusion panel lives in a
        # different space, so its axes are named in its title instead.
        row, col = divmod(k, 3)
        if row == 0 and not (row == 0 and col == 2):
            ax.tick_params(labelbottom=False)
        if col != 0 and not is_ff:
            ax.tick_params(labelleft=False)

        m = summary.loc[summary["method"] == name].iloc[0]
        title = "feature-fusion (own axes: image-only vs. text-only logit)" if is_ff else name
        ax.set_title(f"{title}\nAUC {m.AUC:.3f} | F1 {m.F1:.3f} | Acc {m.Acc:.3f}",
                     fontsize=11 if is_ff else 13)
    axes[0].legend(loc="lower left", frameon=True)
    fig.supxlabel("image logit", fontsize=14)
    fig.supylabel("text logit", fontsize=14)
    fig.tight_layout(h_pad=1.0, w_pad=0.6)

    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"clean_score_space.{ext}"), dpi=200, bbox_inches="tight")

    summary.round(4).to_csv(os.path.join(RESULT_PATH, "fusion_analysis", "clean_fusion_metrics.csv"), index=False)
    write_clean_table(summary.round(4))
    if any(hasattr(h, "best_params_") for h in heads.values()):
        with open(os.path.join(RESULT_PATH, "fusion_analysis", "svm_selection.json"), "w") as f:
            json.dump({n: {"best_params": {k: str(v) for k, v in h.best_params_.items()},
                           "cv_roc_auc": float(h.best_score_)}
                       for n, h in heads.items() if hasattr(h, "best_params_")}, f, indent=4)


if __name__ == "__main__":
    main()
