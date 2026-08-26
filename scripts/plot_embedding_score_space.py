"""Embedding score space: modality-pooled embeddings instead of unimodal logits.

The logit score space (``plot_clean_score_space.py``) cannot host the
feature-fusion model: its decision is not a function of the two unimodal
classifiers' logits, so that panel is stranded on its own ablation axes with no
decision region.

This figure changes the axes to fix that. Themis pools with ``x.mean(dim=1)``
over the concatenated ``[image_tokens ; text_tokens]`` sequence, so the pooled
embedding the classifier head actually consumes decomposes exactly into an
image-token mean and a text-token mean::

    pooled = (n_img * img_pooled + n_txt * txt_pooled) / (n_img + n_txt)

Pushing each half through the classifier head gives two coordinates in logit
units:

    x = lm_head(img_pooled)     y = lm_head(txt_pooled)

Every variant has these coordinates, because every variant uses the same
pooling. For a UNIMODAL checkpoint the sequence is single-modality, so its
coordinate collapses to exactly the logit already plotted -- the late-fusion
panels are unchanged. For the FEATURE-FUSION checkpoint the two coordinates are
internal to the joint model, which is what finally makes its decision region
drawable.

The head is LayerNorm + Linear, and LayerNorm's per-sample mean and scale are
not functions of the two coordinates alone, so the reconstruction is close but
not algebraically exact. How close is measured and reported rather than
assumed.

Usage:
    python3 plot_embedding_score_space.py
    python3 plot_embedding_score_space.py --force     # recompute cached axes
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
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, r2_score

from data_loading import my_datasets
from utils import load_available_datasets, load_model
from configuration_files.configuration import (
    NAME_LLM, NAME_IMG_EMBED, BATCH_SIZE, N_TOKENS, THRESHOLD, DEVICE_EVAL,
    LATE_FUSION_SVM_SEED,
)
from configuration_files.paths import RESULT_PATH

TRAIN_CSV = "data/Recovery/train.csv"
ANALYSIS_DIR = os.path.join(RESULT_PATH, "fusion_analysis")
FIG_DIR = "figures/classification_results/scatter"


def ff_modality_axes(args, device, dataset_classes, load_functions, split, csv_path):
    """Head-projected image-token and text-token pooled embeddings, feature fusion.

    Returns one row per sample: the two coordinates, the token counts behind
    them, and the model's real joint logit for verification.
    """
    cache = os.path.join(ANALYSIS_DIR, f"ff_embedding_axes_{split}.csv")
    if os.path.exists(cache) and not args.force:
        return pd.read_csv(cache)

    # load_model forces the feature-fusion encoder/checkpoint for this modality
    args.modality, args.model_path, args.name_img_embed = "feature-fusion", None, None
    model, tokenizer, processor = load_model(device, args)
    model.eval()
    ds = my_datasets.get_dataset(
        dataset_classes[args.dataset], load_functions[args.dataset],
        args.n_tokens, processor, tokenizer, csv_path, f"data/{args.dataset}/images",
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    cols = {k: [] for k in ("index", "label", "img_axis", "txt_axis", "joint_logit")}
    n_img = n_txt = None
    with torch.no_grad():
        for images, y, texts, _, idx in tqdm(loader, desc=f"{split}/ff-embedding-axes"):
            images, texts = images.to(device), texts.to(device)
            img_pooled, txt_pooled, n_i, n_t = model.modality_pooled_features(
                images=images, texts=texts)
            n_img, n_txt = n_i, n_t
            cols["img_axis"].append(model.lm_head(img_pooled).detach().cpu().flatten())
            cols["txt_axis"].append(model.lm_head(txt_pooled).detach().cpu().flatten())
            _, lg = model(images=images, texts=texts)
            cols["joint_logit"].append(lg.detach().cpu().flatten())
            cols["label"].append(y)
            cols["index"].append(idx)
    del model
    torch.cuda.empty_cache()

    df = pd.DataFrame({k: torch.cat(v).numpy() for k, v in cols.items()})
    df["n_img"], df["n_txt"] = n_img, n_txt
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def unimodal_train_logits(args, device, dataset_classes, load_functions):
    """Clean train-split logits from the text-only and image-only checkpoints.

    For a unimodal checkpoint the pooled embedding covers a single-modality
    sequence, so lm_head(pooled) IS the logit -- these columns are already the
    embedding coordinates for the late-fusion panels.
    """
    from configuration import TEXT_WEIGHTS_PATH, IMAGE_WEIGHTS_PATH
    cache = os.path.join(ANALYSIS_DIR, "train_unimodal_logits_clean.csv")
    if os.path.exists(cache) and not args.force:
        return pd.read_csv(cache)

    out = {}
    for modality, weights in (("text", TEXT_WEIGHTS_PATH), ("image", IMAGE_WEIGHTS_PATH)):
        args.modality, args.model_path = modality, weights
        args.name_img_embed = NAME_IMG_EMBED
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
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def fit_heads(train_df, seed):
    """Learned late-fusion heads on the clean train-split unimodal coordinates."""
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


def fit_ff_boundary(train_ff, seed):
    """Decision region of the feature-fusion model in its own embedding axes.

    Two readings of the same boundary, both fitted on the train split against
    the model's OWN decision (not the labels):

    ``analytic`` reproduces the pooling identity -- the count-weighted average
    of the two coordinates, calibrated to a probability by 1-D logistic
    regression. Its shape is fixed by the token counts, not fitted, so the line
    is a property of the architecture.

    ``free`` lets an RBF-SVM place the boundary anywhere in the plane. The gap
    between the two says how much the LayerNorm in the head bends what the
    pooling identity predicts.
    """
    w = train_ff["n_img"].iloc[0] / (train_ff["n_img"].iloc[0] + train_ff["n_txt"].iloc[0])
    call = (train_ff["joint_logit"].values > 0).astype(int)
    XY = train_ff[["img_axis", "txt_axis"]].values

    analytic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(
        (w * XY[:, 0] + (1 - w) * XY[:, 1]).reshape(-1, 1), call)

    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    free_pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=seed))
    free_pipe.steps[1] = ("svc", free_pipe.steps[1][1])
    free = GridSearchCV(free_pipe, {"svc__C": [0.1, 1.0, 10.0, 100.0],
                                    "svc__gamma": ["scale", 0.01, 0.1, 1.0]},
                        scoring="roc_auc", cv=cv, n_jobs=-1).fit(XY, call)
    return {"analytic": analytic, "free": free, "w": float(w)}


def rule_scores(name, img, txt, heads=None, ff=None):
    """P(Real) for a late-fusion rule, or P(model says Real) for feature fusion."""
    if name == "feature-fusion":
        return ff["analytic"].predict_proba(
            (ff["w"] * img + (1 - ff["w"]) * txt).reshape(-1, 1))[:, 1]
    if name == "feature-fusion-free":
        return ff["free"].best_estimator_.predict_proba(np.column_stack([img, txt]))[:, 1]
    pi, pt = 1 / (1 + np.exp(-img)), 1 / (1 + np.exp(-txt))
    if name == "min":
        return np.minimum(pi, pt)
    if name == "mean":
        return (pi + pt) / 2
    if name == "max":
        return np.maximum(pi, pt)
    if name in ("svm-rbf", "linear"):
        return heads[name].best_estimator_.predict_proba(np.column_stack([img, txt]))[:, 1]
    raise ValueError(name)


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
    p.add_argument("--force", action="store_true", help="recompute cached axes")
    args = p.parse_args()

    device = torch.device(DEVICE_EVAL)
    test_csv = glob.glob(f"data/{args.dataset}/test.*")[0]

    train_uni = unimodal_train_logits(args, device, dataset_classes, load_functions)
    train_ff = ff_modality_axes(args, device, dataset_classes, load_functions, "train", TRAIN_CSV)
    test_ff = ff_modality_axes(args, device, dataset_classes, load_functions, "test", test_csv)

    heads = fit_heads(train_uni, LATE_FUSION_SVM_SEED)
    ff = fit_ff_boundary(train_ff, LATE_FUSION_SVM_SEED)
    for name, h in heads.items():
        print(f"{name}: best={h.best_params_}  CV ROC-AUC={h.best_score_:.4f}")
    print(f"feature-fusion pooling weight on image tokens: w={ff['w']:.4f} "
          f"({train_ff['n_img'].iloc[0]} image vs {train_ff['n_txt'].iloc[0]} text tokens)")

    # ---- test-set unimodal coordinates (== the unimodal logits) ----
    B = os.path.join(RESULT_PATH, "clean")
    t = pd.read_csv(os.path.join(B, "text", "results.csv"))
    i = pd.read_csv(os.path.join(B, "image", "results.csv"))
    te = t.merge(i, on="index", suffixes=("_txt", "_img"))
    assert (te["label_txt"] == te["label_img"]).all()
    y = te["label_txt"].values
    txt, img = te["logit_txt"].values, te["logit_img"].values

    ff_img, ff_txt = test_ff["img_axis"].values, test_ff["txt_axis"].values
    ff_y, ff_call = test_ff["label"].values, (test_ff["joint_logit"].values > 0).astype(int)

    rows = []
    for rule in ("min", "mean", "max", "svm-rbf", "linear"):
        s = rule_scores(rule, img, txt, heads)
        yp = (s > THRESHOLD).astype(int)
        rows.append({"method": rule, "AUC": roc_auc_score(y, s),
                     "F1": f1_score(y, yp), "Acc": accuracy_score(y, yp)})
    ff_s = 1 / (1 + np.exp(-test_ff["joint_logit"].values))
    rows.append({"method": "feature-fusion", "AUC": roc_auc_score(ff_y, ff_s),
                 "F1": f1_score(ff_y, ff_call), "Acc": accuracy_score(ff_y, ff_call)})
    summary = pd.DataFrame(rows)

    # ---- does the pooling identity actually survive the head's LayerNorm? ----
    recon = ff["w"] * ff_img + (1 - ff["w"]) * ff_txt
    fidelity = {
        "n_test": int(len(test_ff)),
        "pooling_weight_image": ff["w"],
        "n_img_tokens": int(test_ff["n_img"].iloc[0]),
        "n_txt_tokens": int(test_ff["n_txt"].iloc[0]),
        "analytic_r2_vs_joint_logit": float(r2_score(test_ff["joint_logit"].values, recon)),
        "analytic_spearman": float(pd.Series(recon).corr(
            pd.Series(test_ff["joint_logit"].values), method="spearman")),
        "analytic_agreement": float(accuracy_score(
            ff_call, (rule_scores("feature-fusion", ff_img, ff_txt, ff=ff) > THRESHOLD).astype(int))),
        "free_agreement": float(accuracy_score(
            ff_call, (rule_scores("feature-fusion-free", ff_img, ff_txt, ff=ff) > THRESHOLD).astype(int))),
    }
    print(f"\nfeature-fusion boundary fidelity to its own decision (test, n={fidelity['n_test']}):")
    print(f"  analytic (pooling identity): agreement={fidelity['analytic_agreement']:.3f}  "
          f"Spearman(recon, joint logit)={fidelity['analytic_spearman']:.3f}  "
          f"R2={fidelity['analytic_r2_vs_joint_logit']:.3f}")
    print(f"  free 2-D RBF-SVM:            agreement={fidelity['free_agreement']:.3f}")

    # ---- figure ----
    panels = [("min", img, txt, y), ("mean", img, txt, y), ("max", img, txt, y),
              ("svm-rbf", img, txt, y), ("linear", img, txt, y),
              ("feature-fusion", ff_img, ff_txt, ff_y)]

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    })
    lo_x, hi_x = img.min() - 1, img.max() + 1
    lo_y, hi_y = txt.min() - 1, txt.max() + 1

    fig, axes2d = plt.subplots(2, 3, figsize=(13.5, 8.8))
    axes = axes2d.ravel()
    for k, (ax, (name, xv, yv, lab)) in enumerate(zip(axes, panels)):
        is_ff = name == "feature-fusion"
        if is_ff:
            pad_x, pad_y = 0.05 * np.ptp(xv), 0.05 * np.ptp(yv)
            xlo, xhi = xv.min() - pad_x, xv.max() + pad_x
            ylo, yhi = yv.min() - pad_y, yv.max() + pad_y
        else:
            xlo, xhi, ylo, yhi = lo_x, hi_x, lo_y, hi_y
        gx, gy = np.meshgrid(np.linspace(xlo, xhi, 300), np.linspace(ylo, yhi, 300))
        gs = rule_scores(name, gx.ravel(), gy.ravel(), heads, ff).reshape(gx.shape)
        ax.contourf(gx, gy, gs, levels=[-1, THRESHOLD, 2], colors=["#f6d5d5", "#d9e8f5"], alpha=0.7)
        ax.contour(gx, gy, gs, levels=[THRESHOLD], colors="k", linewidths=1.3)
        if is_ff:
            # the free-form fit, for comparison against the analytic line
            gf = rule_scores("feature-fusion-free", gx.ravel(), gy.ravel(), ff=ff).reshape(gx.shape)
            ax.contour(gx, gy, gf, levels=[THRESHOLD], colors="k",
                       linewidths=1.1, linestyles="--")
        ax.scatter(xv[lab == 1], yv[lab == 1], s=15, c="#1f77b4", label="Real", alpha=0.75, edgecolors="none")
        ax.scatter(xv[lab == 0], yv[lab == 0], s=20, c="#d62728", label="Fake", alpha=0.9, marker="^", edgecolors="none")
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)

        row, col = divmod(k, 3)
        if row == 0 and col != 2:
            ax.tick_params(labelbottom=False)
        if col != 0 and not is_ff:
            ax.tick_params(labelleft=False)

        m = summary.loc[summary["method"] == name].iloc[0]
        sub = f"AUC {m.AUC:.3f} | F1 {m.F1:.3f} | Acc {m.Acc:.3f}"
        if is_ff:
            title = "feature-fusion (own pooled embedding)"
            sub += (f"\nsolid: pooling identity, {fidelity['analytic_agreement']:.3f} agreement"
                    f"  |  dashed: free fit, {fidelity['free_agreement']:.3f}")
        else:
            title = name
        ax.set_title(f"{title}\n{sub}", fontsize=10 if is_ff else 13)
    axes[0].legend(loc="lower left", frameon=True)
    fig.supxlabel("lm_head(mean-pooled VISION embedding)   [logit units]", fontsize=14)
    fig.supylabel("lm_head(mean-pooled TEXT embedding)", fontsize=14)
    fig.tight_layout(h_pad=1.0, w_pad=0.6)

    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"embedding_score_space.{ext}"), dpi=200, bbox_inches="tight")

    summary.round(4).to_csv(os.path.join(ANALYSIS_DIR, "embedding_space_metrics.csv"), index=False)
    with open(os.path.join(ANALYSIS_DIR, "embedding_space_fidelity.json"), "w") as f:
        json.dump(fidelity, f, indent=4)

    print(f"\nSaved {FIG_DIR}/embedding_score_space.png|pdf\n")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
