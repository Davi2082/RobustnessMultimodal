"""Learned late-fusion heads (svm-rbf, linear), fitted on the validation split.

The unimodal models memorise the training split (a head fitted on their train
outputs reaches train AUC 1.0), so, as in stacking, the heads are fitted on
held-out outputs: the non-augmented validation split, scored beforehand by
``scripts.utils.eval --split val``. The input is [text, image] in the space
``late_fusion_input`` of config.yaml (logits), standardized inside the head.

* svm-rbf: StandardScaler + RBF SVC with fixed C/gamma and Platt probabilities.
* linear:  Linear(2, 1) on standardized features, trained with cross-entropy
           (Adam, ``epochs`` x mini-batches of ``batch_size``), saved as a
           state dict with the standardization folded in.

``out_of_fold_scores`` gives the cross-fitted validation scores used to tune
the decision thresholds (scripts.main_scripts.compute_thresholds).
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from configuration_files.configuration import (
    BATCH_SIZE, HEAD_CV_FOLDS, LATE_FUSION_INPUT, LINEAR_EPOCHS, LINEAR_LEARNING_RATE,
    SVM_RBF_C, SVM_RBF_GAMMA,
)
from configuration_files.paths import dataset_result_root
from models.fusion import (
    DifferentiableRBFSVMFusion, build_linear_fusion_module, fusion_head_path, head_metadata_path,
)

LEARNED_HEADS = ("svm-rbf", "linear")


def val_csv(dataset, method):
    return os.path.join(dataset_result_root(dataset), "val", method, "results.csv")


def validation_outputs(dataset):
    """(labels, text frame, image frame) of the scored validation split."""
    paths = {m: val_csv(dataset, m) for m in ("text", "image")}
    missing = [p for p in paths.values() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"missing validation scores {missing}: run "
            f"python3 -m scripts.main_scripts.compute_thresholds --dataset {dataset}")
    t, i = pd.read_csv(paths["text"]), pd.read_csv(paths["image"])
    assert (t["index"].values == i["index"].values).all(), "text/image validation rows differ"
    assert (t["label"].values == i["label"].values).all()
    return t["label"].values, t, i


def head_features(text, image, space=LATE_FUSION_INPUT):
    """[text, image] head input from two frames with 'score' and 'logit' columns."""
    col = "logit" if space == "logits" else "score"
    return np.column_stack([np.asarray(text[col], dtype=float), np.asarray(image[col], dtype=float)])


def fit_head(rule, X, y, seed):
    """Fit one head; returns the object that is persisted (pipeline or state dict)."""
    if rule == "svm-rbf":
        gamma = SVM_RBF_GAMMA if isinstance(SVM_RBF_GAMMA, str) else float(SVM_RBF_GAMMA)
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=SVM_RBF_C, gamma=gamma,
                                                   probability=True, random_state=seed)).fit(X, y)
    if rule == "linear":
        torch.manual_seed(seed)
        Xt = torch.as_tensor(X, dtype=torch.float32)
        yt = torch.as_tensor(y, dtype=torch.float32)
        mean, std = Xt.mean(dim=0), Xt.std(dim=0).clamp_min(1e-8)
        linear = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(linear.parameters(), lr=LINEAR_LEARNING_RATE)
        criterion = torch.nn.BCEWithLogitsLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset((Xt - mean) / std, yt), batch_size=BATCH_SIZE,
            shuffle=True, generator=torch.Generator().manual_seed(seed))
        for _ in range(LINEAR_EPOCHS):
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(linear(xb).squeeze(-1), yb).backward()
                optimizer.step()
        # Fold the standardization in, so the module takes raw [text, image] features.
        with torch.no_grad():
            weight = linear.weight[0] / std
            bias = linear.bias[0] - (weight * mean).sum()
        return {"weight": weight.unsqueeze(0).clone(), "bias": bias.reshape(1).clone()}
    raise ValueError(f"not a learned head: {rule}")


def head_scores(rule, fitted, X):
    """P(Real) from the same differentiable module the pipeline and attacks use."""
    module = (DifferentiableRBFSVMFusion(fitted) if rule == "svm-rbf"
              else build_linear_fusion_module(fitted)).eval()
    with torch.no_grad():
        return module(torch.as_tensor(X, dtype=torch.float32)).numpy().reshape(-1)


def out_of_fold_scores(rule, X, y, seed, folds=HEAD_CV_FOLDS):
    """Cross-fitted scores: each sample is scored by a head fitted without it."""
    oof = np.empty(len(y))
    for train, held in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        oof[held] = head_scores(rule, fit_head(rule, X[train], y[train], seed), X[held])
    return oof


def fit_and_save(rule, dataset, seed):
    """Fit the deployed head on the whole validation split and persist it."""
    y, t, i = validation_outputs(dataset)
    X = head_features(t, i)
    fitted = fit_head(rule, X, y, seed)
    path = fusion_head_path(rule, dataset)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if rule == "svm-rbf":
        joblib.dump(fitted, path)
        params = {"C": SVM_RBF_C, "gamma": SVM_RBF_GAMMA}
    else:
        torch.save(fitted, path)
        params = {"model": "Linear(2, 1), cross-entropy, Adam", "epochs": LINEAR_EPOCHS,
                  "learning_rate": LINEAR_LEARNING_RATE, "batch_size": BATCH_SIZE}
    oof_auc = roc_auc_score(y, out_of_fold_scores(rule, X, y, seed))

    meta_path = head_metadata_path(dataset)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    meta.update({"split": "val (non-augmented)", "input_space": LATE_FUSION_INPUT,
                 "seed": seed, "n_samples": int(len(y))})
    meta.setdefault("heads", {})[rule] = {
        "params": {k: str(v) for k, v in params.items()},
        "val_roc_auc_in_sample": round(float(roc_auc_score(y, head_scores(rule, fitted, X))), 4),
        f"val_roc_auc_{HEAD_CV_FOLDS}fold_oof": round(float(oof_auc), 4),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
        f.write("\n")
    print(f"wrote {path}  ({rule} on validation {LATE_FUSION_INPUT}, n={len(y)}, "
          f"OOF AUC {oof_auc:.4f})")
