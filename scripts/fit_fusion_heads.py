"""Fit and persist the learned late-fusion heads (svm-rbf, linear).

The only place these are fitted; the clean analysis and the attacks both load
them from disk. Default --split val: the unimodal checkpoints memorised train,
where a fitted head scores a meaningless CV 1.0 and does not transfer.

Writes svm_rbf_head.pkl, linear_head.pkl and fusion_heads.json.
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from tqdm import tqdm

from configuration_files.configuration import (
    BATCH_SIZE,
    DEVICE_EVAL,
    IMAGE_WEIGHTS_PATH,
    LATE_FUSION_SVM_SEED,
    N_TOKENS,
    NAME_IMG_EMBED,
    NAME_LLM,
    TEXT_WEIGHTS_PATH,
)
from configuration_files.paths import RESULT_PATH
from data_loading import my_datasets
from models.fusion import HEAD_METADATA, fusion_head_path
from utils import load_available_datasets, load_model

SPLIT_FILES = {
    "train": "train.csv",
    "val": "val_augmented.csv",
}
CS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def split_path(dataset: str, split: str) -> str:
    """Annotation file for a split, tolerating the two data roots in use."""
    name = SPLIT_FILES[split]
    for root in (f"data/{dataset}", f"data_loading/{dataset}"):
        candidate = os.path.join(root, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No {name} under data/{dataset} or data_loading/{dataset}")


def images_dir(dataset: str) -> str:
    for root in (f"data/{dataset}", f"data_loading/{dataset}"):
        candidate = os.path.join(root, "images")
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"No images directory for {dataset}")


def unimodal_logits(args, device, dataset_classes, load_functions) -> pd.DataFrame:
    """Clean logits of both unimodal branches on one split, cached to disk."""
    cache = os.path.join(
        RESULT_PATH, "fusion_analysis", f"{args.split}_unimodal_logits_clean.csv"
    )
    if os.path.exists(cache) and not args.force:
        print(f"using cached {cache}")
        return pd.read_csv(cache)

    annotations = split_path(args.dataset, args.split)
    columns = {}
    for modality, weights, encoder in (
        ("text", TEXT_WEIGHTS_PATH, NAME_IMG_EMBED),
        ("image", IMAGE_WEIGHTS_PATH, NAME_IMG_EMBED),
    ):
        args.modality, args.model_path, args.name_img_embed = modality, weights, encoder
        model, tokenizer, processor = load_model(device, args)
        model.eval()

        dataset = my_datasets.get_dataset(
            dataset_classes[args.dataset],
            load_functions[args.dataset],
            args.n_tokens,
            processor,
            tokenizer,
            annotations,
            images_dir(args.dataset),
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        logits, labels = [], []
        with torch.no_grad():
            for images, y, texts, _, _ in tqdm(
                loader, desc=f"{args.split}/{modality}"
            ):
                images, texts = images.to(device), texts.to(device)
                if modality == "text":
                    _, batch = model(images=None, texts=texts)
                else:
                    _, batch = model(images=images, texts=None)
                logits.append(batch.detach().cpu().flatten())
                labels.append(y)

        columns[modality] = torch.cat(logits).numpy()
        columns["label"] = torch.cat(labels).numpy()
        del model
        torch.cuda.empty_cache()

    frame = pd.DataFrame(
        {
            "label": columns["label"],
            "text_logit": columns["text"],
            "image_logit": columns["image"],
        }
    )
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    frame.to_csv(cache, index=False)
    print(f"wrote {cache}")
    return frame


def fit_heads(frame: pd.DataFrame, seed: int, input_space: str):
    """Fit both learned heads with cross-validated regularisation."""
    if input_space == "logits":
        features = frame[["image_logit", "text_logit"]].values
    else:
        features = 1.0 / (
            1.0 + np.exp(-frame[["image_logit", "text_logit"]].values)
        )
    labels = frame["label"].values  # 1 = Real (positive class)
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)

    svm_pipe = make_pipeline(
        StandardScaler(), SVC(kernel="rbf", probability=True, random_state=seed)
    )
    svm_pipe.steps[1] = ("svc", svm_pipe.steps[1][1])
    svm = GridSearchCV(
        svm_pipe,
        {"svc__C": CS, "svc__gamma": ["scale", 0.01, 0.1, 1.0]},
        scoring="roc_auc",
        cv=folds,
        n_jobs=-1,
    ).fit(features, labels)

    linear_pipe = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed)
    )
    linear_pipe.steps[1] = ("lr", linear_pipe.steps[1][1])
    linear = GridSearchCV(
        linear_pipe, {"lr__C": CS}, scoring="roc_auc", cv=folds, n_jobs=-1
    ).fit(features, labels)

    return {"svm-rbf": svm, "linear": linear}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Recovery")
    parser.add_argument(
        "--split",
        choices=tuple(SPLIT_FILES),
        default="val",
        help=(
            "Split the heads are fitted on. 'train' reproduces the earlier "
            "analysis but the unimodal models memorised it."
        ),
    )
    parser.add_argument(
        "--input-space",
        choices=("logits", "scores"),
        default="logits",
        help="Feature space of the head; recorded so attacks use the same one.",
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n_tokens", type=int, default=N_TOKENS)
    parser.add_argument("--name_llm", default=NAME_LLM)
    parser.add_argument("--seed", type=int, default=LATE_FUSION_SVM_SEED)
    parser.add_argument("--device", default=DEVICE_EVAL)
    parser.add_argument("--force", action="store_true", help="Recompute logits.")
    parser.add_argument("--set_params", type=bool, default=True)
    parser.add_argument("--merge_tokens", type=int, default=0)
    parser.add_argument("--use_lora", type=bool, default=True)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dataset_classes, load_functions = load_available_datasets()

    frame = unimodal_logits(args, device, dataset_classes, load_functions)
    print(
        f"fitting on {len(frame)} {args.split} samples "
        f"({int((frame.label == 0).sum())} fake / {int((frame.label == 1).sum())} real)"
    )

    heads = fit_heads(frame, args.seed, args.input_space)

    metadata = {
        "split": args.split,
        "input_space": args.input_space,
        "seed": args.seed,
        "n_samples": int(len(frame)),
        "heads": {},
    }
    for rule, head in heads.items():
        path = fusion_head_path(rule)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(head.best_estimator_, path)
        metadata["heads"][rule] = {
            "best_params": {k: str(v) for k, v in head.best_params_.items()},
            "cv_roc_auc": round(float(head.best_score_), 4),
        }
        print(f"wrote {path}  cv_roc_auc={head.best_score_:.4f}  {head.best_params_}")

    with open(HEAD_METADATA, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)
        handle.write("\n")
    print(f"wrote {HEAD_METADATA}")


if __name__ == "__main__":
    main()
