"""Train a text, image, feature-fusion, or late-fusion (svm-rbf/linear) classifier.

Neural models (text, image, feature-fusion) are trained via backpropagation.
Learned fusion heads (svm-rbf, linear) are fitted on unimodal scores from the
validation split and require existing text + image checkpoints.

Usage:
    python3 -m scripts.train --model text --dataset Recovery
    python3 -m scripts.train --model image --dataset Recovery
    python3 -m scripts.train --model feature-fusion --dataset Recovery
    python3 -m scripts.train --model svm-rbf --dataset Recovery
    python3 -m scripts.train --model linear --dataset Recovery
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from tqdm import tqdm

from configuration_files.configuration import (
    BATCH_SIZE, DATASET, FF_WEIGHTS_PATH, IMAGE_WEIGHTS_PATH,
    NAME_IMG_EMBED, NAME_LLM, N_TOKENS, RAND_SEED, TEXT_WEIGHTS_PATH,
)
from configuration_files.paths import DATASET_WEIGHTS_DIR
from data_loading import my_datasets
from models.fusion import HEAD_METADATA, fusion_head_path
from models.themis_model import get_Themis
from utils import load_available_datasets, load_model

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_loading"

NEURAL_MODELS = ("text", "image", "feature-fusion")
FUSION_HEADS = ("svm-rbf", "linear")
MODEL_CHOICES = NEURAL_MODELS + FUSION_HEADS

CHECKPOINT_NAMES = {
    "text": "best_text_only.pt",
    "image": "best_img_only.pt",
    "feature-fusion": "best_feature_fusion.pt",
}

CS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


# ── Dataset discovery ──

def discover_available_datasets():
    if not DATA_ROOT.is_dir():
        return []
    return sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))


def resolve_dataset_implementation(dataset_name):
    class_name = f"{dataset_name}_Dataset"
    loader_name = f"{dataset_name.lower()}_load_annotations_file"
    try:
        return getattr(my_datasets, class_name), getattr(my_datasets, loader_name)
    except AttributeError as error:
        raise ValueError(
            f"Dataset {dataset_name!r} requires {class_name} and {loader_name} in data_loading/my_datasets.py."
        ) from error


def first_existing(dataset_dir, names, split):
    for name in names:
        path = dataset_dir / name
        if path.is_file():
            return path
    expected = ", ".join(names)
    raise FileNotFoundError(f"No {split} annotation file found in {dataset_dir}. Expected one of: {expected}")


def resolve_dataset_assets(dataset_name):
    dataset_dir = DATA_ROOT / dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    dataset_class, annotation_loader = resolve_dataset_implementation(dataset_name)
    train_file = first_existing(dataset_dir, ("train_augmented.csv", "train_augmented.tsv", "train.csv", "train.tsv"), "training")
    val_file = first_existing(dataset_dir, ("val_augmented.csv", "val_augmented.tsv", "val.csv", "val.tsv"), "validation")
    image_dir = next((p for p in (dataset_dir / "images", dataset_dir / "images_copy") if p.is_dir()), None)
    if image_dir is None:
        raise FileNotFoundError(f"No image directory found in {dataset_dir}")
    return dataset_class, annotation_loader, train_file, val_file, image_dir


# ── Neural training helpers ──

def model_inputs(model_type, images, texts):
    if model_type == "text":
        return None, texts
    if model_type == "image":
        return images, None
    return images, texts


def evaluate(model, loader, criterion, device, model_type):
    model.eval()
    running_loss = 0.0
    labels_all, predictions_all = [], []
    with torch.no_grad():
        for images, labels, texts, _, _ in tqdm(loader, desc="Validation", leave=False):
            images, labels, texts = images.to(device), labels.to(device), texts.to(device)
            image_input, text_input = model_inputs(model_type, images, texts)
            scores, _ = model(images=image_input, texts=text_input)
            running_loss += criterion(scores.float(), labels.float().unsqueeze(1)).item()
            labels_all.extend(labels.cpu().numpy())
            predictions_all.extend((scores.detach().cpu().numpy().reshape(-1) > 0.5).astype(int))
    return {
        "loss": running_loss / len(loader),
        "accuracy": accuracy_score(labels_all, predictions_all),
        "precision": precision_score(labels_all, predictions_all, zero_division=0),
        "recall": recall_score(labels_all, predictions_all, zero_division=0),
        "f1": f1_score(labels_all, predictions_all, zero_division=0),
    }


def train_neural(args, dataset_class, annotation_loader, train_file, val_file, image_dir):
    """Train a text, image, or feature-fusion Themis checkpoint."""
    image_encoder = args.name_img_embed or NAME_IMG_EMBED
    merge_tokens = args.merge_tokens or None
    device = torch.device(args.device)

    print(f"  Image encoder: {image_encoder}")

    model, tokenizer, processor = get_Themis(
        name_llm=args.name_llm, name_img_embed=image_encoder,
        use_lora=args.use_lora, is_pythia="pythia" in args.name_llm.lower(),
        lora_alpha=args.lora_alpha, lora_r=args.lora_r,
        lora_dropout=args.lora_dropout, merge_tokens=merge_tokens,
    )
    model.to(device)

    train_dataset = my_datasets.get_dataset(dataset_class, annotation_loader, args.n_tokens, processor, tokenizer, str(train_file), str(image_dir))
    val_dataset = my_datasets.get_dataset(dataset_class, annotation_loader, args.n_tokens, processor, tokenizer, str(val_file), str(image_dir))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    checkpoint = Path(DATASET_WEIGHTS_DIR) / CHECKPOINT_NAMES[args.model]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_f1 = -1.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels, texts, _, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images, labels, texts = images.to(device), labels.to(device), texts.to(device)
            image_input, text_input = model_inputs(args.model, images, texts)
            optimizer.zero_grad()
            scores, _ = model(images=image_input, texts=text_input)
            loss = criterion(scores.float(), labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        train_loss = running_loss / len(train_loader)
        metrics = evaluate(model, val_loader, criterion, device, args.model)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(metrics["loss"])
        history["val_accuracy"].append(metrics["accuracy"])
        print(f"Epoch {epoch + 1}: train loss={train_loss:.4f}, val loss={metrics['loss']:.4f}, F1={metrics['f1']:.4f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), checkpoint)
            metadata = {
                "name_llm": args.name_llm, "name_img_embed": image_encoder,
                "merge_tokens": args.merge_tokens, "lora_alpha": args.lora_alpha,
                "lora_r": args.lora_r, "lora_dropout": args.lora_dropout, "use_lora": args.use_lora,
            }
            checkpoint.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"  Saved new best checkpoint: {checkpoint}")

    plot_dir = ROOT / "results" / args.dataset / "training" / args.model
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.plot(history["train_loss"], label="training loss")
    plt.plot(history["val_loss"], label="validation loss")
    plt.plot(history["val_accuracy"], label="validation accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "training_history.png")
    plt.close()
    print(f"Training completed. Best validation F1: {best_f1:.4f}")
    print(f"Checkpoint: {checkpoint}")


# ── Fusion head fitting ──

def require_unimodal_checkpoints(dataset):
    """Check text + image checkpoints exist before fitting a fusion head."""
    weights_dir = Path(DATASET_WEIGHTS_DIR)
    text_ckpt = weights_dir / CHECKPOINT_NAMES["text"]
    image_ckpt = weights_dir / CHECKPOINT_NAMES["image"]
    missing = []
    if not text_ckpt.exists():
        missing.append(f"text: {text_ckpt}")
    if not image_ckpt.exists():
        missing.append(f"image: {image_ckpt}")
    if missing:
        raise FileNotFoundError(
            "Unimodal checkpoints required to fit fusion heads:\n  "
            + "\n  ".join(missing)
            + "\nTrain them first: python3 -m scripts.train --model text/image"
        )


def collect_unimodal_scores(args, dataset_class, annotation_loader, val_file, image_dir):
    """Run both unimodal models on val to get scores for fitting."""
    device = torch.device(args.device)
    dataset_classes, load_functions = load_available_datasets()

    all_labels, all_text_scores, all_image_scores = [], [], []

    for modality in ("text", "image"):
        ns = argparse.Namespace(
            modality=modality, name_llm=args.name_llm, name_img_embed=NAME_IMG_EMBED,
            model_path=None, n_tokens=args.n_tokens, merge_tokens=0,
            lora_alpha=8, lora_r=8, lora_dropout=0.4, use_lora=True, set_params=True,
        )
        model, tokenizer, processor = load_model(device, ns)
        model.eval()

        dataset = my_datasets.get_dataset(
            dataset_class, annotation_loader, args.n_tokens,
            processor, tokenizer, str(val_file), str(image_dir),
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        scores_list, labels_list = [], []
        with torch.no_grad():
            for images, labels, texts, _, _ in tqdm(loader, desc=f"val/{modality}"):
                images, texts = images.to(device), texts.to(device)
                if modality == "text":
                    batch_scores, _ = model(images=None, texts=texts)
                else:
                    batch_scores, _ = model(images=images, texts=None)
                scores_list.append(batch_scores.detach().cpu().flatten())
                labels_list.append(labels)

        if modality == "text":
            all_text_scores = torch.cat(scores_list).numpy()
            all_labels = torch.cat(labels_list).numpy()
        else:
            all_image_scores = torch.cat(scores_list).numpy()

        del model
        torch.cuda.empty_cache()

    return all_labels, all_text_scores, all_image_scores


def fit_fusion_head(args, dataset_class, annotation_loader, val_file, image_dir):
    """Fit an SVM-RBF or linear fusion head on unimodal scores."""
    require_unimodal_checkpoints(args.dataset)

    labels, text_scores, image_scores = collect_unimodal_scores(
        args, dataset_class, annotation_loader, val_file, image_dir,
    )
    features = np.column_stack([text_scores, image_scores])
    n_fake = int((labels == 0).sum())
    n_real = int((labels == 1).sum())
    print(f"Fitting on {len(labels)} val samples ({n_fake} fake / {n_real} real)")

    seed = args.seed
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)

    if args.model == "svm-rbf":
        pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=seed))
        pipe.steps[1] = ("svc", pipe.steps[1][1])
        head = GridSearchCV(
            pipe, {"svc__C": CS, "svc__gamma": ["scale", 0.01, 0.1, 1.0]},
            scoring="roc_auc", cv=folds, n_jobs=-1,
        ).fit(features, labels)
    else:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed))
        pipe.steps[1] = ("lr", pipe.steps[1][1])
        head = GridSearchCV(
            pipe, {"lr__C": CS}, scoring="roc_auc", cv=folds, n_jobs=-1,
        ).fit(features, labels)

    path = fusion_head_path(args.model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(head.best_estimator_, path)
    print(f"wrote {path}  cv_roc_auc={head.best_score_:.4f}  {head.best_params_}")

    metadata = {
        "split": "val",
        "input_space": "scores",
        "seed": seed,
        "n_samples": int(len(labels)),
        "heads": {
            args.model: {
                "best_params": {k: str(v) for k, v in head.best_params_.items()},
                "cv_roc_auc": round(float(head.best_score_), 4),
            },
        },
    }

    existing = {}
    if os.path.exists(HEAD_METADATA):
        with open(HEAD_METADATA, encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(metadata)
    if "heads" in existing:
        existing["heads"].update(metadata["heads"])

    with open(HEAD_METADATA, "w", encoding="utf-8") as f:
        json.dump(existing, indent=4, fp=f)
        f.write("\n")
    print(f"wrote {HEAD_METADATA}")
    print(f"Fitted {args.model} head. Checkpoint: {path}")


# ── CLI ──

def parse_args():
    datasets = discover_available_datasets()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET, choices=datasets)
    parser.add_argument("--model", choices=MODEL_CHOICES, default=None,
                        help="Classifier to train. Ignored when --train-all is set.")
    parser.add_argument("--train-all", action="store_true",
                        help="Train all models in order: text, image, feature-fusion, svm-rbf, linear. "
                             "Skips any model whose checkpoint already exists.")
    parser.add_argument("--name-llm", default=NAME_LLM)
    parser.add_argument("--name-img-embed", default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-tokens", type=int, default=N_TOKENS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--merge-tokens", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.4)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=RAND_SEED)
    return parser.parse_args()


DEPLOYED_PATHS = {
    "text": TEXT_WEIGHTS_PATH,
    "image": IMAGE_WEIGHTS_PATH,
    "feature-fusion": FF_WEIGHTS_PATH,
    "svm-rbf": fusion_head_path("svm-rbf"),
    "linear": fusion_head_path("linear"),
}


def checkpoint_exists(model_name):
    """Check if the deployed checkpoint for a model already exists."""
    return Path(DEPLOYED_PATHS[model_name]).exists()


def train_one(args, model_name, dataset_class, annotation_loader, train_file, val_file, image_dir):
    """Train or fit a single model, skipping if checkpoint exists."""
    if checkpoint_exists(model_name):
        print(f"  [SKIP] {model_name} — checkpoint exists: {DEPLOYED_PATHS[model_name]}")
        return

    args.model = model_name
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")

    if model_name in NEURAL_MODELS:
        train_neural(args, dataset_class, annotation_loader, train_file, val_file, image_dir)
    else:
        fit_fusion_head(args, dataset_class, annotation_loader, val_file, image_dir)


def main():
    args = parse_args()

    if not args.train_all and args.model is None:
        print("ERROR: either --model or --train-all is required.")
        raise SystemExit(1)

    dataset_class, annotation_loader, train_file, val_file, image_dir = resolve_dataset_assets(args.dataset)

    print("Training configuration")
    print(f"  Dataset: {args.dataset}")
    print(f"  Device:  {args.device}")

    if args.train_all:
        for model_name in MODEL_CHOICES:
            train_one(args, model_name, dataset_class, annotation_loader, train_file, val_file, image_dir)
        print("\nAll models trained.")
    else:
        print(f"  Model:   {args.model}")
        if args.model in NEURAL_MODELS:
            print(f"  Epochs:  {args.epochs}")
            train_neural(args, dataset_class, annotation_loader, train_file, val_file, image_dir)
        else:
            fit_fusion_head(args, dataset_class, annotation_loader, val_file, image_dir)


if __name__ == "__main__":
    main()
