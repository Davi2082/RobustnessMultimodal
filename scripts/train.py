"""Train a text, image, or feature-fusion Themis classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from configuration_files.configuration import (
    BATCH_SIZE,
    DATASET,
    NAME_IMG_EMBED,
    NAME_LLM,
    N_TOKENS,
)
from data_loading import my_datasets
from models.themis_model import get_Themis

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_loading"
MODEL_CHOICES = ("text", "image", "feature-fusion")
CHECKPOINT_NAMES = {
    "text": "best_text_only.pt",
    "image": "best_img_only.pt",
    "feature-fusion": "best_feature_fusion.pt",
}


def discover_available_datasets():
    if not DATA_ROOT.is_dir():
        return []
    return sorted(path.name for path in DATA_ROOT.iterdir() if path.is_dir() and not path.name.startswith((".", "_")))


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
    image_dir = next((path for path in (dataset_dir / "images", dataset_dir / "images_copy") if path.is_dir()), None)
    if image_dir is None:
        raise FileNotFoundError(f"No image directory found in {dataset_dir}")
    return dataset_class, annotation_loader, train_file, val_file, image_dir


def parse_args():
    datasets = discover_available_datasets()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET, choices=datasets, help="Dataset stored below data_loading/.")
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES, help="Classifier type to train.")
    parser.add_argument("--name-llm", default=NAME_LLM, help="Hugging Face language-model identifier.")
    parser.add_argument("--name-img-embed", default=None, help="Image encoder identifier. Defaults depend on --model.")
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
    return parser.parse_args()


def model_inputs(model_type, images, texts):
    if model_type == "text":
        return None, texts
    if model_type == "image":
        return images, None
    return images, texts


def evaluate(model, loader, criterion, device, model_type):
    model.eval()
    running_loss = 0.0
    labels_all = []
    predictions_all = []
    with torch.no_grad():
        for images, labels, texts, _, _ in tqdm(loader, desc="Validation", leave=False):
            images, labels, texts = images.to(device), labels.to(device), texts.to(device)
            image_input, text_input = model_inputs(model_type, images, texts)
            scores, _ = model(images=image_input, texts=text_input)
            running_loss += criterion(scores.float(), labels.float().unsqueeze(1)).item()
            labels_all.extend(labels.cpu().numpy())
            predictions_all.extend((scores.detach().cpu().numpy().reshape(-1) > 0.5).astype(int))
    metrics = {
        "loss": running_loss / len(loader),
        "accuracy": accuracy_score(labels_all, predictions_all),
        "precision": precision_score(labels_all, predictions_all, zero_division=0),
        "recall": recall_score(labels_all, predictions_all, zero_division=0),
        "f1": f1_score(labels_all, predictions_all, zero_division=0),
    }
    return metrics


def main():
    args = parse_args()
    dataset_class, annotation_loader, train_file, val_file, image_dir = resolve_dataset_assets(args.dataset)
    image_encoder = args.name_img_embed or NAME_IMG_EMBED
    merge_tokens = args.merge_tokens or None
    device = torch.device(args.device)

    print("Training configuration")
    print(f"  Dataset:       {args.dataset}")
    print(f"  Model:         {args.model}")
    print(f"  Device:        {device}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Image encoder: {image_encoder}")

    model, tokenizer, processor = get_Themis(
        name_llm=args.name_llm,
        name_img_embed=image_encoder,
        use_lora=args.use_lora,
        is_pythia="pythia" in args.name_llm.lower(),
        lora_alpha=args.lora_alpha,
        lora_r=args.lora_r,
        lora_dropout=args.lora_dropout,
        merge_tokens=merge_tokens,
    )
    model.to(device)
    train_dataset = my_datasets.get_dataset(dataset_class, annotation_loader, args.n_tokens, processor, tokenizer, str(train_file), str(image_dir))
    val_dataset = my_datasets.get_dataset(dataset_class, annotation_loader, args.n_tokens, processor, tokenizer, str(val_file), str(image_dir))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    checkpoint = ROOT / "models" / "weights" / args.dataset / CHECKPOINT_NAMES[args.model]
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
        print(f"Epoch {epoch + 1}: train loss={train_loss:.4f}, validation loss={metrics['loss']:.4f}, F1={metrics['f1']:.4f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), checkpoint)
            metadata = {
                "name_llm": args.name_llm,
                "name_img_embed": image_encoder,
                "merge_tokens": args.merge_tokens,
                "lora_alpha": args.lora_alpha,
                "lora_r": args.lora_r,
                "lora_dropout": args.lora_dropout,
                "use_lora": args.use_lora,
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
    print(f"Pipeline-ready checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
