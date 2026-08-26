"""Modality-ablation probe for any Themis checkpoint.

Measures what the model selected by ``--modality`` predicts when one modality
is removed at test time, under BOTH removal semantics:

``drop``
    The modality's tokens are removed from the sequence entirely. This is what
    ``Themis.forward`` does natively for ``images=None`` / ``texts=None``, but
    it also changes the number of tokens the final ``x.mean(dim=1)`` averages
    over (~50 image + N_TOKENS text vs. one of them alone), so a drop-ablation
    delta mixes "information removed" with "pooling denominator changed".

``blank``
    The modality's token block is zeroed but kept in the sequence, so the
    pooling denominator is identical across all three conditions and the delta
    isolates the information contribution.

Both are reported; ``blank`` is the primary measurement and ``drop`` is the
robustness check.

Column naming is explicit about which modality SURVIVES: ``img_only`` means the
image is present and the text was ablated. (The legacy ``ff_unimodal_logits.csv``
used ``text_logit_*`` for the fed-text-only condition, which is easy to read
backwards.)

Usage:
    python3 -m scripts.modality_ablation --modality feature-fusion
    python3 -m scripts.modality_ablation --modality text
"""

import os
import glob
import json
import argparse

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from data_loading import my_datasets
from utils import load_available_datasets, load_model
from configuration_files.configuration import NAME_LLM, BATCH_SIZE, N_TOKENS, THRESHOLD, DEVICE_EVAL, SUBSET_SIZE
from configuration_files.paths import RESULT_PATH


@torch.no_grad()
def ff_logits(model, images, texts, ablate=None, mode="blank"):
    """Forward the feature-fusion model with one modality optionally ablated.

    ``ablate`` is the modality to REMOVE (``"image"``, ``"text"`` or ``None``).
    ``mode`` selects the removal semantics (``"blank"`` or ``"drop"``).
    Mirrors ``Themis.forward`` exactly apart from the ablation.
    """
    if mode == "drop":
        if ablate == "image":
            return model(images=None, texts=texts)
        if ablate == "text":
            return model(images=images, texts=None)
        return model(images=images, texts=texts)

    if mode != "blank":
        raise ValueError(f"Unknown ablation mode: {mode}")

    # --- blank: keep sequence length, zero the ablated block ---------------
    # Matches Themis.forward: a bare tensor is treated as pixel_values, anything
    # else is a mapping (dict or transformers BatchFeature, which is NOT a dict).
    pixel_values = images if isinstance(images, torch.Tensor) else images["pixel_values"]
    if pixel_values.dim() == 4:
        pixel_values = pixel_values.unsqueeze(1)
    b, k, c, h, w = pixel_values.shape

    image_features = model.img_embed_model(pixel_values=pixel_values.reshape(b * k, c, h, w)).last_hidden_state
    image_features = model.image_proj(image_features)

    text_embeds = model.emb(texts["input_ids"])
    text_embeds = text_embeds.view(text_embeds.shape[0], text_embeds.shape[-2], text_embeds.shape[-1])

    if ablate == "image":
        image_features = torch.zeros_like(image_features)
    elif ablate == "text":
        text_embeds = torch.zeros_like(text_embeds)

    x = torch.cat((image_features, text_embeds), dim=1)
    if model.merge_tokens is not None:
        x = model.patch_merger(x)

    position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
    for i in range(len(model.h)):
        x = model.h[i](x, position_ids=position_ids)[0]
    x = x.mean(dim=1)

    logits = model.lm_head(x)
    return torch.sigmoid(logits), logits


def metrics(y, score, thr):
    """Fake (label 0) is the positive class, matching the rest of the pipeline."""
    y_true = 1 - y
    sc = 1 - score
    y_pred = (sc > thr).astype(int)
    return roc_auc_score(y_true, sc), f1_score(y_true, y_pred), accuracy_score(y_true, y_pred)


def main():
    dataset_classes, load_functions = load_available_datasets()

    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, default="feature-fusion",
                        choices=("feature-fusion", "text", "image"))
    parser.add_argument("--name_llm", type=str, default=NAME_LLM)
    parser.add_argument("--name_img_embed", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--n_tokens", type=int, default=N_TOKENS)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--merge_tokens", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--use_lora", type=bool)
    parser.add_argument("--set_params", type=bool, default=True)
    parser.add_argument("--results_path", type=str, default=RESULT_PATH)
    parser.add_argument("--dataset", type=str, default="Recovery", choices=list(dataset_classes.keys()))
    args = parser.parse_args()

    device = torch.device(DEVICE_EVAL)
    # load_model forces the feature-fusion encoder/checkpoint for this modality
    model, tokenizer, processor = load_model(device, args)
    model.eval()

    dataset_test = my_datasets.get_dataset(
        dataset_classes[args.dataset],
        load_functions[args.dataset],
        args.n_tokens,
        processor,
        tokenizer,
        glob.glob(f"data/{args.dataset}/test.*")[0],
        f"data/{args.dataset}/images",
    )
    if SUBSET_SIZE is not None:
        loader = DataLoader(dataset_test, batch_size=args.batch_size,
                            sampler=list(range(min(SUBSET_SIZE, len(dataset_test)))))
    else:
        loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    # (column suffix, ablated modality) -- the name says which modality SURVIVES.
    # A unimodal checkpoint only has its own modality to remove.
    if args.modality == "feature-fusion":
        conditions = [("both", None), ("img_only", "text"), ("txt_only", "image")]
    elif args.modality == "text":
        conditions = [("both", None), ("img_only", "text")]
    else:
        conditions = [("both", None), ("txt_only", "image")]
    rows = {f"{mode}_logit_{name}": [] for mode in ("blank", "drop") for name, _ in conditions}
    rows.update({f"{mode}_score_{name}": [] for mode in ("blank", "drop") for name, _ in conditions})
    labels_all, indices_all = [], []

    for images, labels, texts, _, indices in tqdm(loader, desc="Ablating"):
        images = images.to(device)
        texts = texts.to(device)
        for mode in ("blank", "drop"):
            for name, ablate in conditions:
                scores, logits = ff_logits(model, images, texts, ablate=ablate, mode=mode)
                rows[f"{mode}_logit_{name}"].append(logits.detach().cpu().flatten())
                rows[f"{mode}_score_{name}"].append(scores.detach().cpu().flatten())
        labels_all.append(labels)
        indices_all.append(indices)

    out = {"index": torch.cat(indices_all).numpy(), "label": torch.cat(labels_all).numpy()}
    for col, chunks in rows.items():
        out[col] = torch.cat(chunks).numpy()
    df = pd.DataFrame(out)

    output_dir = os.path.join(args.results_path, "fusion_analysis")
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{args.modality}_modality_ablation"
    csv_path = os.path.join(output_dir, f"{stem}.csv")
    df.to_csv(csv_path, index=False)

    # ---------- Table 2 ----------
    label = df["label"].values
    summary = []
    for mode in ("blank", "drop"):
        pretty_names = {"both": "both modalities",
                        "img_only": "text ablated (image only)",
                        "txt_only": "image ablated (text only)"}
        for name, pretty in [(n, pretty_names[n]) for n, _ in conditions]:
            auc, f1, acc = metrics(label, df[f"{mode}_score_{name}"].values, args.threshold)
            summary.append({"ablation mode": mode, "condition": pretty,
                            "AUC": round(auc, 4), "F1_fake": round(f1, 4), "Acc": round(acc, 4)})
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(output_dir, f"{stem}_metrics.csv"), index=False)

    # ---------- contribution share (needs both ablations) ----------
    shares = {}
    if args.modality != "feature-fusion":
        print(f"\nWrote {csv_path}\n")
        print(summary.to_string(index=False))
        return

    # Delta(m) = logit(both) - logit(m removed); ||w|| cancels in the ratio.
    for mode in ("blank", "drop"):
        d_img = (df[f"{mode}_logit_both"] - df[f"{mode}_logit_txt_only"]).abs()  # image removed -> text only
        d_txt = (df[f"{mode}_logit_both"] - df[f"{mode}_logit_img_only"]).abs()
        total = d_img + d_txt
        share_img = (d_img / total).replace([float("inf")], float("nan"))
        interaction = df[f"{mode}_logit_both"] - df[f"{mode}_logit_img_only"] - df[f"{mode}_logit_txt_only"]
        shares[mode] = {
            "share_image_mean": round(float(share_img.mean()), 4),
            "share_image_median": round(float(share_img.median()), 4),
            "share_text_mean": round(float(1 - share_img.mean()), 4),
            "interaction_mean_abs": round(float(interaction.abs().mean()), 4),
        }
    with open(os.path.join(output_dir, f"{stem}_shares.json"), "w") as f:
        json.dump(shares, f, indent=4)

    print(f"\nWrote {csv_path}\n")
    print(summary.to_string(index=False))
    print("\nContribution share (share_image = |Delta_image| / (|Delta_image| + |Delta_text|)):")
    print(json.dumps(shares, indent=4))


if __name__ == "__main__":
    main()
