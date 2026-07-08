"""Per-layer logit separation: apply the classification head after every
transformer layer of the text-only and image-only models, and plot one
text-vs-image logit scatter per layer in a single figure (clean data, Real vs
Fake, no attacks). Lets us see how the two modalities separate across depth.

Loads one model at a time (peak memory = a single model) and runs on the GPU
with the most free memory.
"""
import os
import gc
import glob
import json
import argparse
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader

import my_datasets
from utils import load_model, load_available_datasets, plot_layer_logits_grid
from configuration import SOURCE_LABEL, SUBSET_SIZE

CLEAN = "results/Recovery/classification_results/clean"
OUT = "figures/classification_results/scatter/layer_logits_grid.png"


def build_args(params):
    a = argparse.Namespace()
    a.modality = params["Modality"]
    a.name_llm = params.get("Name LLM") or params.get("LLM Name")
    a.name_img_embed = params["Image Embedder Name"]
    a.batch_size = params["Batch Size"]
    a.model_path = params["Model Path"]
    a.n_tokens = params["Number of Tokens"]
    a.merge_tokens = params["Merge Tokens"]
    a.lora_alpha = params["LoRA Alpha"]
    a.lora_r = params["LoRA R"]
    a.lora_dropout = params["LoRA Dropout"]
    a.use_lora = params["Use LoRA"]
    a.dataset = params["Dataset"]
    a.set_params = False
    a.threshold = params["Threshold"]
    return a


def _to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


def _freest_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        ).decode().split()
        return int(np.argmax([int(x) for x in out]))
    except Exception:
        return 0


def collect(modality, params, dataset, batch_size, device):
    """Load one model, run per-layer logits over the dataset, free the model."""
    args = build_args(params)
    model, _, _ = load_model(device, args, args.model_path)

    if SUBSET_SIZE is not None:
        sampler = list(range(min(SUBSET_SIZE, len(dataset))))
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    logits_list, y_list = [], []
    with torch.no_grad():
        for images, labels, texts, _, _ in loader:
            if modality == "text":
                out = model.per_layer_logits(images=None, texts=_to_device(texts, device))
            else:
                out = model.per_layer_logits(images=_to_device(images, device), texts=None)
            logits_list.append(out.cpu().numpy())
            y_list.append(np.asarray(labels))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return np.concatenate(logits_list, axis=0), np.concatenate(y_list, axis=0)


def main():
    device = torch.device(f"cuda:{_freest_gpu()}")
    print(f"using {device}")

    with open(f"{CLEAN}/text/parameters.json") as f:
        params_txt = json.load(f)
    with open(f"{CLEAN}/image/parameters.json") as f:
        params_img = json.load(f)

    # Dataset (shared tokenizer/processor across the two clip-large models)
    dataset_classes, load_functions = load_available_datasets()
    args_i = build_args(params_img)
    tmp_model, tokenizer, processor = load_model(device, args_i, args_i.model_path)
    dataset = my_datasets.get_dataset(
        dataset_classes[args_i.dataset], load_functions[args_i.dataset],
        args_i.n_tokens, processor, tokenizer,
        glob.glob(f"data/{args_i.dataset}/test.*")[0], f"data/{args_i.dataset}/images",
    )
    del tmp_model  # only needed for tokenizer/processor; free the GPU copy
    gc.collect()
    torch.cuda.empty_cache()

    T, y = collect("text", params_txt, dataset, args_i.batch_size, device)
    I, _ = collect("image", params_img, dataset, args_i.batch_size, device)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plot_layer_logits_grid(y, T, I, OUT, source_label=SOURCE_LABEL)
    print(f"saved: {OUT}  (N={len(y)}, layers={T.shape[1]})")


if __name__ == "__main__":
    main()
