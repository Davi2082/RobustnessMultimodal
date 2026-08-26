"""Joint white-box attack: HotFlip on text and PGD on pixels.

One forward/backward pass per iteration; that single gradient drives both a
signed PGD step and one whole-word substitution, so each channel is optimised
against the other channel's current perturbation. --attack-scope text|image
freezes the other channel, giving unimodal baselines from the same family.

    python3 -m attacks.multimodal.joint.attack --fusion min --attack-scope both
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loading import my_datasets
from attacks.attack_algorithms.text.HOTFLIP.hotflip import (
    EmbeddingGradient,
    candidate_mask,
    effective_embedding_table,
    whole_word_positions,
)
from attacks.attack_algorithms.text.common import model_sbert, visible_text_window
from attacks.multimodal.sum.attack import (
    encode_image_batch,
    encode_text_batch,
    move_to_device,
)
from models.fusion import (
    COMPONENT_OUTPUT_COLUMNS,
    COMPONENT_OUTPUT_NAMES,
    FUSION_CHOICES,
    build_classifier,
    model_args_from_parameters,
    read_parameters,
)
from configuration_files.configuration import (
    DEVICE,
    EPSILON,
    MAX_CHANGE_RATIO,
    MIN_TXT_SIMILARITY,
    PGD_ITERS,
    SOURCE_LABEL,
    SUBSET_SIZE,
    TARGET_LABEL,
    THRESHOLD,
)
from configuration_files.paths import (
    dataset_annotations,
    dataset_images_dir,
    CLEAN_IMAGE_PARAMS,
    CLEAN_TEXT_PARAMS,
    LATE_FUSION_DATA_DIR,
    RESULT_PATH,
    late_fusion_scenario_path,
)
from utils import (
    load_available_datasets,
    load_model,
    save_perturbed_image,
    save_perturbed_texts,
    save_predictions,
)
from sentence_transformers import util














def normalise(pixels: torch.Tensor, processor) -> torch.Tensor:
    mean = torch.tensor(
        processor.image_mean, device=pixels.device, dtype=pixels.dtype
    ).view(1, -1, 1, 1)
    std = torch.tensor(
        processor.image_std, device=pixels.device, dtype=pixels.dtype
    ).view(1, -1, 1, 1)
    return (pixels - mean) / std


def to_pil(pixels: torch.Tensor) -> Image.Image:
    array = (
        pixels.squeeze(0).permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy() * 255
    ).astype(np.uint8)
    return Image.fromarray(array)


def joint_attack(
    model,
    tokenizer,
    processor,
    args,
    news: dict[str, Any],
    device: torch.device,
    embedding_table: torch.Tensor,
    allowed_candidates: torch.Tensor,
    label: int,
) -> dict[str, Any]:
    """Run the shared-gradient attack on one sample.

    Untargeted: each sample is pushed toward the opposite class, and every
    decision below is expressed as the probability of that class.
    """
    # 1 = Real, so a fake sample escapes upward and a real sample downward.
    escaping_fake = int(label) == 0
    attack_text = args.attack_scope in {"text", "both"}
    attack_image = args.attack_scope in {"image", "both"}

    # Only the first n_tokens are visible to the classifier; the tail is put
    # back untouched so the dumped text is the original with substitutions,
    # not a truncation of it.
    visible_text, hidden_text = visible_text_window(
        news["txt"], tokenizer, args.n_tokens
    )

    tokenized = tokenizer(
        visible_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_attention_mask=False,
        max_length=args.n_tokens,
    ).to(device)
    input_ids = tokenized.input_ids.clone()

    special = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in tokenizer.all_special_ids:
        special |= input_ids == token_id
    attackable = (~special).squeeze(0)
    if args.word_level:
        attackable &= whole_word_positions(tokenizer, input_ids)
    n_attackable = int(attackable.sum().item())
    max_flips = max(1, int(args.max_change_ratio * n_attackable))

    processed = processor(images=news["img"], return_tensors="pt", do_normalize=False)
    clean_pixels = processed["pixel_values"].to(device)

    delta = torch.zeros_like(clean_pixels)
    if attack_image and args.random_start:
        delta.uniform_(-args.epsilon, args.epsilon)
        delta = (clean_pixels + delta).clamp(0, 1) - clean_pixels

    flipped = torch.zeros_like(attackable)
    text_exhausted = False
    text_failures = 0
    best = {
        "score": -1.0,
        "ids": input_ids.clone(),
        "delta": delta.clone(),
        "flips": 0,
        "iterations": 0,
    }

    capture = EmbeddingGradient(model.text_model.emb)
    try:
        for iteration in range(args.iters):
            pixels = (clean_pixels + delta).clamp(0, 1).detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)

            score, _ = model(
                {"pixel_values": normalise(pixels, processor)},
                {"input_ids": input_ids.unsqueeze(1)},
            )
            fused = score.reshape(-1)[0]

            # Probability of the class the sample is being pushed into.
            target_probability = fused if escaping_fake else 1.0 - fused

            # Ascending log P(target) keeps a usable gradient even when the
            # detector is confident, where the logit itself saturates.
            objective = torch.log(target_probability.clamp_min(1e-12))
            objective.backward()

            current = float(target_probability.detach().item())
            if current > best["score"]:
                best.update(
                    score=current,
                    ids=input_ids.clone(),
                    delta=delta.clone(),
                    flips=int(flipped.sum().item()),
                    iterations=iteration,
                )

            # Success is a flipped prediction, whichever direction it came from.
            fused_value = float(fused.detach().item())
            if (fused_value > args.threshold) != escaping_fake:
                break

            if attack_image and pixels.grad is not None:
                delta = delta + args.alpha * pixels.grad.sign()
                delta = delta.clamp(-args.epsilon, args.epsilon)
                delta = (clean_pixels + delta).clamp(0, 1) - clean_pixels
                delta = delta.detach()

            if attack_text and not text_exhausted and capture.output is not None:
                gradient = capture.output.grad
                if gradient is None:
                    break
                gradient = gradient.reshape(-1, gradient.shape[-1])

                # First-order gain of swapping token w for candidate v.
                gains = gradient @ embedding_table.T
                current_gain = gains.gather(1, input_ids.reshape(-1, 1))
                gains = gains - current_gain
                gains[:, ~allowed_candidates] = -float("inf")

                positions = attackable.clone()
                if int(flipped.sum().item()) >= max_flips:
                    positions &= flipped  # only refine already-changed tokens
                gains[~positions] = -float("inf")

                # The gradient only proposes: a first-order estimate is
                # unreliable for a jump as large as a whole-word substitution,
                # so the top candidates are verified with one batched forward
                # pass and the substitution is kept only if it truly helps.
                # A rejected batch widens the search instead of ending it: the
                # first-order ranking is noisy, so the useful substitution is
                # often just outside the current window.
                width = args.candidates * (1 + text_failures)
                top = torch.topk(gains.reshape(-1), k=min(width, gains.numel()))
                valid = torch.isfinite(top.values)
                if not bool(valid.any()):
                    text_exhausted = True
                else:
                    flat_indices = top.indices[valid]
                    proposals = input_ids.repeat(len(flat_indices), 1)
                    for row, flat in enumerate(flat_indices.tolist()):
                        position, candidate = divmod(flat, gains.shape[1])
                        proposals[row, position] = candidate

                    with torch.inference_mode():
                        candidate_scores, _ = model(
                            {
                                "pixel_values": normalise(
                                    pixels.detach(), processor
                                ).repeat(len(flat_indices), 1, 1, 1)
                            },
                            {"input_ids": proposals.unsqueeze(1)},
                        )
                    candidate_scores = candidate_scores.reshape(-1)
                    winner = int(torch.argmax(candidate_scores).item())

                    if float(candidate_scores[winner].item()) <= current:
                        text_failures += 1
                        if text_failures >= args.text_patience:
                            text_exhausted = True
                    else:
                        text_failures = 0
                        position, candidate = divmod(
                            int(flat_indices[winner].item()), gains.shape[1]
                        )
                        input_ids[0, position] = candidate
                        flipped[position] = True
    finally:
        capture.remove()

    perturbed_ids = best["ids"] if attack_text else input_ids
    perturbed_delta = best["delta"] if attack_image else torch.zeros_like(clean_pixels)

    kept = perturbed_ids.squeeze(0)[~special.squeeze(0)]
    perturbed_text = (
        tokenizer.decode(kept, skip_special_tokens=True) + hidden_text
        if attack_text
        else news["txt"]
    )
    perturbed_image = (
        to_pil((clean_pixels + perturbed_delta).clamp(0, 1))
        if attack_image
        else news["img"]
    )

    similarity = 1.0
    if attack_text:
        with torch.inference_mode():
            original = model_sbert.encode(
                news["txt"], convert_to_tensor=True, device="cpu"
            )
            perturbed = model_sbert.encode(
                perturbed_text, convert_to_tensor=True, device="cpu"
            )
            similarity = float(util.cos_sim(original, perturbed).item())

        if similarity < args.min_txt_similarity:
            perturbed_text = news["txt"]
            similarity = 1.0

    return {
        "txt": perturbed_text,
        "img": perturbed_image,
        "similarity": similarity,
        "flips": best["flips"],
        "n_attackable": n_attackable,
        "best_score": best["score"],
    }


def parse_args() -> tuple[argparse.Namespace, dict[str, Any], dict[str, Any]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-parameters", type=Path, default=Path(CLEAN_TEXT_PARAMS))
    parser.add_argument(
        "--image-parameters", type=Path, default=Path(CLEAN_IMAGE_PARAMS)
    )
    parser.add_argument(
        "--fusion", choices=FUSION_CHOICES, default="mean"
    )
    parser.add_argument(
        "--attack-scope", choices=("text", "image", "both"), default="both"
    )
    parser.add_argument("--iters", type=int, default=PGD_ITERS)
    parser.add_argument(
        "--budget-divisor",
        type=int,
        default=1,
        help=(
            "Divide the iteration budget in the joint scope. One iteration is "
            "one backward pass whatever the scope, so equal iterations already "
            "means equal gradient queries; use 2 to reproduce the halved-budget "
            "convention of the disjoint pipeline."
        ),
    )
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="PGD step size. Defaults to 2.5 * epsilon / iters.",
    )
    parser.add_argument("--no-random-start", dest="random_start", action="store_false")
    parser.add_argument("--max-change-ratio", type=float, default=MAX_CHANGE_RATIO)
    parser.add_argument(
        "--candidates",
        type=int,
        default=8,
        help="Top gradient-proposed substitutions verified per iteration.",
    )
    parser.add_argument(
        "--text-patience",
        type=int,
        default=3,
        help="Consecutive rejected candidate batches before text flips stop.",
    )
    parser.add_argument("--min-txt-similarity", type=float, default=MIN_TXT_SIMILARITY)
    parser.add_argument("--allow-non-ascii", dest="ascii_only", action="store_false")
    parser.add_argument(
        "--subword-flips",
        dest="word_level",
        action="store_false",
        help=(
            "Allow substituting single sub-word pieces. By default a flip "
            "replaces a whole word, which keeps the adversarial text readable."
        ),
    )
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--source-label", type=int, default=SOURCE_LABEL)
    parser.add_argument("--target-label", type=int, default=TARGET_LABEL)
    parser.add_argument("--subset-size", type=int, default=SUBSET_SIZE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dump-dir", type=Path, default=None)
    parser.add_argument("--results-path", type=Path, default=Path(RESULT_PATH))
    args = parser.parse_args()

    text_parameters = read_parameters(args.text_parameters, "Text model")
    image_parameters = read_parameters(args.image_parameters, "Image model")

    args.dataset = text_parameters["Dataset"]
    args.n_tokens = int(text_parameters["Number of Tokens"])
    args.text_model_path = Path(text_parameters["Model Path"])
    args.image_model_path = Path(image_parameters["Model Path"])

    if args.attack_scope == "both" and args.budget_divisor > 1:
        args.iters = max(1, args.iters // args.budget_divisor)
    if args.alpha is None:
        args.alpha = 2.5 * args.epsilon / max(1, args.iters)

    return args, text_parameters, image_parameters


def save_parameters(path: Path, args, text_parameters, image_parameters) -> None:
    payload = {
        "Fusion": {"Type": args.fusion, "Threshold": args.threshold},
        "Scenario": args.attack_scope,
        "Attack": {
            "Objective": "fused score, shared backward pass",
            "Method": "hotflip+pgd",
            "Independent Modalities": False,
            "Joint Optimization": True,
            "Iterations": args.iters,
            "Budget Divisor": args.budget_divisor,
            "Epsilon": args.epsilon,
            "Alpha": args.alpha,
            "Random Start": args.random_start,
            "Max Change Ratio": args.max_change_ratio,
            "Verified Candidates": args.candidates,
            "Text Patience": args.text_patience,
            "ASCII Candidates Only": args.ascii_only,
            "Whole-Word Flips": args.word_level,
            "Minimum Text Similarity": args.min_txt_similarity,
            # This attack is untargeted: the sample filter below is
            # clean_prediction == label, so both classes are attacked and
            # source/target label play no role. Recorded as such.
            "Targeted": False,
            "Source Label": "all (per-sample 1-label)",
            "Target Label": "all (per-sample 1-label)",
            "Only Clean-Correct Samples": True,
            "Number of Tokens": args.n_tokens,
        },
        "Text Model Parameters": text_parameters,
        "Image Model Parameters": image_parameters,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
        handle.write("\n")


def main() -> None:
    args, text_parameters, image_parameters = parse_args()
    device = torch.device(args.device)

    # Model in: late fusion or feature fusion, chosen by --fusion alone.
    args.text_parameters_data = text_parameters
    args.image_parameters_data = image_parameters
    model, tokenizer, processor = build_classifier(args, device)

    vocab_size = int(
        getattr(text_model.emb, "num_embeddings", 0)
        or text_model.emb.weight.shape[0]  # LoRA-wrapped modules expose the weight
    )
    print(f"Building effective embedding table ({vocab_size} x hidden)...")
    embedding_table = effective_embedding_table(text_model.emb, vocab_size, device)
    allowed = candidate_mask(
        tokenizer, vocab_size, args.ascii_only, args.word_level
    ).to(device)
    print(f"HotFlip candidates: {int(allowed.sum().item())}/{vocab_size}")

    dataset_classes, load_functions = load_available_datasets()
    test_data = dataset_annotations(args.dataset, "test")
    dataset_test = my_datasets.get_dataset(
        dataset_classes[args.dataset],
        load_functions[args.dataset],
        args.n_tokens,
        processor,
        tokenizer,
        test_data,
        dataset_images_dir(args.dataset),
    )

    sampler = (
        list(range(min(args.subset_size, len(dataset_test))))
        if args.subset_size is not None
        else None
    )
    dataloader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
    )

    output_dir = args.output_dir or (
        args.results_path / "perturbed" / "late-fusion-joint" / args.fusion
    )
    dump_dir = args.dump_dir or (
        Path(LATE_FUSION_DATA_DIR) / "joint" / args.fusion / args.attack_scope
    )

    output_names = ("scores", "logits", *COMPONENT_OUTPUT_NAMES)
    outputs: dict[str, list[torch.Tensor]] = {name: [] for name in output_names}
    labels_list, indices_list = [], []
    similarities, flips_list, perturbed_text_rows = [], [], []
    attacked = 0

    print(f"Budget: {args.iters} iterations, alpha={args.alpha:.5f}")

    for images, labels, texts, _, indices in tqdm(
        dataloader, desc=f"joint {args.fusion}/{args.attack_scope}"
    ):
        images_device = move_to_device(images, device)
        texts_device = move_to_device(texts, device)

        with torch.inference_mode():
            clean_scores, _ = model(images_device, texts_device)
        clean_predictions = (
            clean_scores.detach().cpu().reshape(-1) > args.threshold
        ).to(torch.int64)

        perturbed_texts, perturbed_images = [], []

        for position, label in enumerate(labels.tolist()):
            index = int(indices[position].item())
            clean_news = {
                "txt": dataset_test.texts[index],
                "img": Image.open(
                    os.path.join(dataset_test.img_dir, dataset_test.imgs_path[index])
                ).convert("RGB"),
            }
            perturbed_text = clean_news["txt"]
            perturbed_image = clean_news["img"]
            similarity, n_flips = 1.0, 0

            clean_is_source = (
                int(clean_predictions[position].item()) == label
            )
            if clean_is_source:  # untargeted: attack every correctly-classified sample
                attacked += 1
                result = joint_attack(
                    model,
                    tokenizer,
                    processor,
                    args,
                    clean_news,
                    device,
                    embedding_table,
                    allowed,
                    label,
                )
                perturbed_text = result["txt"]
                perturbed_image = result["img"]
                similarity = result["similarity"]
                n_flips = result["flips"]

                if args.attack_scope in {"text", "both"}:
                    perturbed_text_rows.append(
                        {
                            "index": index,
                            "original": clean_news["txt"],
                            "perturbed": perturbed_text,
                        }
                    )
                if args.attack_scope in {"image", "both"}:
                    save_perturbed_image(
                        str(dump_dir / "images"), index, perturbed_image
                    )

            perturbed_texts.append(perturbed_text)
            perturbed_images.append(perturbed_image)
            similarities.append(similarity)
            flips_list.append(n_flips)

        encoded_texts = encode_text_batch(
            tokenizer, perturbed_texts, args.n_tokens, device
        )
        encoded_images = encode_image_batch(processor, perturbed_images, device)

        with torch.inference_mode():
            batch_outputs = model(encoded_images, encoded_texts, return_components=True)
        for name, value in zip(output_names, batch_outputs):
            outputs[name].append(value.detach().cpu().reshape(-1))

        labels_list.append(torch.as_tensor(labels).detach().cpu().reshape(-1))
        indices_list.append(torch.as_tensor(indices).detach().cpu().reshape(-1))

    y_true = torch.cat(labels_list).numpy()
    sample_indices = torch.cat(indices_list).numpy()
    scores = torch.cat(outputs["scores"]).numpy().reshape(-1)
    logits = torch.cat(outputs["logits"]).numpy().reshape(-1)
    components = {
        column: torch.cat(outputs[name]).numpy().reshape(-1)
        for column, name in zip(COMPONENT_OUTPUT_COLUMNS, COMPONENT_OUTPUT_NAMES)
    }
    components["hotflip_flips"] = np.asarray(flips_list)

    result_path = Path(late_fusion_scenario_path(output_dir, args.attack_scope))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(
        y_true,
        (scores > args.threshold).astype(np.int64),
        scores,
        logits,
        sample_indices,
        str(result_path),
        txt_similarities=np.asarray(similarities),
        extra_columns=components,
    )
    save_parameters(
        result_path.parent / "parameters.json", args, text_parameters, image_parameters
    )
    if perturbed_text_rows:
        save_perturbed_texts(str(dump_dir), perturbed_text_rows)

    print(f"Saved {result_path}")
    print(f"Attacked {attacked} clean-correct source samples.")


if __name__ == "__main__":
    main()
