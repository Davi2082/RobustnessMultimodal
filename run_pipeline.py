"""Run compatible robustness experiments for a named model type and dataset."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from configuration_files.configuration import DATASET

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
MODEL_TYPES = ("text", "image", "feature-fusion", "late-fusion", "late-fusion-mean", "late-fusion-min", "late-fusion-max")
EXPERIMENT_LABELS = {
    "setup-clean-text": "Prepare clean predictions for the text model",
    "setup-clean-image": "Prepare clean predictions for the image model",
    "clean": "Clean evaluation",
    "pgd": "PGD attack on the image modality",
    "trepat": "TRePAT attack on the text modality",
    "hotflip": "Gradient-guided text attack",
    "pgd-trepat-sum": "Independent PGD + TRePAT attack",
    "pgd-trepat-alternating": "Alternating PGD + TRePAT attack",
    "hotflip-pgd-joint": "Joint gradient-guided text attack + PGD",
}


def experiment(name, module, *arguments):
    return name, (PYTHON, "-m", module, *arguments)


def find_checkpoint(dataset, model_type):
    weights_dir = ROOT / "models" / "weights" / dataset
    patterns = {"text": "*_txt_only.pt", "image": "*_img_only.pt"}
    canonical = {"text": "best_text_only.pt", "image": "best_img_only.pt", "feature-fusion": "best_feature_fusion.pt"}
    preferred = weights_dir / canonical.get(model_type, "")
    if preferred.is_file():
        return preferred.resolve()
    if model_type in patterns:
        candidates = sorted(weights_dir.glob(patterns[model_type]))
    else:
        candidates = sorted(path for path in weights_dir.glob("*.pt") if "_txt_only" not in path.name and "_img_only" not in path.name and not path.name.startswith("LAST_TRAINED"))
    if len(candidates) != 1:
        found = ", ".join(path.name for path in candidates) or "none"
        raise SystemExit(f"Expected exactly one {model_type} checkpoint in {weights_dir}; found: {found}")
    return candidates[0].resolve()


def checkpoint_options(checkpoint):
    metadata_path = checkpoint.with_suffix(".json")
    if not metadata_path.is_file():
        return ()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    options = (
        "--name_llm", str(metadata["name_llm"]),
        "--name_img_embed", str(metadata["name_img_embed"]),
        "--merge_tokens", str(metadata["merge_tokens"]),
        "--lora_alpha", str(metadata["lora_alpha"]),
        "--lora_r", str(metadata["lora_r"]),
        "--lora_dropout", str(metadata["lora_dropout"]),
    )
    return (*options, "--use_lora" if metadata["use_lora"] else "--no-use_lora")


def build_experiments(dataset, model_type):
    results = ROOT / "results" / dataset / "classification_results"
    common = ("--dataset", dataset, "--results_path", str(results))
    if model_type == "text":
        model = find_checkpoint(dataset, "text")
        model_options = checkpoint_options(model)
        clean_params = results / "clean" / "text" / "parameters.json"
        return (
            experiment("clean", "scripts.eval", "--modality", "text", "--model_path", str(model), *model_options, *common),
            experiment("trepat", "attacks.unimodal.text_attack", "--attack_method", "trepat", "--model_path", str(model), "--parameters-path", str(clean_params), "--experiment-name", "trepat", *common),
            experiment("hotflip", "attacks.unimodal.text_attack", "--attack_method", "bertattack", "--model_path", str(model), "--parameters-path", str(clean_params), "--experiment-name", "hotflip", *common),
        )
    if model_type == "image":
        model = find_checkpoint(dataset, "image")
        model_options = checkpoint_options(model)
        clean_params = results / "clean" / "image" / "parameters.json"
        return (
            experiment("clean", "scripts.eval", "--modality", "image", "--model_path", str(model), *model_options, *common),
            experiment("pgd", "attacks.unimodal.image_attack", "--model_path", str(model), "--parameters-path", str(clean_params), "--experiment-name", "pgd", *common),
        )
    if model_type == "feature-fusion":
        model = find_checkpoint(dataset, "feature-fusion")
        model_options = checkpoint_options(model)
        clean_params = results / "clean" / "feature-fusion" / "parameters.json"
        attack = ("attacks.multimodal.multimodal_attack", "--model_path", str(model), "--parameters-path", str(clean_params), *common)
        return (
            experiment("clean", "scripts.eval", "--modality", "feature-fusion", "--model_path", str(model), *model_options, *common),
            experiment("pgd", attack[0], *attack[1:], "--attack-scope", "image", "--experiment-name", "pgd"),
            experiment("trepat", attack[0], *attack[1:], "--attack-scope", "text", "--text-attack", "trepat", "--experiment-name", "trepat"),
            experiment("pgd-trepat-sum", attack[0], *attack[1:], "--attack-scope", "both", "--optimization", "sum", "--text-attack", "trepat", "--experiment-name", "pgd-trepat-sum"),
            experiment("pgd-trepat-alternating", attack[0], *attack[1:], "--attack-scope", "both", "--optimization", "alternating", "--text-attack", "trepat", "--experiment-name", "pgd-trepat-alternating"),
            experiment("hotflip-pgd-joint", attack[0], *attack[1:], "--attack-scope", "both", "--optimization", "alternating", "--text-attack", "hotflip", "--experiment-name", "hotflip-pgd-joint"),
        )
    fusion = model_type.removeprefix("late-fusion-")
    text_model = find_checkpoint(dataset, "text")
    image_model = find_checkpoint(dataset, "image")
    text_options = checkpoint_options(text_model)
    image_options = checkpoint_options(image_model)
    text_params = results / "clean" / "text" / "parameters.json"
    image_params = results / "clean" / "image" / "parameters.json"
    late_common = ("--fusion", fusion, "--text-model-path", str(text_model), "--image-model-path", str(image_model), "--text-parameters", str(text_params), "--image-parameters", str(image_params), *common)
    return (
        experiment("setup-clean-text", "scripts.eval", "--modality", "text", "--model_path", str(text_model), *text_options, *common),
        experiment("setup-clean-image", "scripts.eval", "--modality", "image", "--model_path", str(image_model), *image_options, *common),
        experiment("clean", "scripts.eval", "--modality", "late-fusion", "--late_fusion_mode", fusion, "--text_model_path", str(text_model), "--image_model_path", str(image_model), *image_options, *common),
        experiment("pgd", "scripts.late_fusion_perturbation", *late_common, "--attack-scope", "image"),
        experiment("trepat", "scripts.late_fusion_perturbation", *late_common, "--attack-scope", "text"),
        experiment("pgd-trepat-sum", "scripts.late_fusion_perturbation", *late_common, "--attack-scope", "both"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET, help="Dataset directory below data_loading/.")
    parser.add_argument("--model", required=True, choices=MODEL_TYPES, help="Model/fusion type; checkpoint paths are resolved automatically.")
    parser.add_argument("--fusion", choices=("mean", "min", "max"), default="mean", help="Aggregation used when --model late-fusion.")
    parser.add_argument("--only", nargs="+", help="Run only the named compatible experiments.")
    parser.add_argument("--list", action="store_true", help="List compatible experiments without running them.")
    parser.add_argument("--dry-run", action="store_true", help="Preview the pipeline without running experiments.")
    parser.add_argument("--show-commands", action="store_true", help="Show the complete internal commands.")
    return parser.parse_args()


def run(name, command, dataset, model_type, position, total, dry_run=False, show_commands=False):
    description = EXPERIMENT_LABELS.get(name, name)
    suffix = " - preview, not executed" if dry_run else ""
    print(f"[{position}/{total}] {description}{suffix}", flush=True)
    if show_commands:
        print(f"      Command: {' '.join(command)}", flush=True)
    if dry_run:
        return
    log_dir = ROOT / "logs" / "pipeline" / dataset / model_type
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise SystemExit(f"Experiment {name!r} failed (exit code {result.returncode}). Log: {log_path}")
    print("      Completed.", flush=True)


def main():
    args = parse_args()
    model_type = f"late-fusion-{args.fusion}" if args.model == "late-fusion" else args.model
    experiments = build_experiments(args.dataset, model_type)
    available = {name for name, _ in experiments}
    unknown = set(args.only or ()) - available
    if unknown:
        unknown_names = ", ".join(sorted(unknown))
        available_names = ", ".join(sorted(available))
        raise SystemExit(f"Experiments not compatible with {model_type}: {unknown_names}. Available: {available_names}")
    selected = [(name, command) for name, command in experiments if not args.only or name in args.only]
    if args.list:
        print(f"Available experiments for {model_type} on {args.dataset}:")
        for name, _ in selected:
            print(f"  {name:<28} {EXPERIMENT_LABELS.get(name, name)}")
        return
    print("\nPipeline configuration")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Model:        {model_type}")
    print(f"  Experiments:  {len(selected)}")
    print(f"  Results:      results/{args.dataset}/classification_results")
    if args.dry_run:
        print("  Mode:         preview (no experiments will be executed)")
    print()
    for position, (name, command) in enumerate(selected, start=1):
        run(name, command, args.dataset, model_type, position, len(selected), args.dry_run, args.show_commands)
    message = "Preview completed." if args.dry_run else "Pipeline completed."
    print(f"\n{message}")


if __name__ == "__main__":
    main()
