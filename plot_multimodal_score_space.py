"""Plot clean and fusion-aware attacked samples in multimodal score space.

The script creates one figure for each attack scope:

1. image-only attack;
2. text-only attack;
3. text and image attack with the split multimodal budget.

Each figure contains four panels: MIN, MAX, MEAN, and RBF-SVM fusion. Unlike
the previous version, every panel reads the attack generated specifically
against that fusion rule. In particular, the ``both`` panels read the
dedicated half-budget result and never combine the full-budget text-only and
image-only attacks in memory.

Expected late-fusion result layout::

    results/Recovery/classification_results/perturbed/late-fusion/
    +-- mean/
    |   +-- text-perturbed/perturbed_results.csv
    |   +-- image-perturbed/perturbed_results.csv
    |   `-- perturbed_results.csv
    +-- min/...
    +-- max/...
    `-- svm-rbf/...

Every late-fusion CSV must contain both the fused prediction fields produced
by ``utils.save_predictions`` and the outputs of the two unimodal components::

    index,label,pred,score,logit,
    text_score,text_logit,image_score,image_logit

Here, ``score`` and ``logit`` are the fused outputs. The four component
columns are necessary because a single fused score cannot be mapped back to a
unique point in the two-dimensional text/image score space.

The RBF-SVM is loaded from ``results/Recovery/train/svm_rbf.joblib`` by
default, which is also the default used by ``late_fusion_multimodal_attack.py``.
If the file does not exist, this script generates any missing clean training
predictions, fits the SVM on the clean training set, and saves it there.

Run from the project root with:

    python plot_multimodal_score_space.py
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from configuration import (  # noqa: E402
    BATCH_SIZE,
    DEVICE_EVAL,
    IMAGE_WEIGHTS_PATH,
    NAME_IMG_EMBED,
    NAME_LLM,
    N_TOKENS,
    SOURCE_LABEL,
    TEXT_WEIGHTS_PATH,
    THRESHOLD,
)
from paths import (  # noqa: E402
    CLEAN_IMAGE_CSV,
    CLEAN_TEXT_CSV,
    RESULT_PATH,
    late_fusion_result_path,
)
from utils import (  # noqa: E402
    compute_metrics,
    load_available_datasets,
    load_model,
    plot_score_space_fig7,
    preds_fusion,
    save_predictions,
)


FUSIONS = ("min", "max", "mean", "svm_rbf")

FUSION_TITLES = {
    "min": "MIN fusion",
    "max": "MAX fusion",
    "mean": "MEAN fusion",
    "svm_rbf": "RBF-SVM fusion",
}

FUSION_DIRECTORIES = {
    "min": "min",
    "max": "max",
    "mean": "mean",
    "svm_rbf": "svm-rbf",
}

SCENARIOS = (
    ("image_only", "Fusion-aware image-only attacks", "image"),
    ("text_only", "Fusion-aware text-only attacks", "text"),
    (
        "both",
        "Fusion-aware text + image attacks (split budget)",
        "both",
    ),
)

SCENARIO_RELATIVE_PATHS = {
    "text": Path("text-perturbed") / "perturbed_results.csv",
    "image": Path("image-perturbed") / "perturbed_results.csv",
    "both": Path("perturbed_results.csv"),
}

CLEAN_COLUMNS = {
    "index",
    "label",
    "score",
    "logit",
}

COMPONENT_COLUMNS = (
    "text_score",
    "text_logit",
    "image_score",
    "image_logit",
)

ATTACK_COLUMNS = CLEAN_COLUMNS.union(COMPONENT_COLUMNS)

TRAIN_SOURCE_COLUMNS = {
    "id",
    "title",
    "text",
    "label",
    "image_url",
}

DEFAULT_THRESHOLD = 0.5 if THRESHOLD is None else THRESHOLD

TRAIN_DATA_CSV = "data/Recovery/train_augmented.csv"
TRAIN_IMAGES_DIR = "data/Recovery/images"
TRAIN_TEXT_CSV = "results/Recovery/train/text/results.csv"
TRAIN_IMAGE_CSV = "results/Recovery/train/image/results.csv"
TRAIN_SVM_MODEL = "results/Recovery/train/svm_rbf.joblib"

LATE_FUSION_RESULTS_DIR = str(
    Path(RESULT_PATH)
    / "perturbed"
    / "late-fusion"
)

SCATTER_DIR = "figures/classification_results/scatter"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Create score-space figures from attacks generated directly "
            "against each late-fusion classifier."
        )
    )

    data_group = parser.add_argument_group("evaluation data")
    data_group.add_argument(
        "--test-text-csv",
        type=Path,
        default=Path(CLEAN_TEXT_CSV),
        help=(
            "Clean text predictions "
            f"(default: {CLEAN_TEXT_CSV})."
        ),
    )
    data_group.add_argument(
        "--test-image-csv",
        type=Path,
        default=Path(CLEAN_IMAGE_CSV),
        help=(
            "Clean image predictions "
            f"(default: {CLEAN_IMAGE_CSV})."
        ),
    )
    data_group.add_argument(
        "--late-fusion-results-dir",
        type=Path,
        default=Path(LATE_FUSION_RESULTS_DIR),
        help=(
            "Directory containing one attack subdirectory per fusion "
            f"(default: {LATE_FUSION_RESULTS_DIR})."
        ),
    )

    svm_group = parser.add_argument_group("RBF-SVM")
    svm_group.add_argument(
        "--train-text-csv",
        type=Path,
        default=Path(TRAIN_TEXT_CSV),
        help=(
            "Clean text training predictions "
            f"(default: {TRAIN_TEXT_CSV})."
        ),
    )
    svm_group.add_argument(
        "--train-image-csv",
        type=Path,
        default=Path(TRAIN_IMAGE_CSV),
        help=(
            "Clean image training predictions "
            f"(default: {TRAIN_IMAGE_CSV})."
        ),
    )
    svm_group.add_argument(
        "--svm-model",
        type=Path,
        default=Path(TRAIN_SVM_MODEL),
        help=(
            "Fitted probabilistic RBF-SVM. It is trained and saved here "
            f"when missing (default: {TRAIN_SVM_MODEL})."
        ),
    )
    svm_group.add_argument(
        "--force-svm-refit",
        action="store_true",
        help=(
            "Refit and overwrite the SVM even if --svm-model "
            "already exists."
        ),
    )
    svm_group.add_argument(
        "--svm-c",
        type=float,
        default=1.0,
    )
    svm_group.add_argument(
        "--svm-gamma",
        default="scale",
        help="RBF gamma: 'scale', 'auto', or a positive float.",
    )
    svm_group.add_argument(
        "--svm-input",
        choices=("scores", "logits"),
        default="scores",
        help=(
            "Two-feature input space used by the SVM "
            "(default: scores)."
        ),
    )
    svm_group.add_argument(
        "--svm-batch-size",
        type=int,
        default=50_000,
    )
    svm_group.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    train_group = parser.add_argument_group(
        "automatic training-set inference"
    )
    train_group.add_argument(
        "--train-data-csv",
        type=Path,
        default=Path(TRAIN_DATA_CSV),
        help=(
            "Recovery training annotations "
            f"(default: {TRAIN_DATA_CSV})."
        ),
    )
    train_group.add_argument(
        "--train-images-dir",
        type=Path,
        default=Path(TRAIN_IMAGES_DIR),
        help=(
            "Recovery image directory "
            f"(default: {TRAIN_IMAGES_DIR})."
        ),
    )
    train_group.add_argument(
        "--force-train-inference",
        action="store_true",
        help="Regenerate both clean training prediction CSVs.",
    )
    train_group.add_argument(
        "--train-subset-size",
        type=int,
        help=(
            "Optional number of initial training samples "
            "for a quick test."
        ),
    )
    train_group.add_argument(
        "--text-model-path",
        type=Path,
        default=Path(TEXT_WEIGHTS_PATH),
    )
    train_group.add_argument(
        "--image-model-path",
        type=Path,
        default=Path(IMAGE_WEIGHTS_PATH),
    )
    train_group.add_argument(
        "--name-llm",
        default=NAME_LLM,
    )
    train_group.add_argument(
        "--name-img-embed",
        default=NAME_IMG_EMBED,
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )
    train_group.add_argument(
        "--n-tokens",
        type=int,
        default=N_TOKENS,
    )
    train_group.add_argument(
        "--device",
        default=DEVICE_EVAL,
    )
    train_group.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    train_group.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    train_group.add_argument(
        "--merge-tokens",
        type=int,
        default=0,
    )
    train_group.add_argument(
        "--lora-alpha",
        type=int,
    )
    train_group.add_argument(
        "--lora-r",
        type=int,
    )
    train_group.add_argument(
        "--lora-dropout",
        type=float,
    )
    train_group.add_argument(
        "--use-lora",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    train_group.add_argument(
        "--set-params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Read LoRA parameters from the model filename, "
            "as in evaluation.py."
        ),
    )

    plot_group = parser.add_argument_group(
        "plotting and evaluation"
    )
    plot_group.add_argument(
        "--output-dir",
        type=Path,
        default=Path(SCATTER_DIR),
        help=(
            "Figure output directory "
            f"(default: {SCATTER_DIR})."
        ),
    )
    plot_group.add_argument(
        "--source-label",
        type=int,
        default=SOURCE_LABEL,
    )
    plot_group.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )
    plot_group.add_argument(
        "--grid",
        type=int,
        default=400,
    )
    plot_group.add_argument(
        "--pad",
        type=float,
        default=2.0,
    )
    plot_group.add_argument(
        "--dpi",
        type=int,
        default=200,
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    """Validate scalar arguments before loading any model."""
    if not 0.0 < args.threshold < 1.0:
        raise ValueError(
            "--threshold must be strictly between 0 and 1"
        )

    if args.grid < 2:
        raise ValueError(
            "--grid must be at least 2"
        )

    if args.pad < 0:
        raise ValueError(
            "--pad cannot be negative"
        )

    if args.dpi <= 0:
        raise ValueError(
            "--dpi must be positive"
        )

    if args.svm_c <= 0:
        raise ValueError(
            "--svm-c must be positive"
        )

    if args.svm_batch_size <= 0:
        raise ValueError(
            "--svm-batch-size must be positive"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive"
        )

    if args.n_tokens <= 0:
        raise ValueError(
            "--n-tokens must be positive"
        )

    if args.num_workers < 0:
        raise ValueError(
            "--num-workers cannot be negative"
        )

    if (
        args.train_subset_size is not None
        and args.train_subset_size <= 0
    ):
        raise ValueError(
            "--train-subset-size must be positive"
        )

    if args.source_label not in {0, 1}:
        raise ValueError(
            "--source-label must be 0 or 1"
        )


def model_args_for_modality(
    args: argparse.Namespace,
    modality: str,
) -> argparse.Namespace:
    """Build the namespace expected by ``utils.load_model``."""
    model_args = argparse.Namespace(
        **vars(args)
    )

    model_args.modality = modality
    model_args.model_path = str(
        args.text_model_path
        if modality == "text"
        else args.image_model_path
    )
    model_args.dataset = "Recovery"

    return model_args


def validate_train_inference_inputs(
    args: argparse.Namespace,
    modalities: tuple[str, ...],
) -> None:
    """Validate inputs needed when training inference must be run."""
    if not args.train_data_csv.is_file():
        raise FileNotFoundError(
            "Recovery training CSV does not exist: "
            f"{args.train_data_csv}"
        )

    if not args.train_images_dir.is_dir():
        raise FileNotFoundError(
            "Recovery training image directory does not exist: "
            f"{args.train_images_dir}"
        )

    columns = set(
        pd.read_csv(
            args.train_data_csv,
            nrows=0,
        ).columns
    )

    missing = TRAIN_SOURCE_COLUMNS.difference(
        columns
    )

    if missing:
        raise ValueError(
            "Training annotations are missing required columns: "
            f"{', '.join(sorted(missing))}"
        )

    model_paths = {
        "text": args.text_model_path,
        "image": args.image_model_path,
    }

    for modality in modalities:
        if not model_paths[modality].is_file():
            raise FileNotFoundError(
                f"{modality.capitalize()} model weights do not exist: "
                f"{model_paths[modality]}"
            )


def generate_train_predictions(
    args: argparse.Namespace,
    modality: str,
    output_path: Path,
) -> None:
    """Evaluate one clean model on the Recovery training set."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    import my_datasets

    device = torch.device(
        args.device
    )

    model_args = model_args_for_modality(
        args,
        modality,
    )

    model = None
    tokenizer = None
    processor = None
    dataset = None
    dataloader = None

    try:
        model, tokenizer, processor = load_model(
            device,
            model_args,
        )

        dataset_classes, load_functions = (
            load_available_datasets()
        )

        if "Recovery" not in dataset_classes:
            raise ValueError(
                "Recovery dataset class/load function was not found "
                "by utils.load_available_datasets()"
            )

        dataset = my_datasets.get_dataset(
            dataset_classes["Recovery"],
            load_functions["Recovery"],
            args.n_tokens,
            processor,
            tokenizer,
            str(args.train_data_csv),
            str(args.train_images_dir),
        )

        sampler = None

        if args.train_subset_size is not None:
            sampler = list(
                range(
                    min(
                        args.train_subset_size,
                        len(dataset),
                    )
                )
            )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        label_batches = []
        index_batches = []
        score_batches = []
        logit_batches = []

        for (
            images,
            labels,
            texts,
            _,
            indices,
        ) in tqdm(
            dataloader,
            desc=(
                f"Evaluating Recovery train "
                f"({modality})"
            ),
            total=len(dataloader),
        ):
            with torch.inference_mode():
                if modality == "text":
                    texts = texts.to(
                        device
                    )

                    scores, logits = model(
                        None,
                        texts,
                    )
                else:
                    images = images.to(
                        device
                    )

                    scores, logits = model(
                        images,
                        None,
                    )

            score_batches.append(
                scores
                .detach()
                .cpu()
                .reshape(-1)
            )

            logit_batches.append(
                logits
                .detach()
                .cpu()
                .reshape(-1)
            )

            label_batches.append(
                torch.as_tensor(
                    labels
                )
                .detach()
                .cpu()
                .reshape(-1)
            )

            index_batches.append(
                torch.as_tensor(
                    indices
                )
                .detach()
                .cpu()
                .reshape(-1)
            )

        if not score_batches:
            raise ValueError(
                "The Recovery training dataset produced no batches"
            )

        labels = torch.cat(
            label_batches
        ).numpy()

        indices = torch.cat(
            index_batches
        ).numpy()

        scores = torch.cat(
            score_batches
        ).numpy()

        logits = torch.cat(
            logit_batches
        ).numpy()

        lengths = {
            len(labels),
            len(indices),
            len(scores),
            len(logits),
        }

        if len(lengths) != 1:
            raise RuntimeError(
                f"Mismatched {modality} inference output lengths: "
                f"labels={len(labels)}, "
                f"indexes={len(indices)}, "
                f"scores={len(scores)}, "
                f"logits={len(logits)}"
            )

        predictions = (
            scores > args.threshold
        ).astype(np.int64)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_predictions(
            labels,
            predictions,
            scores,
            logits,
            indices,
            str(output_path),
        )

        print(
            f"Saved {output_path}"
        )

    finally:
        del dataloader
        del dataset
        del processor
        del tokenizer
        del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def ensure_train_predictions(
    args: argparse.Namespace,
) -> None:
    """Create missing clean training predictions."""
    outputs = {
        "text": args.train_text_csv,
        "image": args.train_image_csv,
    }

    if args.force_train_inference:
        modalities = (
            "text",
            "image",
        )
    else:
        modalities = tuple(
            modality
            for modality, path in outputs.items()
            if not path.is_file()
        )

    if not modalities:
        print(
            "Reusing training predictions: "
            f"{args.train_text_csv} and "
            f"{args.train_image_csv}"
        )
        return

    validate_train_inference_inputs(
        args,
        modalities,
    )

    for modality in modalities:
        generate_train_predictions(
            args,
            modality,
            outputs[modality],
        )


def read_prediction_csv(
    path: Path,
    description: str,
    required_columns: set[str],
) -> pd.DataFrame:
    """Read, validate, and index one prediction CSV."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} does not exist: {path}"
        )

    frame = pd.read_csv(
        path
    )

    missing = required_columns.difference(
        frame.columns
    )

    if missing:
        component_missing = set(
            COMPONENT_COLUMNS
        ).intersection(
            missing
        )

        if component_missing:
            raise ValueError(
                f"{description} is missing component columns: "
                f"{', '.join(sorted(component_missing))}. "
                "The fused 'score' and 'logit' columns are not "
                "enough to reconstruct a point in multimodal "
                "score space. Re-run the updated "
                "late_fusion_multimodal_attack.py that saves both "
                "component outputs."
            )

        raise ValueError(
            f"{description} is missing columns: "
            f"{', '.join(sorted(missing))}"
        )

    if frame.empty:
        raise ValueError(
            f"{description} is empty: {path}"
        )

    if (
        frame["index"].isna().any()
        or frame["index"].duplicated().any()
    ):
        raise ValueError(
            f"{description} contains missing or duplicate indexes"
        )

    selected_columns = [
        "index",
        "label",
        "score",
        "logit",
        *[
            column
            for column in COMPONENT_COLUMNS
            if column in frame.columns
        ],
    ]

    frame = frame.loc[
        :,
        selected_columns,
    ].copy()

    numeric_columns = [
        column
        for column in frame.columns
        if column != "index"
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

    labels = frame["label"].to_numpy(
        dtype=np.float64
    )

    if (
        not np.all(np.isfinite(labels))
        or not np.allclose(
            labels,
            np.round(labels),
        )
    ):
        raise ValueError(
            f"{description} labels must be finite integers"
        )

    frame["label"] = np.round(
        labels
    ).astype(np.int64)

    value_columns = [
        column
        for column in numeric_columns
        if column != "label"
    ]

    values = frame[
        value_columns
    ].to_numpy(
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            f"{description} contains non-finite numeric values"
        )

    score_columns = [
        column
        for column in (
            "score",
            "text_score",
            "image_score",
        )
        if column in frame.columns
    ]

    for column in score_columns:
        scores = frame[
            column
        ].to_numpy(
            dtype=np.float64
        )

        if np.any(
            (scores < 0.0)
            | (scores > 1.0)
        ):
            raise ValueError(
                f"{description} column {column!r} "
                "is not probabilistic"
            )

    return frame.set_index(
        "index",
        drop=True,
    )


def read_clean_predictions(
    path: Path,
    description: str,
) -> pd.DataFrame:
    """Read a standard unimodal prediction CSV."""
    return read_prediction_csv(
        path,
        description,
        CLEAN_COLUMNS,
    )


def load_clean_pair(
    text_path: Path,
    image_path: Path,
    description: str,
) -> pd.DataFrame:
    """Align clean text and image predictions."""
    text = read_clean_predictions(
        text_path,
        f"{description} text CSV",
    )

    image = read_clean_predictions(
        image_path,
        f"{description} image CSV",
    )

    text_only = text.index.difference(
        image.index
    )

    image_only = image.index.difference(
        text.index
    )

    if (
        len(text_only)
        or len(image_only)
    ):
        raise ValueError(
            f"{description} text/image indexes differ: "
            f"{len(text_only)} text-only and "
            f"{len(image_only)} image-only rows"
        )

    image = image.reindex(
        text.index
    )

    if not np.array_equal(
        text["label"].to_numpy(),
        image["label"].to_numpy(),
    ):
        raise ValueError(
            f"{description} text/image labels differ"
        )

    return pd.DataFrame(
        {
            "label": text["label"],
            "text_score": text["score"],
            "text_logit": text["logit"],
            "image_score": image["score"],
            "image_logit": image["logit"],
        },
        index=text.index,
    )


def parse_gamma(
    value: str,
) -> str | float:
    """Parse an sklearn-compatible RBF gamma value."""
    if value in {
        "scale",
        "auto",
    }:
        return value

    try:
        gamma = float(
            value
        )
    except ValueError as error:
        raise ValueError(
            "--svm-gamma must be 'scale', 'auto', "
            "or a positive float"
        ) from error

    if gamma <= 0:
        raise ValueError(
            "A numeric --svm-gamma must be positive"
        )

    return gamma


def svm_features(
    data: pd.DataFrame,
    input_space: str,
) -> np.ndarray:
    """Return SVM features ordered as [text, image]."""
    suffix = (
        "score"
        if input_space == "scores"
        else "logit"
    )

    return data[
        [
            f"text_{suffix}",
            f"image_{suffix}",
        ]
    ].to_numpy(
        dtype=np.float64
    )


def validate_svm(
    model: Any,
) -> None:
    """Validate the fitted SVM interface and positive class."""
    if (
        not hasattr(
            model,
            "predict_proba",
        )
        or not hasattr(
            model,
            "classes_",
        )
    ):
        raise TypeError(
            "The RBF-SVM must expose predict_proba and classes_; "
            "train SVC with probability=True"
        )

    classes = np.asarray(
        model.classes_
    )

    if np.count_nonzero(
        classes == 1
    ) != 1:
        raise ValueError(
            "The RBF-SVM must contain positive class 1; "
            f"got {classes}"
        )


def load_or_fit_svm(
    args: argparse.Namespace,
) -> tuple[Any, bool]:
    """Load the attack-time SVM or fit it on clean data."""
    if (
        args.svm_model.is_file()
        and not args.force_svm_refit
        and not args.force_train_inference
    ):
        model = joblib.load(
            args.svm_model
        )

        validate_svm(
            model
        )

        print(
            f"Loaded {args.svm_model}"
        )

        return model, False

    ensure_train_predictions(
        args
    )

    train = load_clean_pair(
        args.train_text_csv,
        args.train_image_csv,
        "training set",
    )

    labels = train[
        "label"
    ].to_numpy(
        dtype=np.int64
    )

    if set(
        np.unique(labels)
    ) != {
        0,
        1,
    }:
        raise ValueError(
            "RBF-SVM training labels must contain classes 0 and 1"
        )

    model = Pipeline(
        steps=(
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=args.svm_c,
                    gamma=parse_gamma(
                        args.svm_gamma
                    ),
                    probability=True,
                    random_state=args.seed,
                ),
            ),
        )
    )

    model.fit(
        svm_features(
            train,
            args.svm_input,
        ),
        labels,
    )

    validate_svm(
        model
    )

    args.svm_model.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        args.svm_model,
    )

    print(
        f"Saved {args.svm_model}"
    )

    return model, True


def svm_scores(
    model: Any,
    data: pd.DataFrame,
    input_space: str,
    batch_size: int,
) -> np.ndarray:
    """Predict positive-class SVM probabilities."""
    positive_index = int(
        np.flatnonzero(
            np.asarray(
                model.classes_
            )
            == 1
        )[0]
    )

    features = svm_features(
        data,
        input_space,
    )

    scores = np.empty(
        len(data),
        dtype=np.float64,
    )

    for start in range(
        0,
        len(data),
        batch_size,
    ):
        stop = min(
            start + batch_size,
            len(data),
        )

        scores[start:stop] = (
            model.predict_proba(
                features[start:stop]
            )[:, positive_index]
        )

    return scores


def fused_scores(
    data: pd.DataFrame,
    fusion: str,
    svm_model: Any,
    svm_input: str,
    svm_batch_size: int,
) -> np.ndarray:
    """Fuse the two component scores."""
    if fusion == "svm_rbf":
        return svm_scores(
            svm_model,
            data,
            svm_input,
            svm_batch_size,
        )

    return np.asarray(
        preds_fusion(
            data[
                "text_score"
            ].to_numpy(
                dtype=np.float64
            ),
            data[
                "image_score"
            ].to_numpy(
                dtype=np.float64
            ),
            fusion,
        ),
        dtype=np.float64,
    ).reshape(-1)


def safe_logit(
    scores: np.ndarray,
) -> np.ndarray:
    """Convert probabilities to finite logits."""
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    eps = np.finfo(
        np.float64
    ).eps

    clipped = np.clip(
        scores,
        eps,
        1.0 - eps,
    )

    return (
        np.log(clipped)
        - np.log1p(-clipped)
    )


def attack_csv_path(
    base_dir: Path,
    fusion: str,
    scenario: str,
) -> Path:
    """Resolve the dedicated CSV for a fusion/scenario pair."""
    path = Path(
        late_fusion_result_path(fusion, scenario, base_dir)
    )

    # Accept the old underscore spelling if present.
    if (
        fusion == "svm_rbf"
        and not path.is_file()
    ):
        alias = (
            base_dir
            / "svm_rbf"
            / SCENARIO_RELATIVE_PATHS[scenario]
        )

        if alias.is_file():
            return alias

    return path


def validate_attack_metadata(
    csv_path: Path,
    fusion: str,
    scenario: str,
    svm_input: str,
) -> None:
    """Check the attack provenance saved beside a CSV."""
    metadata_path = (
        csv_path.parent
        / "parameters.json"
    )

    if not metadata_path.is_file():
        warnings.warn(
            f"Attack metadata is missing: {metadata_path}",
            stacklevel=2,
        )
        return

    try:
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            metadata = json.load(
                handle
            )
    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError(
            f"Cannot read attack metadata: {metadata_path}"
        ) from error

    recorded_scenario = metadata.get(
        "Scenario"
    )

    if (
        recorded_scenario is not None
        and recorded_scenario != scenario
    ):
        raise ValueError(
            f"{metadata_path} records scenario "
            f"{recorded_scenario!r}, expected {scenario!r}"
        )

    fusion_metadata = metadata.get(
        "Fusion",
        {},
    )

    expected_fusion = FUSION_DIRECTORIES[
        fusion
    ]

    recorded_fusion = fusion_metadata.get(
        "Type"
    )

    if recorded_fusion is not None:
        recorded_fusion = str(
            recorded_fusion
        ).replace(
            "_",
            "-",
        )

        if recorded_fusion != expected_fusion:
            raise ValueError(
                f"{metadata_path} records fusion "
                f"{recorded_fusion!r}, "
                f"expected {expected_fusion!r}"
            )

    if fusion == "svm_rbf":
        recorded_input = fusion_metadata.get(
            "SVM Input"
        )

        if (
            recorded_input is not None
            and recorded_input != svm_input
        ):
            raise ValueError(
                f"{metadata_path} records SVM input "
                f"{recorded_input!r}, but the plot uses "
                f"{svm_input!r}"
            )


def load_attacked_panel(
    clean: pd.DataFrame,
    clean_fused_scores: np.ndarray,
    csv_path: Path,
    description: str,
) -> pd.DataFrame:
    """Overlay a fusion-aware attack onto the clean set."""
    attack = read_prediction_csv(
        csv_path,
        description,
        ATTACK_COLUMNS,
    ).rename(
        columns={
            "score": "fused_score",
            "logit": "fused_logit",
        }
    )

    unknown = attack.index.difference(
        clean.index
    )

    if len(unknown):
        raise ValueError(
            f"{description} contains {len(unknown)} indexes "
            "absent from the clean test set"
        )

    common = clean.index.intersection(
        attack.index,
        sort=False,
    )

    if not np.array_equal(
        clean.loc[
            common,
            "label",
        ].to_numpy(),
        attack.loc[
            common,
            "label",
        ].to_numpy(),
    ):
        raise ValueError(
            f"{description} labels differ from the clean test labels"
        )

    missing = clean.index.difference(
        attack.index
    )

    if len(missing):
        warnings.warn(
            f"{description} is missing {len(missing)} test rows; "
            "their clean component and fused values will be retained",
            stacklevel=2,
        )

    result = clean.copy()

    result["fused_score"] = np.asarray(
        clean_fused_scores,
        dtype=np.float64,
    )

    result["fused_logit"] = safe_logit(
        clean_fused_scores
    )

    overlay_columns = [
        *COMPONENT_COLUMNS,
        "fused_score",
        "fused_logit",
    ]

    result.loc[
        common,
        overlay_columns,
    ] = attack.loc[
        common,
        overlay_columns,
    ].to_numpy()

    return result


def validate_stored_fusion(
    attacked: pd.DataFrame,
    fusion: str,
    svm_model: Any,
    args: argparse.Namespace,
    description: str,
) -> None:
    """Check that component outputs reproduce the fused output."""
    recomputed = fused_scores(
        attacked,
        fusion,
        svm_model,
        args.svm_input,
        args.svm_batch_size,
    )

    stored = attacked[
        "fused_score"
    ].to_numpy(
        dtype=np.float64
    )

    max_error = float(
        np.max(
            np.abs(
                recomputed - stored
            )
        )
    )

    if (
        fusion != "svm_rbf"
        and max_error > 1e-4
    ):
        raise ValueError(
            f"{description} component scores do not reproduce "
            "its fused score "
            f"(maximum absolute error={max_error:.6f})"
        )

    # The attack uses a differentiable PyTorch reproduction of
    # libsvm's calibrated probability. A small difference from
    # sklearn can therefore occur.
    if (
        fusion == "svm_rbf"
        and max_error > 0.05
    ):
        warnings.warn(
            f"{description} differs noticeably from sklearn "
            "SVM probabilities "
            f"(maximum absolute error={max_error:.4f})",
            stacklevel=2,
        )


def evaluate(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute the project's binary classification metrics."""
    predictions = (
        scores > threshold
    ).astype(np.int64)

    metrics, _, _, _ = compute_metrics(
        labels,
        predictions,
        scores,
    )

    return {
        key: float(value)
        for key, value in metrics.items()
    }


def panel_title(
    fusion: str,
    clean_metrics: dict[str, float],
    attacked_metrics: dict[str, float],
) -> str:
    """Build the clean-to-attacked metric summary."""
    return (
        f"{FUSION_TITLES[fusion]}\n"
        f"F1: {clean_metrics['f1']:.3f} "
        f"→ {attacked_metrics['f1']:.3f}   |   "
        f"AUC: {clean_metrics['auc']:.3f} "
        f"→ {attacked_metrics['auc']:.3f}"
    )


def validate_test_labels(
    test: pd.DataFrame,
) -> None:
    """Require a binary test set."""
    labels = test[
        "label"
    ].to_numpy(
        dtype=np.int64
    )

    if set(
        np.unique(labels)
    ) != {
        0,
        1,
    }:
        raise ValueError(
            "Test labels must contain exactly classes 0 and 1"
        )


def validate_attack_artifacts(base_dir: Path) -> None:
    """Report every missing or legacy fusion/scenario CSV at once."""
    problems: list[str] = []
    for _, _, scenario in SCENARIOS:
        for fusion in FUSIONS:
            path = attack_csv_path(base_dir, fusion, scenario)
            if not path.is_file():
                problems.append(f"missing file: {path}")
                continue

            columns = set(pd.read_csv(path, nrows=0).columns)
            missing_columns = ATTACK_COLUMNS.difference(columns)
            if missing_columns:
                problems.append(
                    f"{path}: missing columns "
                    f"{', '.join(sorted(missing_columns))}"
                )

    if not problems:
        return

    details = "\n".join(f"  - {problem}" for problem in problems)
    raise ValueError(
        "Late-fusion plotting requires 12 complete attack CSVs:\n"
        f"{details}\n"
        "Re-run run_late_fusion_attacks.py to regenerate them."
    )


def main() -> None:
    """Load the 12 attack files and create three figures."""
    args = parse_args()

    validate_args(
        args
    )

    validate_attack_artifacts(args.late_fusion_results_dir)

    clean_test = load_clean_pair(
        args.test_text_csv,
        args.test_image_csv,
        "clean test set",
    )

    validate_test_labels(
        clean_test
    )

    # Use the same fitted classifier used by the SVM-aware attack.
    svm_model, _ = load_or_fit_svm(
        args
    )

    labels = clean_test[
        "label"
    ].to_numpy(
        dtype=np.int64
    )

    clean_scores = {
        fusion: fused_scores(
            clean_test,
            fusion,
            svm_model,
            args.svm_input,
            args.svm_batch_size,
        )
        for fusion in FUSIONS
    }

    clean_metrics = {
        fusion: evaluate(
            labels,
            scores,
            args.threshold,
        )
        for fusion, scores in clean_scores.items()
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_output: dict[
        str,
        dict[
            str,
            dict[str, Any],
        ],
    ] = {}

    for (
        scenario_key,
        scenario_title,
        scenario,
    ) in SCENARIOS:
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(18, 15),
            constrained_layout=True,
        )

        figure.suptitle(
            scenario_title,
            fontsize=18,
            fontweight="bold",
        )

        scenario_metrics: dict[
            str,
            dict[str, Any],
        ] = {}

        for axis, fusion in zip(
            axes.flat,
            FUSIONS,
        ):
            csv_path = attack_csv_path(
                args.late_fusion_results_dir,
                fusion,
                scenario,
            )

            validate_attack_metadata(
                csv_path,
                fusion,
                scenario,
                args.svm_input,
            )

            description = (
                f"{FUSION_TITLES[fusion]} "
                f"{scenario} attack CSV"
            )

            attacked = load_attacked_panel(
                clean_test,
                clean_scores[fusion],
                csv_path,
                description,
            )

            validate_stored_fusion(
                attacked,
                fusion,
                svm_model,
                args,
                description,
            )

            attacked_scores = attacked[
                "fused_score"
            ].to_numpy(
                dtype=np.float64
            )

            attacked_metrics = evaluate(
                labels,
                attacked_scores,
                args.threshold,
            )

            scenario_metrics[fusion] = {
                "attack_csv": str(
                    csv_path
                ),
                "clean": clean_metrics[
                    fusion
                ],
                "attacked": attacked_metrics,
            }

            plot_score_space_fig7(
                y_true=labels,
                text_clean=clean_test[
                    "text_logit"
                ].to_numpy(
                    dtype=np.float64
                ),
                image_clean=clean_test[
                    "image_logit"
                ].to_numpy(
                    dtype=np.float64
                ),
                text_pert=attacked[
                    "text_logit"
                ].to_numpy(
                    dtype=np.float64
                ),
                image_pert=attacked[
                    "image_logit"
                ].to_numpy(
                    dtype=np.float64
                ),
                fusion=fusion,
                threshold=args.threshold,
                source_label=args.source_label,
                grid=args.grid,
                pad=args.pad,
                svm_model=(
                    svm_model
                    if fusion == "svm_rbf"
                    else None
                ),
                svm_input=args.svm_input,
                svm_positive_label=1,
                svm_batch_size=args.svm_batch_size,
                attack_mode=scenario,
                title=panel_title(
                    fusion,
                    clean_metrics[fusion],
                    attacked_metrics,
                ),
                ax=axis,
                add_colorbar=True,
            )

            print(
                f"Loaded {csv_path}"
            )

        output_path = (
            args.output_dir
            / f"score_space_{scenario_key}.png"
        )

        figure.savefig(
            output_path,
            dpi=args.dpi,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )

        metrics_output[
            scenario_key
        ] = scenario_metrics

        print(
            f"Saved {output_path}"
        )

    metrics_path = (
        args.output_dir
        / "score_space_metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metrics_output,
            handle,
            indent=2,
        )
        handle.write(
            "\n"
        )

    print(
        f"Saved {metrics_path}"
    )


if __name__ == "__main__":
    main()