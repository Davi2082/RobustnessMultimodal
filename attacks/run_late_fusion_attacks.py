"""Run the complete late-fusion robustness experiment.

The runner first fits the RBF-SVM on clean training predictions, then runs the
three independent attack scopes for every fusion method, and finally creates
the three requested four-panel figures. Single-modality scenarios use the
full corresponding attack budget; only ``both`` uses half per modality.
"""

import joblib
import numpy as np
import pandas as pd
import subprocess
import sys
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from configuration_files.configuration import (
    LATE_FUSION_ATTACK_SCOPES,
    LATE_FUSION_METHODS,
    LATE_FUSION_SVM_C,
    LATE_FUSION_SVM_GAMMA,
    LATE_FUSION_SVM_INPUT,
    LATE_FUSION_SVM_SEED,
    late_fusion_attack_budgets,
)
from configuration_files.paths import (
    LATE_FUSION_FIGURES_DIR,
    LATE_FUSION_LOG_DIR,
    LATE_FUSION_RESULTS_DIR,
    TRAIN_IMAGE_CSV,
    TRAIN_SVM_MODEL,
    TRAIN_TEXT_CSV,
)


PYTHON = sys.executable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACK_SCRIPT = PROJECT_ROOT / "scripts" / "late_fusion_perturbation.py"
PLOT_SCRIPT = PROJECT_ROOT / "scripts" / "plot_multimodal_score_space.py"
LOG_DIR = PROJECT_ROOT / LATE_FUSION_LOG_DIR
TRAIN_TEXT_PATH = PROJECT_ROOT / TRAIN_TEXT_CSV
TRAIN_IMAGE_PATH = PROJECT_ROOT / TRAIN_IMAGE_CSV
SVM_MODEL_PATH = PROJECT_ROOT / TRAIN_SVM_MODEL
RESULTS_PATH = PROJECT_ROOT / LATE_FUSION_RESULTS_DIR
FIGURES_PATH = PROJECT_ROOT / LATE_FUSION_FIGURES_DIR

ATTACK_CONFIGURATIONS = tuple(
    (fusion, attack_scope)
    for fusion in LATE_FUSION_METHODS
    for attack_scope in LATE_FUSION_ATTACK_SCOPES
)


def read_training_predictions(
    path: Path,
    modality: str,
) -> pd.DataFrame:
    """Read one clean training prediction CSV and index it by sample id."""
    if not path.is_file():
        raise FileNotFoundError(f"{modality} training predictions not found: {path}")

    frame = pd.read_csv(path)
    value_column = "score" if LATE_FUSION_SVM_INPUT == "scores" else "logit"
    required = {"index", "label", value_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{modality} training predictions are missing: "
            f"{', '.join(sorted(missing))}"
        )

    if frame.empty:
        raise ValueError(f"{modality} training predictions are empty: {path}")
    if frame["index"].isna().any() or frame["index"].duplicated().any():
        raise ValueError(f"{modality} training predictions have invalid indexes")

    for column in ("label", value_column):
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    values = frame[["label", value_column]].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{modality} training predictions contain non-finite values")

    return frame.set_index("index").loc[:, ["label", value_column]]


def fit_late_fusion_svm() -> None:
    """Fit the attack-time RBF-SVM only on clean training predictions."""
    text = read_training_predictions(TRAIN_TEXT_PATH, "text")
    image = read_training_predictions(TRAIN_IMAGE_PATH, "image")

    text_only = text.index.difference(image.index)
    image_only = image.index.difference(text.index)
    if len(text_only) or len(image_only):
        raise ValueError(
            "Training prediction indexes differ: "
            f"{len(text_only)} text-only and {len(image_only)} image-only rows"
        )

    image = image.reindex(text.index)
    if not np.array_equal(text["label"].to_numpy(), image["label"].to_numpy()):
        raise ValueError("Text/image training labels differ")

    labels = text["label"].to_numpy(dtype=np.int64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("SVM training labels must contain classes 0 and 1")

    value_column = "score" if LATE_FUSION_SVM_INPUT == "scores" else "logit"
    features = np.column_stack(
        (
            text[value_column].to_numpy(dtype=np.float64),
            image[value_column].to_numpy(dtype=np.float64),
        )
    )

    model = Pipeline(
        steps=(
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=LATE_FUSION_SVM_C,
                    gamma=LATE_FUSION_SVM_GAMMA,
                    probability=True,
                    random_state=LATE_FUSION_SVM_SEED,
                ),
            ),
        )
    )
    model.fit(features, labels)

    SVM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SVM_MODEL_PATH)
    print(f"[pipeline] fitted training-set RBF-SVM -> {SVM_MODEL_PATH}")


def run(command: list[str], log_file: str, label: str) -> None:
    """Run one command, save its output, and stop on failure."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_file

    print(f"[pipeline] {label} ...", flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if result.returncode != 0:
        print(
            f"[pipeline] FAILED: {label} "
            f"(exit {result.returncode}) -> {log_path}"
        )
        sys.exit(result.returncode)

    print(f"[pipeline] done:   {label}")



def attack_budget_arguments(attack_scope: str) -> tuple[list[str], str]:
    """Return the active, scope-specific budget arguments."""
    budgets = late_fusion_attack_budgets(attack_scope)

    arguments: list[str] = []
    labels: list[str] = []

    if attack_scope in {"text", "both"}:
        max_variants = budgets["max_variants"]
        arguments.extend(
            ["--max-variants", str(max_variants)]
        )
        labels.append(
            f"TRePAT max variants={max_variants}"
        )

    if attack_scope in {"image", "both"}:
        pgd_iters = budgets["pgd_iters"]
        arguments.extend(
            ["--pgd-iters", str(pgd_iters)]
        )
        labels.append(
            f"PGD iterations={pgd_iters}"
        )

    return arguments, ", ".join(labels)


def main() -> None:
    """Fit the SVM, run all attacks, and generate the three figures."""
    if not ATTACK_SCRIPT.is_file():
        raise FileNotFoundError(f"Attack script not found: {ATTACK_SCRIPT}")
    if not PLOT_SCRIPT.is_file():
        raise FileNotFoundError(f"Plot script not found: {PLOT_SCRIPT}")

    fit_late_fusion_svm()

    for fusion, attack_scope in ATTACK_CONFIGURATIONS:
        budget_arguments, budget_label = attack_budget_arguments(attack_scope)
        run(
            [
                PYTHON,
                str(ATTACK_SCRIPT),
                "--fusion",
                fusion,
                "--attack-scope",
                attack_scope,
                "--svm-model",
                str(SVM_MODEL_PATH),
                "--svm-input",
                LATE_FUSION_SVM_INPUT,
                "--output-dir",
                str(RESULTS_PATH / fusion),
                *budget_arguments,
            ],
            log_file=f"{fusion}_{attack_scope}.log",
            label=(
                f"Late-fusion attack - {fusion} - {attack_scope} "
                f"({budget_label})"
            ),
        )

    run(
        [
            PYTHON,
            str(PLOT_SCRIPT),
            "--svm-model",
            str(SVM_MODEL_PATH),
            "--svm-input",
            LATE_FUSION_SVM_INPUT,
            "--late-fusion-results-dir",
            str(RESULTS_PATH),
            "--output-dir",
            str(FIGURES_PATH),
        ],
        log_file="plot_score_spaces.log",
        label="Late-fusion score-space figures",
    )

    print(f"\n[pipeline] Complete. Figures saved in {FIGURES_PATH}")


if __name__ == "__main__":
    main()