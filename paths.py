import os

# Results root
RESULT_PATH = "results/Recovery/classification_results"
CLEAN_BASE  = os.path.join(RESULT_PATH, "clean")
PERT_BASE   = os.path.join(RESULT_PATH, "perturbed")

# Clean training-set predictions used to fit late-fusion classifiers
TRAIN_BASE = "results/Recovery/train"
TRAIN_TEXT_CSV = os.path.join(TRAIN_BASE, "text", "results.csv")
TRAIN_IMAGE_CSV = os.path.join(TRAIN_BASE, "image", "results.csv")
TRAIN_SVM_MODEL = os.path.join(TRAIN_BASE, "svm_rbf.joblib")

# Recovery training data
TRAIN_DATA_CSV = "data/Recovery/train_augmented.csv"
TRAIN_IMAGES_DIR = "data/Recovery/images"

# Clean - CSVs
CLEAN_TEXT_CSV  = os.path.join(CLEAN_BASE, "text",  "results.csv")
CLEAN_IMAGE_CSV = os.path.join(CLEAN_BASE, "image", "results.csv")

# Clean - parameters.json
CLEAN_TEXT_PARAMS  = os.path.join(CLEAN_BASE, "text",           "parameters.json")
CLEAN_IMAGE_PARAMS = os.path.join(CLEAN_BASE, "image",          "parameters.json")
CLEAN_FF_PARAMS    = os.path.join(CLEAN_BASE, "feature-fusion", "parameters.json")

# Perturbed - output directories
LATE_FUSION_RESULTS_DIR = os.path.join(PERT_BASE, "late-fusion")
PERT_IMAGE_DIR = os.path.join(PERT_BASE, "image")
PERT_TEXT_DIR  = os.path.join(PERT_BASE, "text")
PERT_FF_DIR    = os.path.join(PERT_BASE, "feature-fusion")

# Perturbed - CSVs
PER_TEXT_CSV  = os.path.join(PERT_TEXT_DIR,  "perturbed_results.csv")
PER_IMAGE_CSV = os.path.join(PERT_IMAGE_DIR, "perturbed_results.csv")

# Perturbed - parameters.json
PER_TEXT_PARAMS  = os.path.join(PERT_TEXT_DIR,  "parameters.json")
PER_IMAGE_PARAMS = os.path.join(PERT_IMAGE_DIR, "parameters.json")

# Perturbed sample dumps - generated images + texts
DATA_PERTURBED_BASE  = "data_perturbed"
DATA_PERTURBED_IMAGE = os.path.join(DATA_PERTURBED_BASE, "image")
DATA_PERTURBED_TEXT  = os.path.join(DATA_PERTURBED_BASE, "text")
LATE_FUSION_DATA_DIR = os.path.join(DATA_PERTURBED_BASE, "late-fusion")
DATA_PERTURBED_FF    = os.path.join(DATA_PERTURBED_BASE, "feature-fusion")

# ROC directories
ROC_BASE     = "figures/classification_results/rocs"
ROC_SETS_DIR  = os.path.join(ROC_BASE, "roc_sets")
ROC_PLOTS_DIR = os.path.join(ROC_BASE, "roc_plots")

# Late-fusion pipeline artifacts
LATE_FUSION_FIGURES_DIR = "figures/classification_results/scatter"
LATE_FUSION_LOG_DIR = "logs/late_fusion_attacks"

LATE_FUSION_SCENARIO_FILES = {
    "text": os.path.join("text-perturbed", "perturbed_results.csv"),
    "image": os.path.join("image-perturbed", "perturbed_results.csv"),
    "both": "perturbed_results.csv",
}


def late_fusion_directory_name(fusion):
    """Return the canonical directory/CLI spelling for a fusion method."""
    return str(fusion).replace("_", "-")


def late_fusion_scenario_path(fusion_dir, attack_scope):
    """Return one scenario CSV below an already resolved fusion directory."""
    try:
        relative_path = LATE_FUSION_SCENARIO_FILES[attack_scope]
    except KeyError as error:
        valid = ", ".join(LATE_FUSION_SCENARIO_FILES)
        raise ValueError(
            f"Unknown late-fusion attack scope {attack_scope!r}; "
            f"expected one of: {valid}"
        ) from error
    return os.path.join(os.fspath(fusion_dir), relative_path)


def late_fusion_result_path(
    fusion,
    attack_scope,
    base_dir=LATE_FUSION_RESULTS_DIR,
):
    """Return the canonical CSV path for a fusion/scenario pair."""
    fusion_dir = os.path.join(
        os.fspath(base_dir),
        late_fusion_directory_name(fusion),
    )
    return late_fusion_scenario_path(fusion_dir, attack_scope)
