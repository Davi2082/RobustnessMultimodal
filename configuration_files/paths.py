import os
from configuration_files.configuration import DATASET

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CONFIG_DIR)
CHECKPOINTS_ROOT = os.path.join(PROJECT_ROOT, "checkpoints")


def dataset_weights_dir(dataset=None):
    return os.path.join(CHECKPOINTS_ROOT, dataset or DATASET)


# ---------------------------------------------------------------------------
# New result layout:
#   results/<dataset>/<model_or_fusion>/clean/
#   results/<dataset>/<model_or_fusion>/perturbed/<attack>/
#   results/<dataset>/<model_or_fusion>/feature_ablation/
# ---------------------------------------------------------------------------

def dataset_result_root(dataset=None):
    """Top-level results directory for a dataset."""
    return f"results/{dataset or DATASET}"


def model_clean_dir(model_or_fusion, dataset=None):
    """results/<dataset>/<model_or_fusion>/clean/"""
    return os.path.join(dataset_result_root(dataset), model_or_fusion, "clean")


def model_perturbed_dir(model_or_fusion, attack, dataset=None):
    """results/<dataset>/<model_or_fusion>/perturbed/<attack>/"""
    return os.path.join(dataset_result_root(dataset), model_or_fusion, "perturbed", attack)


def model_ablation_dir(model_or_fusion, dataset=None):
    """results/<dataset>/<model_or_fusion>/feature_ablation/"""
    return os.path.join(dataset_result_root(dataset), model_or_fusion, "feature_ablation")


def clean_text_params(dataset=None):
    return os.path.join(model_clean_dir("text", dataset), "parameters.json")


def clean_image_params(dataset=None):
    return os.path.join(model_clean_dir("image", dataset), "parameters.json")


def clean_ff_params(dataset=None):
    return os.path.join(model_clean_dir("feature-fusion", dataset), "parameters.json")


def dataset_train_base(dataset=None):
    return f"results/{dataset or DATASET}/train"


def train_svm_model(dataset=None):
    return os.path.join(dataset_train_base(dataset), "svm_rbf.joblib")


# Convenience constants for the active DATASET
DATASET_WEIGHTS_DIR = dataset_weights_dir()
RESULT_PATH = dataset_result_root()  # backward compat for plot scripts
CLEAN_TEXT_PARAMS = clean_text_params()
CLEAN_IMAGE_PARAMS = clean_image_params()
CLEAN_FF_PARAMS = clean_ff_params()

# Training data
TRAIN_BASE = dataset_train_base()
TRAIN_TEXT_CSV = os.path.join(TRAIN_BASE, "text", "results.csv")
TRAIN_IMAGE_CSV = os.path.join(TRAIN_BASE, "image", "results.csv")
TRAIN_SVM_MODEL = train_svm_model()
TRAIN_DATA_CSV = f"data/{DATASET}/train_augmented.csv"
TRAIN_IMAGES_DIR = f"data/{DATASET}/images"

# Perturbed sample dumps (generated images + texts), grouped per dataset
def dataset_perturbed_base(dataset=None):
    return os.path.join("data_perturbed", dataset or DATASET)


def late_fusion_data_dir(dataset=None):
    return os.path.join(dataset_perturbed_base(dataset), "late-fusion")


DATA_PERTURBED_BASE = dataset_perturbed_base()
DATA_PERTURBED_IMAGE = os.path.join(DATA_PERTURBED_BASE, "image")
DATA_PERTURBED_TEXT = os.path.join(DATA_PERTURBED_BASE, "text")
LATE_FUSION_DATA_DIR = late_fusion_data_dir()
DATA_PERTURBED_FF = os.path.join(DATA_PERTURBED_BASE, "feature-fusion")

# ROC / figures
ROC_BASE = "figures/classification_results/rocs"
ROC_SETS_DIR = os.path.join(ROC_BASE, "roc_sets")
ROC_PLOTS_DIR = os.path.join(ROC_BASE, "roc_plots")
LATE_FUSION_FIGURES_DIR = "figures/classification_results/scatter"
LATE_FUSION_LOG_DIR = "logs/late_fusion_attacks"

# Dataset annotations / images
DATASET_ROOTS = ("data",)


def dataset_images_dir(dataset):
    for root in DATASET_ROOTS:
        candidate = os.path.join(root, dataset, "images")
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"No images directory for {dataset} under {' or '.join(DATASET_ROOTS)}"
    )


def dataset_annotations(dataset, split="test"):
    import glob as _glob
    for root in DATASET_ROOTS:
        matches = sorted(_glob.glob(os.path.join(root, dataset, f"{split}.*")))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No {split} annotations for {dataset} under {' or '.join(DATASET_ROOTS)}"
    )
