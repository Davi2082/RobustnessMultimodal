"""Pipeline settings, loaded from config.yaml at the project root.

Edit values in ../config.yaml, not here — this module only turns that file
into the constants/functions the rest of the codebase imports.
"""

import os
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f)


def _number(value):
    """Accept a plain number or an 'a/b' fraction string (e.g. '16/255')."""
    if isinstance(value, str) and "/" in value:
        num, denom = value.split("/")
        return float(num) / float(denom)
    return value


RAND_SEED = _CFG["rand_seed"]

# Pipeline stage toggles (0/1 in config.yaml)
TRAIN_PIPELINE = bool(_CFG.get("train_pipeline", 0))
CLEAN_PIPELINE = bool(_CFG.get("clean_pipeline", 1))
ABLATION_PIPELINE = bool(_CFG.get("ablation_pipeline", 0))
ADVERSARIAL_PIPELINE = bool(_CFG.get("adversarial_pipeline", 0))

# Dataset selection
DATASETS = _CFG["datasets"]  # datasets available for full-pipeline sweeps
DATASET = DATASETS[0]  # active dataset for scripts without their own --dataset override

# Full-pipeline sweep membership (run_clean.py, run_multimodal_attacks.py, run_scripts.sh)
PIPELINE_LATE_FUSION_MODES = _CFG["pipeline_late_fusion_modes"]
PIPELINE_FUSIONS = PIPELINE_LATE_FUSION_MODES + ["feature-fusion"]
PIPELINE_ATTACKS = _CFG["pipeline_attacks"]

# Which modality each attack type perturbs.
ATTACK_SCOPE = {
    "pgd": "image",
    "trepat": "text",
    "sum": "both",
    "interleaved": "both",
    "joint": "both",
}

# Models
NAME_LLM = _CFG["name_llm"]
NAME_IMG_EMBED = _CFG["name_img_embed"]  # feature-fusion image encoder (same for every dataset)

# Per-dataset settings: CUDA devices + checkpoint filenames (see config.yaml).
DATASET_CONFIGS = _CFG["dataset_configs"]


def dataset_config(dataset=None):
    """Per-dataset settings dict; defaults to the active DATASET."""
    dataset = dataset or DATASET
    if dataset not in DATASET_CONFIGS:
        raise KeyError(f"No DATASET_CONFIGS entry for dataset {dataset!r}")
    return DATASET_CONFIGS[dataset]


def dataset_devices(dataset=None):
    return dataset_config(dataset)["devices"]


def dataset_device_mlm(dataset=None):
    return dataset_config(dataset)["device_mlm"]


def text_weights_path(dataset=None):
    dataset = dataset or DATASET
    return os.path.join("checkpoints", dataset, dataset_config(dataset)["text_checkpoint"])


def image_weights_path(dataset=None):
    dataset = dataset or DATASET
    return os.path.join("checkpoints", dataset, dataset_config(dataset)["image_checkpoint"])


def ff_weights_path(dataset=None):
    dataset = dataset or DATASET
    return os.path.join("checkpoints", dataset, dataset_config(dataset)["ff_checkpoint"])


# Convenience constants for the active DATASET. Dataset-aware scripts (those
# with their own --dataset CLI arg) must call the functions above with the
# parsed args.dataset instead of importing these, or they silently fall back
# to DATASET's devices/checkpoints regardless of what dataset was requested.
DEVICES = dataset_devices()
DEVICE_MLM = dataset_device_mlm()
TEXT_WEIGHTS_PATH = text_weights_path()
IMAGE_WEIGHTS_PATH = image_weights_path()
FF_WEIGHTS_PATH = ff_weights_path()

# Training parameters
EPOCHS = _CFG["epochs"]
LEARNING_RATE = _CFG["learning_rate"]
LORA_ALPHA = _CFG["lora_alpha"]
LORA_R = _CFG["lora_r"]
LORA_DROPOUT = _CFG["lora_dropout"]
USE_LORA = _CFG["use_lora"]
MERGE_TOKENS = _CFG["merge_tokens"]
NUM_WORKERS = _CFG["num_workers"]

# Model parameters
BATCH_SIZE = _CFG["batch_size"]
N_TOKENS = _CFG["n_tokens"]
THRESHOLD = _CFG["threshold"]

# Per-dataset subset settings (None = full dataset)
def dataset_subset_size(dataset=None):
    return dataset_config(dataset).get("subset_size", None)

def dataset_balanced_subset(dataset=None):
    return dataset_config(dataset).get("balanced_subset", False)

SUBSET_SIZE = dataset_subset_size()

# Attack parameters
SOURCE_LABEL = _CFG["source_label"]  # Fake
TARGET_LABEL = _CFG["target_label"]  # Real
## Image attack parameters
PGD_ITERS = _CFG["pgd_iters"]  # 50 steps
EPSILON = _number(_CFG["epsilon"])  # eps = 16/255
ALPHA_FACTOR = _number(_CFG["alpha_factor"])  # alpha = eps/(iters*factor) = 0.8/255, the paper's step size
## TrePat attack parameters (matched to Przybyła et al. 2025, arXiv:2410.20940 final-evaluation config)
ATTACK_MODEL = _CFG["attack_model"]  # meta-llama/Llama-3.2-3B-Instruct, as reported in the paper.
                        # TrePat's own paper used OLMo; switching rewriters changes
                        # every TrePat result, so runs must not be mixed across models.
COMMAND = _CFG["command"]  # paper's best prompt for journalistic/news text (their HN task): "INFORMAL rephrasing for text from journalistic ... sources"
                   # options: "REPHRASE": "Rephrase the provided input text.",
                   # "PARAPHRASE": "Paraphrase the provided input text.",
                   # "SIMPLIFY": "Simplify the provided input text.",
                   # "FORMAL": "Rewrite the provided input text in a more formal style.",
                   # "INFORMAL": "Rewrite the provided input text in a less formal style.",
                   # "CHANGE": "Make changes to the provided input text."
MAX_CHANGE_TOTAL = _CFG["max_change_total"]  # discard changes modifying more than 1/3 of the whole text
MAX_CHANGE_FRAGMENT = _CFG["max_change_fragment"]  # discard changes modifying more than 2/3 of the fragment
MAX_VARIANTS = _CFG["max_variants"]  # paper's default query limit
MIN_CHUNK_OR_SENTENCE_LENGTH = _CFG["min_chunk_or_sentence_length"]  # fragments shorter than 60 characters lack context for the LLM to rephrase
RESPONSES_EXPECTED = _CFG["responses_expected"]  # paper's REPHRASE prompt: "Return five different rephrasings, separated by newline"
## Bert-Attack attack parameters (matched to Li et al. 2020 original repo, cmd.txt)
K_BERT_ATTACK = _CFG["k_bert_attack"]  # --k 48
THRESHOLD_PRED_SCORE = _CFG["threshold_pred_score"] # --threshold_pred_score 0
MAX_WORDS_TO_ATTACK = _CFG["max_words_to_attack"]
MAX_CANDIDATES_PER_WORD = _CFG["max_candidates_per_word"]  # Maximum number of candidates to consider for each word in the attack
MAX_WORDS_FOR_IMPORTANCE = _CFG["max_words_for_importance"]
MAX_CHANGE_RATIO = _CFG["max_change_ratio"]  # perturbable word cap hardcoded as 0.4 * len(words)
MIN_TXT_SIMILARITY = _CFG["min_txt_similarity"]  # Post-hoc USE semantic similarity floor; revert to original if below
USE_BPE = _CFG["use_bpe"]  # --use_bpe 1
## Multimodal attack parameters
ALTERNATION_ROUNDS = _CFG["alternation_rounds"]  # Rounds of interleaved image-PGD + text-BERTAttack (1 = single biperturbed pass)

# Late-fusion experiments
#
# PGD_ITERS and MAX_VARIANTS are the budgets of the corresponding unimodal
# attacks. A single-modality scenario keeps that full budget. Only the "both"
# scenario splits the comparison budget and therefore uses half for each
# independently attacked modality.
LATE_FUSION_ATTACK_SCOPES = tuple(_CFG["late_fusion_attack_scopes"])
LATE_FUSION_BUDGET_DIVISOR = _CFG["late_fusion_budget_divisor"]


# The RBF-SVM is fitted on clean text/image predictions from the training set.
def build_subset_sampler(dataset, subset_size, balanced=None):
    """Return a list of indices for a balanced or sequential subset.

    ``balanced`` defaults to the active dataset's ``balanced_subset`` setting.
    """
    import numpy as np
    if subset_size is None:
        return None
    if balanced is None:
        balanced = dataset_balanced_subset()
    n = min(subset_size, len(dataset))
    if not balanced:
        return list(range(n))
    try:
        labels = dataset.img_labels.iloc[:, 1].values
    except AttributeError:
        return list(range(n))
    rng = np.random.RandomState(RAND_SEED)
    per_class = n // 2
    idx_0 = np.where(labels == 0)[0]
    idx_1 = np.where(labels == 1)[0]
    pick_0 = rng.choice(idx_0, size=min(per_class, len(idx_0)), replace=False)
    pick_1 = rng.choice(idx_1, size=min(per_class, len(idx_1)), replace=False)
    sampler = sorted(np.concatenate([pick_0, pick_1]).tolist())
    return sampler


LATE_FUSION_INPUT = _CFG["late_fusion_input"]
LATE_FUSION_SVM_C = _CFG["late_fusion_svm_c"]
LATE_FUSION_SVM_GAMMA = _CFG["late_fusion_svm_gamma"]
