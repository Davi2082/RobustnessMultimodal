# Models
NAME_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NAME_IMG_EMBED = "openai/clip-vit-large-patch14"
TEXT_WEIGHTS_PATH = "model/clip-vit-large-patch14_None_8_8_0.4_True10_best_txt_only.pt"
IMAGE_WEIGHTS_PATH = "model/clip-vit-large-patch14_None_8_8_0.4_True10_best_img_only.pt"
FF_WEIGHTS_PATH = "model/clip-vit-base-patch32_None_8_8_0.4_True10_best.pt"
FF_NAME_IMG_EMBED = "openai/clip-vit-base-patch32"

# CUDA devices
DEVICE = "cuda:1"      # main model (eval + attacks)
DEVICE_EVAL = "cuda:1" # clean eval
DEVICE_MLM = "cuda:1"  # BERT MLM (text/multimodal attacks only)

# Model parameters
BATCH_SIZE = 128
N_TOKENS = 256
THRESHOLD = 0.5

# Testing — restrict clean eval + attacks to the first N samples (None = full dataset)
SUBSET_SIZE = None

# Attack parameters
SOURCE_LABEL = 0 # Fake
TARGET_LABEL = 1 # Real
## Image attack parameters
PGD_ITERS = 20
EPSILON = 8 / 255
ALPHA_FACTOR = 0.01
## TrePat attack parameters (matched to Przybyła et al. 2025, arXiv:2410.20940 final-evaluation config)
ATTACK_MODEL = "OLMO7B" # paper's chosen LLM ("we have decided to use OLMO due to its open and transparent features")
COMMAND = "INFORMAL" # paper's best prompt for journalistic/news text (their HN task): "INFORMAL rephrasing for text from journalistic ... sources"
                   # options: "REPHRASE": "Rephrase the provided input text.",
                   # "PARAPHRASE": "Paraphrase the provided input text.",
                   # "SIMPLIFY": "Simplify the provided input text.",
                   # "FORMAL": "Rewrite the provided input text in a more formal style.",
                   # "INFORMAL": "Rewrite the provided input text in a less formal style.",
                   # "CHANGE": "Make changes to the provided input text."
MAX_CHANGE_TOTAL = 0.333 # paper: discard changes modifying more than 1/3 of the whole text
MAX_CHANGE_FRAGMENT = 0.667 # paper: discard changes modifying more than 2/3 of the fragment
MAX_VARIANTS = 50 # paper's default query limit (Experiments 1-3 and final evaluation; Fig.2 also tests 10/100/250)
MIN_CHUNK_OR_SENTENCE_LENGTH = 60 # paper: fragments shorter than 60 characters lack context for the LLM to rephrase
RESPONSES_EXPECTED = 5 # paper's REPHRASE prompt: "Return five different rephrasings, separated by newline"
## Bert-Attack attack parameters (matched to Li et al. 2020 original repo, cmd.txt)
K_BERT_ATTACK = 48 # paper's exact value: "--k 48"
THRESHOLD_PRED_SCORE = 0 # paper's exact value: "--threshold_pred_score 0"
MAX_WORDS_TO_ATTACK = 1024
MAX_CANDIDATES_PER_WORD = 64 # Maximum number of candidates to consider for each word in the attack
MAX_WORDS_FOR_IMPORTANCE = 1024
MAX_CHANGE_RATIO = 0.4 # paper's exact value: perturbable word cap hardcoded as 0.4 * len(words)
MIN_TXT_SIMILARITY = 0.0 # Post-hoc USE semantic similarity floor; revert to original if below
USE_BPE = 1 # paper's exact value: "--use_bpe 1"
## Multimodal attack parameters
ALTERNATION_ROUNDS = 1 # Rounds of interleaved image-PGD + text-BERTAttack (1 = single biperturbed pass)

# Late-fusion experiments
#
# PGD_ITERS and MAX_VARIANTS are the budgets of the corresponding unimodal
# attacks. A late-fusion attack uses half of each modality budget, regardless
# of whether the requested scenario attacks text, image, or both. Keeping the
# effective budgets here also makes direct invocations consistent with the
# batch runner.
LATE_FUSION_METHODS = ("min", "max", "mean", "svm-rbf")
LATE_FUSION_ATTACK_SCOPES = ("image", "text", "both")
LATE_FUSION_BUDGET_DIVISOR = 2
LATE_FUSION_PGD_ITERS = max(1, PGD_ITERS // LATE_FUSION_BUDGET_DIVISOR)
LATE_FUSION_MAX_VARIANTS = max(
    1,
    MAX_VARIANTS // LATE_FUSION_BUDGET_DIVISOR,
)

# The RBF-SVM is fitted on clean text/image predictions from the training set.
LATE_FUSION_SVM_INPUT = "scores"
LATE_FUSION_SVM_C = 1.0
LATE_FUSION_SVM_GAMMA = "scale"
LATE_FUSION_SVM_SEED = 42
