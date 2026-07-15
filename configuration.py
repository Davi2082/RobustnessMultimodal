# Models
NAME_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NAME_IMG_EMBED = "openai/clip-vit-large-patch14"
TEXT_WEIGHTS_PATH = "model/clip-vit-large-patch14_None_8_8_0.4_True10_best_txt_only.pt"
IMAGE_WEIGHTS_PATH = "model/clip-vit-large-patch14_None_8_8_0.4_True10_best_img_only.pt"
FF_WEIGHTS_PATH = "model/clip-vit-base-patch32_None_8_8_0.4_True10_best.pt"
FF_NAME_IMG_EMBED = "openai/clip-vit-base-patch32"

# CUDA devices
DEVICE = "cuda:0"      # main model (eval + attacks)
DEVICE_EVAL = "cuda:0" # clean eval
DEVICE_MLM = "cuda:0"  # BERT MLM (text/multimodal attacks only)

# Model parameters
BATCH_SIZE = 128
N_TOKENS = 1024
THRESHOLD = 0.5

# Testing — restrict clean eval + attacks to the first N samples (None = full dataset)
<<<<<<< HEAD
SUBSET_SIZE = 2
=======
SUBSET_SIZE = 28
>>>>>>> refs/remotes/origin/main

# Attack parameters
SOURCE_LABEL = 0 # Fake
TARGET_LABEL = 1 # Real
## Image attack parameters
PGD_ITERS = 25
EPSILON = 255 / 255
ALPHA_FACTOR = 2.0
## TrePat attack parameters
ATTACK_MODEL = "LLAMA8B" # options: "OLDGEMMA", "LLAMA1B", "LLAMA3B", "LLAMA8B", "GEMMA2B", "GEMMA9B", "OLMO7B"
COMMAND = "PARAPHRASE" # options: "REPHRASE": "Rephrase the provided input text.", 
                   # "PARAPHRASE": "Paraphrase the provided input text.", 
                   # "SIMPLIFY": "Simplify the provided input text.", 
                   # "FORMAL": "Rewrite the provided input text in a more formal style.", 
                   # "INFORMAL": "Rewrite the provided input text in a less formal style.", 
                   # "CHANGE": "Make changes to the provided input text."
MAX_CHANGE_TOTAL = 1 # Maximum change size relative to the full text
MAX_CHANGE_FRAGMENT = 1 # Maximum change size relative to the current fragment
MAX_VARIANTS = 10000 # Maximum number of candidate variants evaluated
MIN_CHUNK_OR_SENTENCE_LENGTH = 60 # Merge fragments shorter than MIN_CHUNK_OR_SENTENCE_LENGTH characters
RESPONSES_EXPECTED = 10 # Number of paraphrases requested per fragment
## Bert-Attack attack parameters
K_BERT_ATTACK = 100 # Number of candidates to consider for each word in the attack
THRESHOLD_PRED_SCORE = 0
MAX_WORDS_TO_ATTACK = 1024
MAX_CANDIDATES_PER_WORD = 64 # Maximum number of candidates to consider for each word in the attack
MAX_WORDS_FOR_IMPORTANCE = 1024
MAX_CHANGE_RATIO = 1.0 # Max fraction of words BERTAttack may substitute before giving up (default 0.4)
MIN_TXT_SIMILARITY = 0.0 # Post-hoc USE semantic similarity floor; revert to original if below
USE_BPE = 1 # 1 = also attack multi-subword words (native BPE reconstruction); 0 = single-token words only
## Multimodal attack parameters
ALTERNATION_ROUNDS = 1 # Rounds of interleaved image-PGD + text-BERTAttack (1 = single biperturbed pass)
