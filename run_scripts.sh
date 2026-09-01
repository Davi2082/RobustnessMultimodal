#!/usr/bin/env bash
# ── Set this ──
DATASET="Recovery"
# ──────────────

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate multimodal
# set -e

# 1. Clean eval (text, image, feature-fusion, all late-fusion modes)
python3 -m scripts.run_clean --dataset $DATASET

# 2. Missing-modality ablation (feature-fusion + late-fusion)
python3 -m scripts.run_ablation --dataset $DATASET

# 3. All adversarial attacks (5 attack types × 6 fusion methods)
python3 -m scripts.run_multimodal_attacks --dataset $DATASET
