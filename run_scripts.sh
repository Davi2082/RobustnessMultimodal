#!/usr/bin/env bash
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate multimodal
set -e

SECONDS=0

run_pipeline() {
    local DATASET=$1
    local DEVICE=$2

    echo ""
    echo "======================================================================"
    echo "PIPELINE: $DATASET on $DEVICE"
    echo "======================================================================"

    # # 0. Train if necessary
    # # python3 -m scripts.train --train-all --dataset $DATASET --device $DEVICE
    # python3 -m scripts.train --model svm-rbf --dataset $DATASET --device $DEVICE --force
    # python3 -m scripts.train --model linear --dataset $DATASET --device $DEVICE --force

    # # 1. Clean eval (text, image, feature-fusion, all late-fusion modes)
    # python3 -m scripts.run_clean --dataset $DATASET --device $DEVICE --force

    # # 2. Missing-modality ablation (feature-fusion + late-fusion)
    # python3 -m scripts.run_ablation --dataset $DATASET --device $DEVICE --force

    # 3. All adversarial attacks (5 attack types × 6 fusion methods)
    python3 -m scripts.run_multimodal_attacks --dataset $DATASET --device $DEVICE --force
}

run_pipeline "Recovery" "cuda:1" &
# run_pipeline "Fakeddit" "cuda:1" &

wait

echo ""
echo "======================================================================"
echo "FULL PIPELINE COMPLETE — $((SECONDS / 60)) min $((SECONDS % 60)) sec"
echo "======================================================================"
