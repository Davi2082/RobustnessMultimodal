#!/usr/bin/env bash
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate multimodal
set -e

SECONDS=0

FUSIONS=(min mean max svm-rbf linear feature-fusion)
ATTACKS=(pgd trepat sum interleaved joint)
M="python3 -m scripts.utils.metrics"

run_pipeline() {
    local DATASET=$1
    local DEVICES=$2  # space-separated device list or "all"

    echo ""
    echo "======================================================================"
    echo "PIPELINE: $DATASET  devices=$DEVICES"
    echo "======================================================================"

    # # 0. Train if necessary
    # python3 -m scripts.train_scripts.train --model svm-rbf --dataset $DATASET --devices $DEVICES --force
    # python3 -m scripts.train_scripts.train --model linear --dataset $DATASET --devices $DEVICES --force

    # # 1. Clean eval (text, image, feature-fusion, all late-fusion modes)
    # python3 -m scripts.main_scripts.run_clean --dataset $DATASET --devices $DEVICES --force

    # # 2. Missing-modality ablation (feature-fusion + late-fusion)
    # python3 -m scripts.main_scripts.run_ablation --dataset $DATASET --devices $DEVICES --force

    # 3. All adversarial attacks (5 attack types × 6 fusion methods)
    python3 -m scripts.main_scripts.run_multimodal_attacks --dataset $DATASET --devices $DEVICES --force

    # 4. Compute metrics
    echo ""
    echo "======================================================================"
    echo "METRICS: $DATASET"
    echo "======================================================================"

    # Clean — unimodal + feature-fusion
    $M --type clean --modality text
    $M --type clean --modality image
    $M --type clean --modality feature-fusion

    # Clean — late-fusion (5 modes)
    for f in "${FUSIONS[@]:0:5}"; do
        $M --type clean --modality late-fusion --mode "$f"
    done

    # Perturbed — unimodal
    $M --type perturbed --modality text
    $M --type perturbed --modality image

    # Perturbed — feature-fusion (3 scopes)
    for pt in biperturbed text-perturbed image-perturbed; do
        $M --type perturbed --modality feature-fusion --perturbation_type "$pt"
    done

    # Perturbed — late-fusion (5 modes × 5 attacks)
    for atk in "${ATTACKS[@]}"; do
        for f in "${FUSIONS[@]:0:5}"; do
            $M --type perturbed --modality late-fusion --mode "$f" --attack "$atk"
        done
    done
}

run_pipeline "Recovery" "all" &
# run_pipeline "Fakeddit" "all" &

wait

echo ""
echo "======================================================================"
echo "FULL PIPELINE COMPLETE — $((SECONDS / 60)) min $((SECONDS % 60)) sec"
echo "======================================================================"
