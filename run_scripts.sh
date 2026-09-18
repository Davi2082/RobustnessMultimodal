#!/usr/bin/env bash
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"

# ── All settings come from config.yaml ──
SECONDS=0
FAILURES=()

run_step() {
    local desc="$1"; shift
    echo ""
    echo "--- $desc ---"
    "$@" || { echo "FAILED: $desc" >&2; FAILURES+=("$desc"); }
}

# Read pipeline toggles and lists from config.yaml via Python
RUN_TRAIN=$(python3 -c 'from configuration_files.configuration import TRAIN_PIPELINE; print(int(TRAIN_PIPELINE))')
RUN_CLEAN=$(python3 -c 'from configuration_files.configuration import CLEAN_PIPELINE; print(int(CLEAN_PIPELINE))')
RUN_ABLATION=$(python3 -c 'from configuration_files.configuration import ABLATION_PIPELINE; print(int(ABLATION_PIPELINE))')
RUN_ADVERSARIAL=$(python3 -c 'from configuration_files.configuration import ADVERSARIAL_PIPELINE; print(int(ADVERSARIAL_PIPELINE))')
OVERWRITE=$(python3 -c 'from configuration_files.configuration import OVERWRITE; print(int(OVERWRITE))')

mapfile -t DATASETS < <(python3 -c 'from configuration_files.configuration import DATASETS; print("\n".join(DATASETS))')
mapfile -t TRAIN_MODELS < <(python3 -c 'from configuration_files.configuration import PIPELINE_TRAIN; print("\n".join(PIPELINE_TRAIN))')
mapfile -t LATE_FUSIONS < <(python3 -c 'from configuration_files.configuration import PIPELINE_LATE_FUSION_MODES; print("\n".join(PIPELINE_LATE_FUSION_MODES))')
mapfile -t FUSIONS < <(python3 -c 'from configuration_files.configuration import PIPELINE_FUSIONS; print("\n".join(PIPELINE_FUSIONS))')
mapfile -t ATTACKS < <(python3 -c 'from configuration_files.configuration import PIPELINE_ATTACKS; print("\n".join(PIPELINE_ATTACKS))')
M="python3 -m scripts.utils.metrics"

FORCE_FLAG=""
if [ "$OVERWRITE" -eq 1 ]; then FORCE_FLAG="--force"; fi

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "====== PIPELINE: $DATASET ======"

    # 0. Training (models from pipeline_train in config.yaml)
    if [ "$RUN_TRAIN" -eq 1 ]; then
        for model in "${TRAIN_MODELS[@]}"; do
            run_step "train $model ($DATASET)" python3 -m scripts.train_scripts.train --model "$model" --dataset "$DATASET"
        done
    fi

    # 1. Clean eval
    if [ "$RUN_CLEAN" -eq 1 ]; then
        run_step "clean ($DATASET)" python3 -m scripts.main_scripts.run_clean --dataset "$DATASET" $FORCE_FLAG

        run_step "metrics clean text"  $M --type clean --modality text --dataset "$DATASET"
        run_step "metrics clean image" $M --type clean --modality image --dataset "$DATASET"
        run_step "metrics clean feature-fusion" $M --type clean --modality feature-fusion --dataset "$DATASET"
        for f in "${LATE_FUSIONS[@]}"; do
            run_step "metrics clean $f" $M --type clean --modality late-fusion --mode "$f" --dataset "$DATASET"
        done
    fi

    # 2. Ablation
    if [ "$RUN_ABLATION" -eq 1 ]; then
        run_step "ablation ($DATASET)" python3 -m scripts.main_scripts.run_ablation --dataset "$DATASET" $FORCE_FLAG
    fi

    # 3. Adversarial attacks + metrics
    if [ "$RUN_ADVERSARIAL" -eq 1 ]; then
        # Unimodal attacks
        run_step "image attack ($DATASET)" python3 -m attacks.unimodal.image.attack --dataset "$DATASET"
        run_step "metrics image/pgd" $M --type perturbed --modality image --attack pgd --dataset "$DATASET"

        run_step "text attack ($DATASET)" python3 -m attacks.unimodal.text.attack --dataset "$DATASET"
        run_step "metrics text/trepat" $M --type perturbed --modality text --attack trepat --dataset "$DATASET"

        # Multimodal attacks
        run_step "multimodal attacks ($DATASET)" python3 -m scripts.main_scripts.run_multimodal_attacks --dataset "$DATASET" $FORCE_FLAG

        for atk in "${ATTACKS[@]}"; do
            for f in "${FUSIONS[@]}"; do
                if [ "$f" = "feature-fusion" ]; then
                    run_step "metrics $f/$atk" $M --type perturbed --modality feature-fusion --attack "$atk" --dataset "$DATASET"
                else
                    run_step "metrics $f/$atk" $M --type perturbed --modality late-fusion --mode "$f" --attack "$atk" --dataset "$DATASET"
                fi
            done
        done
    fi
done

echo ""
echo "====== DONE — $((SECONDS / 60))m $((SECONDS % 60))s | ${#FAILURES[@]} failure(s) ======"
for f in "${FAILURES[@]}"; do echo "  - $f"; done
[ "${#FAILURES[@]}" -eq 0 ]
