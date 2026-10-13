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
mapfile -t SEEDS < <(python3 -c 'from configuration_files.configuration import RAND_SEEDS; print("\n".join(map(str, RAND_SEEDS)))')
mapfile -t TRAIN_MODELS < <(python3 -c 'from configuration_files.configuration import PIPELINE_TRAIN; print("\n".join(PIPELINE_TRAIN))')
mapfile -t LATE_FUSIONS < <(python3 -c 'from configuration_files.configuration import PIPELINE_LATE_FUSION_MODES; print("\n".join(PIPELINE_LATE_FUSION_MODES))')
mapfile -t FUSIONS < <(python3 -c 'from configuration_files.configuration import PIPELINE_FUSIONS; print("\n".join(PIPELINE_FUSIONS))')
mapfile -t ATTACKS < <(python3 -c 'from configuration_files.configuration import PIPELINE_ATTACKS; print("\n".join(PIPELINE_ATTACKS))')
M="python3 -m scripts.utils.metrics"

FORCE_FLAG=""
if [ "$OVERWRITE" -eq 1 ]; then FORCE_FLAG="--force"; fi

# One background job per dataset, each on the GPUs in its dataset_configs
# entry; output goes to logs/pipeline_<dataset>.log.
run_dataset() {
    local DATASET="$1"
    FAILURES=()
    SECONDS=0
    echo ""
    echo "====== PIPELINE: $DATASET ======"

    # 0. Training of the neural models in pipeline_train (config.yaml). The
    #    learned fusion heads (svm-rbf, linear) are fitted in step 1, on the
    #    validation outputs of the unimodal models.
    if [ "$RUN_TRAIN" -eq 1 ]; then
        for model in "${TRAIN_MODELS[@]}"; do
            case "$model" in svm-rbf|linear) continue ;; esac
            run_step "train $model ($DATASET)" python3 -m scripts.train_scripts.train --model "$model" --dataset "$DATASET" $FORCE_FLAG
        done
    fi

    # 1. Validation scores, learned heads on validation logits, calibrated
    #    per-method thresholds (checkpoints/<ds>/thresholds.json). Needed
    #    before any evaluation when config threshold is "calibrated".
    if [ "$RUN_TRAIN" -eq 1 ] || [ "$RUN_CLEAN" -eq 1 ]; then
        run_step "heads + thresholds ($DATASET)" python3 -m scripts.main_scripts.compute_thresholds --dataset "$DATASET"
    fi

    # 2. Clean eval
    if [ "$RUN_CLEAN" -eq 1 ]; then
        run_step "clean ($DATASET)" python3 -m scripts.main_scripts.run_clean --dataset "$DATASET" $FORCE_FLAG

        run_step "metrics clean text"  $M --type clean --modality text --dataset "$DATASET"
        run_step "metrics clean image" $M --type clean --modality image --dataset "$DATASET"
        run_step "metrics clean feature-fusion" $M --type clean --modality feature-fusion --dataset "$DATASET"
        for f in "${LATE_FUSIONS[@]}"; do
            run_step "metrics clean $f" $M --type clean --modality late-fusion --mode "$f" --dataset "$DATASET"
        done
    fi

    # 3. Ablation
    if [ "$RUN_ABLATION" -eq 1 ]; then
        run_step "ablation ($DATASET)" python3 -m scripts.main_scripts.run_ablation --dataset "$DATASET" $FORCE_FLAG
    fi

    # 4. Adversarial attacks + metrics
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

    echo ""
    echo "====== DONE $DATASET — $((SECONDS / 60))m $((SECONDS % 60))s | ${#FAILURES[@]} failure(s) ======"
    for f in "${FAILURES[@]}"; do echo "  - $f"; done
    [ "${#FAILURES[@]}" -eq 0 ]
}

mkdir -p logs
FAILED=()
# One pipeline per test-subset seed (rand_seed in config.yaml, one value or a
# list), run one after the other; every step sees the seed via RM_RAND_SEED
# and writes to results/<dataset>/<subset>/<seed>/.
for SEED in "${SEEDS[@]}"; do
    export RM_RAND_SEED="$SEED"
    echo "====== SEED $SEED ======"
    declare -A PIDS=()
    for DATASET in "${DATASETS[@]}"; do
        LOG="logs/pipeline_${DATASET}_seed${SEED}.log"
        run_dataset "$DATASET" > "$LOG" 2>&1 &
        PIDS[$DATASET]=$!
        echo "started $DATASET seed $SEED (pid ${PIDS[$DATASET]}) -> $LOG"
    done
    for DATASET in "${!PIDS[@]}"; do
        wait "${PIDS[$DATASET]}" || FAILED+=("$DATASET seed $SEED")
        # Clean / ablation / adversarial tables for this seed (CSV + Markdown in its results folder).
        python3 scripts/results_table.py --dataset "$DATASET" >> "logs/pipeline_${DATASET}_seed${SEED}.log" 2>&1 \
            || FAILED+=("$DATASET seed $SEED tables")
    done
    unset PIDS
done

echo ""
echo "====== ALL DONE — $((SECONDS / 60))m $((SECONDS % 60))s | failed datasets: ${#FAILED[@]} ======"
for d in "${FAILED[@]}"; do echo "  - $d (see logs/pipeline_<dataset>_seed<seed>.log)"; done
[ "${#FAILED[@]}" -eq 0 ]
