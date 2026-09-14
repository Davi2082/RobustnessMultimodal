#!/usr/bin/env bash
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate multimodal

SECONDS=0
FAILURES=()

run_step() {
    local desc="$1"; shift
    local rc=0
    echo ""
    echo "--- $desc ---"
    "$@" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "FAILED (exit $rc): $desc" >&2
        FAILURES+=("$desc")
    fi
}

# Datasets come from config.yaml. Devices are resolved per-dataset by
# train.py itself (configuration_files/configuration.py -> config.yaml
# dataset_configs.<dataset>.devices) — never hardcode a device here.
mapfile -t DATASETS < <(python3 -c 'from configuration_files.configuration import DATASETS; print("\n".join(DATASETS))')

T="python3 -m scripts.train_scripts.train"

# Only the unimodal detectors + feature-fusion model — no late-fusion heads.
MODELS=(text image feature-fusion)
FORCE="--force" # set to "" to not override the checkpoints

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "TRAIN: $DATASET"
    echo "======================================================================"
    for MODEL in "${MODELS[@]}"; do
        run_step "train $MODEL ($DATASET)" $T --model "$MODEL" --dataset "$DATASET" "$FORCE"
    done
done

echo ""
echo "======================================================================"
echo "TRAINING COMPLETE — $((SECONDS / 60)) min $((SECONDS % 60)) sec"
if [ "${#FAILURES[@]}" -gt 0 ]; then
    echo "${#FAILURES[@]} step(s) FAILED:"
    for f in "${FAILURES[@]}"; do
        echo "  - $f"
    done
else
    echo "All steps succeeded."
fi
echo "======================================================================"

[ "${#FAILURES[@]}" -eq 0 ]
