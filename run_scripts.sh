#!/usr/bin/env bash
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate multimodal

SECONDS=0
FAILURES=()

# Run one step; on failure, record it and keep going instead of aborting the
# whole pipeline.
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

# Datasets, fusion methods, attack types and per-dataset devices all come
# from config.yaml.
mapfile -t DATASETS < <(python3 -c 'from configuration_files.configuration import DATASETS; print("\n".join(DATASETS))')
mapfile -t LATE_FUSIONS < <(python3 -c 'from configuration_files.configuration import PIPELINE_LATE_FUSION_MODES; print("\n".join(PIPELINE_LATE_FUSION_MODES))')
mapfile -t FUSIONS < <(python3 -c 'from configuration_files.configuration import PIPELINE_FUSIONS; print("\n".join(PIPELINE_FUSIONS))')
mapfile -t ATTACKS < <(python3 -c 'from configuration_files.configuration import PIPELINE_ATTACKS; print("\n".join(PIPELINE_ATTACKS))')
M="python3 -m scripts.utils.metrics"

run_pipeline() {
    local DATASET="$1"

    echo ""
    echo "======================================================================"
    echo "PIPELINE: $DATASET"
    echo "======================================================================"

    # 0. Train if necessary (fusion heads only — full dataset, not subset-gated)
    # run_step "train svm-rbf ($DATASET)" python3 -m scripts.train_scripts.train --model svm-rbf --dataset "$DATASET" --force
    # run_step "train linear ($DATASET)"  python3 -m scripts.train_scripts.train --model linear  --dataset "$DATASET" --force

    # 1. Clean eval (text, image, feature-fusion, all late-fusion modes)
    run_step "clean eval ($DATASET)" python3 -m scripts.main_scripts.run_clean --dataset "$DATASET" --force

    # # 2. Missing-modality ablation (feature-fusion + late-fusion)
    run_step "ablation ($DATASET)" python3 -m scripts.main_scripts.run_ablation --dataset "$DATASET" --force

    # 3. All multimodal/late-fusion adversarial attacks (attack types × fusion methods)
    run_step "multimodal attacks ($DATASET)" python3 -m scripts.main_scripts.run_multimodal_attacks --dataset "$DATASET" --force

    # 4. Compute metrics — each call is independent; a missing result file
    echo ""
    echo "======================================================================"
    echo "METRICS: $DATASET"
    echo "======================================================================"

    # Clean — unimodal + feature-fusion
    run_step "metrics clean text"           $M --type clean --modality text --dataset "$DATASET"
    run_step "metrics clean image"          $M --type clean --modality image --dataset "$DATASET"
    run_step "metrics clean feature-fusion" $M --type clean --modality feature-fusion --dataset "$DATASET"

    # Clean — late-fusion
    for f in "${LATE_FUSIONS[@]}"; do
        run_step "metrics clean late-fusion/$f" $M --type clean --modality late-fusion --mode "$f" --dataset "$DATASET"
    done

    # Perturbed — late-fusion + feature-fusion (per mode × per attack)
    for atk in "${ATTACKS[@]}"; do
        for f in "${FUSIONS[@]}"; do
            run_step "metrics perturbed late-fusion/$f/$atk" \
                $M --type perturbed --modality late-fusion --mode "$f" --attack "$atk" --dataset "$DATASET"
        done
    done
}

for DATASET in "${DATASETS[@]}"; do
    run_pipeline "$DATASET"
done

echo ""
echo "======================================================================"
echo "FULL PIPELINE COMPLETE — $((SECONDS / 60)) min $((SECONDS % 60)) sec"
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
