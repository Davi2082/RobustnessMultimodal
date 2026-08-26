#!/usr/bin/env bash
# Fusion-aware attack grid: 6 fusion methods x 3 attack scopes.
#
# All runs are untargeted (every correctly classified sample is pushed toward
# the opposite class) and use the TREPAT rewriter named by ATTACK_MODEL in
# configuration.py. Image-only scopes run first because they are cheap and
# complete a third of the table quickly; text-only runs last because it is the
# most expensive scope (full variant budget, no budget divisor).
set -u

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multimodal

DEVICE="${DEVICE:-cuda:0}"
OPT="${OPT:-sum}"
SUFFIX="${SUFFIX:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOG_DIR="logs/attack_grid${SUFFIX}"
mkdir -p "$LOG_DIR"

FUSIONS="${FUSIONS:-min mean max svm-rbf linear feature-fusion}"
SCOPES="${SCOPES:-image both text}"

echo "[$(date +%F' '%H:%M:%S)] grid start | device=$DEVICE opt=$OPT"
echo "  fusions: $FUSIONS"
echo "  scopes : $SCOPES"

for scope in $SCOPES; do
  for fusion in $FUSIONS; do
    log="$LOG_DIR/${fusion}_${scope}.log"
    out="results/Recovery/classification_results/perturbed/late-fusion${SUFFIX}/${fusion}"
    echo "[$(date +%H:%M:%S)] $fusion / $scope -> $log"

    python3 -m attacks.multimodal.sum.attack \
      --fusion "$fusion" \
      --attack-scope "$scope" \
      --optimization "$OPT" \
      --output-dir "$out" \
      $EXTRA_ARGS \
      >"$log" 2>&1

    status=$?
    if [ $status -ne 0 ]; then
      echo "[$(date +%H:%M:%S)] FAILED ($status): $fusion / $scope -- see $log"
    else
      echo "[$(date +%H:%M:%S)] done: $fusion / $scope"
    fi
  done
done

echo "[$(date +%F' '%H:%M:%S)] grid complete"
