#!/usr/bin/env bash
# Joint HotFlip+PGD attack over every fusion rule and attack scope.
# One iteration is one shared backward pass, so all scopes get equal budget.
set -u

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multimodal

DEVICE="${DEVICE:-cuda:1}"
ITERS="${ITERS:-20}"
LOG_DIR="logs/joint_hotflip_pgd"
mkdir -p "$LOG_DIR"

for fusion in mean min max; do
  for scope in text image both; do
    log="$LOG_DIR/${fusion}_${scope}.log"
    echo "[$(date +%H:%M:%S)] $fusion / $scope -> $log"
    python3 attacks/joint_hotflip_pgd.py \
      --fusion "$fusion" \
      --attack-scope "$scope" \
      --iters "$ITERS" \
      --device "$DEVICE" >"$log" 2>&1
    status=$?
    if [ $status -ne 0 ]; then
      echo "[$(date +%H:%M:%S)] FAILED ($status): $fusion / $scope -- see $log"
    fi
  done
done

echo "[$(date +%H:%M:%S)] done"
