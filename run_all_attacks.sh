#!/usr/bin/env bash
# Full untargeted attack matrix. Every attack targets the FINAL score of the
# model it attacks: the fused score for late fusion, the joint score for
# feature fusion. See the threat-model section of CLAUDE.md.
set -u
cd /storageB/home/heddoubi/projects/RobustnessMultimodal
source ~/miniconda3/etc/profile.d/conda.sh && conda activate multimodal
LOG=/storageB/home/heddoubi/projects/RobustnessMultimodal/logs/attacks_$(date +%m%d_%H%M).log
mkdir -p "$(dirname "$LOG")"

run() { echo "=== $* :: $(date +%H:%M) ===" >>"$LOG"; "$@" >>"$LOG" 2>&1 \
        && echo "--- OK  $* :: $(date +%H:%M)" >>"$LOG" \
        || echo "--- FAILED  $* :: $(date +%H:%M)" >>"$LOG"; }

# feature fusion: one run per scope
for s in image text both; do run python3 attacks/multimodal_attack.py --attack_scope "$s"; done

# late fusion: --attack_scope both emits all three scenario CSVs per fusion
for f in min mean max svm-rbf; do run python3 attacks/late_fusion_multimodal_attack.py --fusion "$f" --attack_scope both; done

echo "ALL DONE $(date +%m-%d_%H:%M)" >>"$LOG"
python3 adv_robustness_table.py >>"$LOG" 2>&1
echo "TABLE REGENERATED" >>"$LOG"
