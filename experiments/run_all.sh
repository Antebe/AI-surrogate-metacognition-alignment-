#!/usr/bin/env bash
# Run all experiments sequentially in tmux.
# Usage: bash run_all.sh
#   or:  tmux new -s experiments 'bash run_all.sh'
set -e

cd "$(dirname "$0")"

echo "============================================"
echo " SAE Steering Experiments"
echo " Started: $(date)"
echo "============================================"

for exp in E1 E2 E3 E5 E6 E7; do
    echo ""
    echo "────────────────────────────────────────────"
    echo " Running ${exp}  ($(date +%H:%M:%S))"
    echo "────────────────────────────────────────────"
    python run_${exp}.py
    echo "  ${exp} complete  ($(date +%H:%M:%S))"
done

echo ""
echo "============================================"
echo " All experiments complete"
echo " Finished: $(date)"
echo "============================================"
