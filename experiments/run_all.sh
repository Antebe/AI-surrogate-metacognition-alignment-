#!/usr/bin/env bash
# Option B pipeline - tmux-friendly, resumable, pilot-gated.
# Usage:
#   tmux new -s mcog 'cd experiments && bash run_all.sh; exec bash'
set -euo pipefail
cd "$(dirname "$0")"

source /home/cs29824/.venv/bin/activate
: "${HF_TOKEN:?HF_TOKEN not set - see README 'API tokens' section}"
: "${ANTHROPIC_API_KEY:=${ANTHROPIC_KEY:-}}"
export ANTHROPIC_API_KEY
[ -n "$ANTHROPIC_API_KEY" ] || { echo "ANTHROPIC_API_KEY not set"; exit 1; }

LOG_ROOT="../logs/run_all_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
echo "Logs: $LOG_ROOT"

run () {
    local name=$1
    echo "---- $name  $(date +%H:%M:%S) ----"
    python "run_${name}.py" 2>&1 | tee "$LOG_ROOT/${name}.log"
    echo "---- $name done $(date +%H:%M:%S) ----"
}

run E0_pilot
run E1
run E2
run E8
run E5
run E3
run E6
run E7
run E9

echo "ALL DONE $(date)"
