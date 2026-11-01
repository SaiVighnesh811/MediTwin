#!/usr/bin/env bash
# run_nlp_training.sh
# Runs Chapter 5 NLP training in the background, writing to logs/nlp_training.log.
# Re-running the same command will RESUME from the latest epoch checkpoint.

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$PROJECT/logs/nlp_training.log"
VENV="$PROJECT/venv/bin/activate"

if [ ! -f "$VENV" ]; then
    echo "ERROR: venv not found at $VENV"; exit 1
fi

source "$VENV"
mkdir -p "$PROJECT/logs" "$PROJECT/models/nlp"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting / resuming NLP training. Log: $LOG"

nohup python "$PROJECT/agents/nlp_agent/train.py" >> "$LOG" 2>&1 &
PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID=$PID. Monitor with: tail -f $LOG"
echo "To check if still running: kill -0 $PID 2>/dev/null && echo running || echo done"
