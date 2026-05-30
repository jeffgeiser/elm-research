#!/usr/bin/env bash
# Wrapper for train_lora.py — tmux session + log file, survives SSH disconnect.
#
# Usage (on the DGX Spark):
#     cd ~/elm-research
#     ./train/run.sh                          # default: timestamped run name
#     ./train/run.sh my-run-name              # custom run name
#     ./train/run.sh round4 --resume-from-checkpoint train/runs/round3-clean/checkpoint-10
#         # resume: any args after the run name are forwarded to train_lora.py
#
# Attach later:
#     tmux attach -t elm-train
#
# View live training log:
#     tail -f train/runs/<run-name>/train.log
#
# View VRAM usage:
#     tail -f train/runs/<run-name>/vram.log
#
# Stop the run:
#     tmux send-keys -t elm-train C-c
#     # or kill the session entirely:
#     tmux kill-session -t elm-train

set -euo pipefail

SESSION="elm-train"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

RUN_NAME="${1:-$(date +%Y%m%d-%H%M%S)}"
# Any args after the run name are forwarded verbatim to train_lora.py
# (e.g. --resume-from-checkpoint, --force).
EXTRA_ARGS=("${@:2}")
RUN_DIR="$HERE/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"
LOG_PATH="$RUN_DIR/train.log"

# Refuse to clobber an active session
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' is already running. Attach with:"
    echo "    tmux attach -t $SESSION"
    echo "Or kill it first: tmux kill-session -t $SESSION"
    exit 1
fi

# Confirm data files exist before burning a tmux start
for f in "$HERE/data/train.jsonl" "$HERE/data/eval.jsonl"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: $f not found. Run format_jsonl.py first."
        exit 1
    fi
done

echo "Starting training in tmux session '$SESSION'"
echo "Run dir:  $RUN_DIR"
echo "Log:      $LOG_PATH"
echo "VRAM log: $RUN_DIR/vram.log"
echo

# Use tee for cross-platform logging. PYTHONUNBUFFERED keeps progress
# lines flushing in real time without needing `script`.
# Build the forwarded-args string (quoted) for the tmux command line.
EXTRA_STR=""
for a in "${EXTRA_ARGS[@]}"; do
    EXTRA_STR+=" '$a'"
done

tmux new-session -d -s "$SESSION" -c "$ROOT" \
    "PYTHONUNBUFFERED=1 uv run python train/train_lora.py --run-name '$RUN_NAME'$EXTRA_STR 2>&1 | tee '$LOG_PATH'"

echo "Started. Attach with:  tmux attach -t $SESSION"
echo "Follow log with:       tail -f $LOG_PATH"
