#!/usr/bin/env bash
# Clean progress view for continue_from_adapter.py: strips the \r progress bars
# and prints one line per logged train step / eval.
#   ./watch_train.sh logs/continue_multi_glm_ds.out [interval_s]
set -uo pipefail
LOG=${1:?usage: watch_train.sh <log> [interval_s]}
INTERVAL=${2:-30}
SEEN=""

while :; do
  # Progress bars are \r-separated; split them into lines before filtering.
  CUR=$(tr '\r' '\n' < "$LOG" 2>/dev/null \
        | grep -E "^\{'(loss|eval_loss)|'eval_loss'|\[TRAIN\]|\[TIME\]|\[SAVE\]|Error|Traceback" \
        | tail -60)
  if [ "$CUR" != "$SEEN" ]; then
    comm -13 <(printf '%s\n' "$SEEN") <(printf '%s\n' "$CUR") 2>/dev/null \
      || printf '%s\n' "$CUR"
    SEEN="$CUR"
  fi
  STEP=$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -oE "[0-9]+/[0-9]+ \[" | tail -1 | tr -d ' [')
  echo "[$(date +%H:%M:%S)] step=${STEP:-?}"
  pgrep -f continue_from_adapter.py >/dev/null || { echo "[$(date +%H:%M:%S)] training process gone"; break; }
  sleep "$INTERVAL"
done
