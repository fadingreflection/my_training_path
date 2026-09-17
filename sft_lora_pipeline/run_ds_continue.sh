#!/usr/bin/env bash
# Wait for DeepSeek teacher → build DS-hit SFT → continue LoRA into a NEW dir → holdout eval.
# Never writes into sft_lora_pipeline/output_explain_cwe_qwen17b.
set -euo pipefail

ROOT="/home/afedotova/my_training_path"
PY="$ROOT/.venv/bin/python"
PIPE="$ROOT/sft_lora_pipeline"
RUNS="$ROOT/explain_cwe_map/runs"
LOG_DIR="$PIPE/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/ds_continue_pipeline.log"
BASELINE="$PIPE/output_explain_cwe_qwen17b/adapter_model.safetensors"
OUT_ADAPTER="$PIPE/output_explain_cwe_ds_continue"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

exec > >(tee -a "$LOG") 2>&1

ts() { date "+%Y-%m-%d %H:%M:%S"; }
echo "[$(ts)] START ds_continue_pipeline cwd=$ROOT gpu=$CUDA_VISIBLE_DEVICES"

if [[ ! -x "$PY" ]]; then
  echo "[$(ts)] missing venv python $PY" >&2
  exit 1
fi
if [[ ! -f "$BASELINE" ]]; then
  echo "[$(ts)] missing baseline adapter $BASELINE" >&2
  exit 1
fi

BASE_SHA=$(sha256sum "$BASELINE" | awk '{print $1}')
CKPT_SHA=$(sha256sum "$PIPE/output_explain_cwe_qwen17b/checkpoint-204/adapter_model.safetensors" | awk '{print $1}')
echo "[$(ts)] baseline_sha=$BASE_SHA checkpoint-204_sha=$CKPT_SHA"
if [[ "$BASE_SHA" != "$CKPT_SHA" ]]; then
  echo "[$(ts)] WARNING baseline adapter != checkpoint-204" >&2
fi
echo "$BASE_SHA  $BASELINE" > "$PIPE/output_explain_cwe_qwen17b.sha256"

wait_teacher() {
  local last_scored=-1
  local idle=0
  echo "[$(ts)] waiting for deepseek teacher v41flash_full"
  while true; do
    local scored=0 n=0 hits=0 running=0
    if [[ -f "$RUNS/v41flash_full_progress.json" ]]; then
      read -r scored n hits < <("$PY" -c "
import json
p=json.load(open('$RUNS/v41flash_full_progress.json'))
print(int(p.get('scored') or 0), int(p.get('n') or 0), int(p.get('hits') or 0))
")
    fi
    if pgrep -f "explain_cwe_map/run.py .*--tag v41flash_full" >/dev/null 2>&1; then
      running=1
    fi
    echo "[$(ts)] teacher scored=$scored/$n hits=$hits running=$running idle=${idle}s"
    if [[ "$n" -gt 0 && "$scored" -ge "$n" ]]; then
      if [[ "$running" -eq 0 ]]; then
        echo "[$(ts)] teacher complete"
        return 0
      fi
      sleep 15
      if ! pgrep -f "explain_cwe_map/run.py .*--tag v41flash_full" >/dev/null 2>&1; then
        echo "[$(ts)] teacher complete (process exited)"
        return 0
      fi
    fi
    if [[ "$running" -eq 0 ]]; then
      if [[ "$scored" -eq "$last_scored" ]]; then
        idle=$((idle + 30))
      else
        idle=0
      fi
      if [[ "$idle" -ge 120 && "$scored" -gt 0 ]]; then
        echo "[$(ts)] teacher process gone; proceeding with partial scored=$scored/$n"
        return 0
      fi
    else
      idle=0
    fi
    last_scored=$scored
    sleep 30
  done
}

wait_teacher

echo "[$(ts)] building DS-hit SFT"
"$PY" "$ROOT/explain_cwe_map/make_sft_ds_continue.py"
echo "[$(ts)] dataset ready"
"$PY" -c "import json; m=json.load(open('$ROOT/explain_cwe_map/sft/ds41flash_continue/manifest.json')); t=m['train']; print('train_ids', t['n_ids'], 'turns', t['n_turns'], 'review', t['n_review'], 'map', t['n_map'], 'map_source', t.get('map_source')); print('glm_map_priority', t.get('glm_map_priority'), 'ds_map_no_glm', t.get('ds_map_no_glm'), 'cwe_collision', t.get('mapping_cwe_set_collision'))"

AFTER_SHA=$(sha256sum "$BASELINE" | awk '{print $1}')
if [[ "$AFTER_SHA" != "$BASE_SHA" ]]; then
  echo "[$(ts)] FATAL baseline adapter changed before train: $AFTER_SHA" >&2
  exit 2
fi

echo "[$(ts)] continue-train start out=$OUT_ADAPTER"
cd "$PIPE"
mkdir -p logs
"$PY" -u continue_from_adapter.py config_explain_cwe_ds_continue.yaml
echo "[$(ts)] continue-train done"

AFTER_SHA=$(sha256sum "$BASELINE" | awk '{print $1}')
if [[ "$AFTER_SHA" != "$BASE_SHA" ]]; then
  echo "[$(ts)] FATAL baseline adapter changed during train: $AFTER_SHA" >&2
  exit 2
fi
echo "[$(ts)] baseline still $BASE_SHA"

if [[ ! -f "$OUT_ADAPTER/adapter_model.safetensors" ]]; then
  echo "[$(ts)] missing new adapter weights" >&2
  exit 3
fi

echo "[$(ts)] holdout eval start"
cd "$ROOT"
"$PY" -u explain_cwe_map/eval_local.py \
  --adapter "$OUT_ADAPTER" \
  --tag qwen17b_lora_ds_continue_holdout

echo "[$(ts)] compare vs 0.421"
"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("/home/afedotova/my_training_path/explain_cwe_map/runs")
base = json.loads((root / "qwen17b_lora_holdout_summary.json").read_text())
newp = root / "qwen17b_lora_ds_continue_holdout_summary.json"
new = json.loads(newp.read_text())
bm = base["metrics"]
nm = new["metrics"]
keys = ["n", "said_safe", "said_insufficient", "hit_raw", "hit_canonical", "hit_family", "any_level_hit"]
delta = {k: nm.get(k) - bm.get(k) if isinstance(nm.get(k), (int, float)) and isinstance(bm.get(k), (int, float)) else None for k in keys}
cmp = {
    "baseline_adapter": "/home/afedotova/my_training_path/sft_lora_pipeline/output_explain_cwe_qwen17b",
    "new_adapter": "/home/afedotova/my_training_path/sft_lora_pipeline/output_explain_cwe_ds_continue",
    "baseline_any_level_hit": bm.get("any_level_hit"),
    "new_any_level_hit": nm.get("any_level_hit"),
    "delta_any_level_hit": (nm.get("any_level_hit") or 0) - (bm.get("any_level_hit") or 0),
    "baseline_hits": round((bm.get("any_level_hit") or 0) * (bm.get("n") or 0)),
    "new_hits": round((nm.get("any_level_hit") or 0) * (nm.get("n") or 0)),
    "baseline_metrics": {k: bm.get(k) for k in keys},
    "new_metrics": {k: nm.get(k) for k in keys},
    "delta": delta,
}
out = root / "qwen17b_lora_ds_continue_holdout_compare.json"
out.write_text(json.dumps(cmp, indent=2) + "\n")
print(json.dumps(cmp, indent=2))
PY

AFTER_SHA=$(sha256sum "$BASELINE" | awk '{print $1}')
echo "[$(ts)] DONE baseline_sha=$AFTER_SHA unchanged=$([[ $AFTER_SHA == $BASE_SHA ]] && echo yes || echo NO)"
