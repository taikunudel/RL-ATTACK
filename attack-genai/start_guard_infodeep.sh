#!/usr/bin/env bash
# Start a 2nd Llama-Guard-4-12B guard on infodeep GPU1 (port 8001), native bf16,
# in a detached screen. Env 'llama_guard' has flask+transformers+torch. Idempotent.
set -u
ROOT=/usa/taikun/rl-attack/rl_atk/attack-genai
LOG=$ROOT/serve_infodeep_gpu1.log
source ~/miniconda3/etc/profile.d/conda.sh
pkill -f 'GUARD_PORT=8001' 2>/dev/null
screen -S guard8001 -X quit 2>/dev/null
sleep 2
: > "$LOG"
screen -dmS guard8001 bash -lc "source ~/miniconda3/etc/profile.d/conda.sh; conda activate llama_guard; cd $ROOT; CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 GUARD_DTYPE=bfloat16 GUARD_PORT=8001 GUARD_MICRO_BATCH=8 python -u serve_guard_cfg.py > $LOG 2>&1"
sleep 6
echo "SCREENS=$(screen -ls 2>/dev/null | grep -c guard8001)"
echo "PROC=$(pgrep -f 'GUARD_PORT=8001' | head -1)"
echo "LOGBYTES=$(wc -c < "$LOG" 2>/dev/null)"
