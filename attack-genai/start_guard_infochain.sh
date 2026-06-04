#!/usr/bin/env bash
# Run ON infochain. Starts the Llama-Guard-4-12B judge in a detached screen
# at NATIVE bf16, so it survives SSH logout. Idempotent: kills any prior instance.
set -u
LOG=/usa/taikun/serve_cfg_bf16.log
SRV=/usa/taikun/serve_guard_cfg.py

source ~/miniconda3/etc/profile.d/conda.sh

pkill -f serve_guard_cfg.py 2>/dev/null
screen -S guard -X quit 2>/dev/null
sleep 2
: > "$LOG"

screen -dmS guard bash -lc "source ~/miniconda3/etc/profile.d/conda.sh; conda activate llama; cd /usa/taikun; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 GUARD_DTYPE=bfloat16 GUARD_PORT=8000 GUARD_MICRO_BATCH=${GUARD_MICRO_BATCH:-4} python -u $SRV > $LOG 2>&1"

sleep 6
echo "SCREENS=$(screen -ls 2>/dev/null | grep -c guard)"
echo "PROC=$(pgrep -f serve_guard_cfg.py | head -1)"
echo "LOGBYTES=$(wc -c < "$LOG" 2>/dev/null)"
