#!/usr/bin/env bash
# Start the LOCAL Llama-Guard-4-12B server on the A40 GPU, native bf16, inside the
# container venv_guard. Shares host network -> reachable at localhost:8000.
set +u   # VALET's vpkg_require references unset vars; nounset would kill the script
source /etc/profile.d/valet.sh 2>/dev/null
vpkg_require apptainer/1.4.1 >/dev/null 2>&1
set -u
export APPTAINER_CACHEDIR=/tmp/a40/apptainer/cache APPTAINER_TMPDIR=/tmp/a40/apptainer/tmp
CTR=/tmp/a40/ctr
ROOT=/work/weiqian_stat/taikun/a40_rlatk
mkdir -p "$ROOT/grid_runs"
LOG="$ROOT/grid_runs/guard.log"
: > "$LOG"
APPTAINERENV_HF_HOME=/tmp/a40/hf \
APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
APPTAINERENV_GUARD_DTYPE=bfloat16 \
APPTAINERENV_GUARD_PORT=8000 \
APPTAINERENV_GUARD_MICRO_BATCH=16 \
APPTAINERENV_PYTHONUNBUFFERED=1 \
setsid nohup apptainer exec --nv -B /tmp/a40,/work "$CTR" \
  /tmp/a40/venv_guard/bin/python -u "$ROOT/serve_guard_cfg.py" > "$LOG" 2>&1 < /dev/null &
echo "guard launching pid $! ; log=$LOG"
