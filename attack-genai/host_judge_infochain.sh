#!/usr/bin/env bash
# Host a Llama Guard judge on infochain via vLLM (OpenAI-compatible server).
# Companion to `--judge infochain` in evaluation_attacker_genai_llama_itself.py.
#
# Run this ON infochain (e.g. inside a `screen`/`tmux` so it survives logout):
#     bash host_judge_infochain.sh
# Then, from the attack host, point the judge at it:
#     --judge infochain                         # default endpoint http://infochain:8000/v1/chat/completions
#     # or override: --judge_url http://infochain:8000/v1/chat/completions --judge_model llama-guard-3
#
# NOTE: served-model-name MUST match --judge_model on the client side.
set -euo pipefail

# NATIVE PRECISION ONLY: --dtype auto makes vLLM use the checkpoint's own dtype
# (bf16 for Llama-Guard). Do NOT set float16 on a bf16 model — it can change results.
#
# VRAM at native bf16 (rough, weights+KV on one GPU):
#   Llama-Guard-3-8B  ~16-20 GB -> fits a single 24 GB GPU (TP=1).
#   Llama-Guard-4-12B ~24-30 GB -> does NOT fit one 24 GB GPU; set GPU="0,1" and TP=2.
# If it won't fit at native precision, stop and tell the user (don't quantize).

GPU="${GPU:-0}"                                   # which GPU(s); e.g. "0" or "0,1" for TP=2
MODEL="${MODEL:-meta-llama/Llama-Guard-3-8B}"     # HF model id to load
SERVED_NAME="${SERVED_NAME:-llama-guard-3}"       # MUST match client --judge_model
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.90}"
DTYPE="${DTYPE:-auto}"                             # auto = native checkpoint dtype (keep it)
# tensor-parallel size = number of GPUs listed in $GPU
TP="${TP:-$(awk -F, '{print NF}' <<<"$GPU")}"

echo "[host_judge_infochain] GPU=$GPU TP=$TP dtype=$DTYPE model=$MODEL served-as=$SERVED_NAME port=$PORT"
CUDA_VISIBLE_DEVICES="$GPU" python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAXLEN" \
    --gpu-memory-utilization "$GPU_UTIL"
