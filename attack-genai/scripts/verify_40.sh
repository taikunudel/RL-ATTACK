#!/usr/bin/env bash
# Deferred VERIFICATION run: config 40 (B_a0.8_K5) on infodeep, to cross-check the A40's config 40.
# Waits until the 0-39 grid finishes (frees a GPU), then runs the IDENTICAL pipeline into verify_40/
# (isolated; never touches the canonical 0-39 grid_runs). Same args as the grid -> apples-to-apples.
set -uo pipefail
cd /usa/taikun/rl-attack/rl_atk/attack-genai
source ~/miniconda3/etc/profile.d/conda.sh; conda activate 03_transf_py311
ROOT=/usa/taikun/rl-attack/rl_atk/attack-genai
VDIR="$ROOT/grid_runs/verify_40"; mkdir -p "$VDIR"
LOG="$VDIR/verify.log"
GUARD="http://localhost:8001/v1"
TARGET="meta-llama/Llama-Guard-4-12B"; ATKER="bert-base-uncased"; DATA="harmul_strings"
R=B; A=0.8; K=5; TAG="B_a0.8_K5"
echo "[verify $(date +%F_%T)] queued; waiting for 0-39 grid to complete (40/40 .DONE) before starting" >> "$LOG"

# 1. wait for the live grid to free the GPUs (don't compete)
while :; do
  n=$(ls "$ROOT/grid_runs"/*.DONE 2>/dev/null | wc -l)
  if [ "$n" -ge 40 ]; then echo "[verify $(date +%F_%T)] 0-39 done ($n/40) -> starting config 40" >> "$LOG"; break; fi
  sleep 600
done

# 2. guard must be healthy
for i in $(seq 1 30); do curl -s --max-time 6 "${GUARD%/v1}/health" 2>/dev/null | grep -q status && break; sleep 10; done

# 3. TRAIN config 40 (identical args to the grid), GPU0
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u train_attacker_genai.py \
  --atker_path "$ATKER" --target_path "$TARGET" --save_to_path "$VDIR/attacker" \
  --len_doc_max 512 --atk_what doc --linear_head True \
  --num_doc_masks "$K" --alpha "$A" --reward_type "$R" --epochs 1 --eval_interval 1000000 \
  --server_url "$GUARD" > "$VDIR/train_$TAG.log" 2>&1
BEST=$(grep -aoE "Best model saved.*path: \S+\.pth" "$VDIR/train_$TAG.log" | tail -1 | grep -aoE "/\S+\.pth")
if [ -z "$BEST" ] || [ ! -f "$BEST" ]; then echo "[verify $(date +%F_%T)] TRAIN FAIL (no best ckpt)" >> "$LOG"; exit 1; fi
echo "[verify $(date +%F_%T)] trained; best=$BEST" >> "$LOG"

# 4. EVAL trained + untrained (identical args)
for M in trained untrained; do
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u evaluation_attacker_genai.py \
    --atker_path "$ATKER" --atker_mode "$M" --target_path "$TARGET" --data_name "$DATA" \
    --save_to_path "$BEST" --num_doc_masks "$K" --samples_per_tok 20 --eval_limit 100 \
    --atk_json_log "$VDIR/eval_${M}_$TAG.json" --server_url "$GUARD" > "$VDIR/eval${M}_$TAG.log" 2>&1
done
echo "[verify $(date +%F_%T)] VERIFY_DONE — eval JSONs in $VDIR (compare vs A40 grid_runs_a40/)" >> "$LOG"
touch "$VDIR/VERIFY_DONE"
