#!/usr/bin/env bash
# Autonomous RL-ATTACK grid driver vs LLaMA Guard 4 12B (self-hosted, free).
# Grid: reward {A,B,C} x alpha {1.0..0.0 step 0.1} x K(num_doc_masks) {3,5,10} = 99 configs.
# For each config: TRAIN (best-only ckpt) -> EVAL (trained + untrained) -> quality metrics.
# Resumable: skips a config whose DONE marker exists. Sequential (one GPU / one guard).
set -u
cd /usa/taikun/rl-attack/rl_atk/attack-genai
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate 03_transf_py311 2>/dev/null

ROOT=/usa/taikun/rl-attack/rl_atk/attack-genai
RUNDIR=$ROOT/grid_runs
mkdir -p "$RUNDIR" "$ROOT/trained_attacker"
SERVER_URL="http://localhost:8000/v1"
TARGET="meta-llama/Llama-Guard-4-12B"
ATKER="bert-base-uncased"
DATA="harmul_strings"
SAMPLES_PER_TOK=20         # eval-time Q-candidates (attack-time knob)
MASTER=$RUNDIR/grid_master.csv
[ -f "$MASTER" ] || echo "reward,alpha,K,train_log,best_ckpt,eval_trained_json,eval_untrained_json,status" > "$MASTER"

REWARDS=(A B C)
ALPHAS=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)
KS=(3 5 10)

guard_ok () { curl -s --max-time 8 "$SERVER_URL/health" 2>/dev/null | grep -q status; }

for R in "${REWARDS[@]}"; do
 for A in "${ALPHAS[@]}"; do
  for K in "${KS[@]}"; do
    TAG="${R}_a${A}_K${K}"
    DONE="$RUNDIR/$TAG.DONE"
    if [ -f "$DONE" ]; then echo "[skip] $TAG (done)"; continue; fi
    echo "===== [$(date +%F_%T)] CONFIG $TAG ====="
    if ! guard_ok; then echo "[$TAG] guard DOWN -> aborting grid"; exit 3; fi

    TLOG="$RUNDIR/train_$TAG.log"
    # TRAIN (best-only checkpoint; stable filename built from timestamp inside the code,
    # so capture the printed path)
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u train_attacker_genai.py \
      --atker_path "$ATKER" --target_path "$TARGET" \
      --save_to_path "$ROOT/trained_attacker" \
      --len_doc_max 512 --atk_what doc --linear_head True \
      --num_doc_masks "$K" --alpha "$A" --reward_type "$R" \
      --server_url "$SERVER_URL" > "$TLOG" 2>&1
    TRC=$?
    BEST=$(grep -aoE "Best model saved.*path: \S+\.pth" "$TLOG" | tail -1 | grep -aoE "\S+\.pth")
    if [ "$TRC" -ne 0 ] || [ -z "$BEST" ] || [ ! -f "$BEST" ]; then
      echo "[$TAG] TRAIN FAILED (rc=$TRC best=$BEST)"; echo "$R,$A,$K,$TLOG,,,,train_failed" >> "$MASTER"; continue
    fi
    echo "[$TAG] trained -> $BEST"

    # EVAL trained + untrained (classifier-evasion eval, like the codebase)
    ETJ="$RUNDIR/eval_trained_$TAG.json"
    EUJ="$RUNDIR/eval_untrained_$TAG.json"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u evaluation_attacker_genai.py \
      --atker_path "$ATKER" --atker_mode trained --target_path "$TARGET" --data_name "$DATA" \
      --save_to_path "$BEST" --num_doc_masks "$K" --samples_per_tok "$SAMPLES_PER_TOK" \
      --atk_json_log "$ETJ" --server_url "$SERVER_URL" > "$RUNDIR/evalT_$TAG.log" 2>&1
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u evaluation_attacker_genai.py \
      --atker_path "$ATKER" --atker_mode untrained --target_path "$TARGET" --data_name "$DATA" \
      --save_to_path "$BEST" --num_doc_masks "$K" --samples_per_tok "$SAMPLES_PER_TOK" \
      --atk_json_log "$EUJ" --server_url "$SERVER_URL" > "$RUNDIR/evalU_$TAG.log" 2>&1

    # QUALITY METRICS (PPL + CoLA) on the trained-eval pairs (local, free, eval-only)
    if [ -f "$ETJ" ]; then
      CUDA_VISIBLE_DEVICES=0 python -u quality_metrics.py --inputs "$ETJ" \
        --out-prefix "$RUNDIR/quality_$TAG" --device cpu > "$RUNDIR/quality_$TAG.log" 2>&1 || true
    fi

    echo "$R,$A,$K,$TLOG,$BEST,$ETJ,$EUJ,ok" >> "$MASTER"
    touch "$DONE"
    echo "[$TAG] DONE"
  done
 done
done
echo "ALL_GRID_DONE $(date +%F_%T)"
