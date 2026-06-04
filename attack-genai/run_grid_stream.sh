#!/usr/bin/env bash
# One stream of the RL-ATTACK grid. Args: STREAM_ID(0|1)  SERVER_URL
# Splits the 99 configs round-robin by global index % 2 == STREAM_ID.
# Each config: TRAIN (1 epoch, best-only) -> EVAL trained+untrained -> quality metrics.
# Resumable via per-config .DONE marker. FREE/local only.
set -u
STREAM_ID="${1:?stream id 0|1}"
SERVER_URL="${2:?server url}"

cd /usa/taikun/rl-attack/rl_atk/attack-genai
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate 03_transf_py311 2>/dev/null

ROOT=/usa/taikun/rl-attack/rl_atk/attack-genai
RUNDIR=$ROOT/grid_runs
mkdir -p "$RUNDIR" "$ROOT/trained_attacker"
TARGET="meta-llama/Llama-Guard-4-12B"; ATKER="bert-base-uncased"; DATA="harmul_strings"
SAMPLES_PER_TOK=20
EPOCHS=1
EVAL_INTERVAL=1000000     # effectively disable in-loop validation (1 epoch=996 steps)
MASTER=$RUNDIR/grid_master.csv
[ -f "$MASTER" ] || echo "reward,alpha,K,best_ckpt,eval_trained_json,eval_untrained_json,status,stream,server" > "$MASTER"

REWARDS=(A B C); ALPHAS=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0); KS=(3 5 10)
HEALTH_URL="${SERVER_URL%/v1}/health"      # health lives at /health, not /v1/health
guard_ok () { curl -s --max-time 8 "$HEALTH_URL" 2>/dev/null | grep -q status; }

# WORK-STEALING within an index RANGE: any stream runs ANY not-yet-claimed config
# whose global index is in [CONFIG_LO, CONFIG_HI]. Default = full 0..98. Set these to
# split across machines (e.g. infodeep CONFIG_LO=0 CONFIG_HI=39; A40 CONFIG_LO=40 CONFIG_HI=98).
CONFIG_LO="${CONFIG_LO:-0}"; CONFIG_HI="${CONFIG_HI:-98}"
GIDX=-1
for R in "${REWARDS[@]}"; do for A in "${ALPHAS[@]}"; do for K in "${KS[@]}"; do
  GIDX=$((GIDX+1))
  [ "$GIDX" -lt "$CONFIG_LO" ] && continue
  [ "$GIDX" -gt "$CONFIG_HI" ] && continue
  TAG="${R}_a${A}_K${K}"; DONE="$RUNDIR/$TAG.DONE"; CLAIM="$RUNDIR/$TAG.claim"
  [ -f "$DONE" ] && continue
  mkdir "$CLAIM" 2>/dev/null || { continue; }            # someone else is running it
  echo "$STREAM_ID $(date +%s)" > "$CLAIM/owner"
  echo "===== [s$STREAM_ID $(date +%F_%T)] $TAG ====="
  if ! guard_ok; then echo "[s$STREAM_ID] guard $SERVER_URL DOWN; sleeping 60s"; sleep 60; guard_ok || { echo "still down, releasing claim & abort"; rmdir "$CLAIM" 2>/dev/null; rm -f "$CLAIM/owner" 2>/dev/null; rm -rf "$CLAIM"; exit 3; }; fi

  TLOG="$RUNDIR/train_$TAG.log"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u train_attacker_genai.py \
    --atker_path "$ATKER" --target_path "$TARGET" --save_to_path "$ROOT/trained_attacker" \
    --len_doc_max 512 --atk_what doc --linear_head True \
    --num_doc_masks "$K" --alpha "$A" --reward_type "$R" \
    --epochs "$EPOCHS" --eval_interval "$EVAL_INTERVAL" \
    --server_url "$SERVER_URL" > "$TLOG" 2>&1
  TRC=$?
  BEST=$(grep -aoE "Best model saved.*path: \S+\.pth" "$TLOG" | tail -1 | grep -aoE "/\S+\.pth")
  if [ "$TRC" -ne 0 ] || [ -z "$BEST" ] || [ ! -f "$BEST" ]; then
    echo "[s$STREAM_ID $TAG] TRAIN FAIL rc=$TRC best=$BEST"; echo "$R,$A,$K,,,,train_failed,$STREAM_ID,$SERVER_URL" >> "$MASTER"
    rm -rf "$CLAIM"; continue   # release so another stream/retry can pick it up
  fi

  ETJ="$RUNDIR/eval_trained_$TAG.json"; EUJ="$RUNDIR/eval_untrained_$TAG.json"
  EXPECT=100                                   # must match --eval_limit
  rm -f "$ETJ" "$EUJ" 2>/dev/null              # never reuse stale/partial JSONs
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u evaluation_attacker_genai.py \
    --atker_path "$ATKER" --atker_mode trained --target_path "$TARGET" --data_name "$DATA" \
    --save_to_path "$BEST" --num_doc_masks "$K" --samples_per_tok "$SAMPLES_PER_TOK" --eval_limit $EXPECT \
    --atk_json_log "$ETJ" --server_url "$SERVER_URL" > "$RUNDIR/evalT_$TAG.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u evaluation_attacker_genai.py \
    --atker_path "$ATKER" --atker_mode untrained --target_path "$TARGET" --data_name "$DATA" \
    --save_to_path "$BEST" --num_doc_masks "$K" --samples_per_tok "$SAMPLES_PER_TOK" --eval_limit $EXPECT \
    --atk_json_log "$EUJ" --server_url "$SERVER_URL" > "$RUNDIR/evalU_$TAG.log" 2>&1

  # Completeness gate: BOTH eval JSONs must have the full EXPECT records, else this
  # config is NOT done (release claim so it re-runs). Prevents partial-eval -> DONE.
  NT=$(python3 -c "import json;print(len(json.load(open('$ETJ'))))" 2>/dev/null || echo 0)
  NU=$(python3 -c "import json;print(len(json.load(open('$EUJ'))))" 2>/dev/null || echo 0)
  if [ "$NT" -lt "$EXPECT" ] || [ "$NU" -lt "$EXPECT" ]; then
    echo "[s$STREAM_ID $TAG] EVAL INCOMPLETE (trained=$NT untrained=$NU, want $EXPECT) -> releasing"
    echo "$R,$A,$K,$BEST,$ETJ,$EUJ,eval_incomplete,$STREAM_ID,$SERVER_URL" >> "$MASTER"
    rm -rf "$CLAIM"; continue
  fi

  # Quality metrics (fluency=GPT-2 PPL + grammar=CoLA) on BOTH trained and untrained
  # adversarial docs — so every config reports attack + semantic + fluency + grammar.
  CUDA_VISIBLE_DEVICES=0 python -u quality_metrics.py --inputs "$ETJ" \
      --out-prefix "$RUNDIR/quality_trained_$TAG" --device cpu > "$RUNDIR/quality_trained_$TAG.log" 2>&1 || true
  CUDA_VISIBLE_DEVICES=0 python -u quality_metrics.py --inputs "$EUJ" \
      --out-prefix "$RUNDIR/quality_untrained_$TAG" --device cpu > "$RUNDIR/quality_untrained_$TAG.log" 2>&1 || true

  echo "$R,$A,$K,$BEST,$ETJ,$EUJ,ok,$STREAM_ID,$SERVER_URL" >> "$MASTER"
  touch "$DONE"; echo "[s$STREAM_ID $TAG] DONE"   # keep $CLAIM dir as the 'taken' marker
done; done; done
echo "STREAM_${STREAM_ID}_PASS_DONE $(date +%F_%T)  (done=$(ls "$RUNDIR"/*.DONE 2>/dev/null|wc -l)/99)"
