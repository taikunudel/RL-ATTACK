#!/bin/bash

MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-1024}
STREAM_LOGS=${STREAM_LOGS:-0}
SAVE_LOGS=${SAVE_LOGS:-0}

run_attack() {
  local samples_per_tok=$1
  local atker_mode=$2
  local data_name=$3

  export ATKER_PATH='bert-base-uncased'
  export ATKER_MODE="$atker_mode"
  export ATTACK_WHAT='doc'
  export DATA_NAME="$data_name"
  export SAVE_TO_PATH='/usa/taikun/rl-attack/rl_atk/attack-genai/trained_attacker/attacker_05082025_162746_llama-guard_doc_0.3_6_6000_0.8100.pth'
  export TARGET_PATH='qwen3-8b'
  export SERVER_URL='http://localhost:8002/v1'
  export LEN_DOC_MAX=512
  export NUM_DOC_MASKS=5
  export SAMPLES_PER_TOK=$samples_per_tok

  timestamp=$(date +%m%d_%H%M%S)
  export EVALUATION_PREFIX="eva_${timestamp}_${ATKER_PATH}_${TARGET_PATH}_${ATTACK_WHAT}_${ATKER_MODE}_${DATA_NAME}_${NUM_DOC_MASKS}_${SAMPLES_PER_TOK}"

  if [[ "$SAVE_LOGS" == "1" ]]; then
    mkdir -p /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results
    export OUTPUT_TXT="/usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/${EVALUATION_PREFIX}.txt"
    export ATK_JSON_LOG="/usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/${EVALUATION_PREFIX}.json"
  else
    export OUTPUT_TXT="/dev/null"
    export ATK_JSON_LOG="/tmp/${EVALUATION_PREFIX}.json"
  fi

  echo "=== Running attack: mode=$ATKER_MODE, data=$DATA_NAME, samples_per_tok=$SAMPLES_PER_TOK, max_gen_tokens=$MAX_GEN_TOKENS ==="
  if [[ "$SAVE_LOGS" == "1" ]]; then
    echo "  Log → $OUTPUT_TXT"
  else
    echo "  Not saving attack logs (SAVE_LOGS=0)"
  fi

  cmd=(conda run -n hqaattack python -u /usa/taikun/rl-attack/rl_atk/attack-genai/evaluation_attacker_genai_llama_itself.py \
    --atker_path "$ATKER_PATH" \
    --atker_mode "$ATKER_MODE" \
    --target_path "$TARGET_PATH" \
    --data_name "$DATA_NAME" \
    --save_to_path "$SAVE_TO_PATH" \
    --len_doc_max "$LEN_DOC_MAX" \
    --num_doc_masks "$NUM_DOC_MASKS" \
    --atk_json_log "$ATK_JSON_LOG" \
    --server_url "$SERVER_URL" \
    --samples_per_tok "$SAMPLES_PER_TOK" \
    --max_gen_tokens "$MAX_GEN_TOKENS")

  if [[ "$STREAM_LOGS" == "1" ]]; then
    echo "  Streaming output..."
    if [[ "$SAVE_LOGS" == "1" ]]; then
      stdbuf -oL -eL "${cmd[@]}" 2>&1 | stdbuf -oL tr '\r' '\n' | tee "$OUTPUT_TXT"
    else
      stdbuf -oL -eL "${cmd[@]}" 2>&1 | stdbuf -oL tr '\r' '\n'
    fi
  else
    if [[ "$SAVE_LOGS" == "1" ]]; then
      "${cmd[@]}" > "$OUTPUT_TXT" 2>&1 &
    else
      "${cmd[@]}" > /dev/null 2>&1 &
    fi
  fi
}

#─ Arrays of configurations ──────────────────────────────────────────────────────
# atker_modes=(trained untrained random)
# atker_modes=(trained untrained)
# atker_modes=(random)
atker_modes=(trained)
data_names=(harmul_strings)
# samples_per_toks=(150)
samples_per_toks=(5)

#─ Launch jobs ───────────────────────────────────────────────────────────────────
for atker_mode in "${atker_modes[@]}"; do
  for data_name in "${data_names[@]}"; do
    for samples_per_tok in "${samples_per_toks[@]}"; do
      run_attack "$samples_per_tok" "$atker_mode" "$data_name"
      if [[ "$STREAM_LOGS" == "1" ]]; then
        echo "=== Completed attack: mode=$atker_mode, data=$data_name, samples_per_tok=$samples_per_tok ==="
      else
        sleep 5
      fi
    done
  done
done

#─ Wait for everything to finish ─────────────────────────────────────────────────
if [[ "$STREAM_LOGS" != "1" ]]; then
  wait
  echo "=== All attacks completed ==="
else
  echo "=== Streaming run(s) finished ==="
fi

