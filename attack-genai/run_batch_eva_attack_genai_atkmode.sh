#!/bin/bash

run_attack() {
  local samples_per_tok=$1
  local atker_mode=$2
  local data_name=$3

  export ATKER_PATH='bert-base-uncased'
  export ATKER_MODE="$atker_mode"
  export ATTACK_WHAT='doc'
  export DATA_NAME="$data_name"
  export SAVE_TO_PATH='/usa/taikun/07_transencoder/rl_atk/attack-genai/trained_attacker/attacker_05082025_162746_llama-guard_doc_0.3_6_6000_0.8100.pth'
  export TARGET_PATH='llama-guard-3-8b'
  export SERVER_URL='http://infodeep:8002/v1'
  export LEN_DOC_MAX=512
  export NUM_DOC_MASKS=3
  export SAMPLES_PER_TOK=$samples_per_tok

  timestamp=$(date +%m%d_%H%M%S)
  export EVALUATION_PREFIX="eva_${timestamp}_${ATKER_PATH}_${TARGET_PATH}_${ATTACK_WHAT}_${ATKER_MODE}_${DATA_NAME}_${NUM_DOC_MASKS}_${SAMPLES_PER_TOK}"
  export OUTPUT_TXT="/usa/taikun/07_transencoder/rl_atk/attack-genai/${EVALUATION_PREFIX}.txt"
  export ATK_JSON_LOG="/usa/taikun/07_transencoder/rl_atk/attack-genai/${EVALUATION_PREFIX}.json"

  echo "=== Running attack: mode=$ATKER_MODE, data=$DATA_NAME, samples_per_tok=$SAMPLES_PER_TOK ==="
  echo "  Log → $OUTPUT_TXT"

  python -u /usa/taikun/07_transencoder/rl_atk/attack-genai/evaluation_attacker_genai.py \
    --atker_path    "$ATKER_PATH" \
    --atker_mode    "$ATKER_MODE" \
    --target_path   "$TARGET_PATH" \
    --data_name     "$DATA_NAME" \
    --save_to_path  "$SAVE_TO_PATH" \
    --len_doc_max   "$LEN_DOC_MAX" \
    --num_doc_masks "$NUM_DOC_MASKS" \
    --atk_json_log  "$ATK_JSON_LOG" \
    --server_url    "$SERVER_URL" \
    --samples_per_tok "$SAMPLES_PER_TOK" \
  > "$OUTPUT_TXT" 2>&1 &
}

#─ Arrays of configurations ──────────────────────────────────────────────────────
# atker_modes=(trained untrained random)
atker_modes=(random)
data_names=(harmul_strings harmful_behaviors)
samples_per_toks=(100 200 300)

#─ Launch jobs ───────────────────────────────────────────────────────────────────
for atker_mode in "${atker_modes[@]}"; do
  for data_name in "${data_names[@]}"; do
    for samples_per_tok in "${samples_per_toks[@]}"; do
      run_attack "$samples_per_tok" "$atker_mode" "$data_name"
      sleep 5
    done
  done
done

#─ Wait for everything to finish ─────────────────────────────────────────────────
wait
echo "=== All attacks completed ==="
