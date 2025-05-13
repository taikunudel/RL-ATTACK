#!/bin/bash

run_attack() {
  local samples_per_tok=$1
  export ATKER_PATH='bert-base-uncased'
  # export ATKER_MODE='trained'
  export ATKER_MODE='untrained' # trained, untrained, random
  export ATTACK_WHAT='doc'
  export TARGET_PATH='llama-guard-3-8b'
  export SERVER_URL='http://infodeep:8001/v1'
  # export SAVE_TO_PATH='/usa/taikun/07_transencoder/attack-genai/trained_attacker/attacker_05082025_162359_llama-guard_doc_0.5_9_9100_0.7800.pth'
  export SAVE_TO_PATH='/usa/taikun/07_transencoder/attack-genai/trained_attacker/attacker_05082025_162746_llama-guard_doc_0.3_6_6000_0.8100.pth'
  # export SAVE_TO_PATH='/usa/taikun/07_transencoder/attack-genai/trained_attacker/attacker_05092025_205007_llama-guard_doc_1.0_3_3400_0.7800.pth'
  export LEN_DOC_MAX=512
  export NUM_DOC_MASKS=10

  export SAMPLES_PER_TOK=$samples_per_tok

  timestamp=$(date +%m%d_%H%M%S)
  export EVALUATION_PREFIX="eva_${timestamp}_${ATKER_PATH}_${TARGET_PATH}_${ATTACK_WHAT}_${ATKER_MODE}_${NUM_DOC_MASKS}_${SAMPLES_PER_TOK}"
  export OUTPUT_TXT="/usa/taikun/07_transencoder/attack-genai/${EVALUATION_PREFIX}.txt"
  export ATK_JSON_LOG="/usa/taikun/07_transencoder/attack-genai/${EVALUATION_PREFIX}.json"

  echo "=== Running attack with SAMPLES_PER_TOK=$SAMPLES_PER_TOK ==="
  echo "Log: $OUTPUT_TXT"

  python -u /usa/taikun/07_transencoder/attack-genai/evaluation_attacker_genai.py \
    --atker_path "$ATKER_PATH" \
    --atker_mode "$ATKER_MODE" \
    --target_path "$TARGET_PATH" \
    --save_to_path "$SAVE_TO_PATH" \
    --len_doc_max "$LEN_DOC_MAX" \
    --num_doc_masks "$NUM_DOC_MASKS" \
    --atk_json_log "$ATK_JSON_LOG" \
    --server_url "$SERVER_URL" \
    --samples_per_tok "$SAMPLES_PER_TOK" > "$OUTPUT_TXT" 2>&1 &
}

# Loop through values and run in parallel
for tok in 100 200 300 400 500; do
  run_attack "$tok"
  sleep 60  # prevent timestamp collisions
done

# Wait for all background jobs to complete
wait

echo "=== All attacks completed ==="
