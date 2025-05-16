#!/bin/bash

run_attack() {
  local samples_per_tok=$1
  local save_path=$2

  export ATKER_PATH='bert-base-uncased'
  export ATKER_MODE='trained' # trained, untrained, random
  export ATKER_MODE='untrained'
  # export ATKER_MODE='random'
  export ATTACK_WHAT='doc'
  export TARGET_PATH='llama-guard-3-8b'
  export SERVER_URL='http://infodeep:8000/v1'
  export SAVE_TO_PATH="$save_path"
  export LEN_DOC_MAX=512
  export NUM_DOC_MASKS=10

  export SAMPLES_PER_TOK=$samples_per_tok

  timestamp=$(date +%m%d_%H%M%S)
  # Extract a descriptive name from the save path for the evaluation prefix
  local save_name=$(basename "$save_path" .pth)
  export EVALUATION_PREFIX="eva_${timestamp}_${ATKER_PATH}_${TARGET_PATH}_${ATTACK_WHAT}_${ATKER_MODE}_${NUM_DOC_MASKS}_${SAMPLES_PER_TOK}_${save_name}"
  export OUTPUT_TXT="/usa/taikun/07_transencoder/rl_atk/attack-genai/${EVALUATION_PREFIX}.txt"
  export ATK_JSON_LOG="/usa/taikun/07_transencoder/rl_atk/attack-genai/${EVALUATION_PREFIX}.json"

  echo "=== Running attack with SAVE_TO_PATH=$SAVE_TO_PATH, SAMPLES_PER_TOK=$SAMPLES_PER_TOK ==="
  echo "Log: $OUTPUT_TXT"

  python -u /usa/taikun/07_transencoder/rl_atk/attack-genai/evaluation_attacker_genai.py \
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

# Define the different SAVE_TO_PATH values
save_paths=(
  '/usa/taikun/07_transencoder/rl_atk/attack-genai/trained_attacker/attacker_05092025_204952_llama-guard_doc_0.0_3_3000_0.8000.pth'
  '/usa/taikun/07_transencoder/rl_atk/attack-genai/trained_attacker/attacker_05082025_162359_llama-guard_doc_0.5_9_9100_0.7800.pth'
  '/usa/taikun/07_transencoder/rl_atk/attack-genai/trained_attacker/attacker_05082025_162758_llama-guard_doc_0.7_5_5800_0.8100.pth'
  '/usa/taikun/07_transencoder/rl_atk/attack-genai/trained_attacker/attacker_05092025_205007_llama-guard_doc_1.0_3_3400_0.7800.pth'
)

# Loop through the different SAVE_TO_PATH values and samples_per_tok
for save_path in "${save_paths[@]}"; do
  for tok in 80; do
    run_attack "$tok" "$save_path"
    sleep 5 # prevent timestamp collisions
  done
done

# Wait for all background jobs to complete
wait

echo "=== All attacks completed ==="