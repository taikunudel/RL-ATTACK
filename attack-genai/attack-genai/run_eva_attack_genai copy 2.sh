#!/bin/bash
export ATKER_PATH='bert-base-uncased'
export TARGET_PATH='defult' # default is LLaMA Guard 3 1b
export SAVE_TO_PATH='/usa/taikun/07_transencoder/1training/llama-guard-attacker/attacker_llama-guard_4_5100_0.6450.pth'
export LEN_DOC_MAX=512
export PREFIX_LENGTH=10
export SAMPLES_PER_TOK=3
export MAX_QUERIES_PER_DOC=100

export EVALUATION_PREFIX="eva_$(date +%m%d_%H%M%S)_${ATKER_PATH}_${TARGET_PATH}_${PREFIX_LENGTH}_${MAX_QUERIES_PER_DOC}_${SAMPLES_PER_TOK}"
export OUTPUT_TXT="/usa/taikun/07_transencoder/attack-genai/${EVALUATION_PREFIX}.txt"
export ATK_JSON_LOG="/usa/taikun/07_transencoder/attack-genai/${EVALUATION_PREFIX}.json"

# # Print the summary message
echo "Attacking with the following parameters:"
echo "Attacker Name: $ATKER_PATH"
echo "Target Model: $TARGET_PATH"
echo "JSON File: $ATK_JSON_LOG"

SCRIPT_PATH="/usa/taikun/07_transencoder/attack-genai/evaluation_attacker_genai.py"
# Run the python script with the defined variables
python -u $SCRIPT_PATH \
  --atker_path $ATKER_PATH \
  --target_path $TARGET_PATH \
  --save_to_path $SAVE_TO_PATH \
  --len_doc_max $LEN_DOC_MAX \
  --prefix_length $PREFIX_LENGTH \
  --atk_json_log $ATK_JSON_LOG \
  --max_queries_per_doc $MAX_QUERIES_PER_DOC \
  --samples_per_tok $SAMPLES_PER_TOK > $OUTPUT_TXT 2>&1
