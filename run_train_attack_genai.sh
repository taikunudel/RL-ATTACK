#!/bin/bash
export ATKER_PATH='bert-base-uncased'
export TARGET_PATH='defult' # default is LLaMA Guard 3 1b
export SAVE_TO_PATH='/usa/taikun/07_transencoder/1training/llama-guard-attacker'
export LEN_DOC_MAX=512
export PREFIX_LENGTH=10
export SAMPLES_PER_TOK=3
export ATK_WHAT='doc'
export ALPHA=0.7
export NUM_DOC_MASKS=10

# export TRAIN_PREFIX="train_$(date +%m%d_%H%M%S)_${ATKER_PATH}_${TARGET_PATH}_${PREFIX_LENGTH}_${MAX_QUERIES_PER_DOC}_${SAMPLES_PER_TOK}"
export TRAIN_PREFIX="train_$(date +%m%d_%H%M%S)_${ATKER_PATH}_${TARGET_PATH}_${ALPHA}_${NUM_DOC_MASKS}_${SAMPLES_PER_TOK}"
export OUTPUT_TXT="/usa/taikun/07_transencoder/attack-genai/${TRAIN_PREFIX}.txt"

# # Print the summary message
echo "Attacking with the following parameters:"
echo "Attacker Name: $ATKER_PATH"
echo "Target Model: $TARGET_PATH"

SCRIPT_PATH="/usa/taikun/07_transencoder/attack-genai/train_attacker_genai.py"
# Run the python script with the defined variables
python -u $SCRIPT_PATH \
  --atker_path $ATKER_PATH \
  --target_path $TARGET_PATH \
  --save_to_path $SAVE_TO_PATH \
  --len_doc_max $LEN_DOC_MAX \
  --prefix_length $PREFIX_LENGTH \
  --num_doc_masks $NUM_DOC_MASKS \
  --atk_what $ATK_WHAT \
  --alpha $ALPHA > $OUTPUT_TXT 2>&1