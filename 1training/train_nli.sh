#!/bin/bash

TODAY=$(date +"%Y%m%d")
PREFIX="training_${TODAY}_${ATTACKER_NAME}_${DATA_NAME}_${ALPHA}_${ON}"
TXT_FILE="${PREFIX}.txt"

# Print the summary message
echo "Attacking classification with the following parameters:"
echo "Attacker Name: $ATTACKER_NAME"
echo "Target Model: $TGT_MODEL"
echo "Data Name: $DATA_NAME"

# Run the python script with the defined variables
python /usa/taikun/07_transencoder/1training/attacker_training_nli.py \
  --dataName $DATA_NAME \
  --dataPath $DATA_PATH \
  --importantTokensFile $IMPORTANT_TOKENS_FILE \
  --tgtModel $TGT_MODEL \
  --attackerPath $ATTACKER_PATH \
  --numsAttackedTokens $NUMS_ATTACKED_TOKENS \
  --maxLenDoc $MAX_LEN_DOC \
  --alpha $ALPHA > $TXT_FILE