#!/bin/bash

# Load specific configuration
# source ../config_yelp_val.sh
# source ../config_imdb_val.sh
# source ../config_mnli_matched_val.sh
# source ../config_mnli_mismatched_val.sh
# prefix for evaluation results txt and json files

# # Print the summary message
echo "Attacking with the following parameters:"
echo "Attacker Name: $ATTACKER_NAME"
echo "Target Model: $TGT_MODEL"
echo "Data Name: $DATA_NAME"
echo "JSON File: $EVALUATION_JSON"


# (moved) entry point is now: python -m rlatk.classifier.eval.evaluation_nli
# Run the python script with the defined variables
python -u -m rlatk.classifier.eval.evaluation_nli \
  --dataName $DATA_NAME \
  --mode $MODE \
  --dataPath $DATA_PATH \
  --importantTokensFile $IMPORTANT_TOKENS_FILE \
  --tgtModel $TGT_MODEL \
  --tgt_model_name $TGT_MODEL_NAME \
  --numsAttackedTokens $NUMS_ATTACKED_TOKENS \
  --numsMaxCandidates $NUMS_MAX_CANDIDATES \
  --maxLenDoc $MAX_LEN_DOC \
  --attackerName $ATTACKER_NAME \
  --attackerFile $ATTACKER_FILE \
  --numCandidatesEachToken $NUMS_CANDIDATES_EACH_TOKEN \
  --ifPrintAttackProcess $IF_PRINT_ATTACK_PROCESS \
  --startFromSample $STARTFROMSAMPLE\
  --evaluationPartition $EVALUATION_JSON \
  --attacker_name $ATTACKER_NAME > $OUTPUT_FILE
