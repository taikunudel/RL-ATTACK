#!/bin/bash

# # For Yelp Validation Set
# # Attacker BERTFineTuned
# DATA_NAME='yelp_val'
# DATA_PATH='/usa/taikun/07_transencoder/yelp/yelp/yelp.txt'
# IMPORTANT_TOKENS_FILE='/usa/taikun/07_transencoder/0dataProcessing/NImportantTokens_yelp_val_0_updated.json'
# TGT_MODEL="textattack/bert-base-uncased-yelp-polarity"
# ATTACKER_NAME='BERTFineTuned'
# ATTACKER_PATH='/usa/taikun/07_transencoder/yelp/attackerModels'
# NUMS_ATTACKED_TOKENS=1.0 # useless for now, percentage of importantce to attack
# MAX_LEN_DOC=512

# ALPHA=-0.1

# ATTACKER_FILE='/usa/taikun/07_transencoder/yelp/attackerModelsattacker_yelp_0.7_6_133_0.6667.pth'
# NUMS_CANDIDATES_EACH_TOKEN=400 # total numbers of desire attacked tokens queries
# NUMS_MAX_CANDIDATES=24 # change here to change thedesired attacked tokens

# IF_PRINT_ATTACK_PROCESS=True
# STARTFROMSAMPLE=128


# # For Yelp Validation Set
# # Attacker BERTMaskedLM
# DATA_NAME='yelp_val'
# DATA_PATH='/usa/taikun/07_transencoder/yelp/yelp/yelp.txt'
# IMPORTANT_TOKENS_FILE='/usa/taikun/07_transencoder/0dataProcessing/NImportantTokens_yelp_val_0_updated.json'
# TGT_MODEL="textattack/bert-base-uncased-yelp-polarity"
# ATTACKER_NAME='BERTMaskedLM'
# ATTACKER_PATH='/usa/taikun/07_transencoder/yelp/attackerModels'
# NUMS_ATTACKED_TOKENS=1.0
# MAX_LEN_DOC=512

# ALPHA=-0.1

# ATTACKER_FILE='None'
# NUMS_CANDIDATES_EACH_TOKEN=400 # total numbers of desire attacked tokens
# NUMS_MAX_CANDIDATES=24
# IF_PRINT_ATTACK_PROCESS=True
# STARTFROMSAMPLE=128


# # For Yelp Validation Set
# # Attacker BERTMaskedLMRandom
# DATA_NAME='yelp_val'
# DATA_PATH='/usa/taikun/07_transencoder/yelp/yelp/yelp.txt'
# IMPORTANT_TOKENS_FILE='/usa/taikun/07_transencoder/0dataProcessing/NImportantTokens_yelp_val_0_updated.json'
# TGT_MODEL="textattack/bert-base-uncased-yelp-polarity"
# ATTACKER_NAME='BERTMaskedLMRandom'
# ATTACKER_PATH='/usa/taikun/07_transencoder/yelp/attackerModels'
# NUMS_ATTACKED_TOKENS=1.0
# MAX_LEN_DOC=512

# ALPHA=-0.1

# ATTACKER_FILE='None'
# NUMS_CANDIDATES_EACH_TOKEN=400 # total numbers of desire attacked tokens
# NUMS_MAX_CANDIDATES=24
# IF_PRINT_ATTACK_PROCESS=True
# STARTFROMSAMPLE=35


# For IMDB Validation Set
# # Attacker BERTFineTuned
DATA_NAME='imdb_val'
MODE='classification'
DATA_PATH='/usa/taikun/07_transencoder/imdb/imdb.txt'
IMPORTANT_TOKENS_FILE='/usa/taikun/07_transencoder/0dataProcessing/NImportantTokens_imdb_val_0_updated.json'
TGT_MODEL="textattack/bert-base-uncased-imdb"
ATTACKER_NAME='BERTFineTuned'
ATTACKER_PATH='./imdb/attackerModels/'
NUMS_ATTACKED_TOKENS=1.0
MAX_LEN_DOC=512

ALPHA=-0.1

ATTACKER_FILE='/usa/taikun/07_transencoder/imdb/attackerModels/attacker_imdb_2_27_0.5000.pth'
NUMS_CANDIDATES_EACH_TOKEN=1000 # total numbers of desire attacked tokens
NUMS_MAX_CANDIDATES=48
IF_PRINT_ATTACK_PROCESS=True

STARTFROMSAMPLE=0


# prefix for evaluation results txt and json files
EVALUATION_PREFIX="${ATTACKER_NAME}_${DATA_NAME}_${NUMS_ATTACKED_TOKENS}_${NUMS_MAX_CANDIDATES}_${NUMS_CANDIDATES_EACH_TOKEN}_${STARTFROMSAMPLE}"
EVALUATION_JSON="${EVALUATION_PREFIX}.json"
OUTPUT_FILE="${EVALUATION_PREFIX}.txt"

# Print the summary message
echo "Attacking with the following parameters:"
echo "Attacker Name: $ATTACKER_NAME"
echo "Target Model: $TGT_MODEL"
echo "Data Name: $DATA_NAME"
echo "JSON File: $EVALUATION_JSON"

# Run the python script with the defined variables
python attackerEvaluation.py \
  --dataName $DATA_NAME \
  --mode $MODE \
  --dataPath $DATA_PATH \
  --importantTokensFile $IMPORTANT_TOKENS_FILE \
  --tgtModel $TGT_MODEL \
  --attackerPath $ATTACKER_PATH \
  --numsAttackedTokens $NUMS_ATTACKED_TOKENS \
  --numsMaxCandidates $NUMS_MAX_CANDIDATES \
  --maxLenDoc $MAX_LEN_DOC \
  --alpha $ALPHA \
  --attackerName $ATTACKER_NAME \
  --attackerFile $ATTACKER_FILE \
  --numCandidatesEachToken $NUMS_CANDIDATES_EACH_TOKEN \
  --ifPrintAttackProcess $IF_PRINT_ATTACK_PROCESS \
  --startFromSample $STARTFROMSAMPLE\
  --evaluationPartition $EVALUATION_JSON > $OUTPUT_FILE
