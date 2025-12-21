#!/bin/bash

# For mnli Validation Set
export DATA_NAME='mnli_matched_val'
export DATA_PATH='/usa/taikun/07_transencoder/MNLI/mnli/mnli_matched.txt'
export IMPORTANT_TOKENS_FILE='/usa/taikun/07_transencoder/0dataProcessing/mnli/tokens_mnli_matched_val_0.json_premise.json'
# export IMPORTANT_TOKENS_FILE='/usa/taikun/07_transencoder/0dataProcessing/mnli/tokens_mnli_matched_val_0.json_hypothesis.json'

export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/mnli/premise/02Sep/attacker_mnlimatched_0.7_2_1000_2.9903.pth'
# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/mnli/premise/02Sep/attacker_mnlimatched_0.8_2_1000_2.9205.pth'
# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/mnli/premise/02Sep/attacker_mnlimatched_0.9_2_1000_2.6550.pth'
# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/mnli/hypothesis/9Sep/attacker_mnlimatched_0.5_1_500_3.1694.pth'
# export ATTACKER_FILE='None'

export MODE='nli'
export ON='premise'
# export ON='hypothesis'

# export TGT_MODEL_NAME="BERT_MNLI"
# export TGT_MODEL="textattack/bert-base-uncased-MNLI"
export TGT_MODEL_NAME="ROBERTA_MNLI"
export TGT_MODEL="textattack/roberta-base-MNLI"
# export TGT_MODEL_NAME='SBERT_NLI'
# export TGT_MODEL='cross-encoder/nli-roberta-base'

export ATTACKER_NAME='BERTFineTuned'
# export ATTACKER_NAME='BERTMaskedLMRandom'
# export ATTACKER_NAME='BERTMaskedLM'

export NUMS_ATTACKED_TOKENS=0.7
export MAX_LEN_DOC=512

# export NUMS_CANDIDATES_EACH_TOKEN=20 # total numbers of desired attacked tokens
# export NUMS_CANDIDATES_EACH_TOKEN=40 # total numbers of desired attacked tokens
export NUMS_CANDIDATES_EACH_TOKEN=60
# export NUMS_CANDIDATES_EACH_TOKEN=80
# export NUMS_CANDIDATES_EACH_TOKEN=120
# export NUMS_CANDIDATES_EACH_TOKEN=160
# export NUMS_CANDIDATES_EACH_TOKEN=40

export NUMS_MAX_CANDIDATES=8
# export NUMS_MAX_CANDIDATES=12
export IF_PRINT_ATTACK_PROCESS=True

export STARTFROMSAMPLE=0

# Print the summary message
echo "Attacking classification with the following parameters:"
echo "Attacker Name: $ATTACKER_NAME"
echo "Target Model: $TGT_MODEL"
echo "Data Name: $DATA_NAME"
echo "Start From Sample: $STARTFROMSAMPLE"

# export EVALUATION_PREFIX="evaluation_${ATTACKER_NAME}_${TGT_MODEL_NAME}_${DATA_NAME}_${NUMS_ATTACKED_TOKENS}_${NUMS_MAX_CANDIDATES}_${NUMS_CANDIDATES_EACH_TOKEN}_${STARTFROMSAMPLE}"
export EVALUATION_PREFIX="eva_$(date +%m%d_%H%M%S)${ATTACKER_NAME}_${TGT_MODEL_NAME}_${DATA_NAME}_${NUMS_ATTACKED_TOKENS}_${NUMS_MAX_CANDIDATES}_${NUMS_CANDIDATES_EACH_TOKEN}_${STARTFROMSAMPLE}"

export EVALUATION_JSON="/usa/taikun/07_transencoder/2evaluation/mnli/${ON}/${EVALUATION_PREFIX}.json"
export OUTPUT_FILE="/usa/taikun/07_transencoder/2evaluation/mnli/${ON}/${EVALUATION_PREFIX}.txt"
echo "OUTPUT_FILE: $OUTPUT_FILE"

# Ensure the second script has execute permissions
chmod +x /usa/taikun/07_transencoder/2evaluation/evaluation_nli.sh

# Call the second script
/usa/taikun/07_transencoder/2evaluation/evaluation_nli.sh
