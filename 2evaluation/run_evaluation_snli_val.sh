#!/bin/bash

# For SNLI Validation Set
export DATA_NAME='snli_val'
export DATA_PATH='/usa/taikun/07_transencoder/3datasets/snli/snli/snli.txt'
export IMPORTANT_TOKENS_FILE='/usa/taikun/07_transencoder/0dataProcessing/snli/tokens_snli_val_0.json_premise.json'
export IMPORTANT_TOKENS_FILE='/usa/taikun/07_transencoder/0dataProcessing/snli/tokens_snli_val_0.json_hypothesis.json'

# 4 Apr 2025, 2k samples
# export DATA_NAME='snli_test'
# export DATA_PATH='/usa/taikun/07_transencoder/3datasets/snli/snli_test.txt'
# export IMPORTANT_TOKENS_FILE='/usa/taikun/07_transencoder/0dataProcessing/tokens_snli_test_full.json_premise.json'
# export IMPORTANT_TOKENS_FILE='/usa/taikun/07_transencoder/0dataProcessing/tokens_snli_test_full.json_hypothesis.json'


# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/premise/attacker_snli_0.5_0_400_2.9763.pth'
# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/premise/1125/attacker_snli_0.5_0_240_2.9472.pth'

# testing models with varable alpha, per request of UAI reviewers
# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/premise/attacker_snli_0.0_0_400_2.9743.pth'
# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/premise/alpha_exp/attacker_snli_0.1_0_400_2.9750.pth'
# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/premise/alpha_exp/attacker_snli_0.3_0_400_2.9570.pth'
# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/premise/alpha_exp/attacker_snli_0.6_7_300_3.1881.pth'
# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/premise/attacker_snli_0.7_0_400_2.8756.pth'

# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/hypothesis/attacker_snli_0.5_1_500_2.9032.pth'
# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/hypothesis/1125/attacker_snli_0.5_0_400_2.9243.pth'
# export ATTACKER_FILE='None'

# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/premise/0207non_linear/attacker_snli_0.5_2_1000_2.9262.pth'

# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/hypothesis/attacker_snli_0.5_2_1000_2.8892.pth'
# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/hypothesis/0209bertnonlinear/attacker_snli_0.5_2_900_2.8909.pth'

# evaluate on diff alpha, per UAi reviewers' request.
# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/premise/alpha_exp/attacker_snli_0.0_0_400_2.9743.pth'
# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/premise/alpha_exp/attacker_snli_0.1_0_400_2.9750.pth'
# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/premise/alpha_exp/attacker_snli_0.3_0_400_2.9570.pth'
# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/premise/alpha_exp/attacker_snli_0.4_0_400_2.9462.pth'
# export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/premise/alpha_exp/attacker_snli_0.5_0_400_2.9763.pth'
export ATTACKER_FILE='/usa/taikun/07_transencoder/1training/snli/premise/alpha_exp/attacker_snli_0.6_7_300_3.1881.pth'

export MODE='nli'
export ON='premise'
# export ON='hypothesis'

export TGT_MODEL_NAME="BERT_SNLI"
export TGT_MODEL="textattack/bert-base-uncased-snli"

# export TGT_MODEL_NAME="ROBERTA_SNLI"
# export TGT_MODEL="pepa/roberta-base-snli"

# export TGT_MODEL_NAME='SBERT_NLI'
# export TGT_MODEL='cross-encoder/nli-roberta-base'

export ATTACKER_NAME='BERTFineTuned'
# export ATTACKER_NAME='BERT_distill'
# export ATTACKER_NAME='BERTMaskedLMRandom'
# export ATTACKER_NAME='BERTMaskedLM'
# export ATTACKER_NAME='BERT_nonlinear'

export NUMS_ATTACKED_TOKENS=0.7
export MAX_LEN_DOC=512

# export NUMS_CANDIDATES_EACH_TOKEN=20
# export NUMS_CANDIDATES_EACH_TOKEN=40
export NUMS_CANDIDATES_EACH_TOKEN=60 # total numbers of desired attacked tokens
# export NUMS_CANDIDATES_EACH_TOKEN=80 # total numbers of desired attacked tokens
# export NUMS_CANDIDATES_EACH_TOKEN=100
# export NUMS_CANDIDATES_EACH_TOKEN=120 # total numbers of desired attacked tokens
# export NUMS_CANDIDATES_EACH_TOKEN=160 # total numbers of desired attacked tokens
# export NUMS_CANDIDATES_EACH_TOKEN=40 # total numbers of desired attacked tokens

# export NUMS_MAX_CANDIDATES=4
# export NUMS_MAX_CANDIDATES=6
export NUMS_MAX_CANDIDATES=8
# export NUMS_MAX_CANDIDATES=10
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
export EVALUATION_PREFIX="eva_$(date +%m%d_%H%M%S)_${ON}_${ATTACKER_NAME}_${TGT_MODEL_NAME}_${DATA_NAME}_${NUMS_ATTACKED_TOKENS}_${NUMS_MAX_CANDIDATES}_${NUMS_CANDIDATES_EACH_TOKEN}_${STARTFROMSAMPLE}"

export EVALUATION_JSON="/usa/taikun/07_transencoder/2evaluation/snli/${ON}/${EVALUATION_PREFIX}.json"
export OUTPUT_FILE="/usa/taikun/07_transencoder/2evaluation/snli/${ON}/${EVALUATION_PREFIX}.txt"
echo "OUTPUT_FILE: $OUTPUT_FILE"

# Ensure the second script has execute permissions
chmod +x /usa/taikun/07_transencoder/2evaluation/evaluation_nli.sh

# Call the second script
/usa/taikun/07_transencoder/2evaluation/evaluation_nli.sh

# 0207nonlinear
