#!/bin/bash

# For SNLI Set
export DATA_NAME='snli'
export DATA_PATH='/usa/taikun/rl-attack/3datasets/snli/snli_1.0_train.jsonl'
export IMPORTANT_TOKENS_FILE='/usa/taikun/rl-attack/0dataProcessing/snli/tokens_snli_10000.json_premise.json'
# export IMPORTANT_TOKENS_FILE='/usa/taikun/rl-attack/0dataProcessing/snli/tokens_snli_10000.json_hypothesis.json'

export ON='premise'
# export ON='hypothesis'

export TGT_MODEL='textattack/bert-base-uncased-snli'
export ATTACKER_NAME='BERTFineTuned'
# export ATTACKER_NAME='BERT_distill'
# export ATTACKER_NAME='BERT_large'

export ATTACKER_PATH="/usa/taikun/rl-attack/1training/${DATA_NAME}/${ON}/"

export NUMS_ATTACKED_TOKENS=5
export MAX_LEN_DOC=512

export ALPHA=0.0

# Ensure the second script has execute permissions
chmod +x /usa/taikun/rl-attack/1training/train_nli.sh

# Call the second script
/usa/taikun/rl-attack/1training/train_nli.sh
