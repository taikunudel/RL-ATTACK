#!/bin/bash

# For MNLI matched Set
export DATA_NAME='mnlimatched'
export DATA_PATH='/usa/taikun/rl-attack/MNLI/multinli_1.0_train.jsonl'
export IMPORTANT_TOKENS_FILE='/usa/taikun/rl-attack/0dataProcessing/mnli/tokens_mnlimatched_0.json_premise.json'
# export IMPORTANT_TOKENS_FILE='/usa/taikun/rl-attack/0dataProcessing/tokens_mnlimatched_0.json_hypothesis.json'

export ON='premise'
# export ON='hypothesis'

export TGT_MODEL='textattack/bert-base-uncased-MNLI'
export ATTACKER_NAME='BERTFineTuned'
export ATTACKER_PATH="/usa/taikun/rl-attack/1training/mnli/${ON}/"

export NUMS_ATTACKED_TOKENS=5
export MAX_LEN_DOC=512

# Ensure the second script has execute permissions
chmod +x /usa/taikun/rl-attack/1training/train_nli.sh

for ALPHA in 1.0 0.9 0.7; do
  echo "Starting train_nli.sh with ALPHA=$ALPHA"
  (
    export ALPHA
    /usa/taikun/rl-attack/1training/train_nli.sh
  ) &
  
done
