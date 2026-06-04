#!/bin/bash
TODAY=$(date +"%Y%m%d")
PREFIX=" /usa/taikun/rl-attack/1training/llama-guard-attacker/attacker_training_genai_${TODAY}"
TXT_FILE="${PREFIX}.txt"
python /usa/taikun/rl-attack/1training/attacker_training_genai.py > $TXT_FILE
