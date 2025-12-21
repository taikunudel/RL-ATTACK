#!/bin/bash
TODAY=$(date +"%Y%m%d")
PREFIX=" /usa/taikun/07_transencoder/1training/llama-guard-attacker/attacker_training_genai_${TODAY}"
TXT_FILE="${PREFIX}.txt"
python /usa/taikun/07_transencoder/1training/attacker_training_genai.py > $TXT_FILE
