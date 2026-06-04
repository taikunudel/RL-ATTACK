#!/bin/bash
# Resume remaining 2 runs with taikunchen1994 key

source /usa/taikun/miniconda3/etc/profile.d/conda.sh
conda activate hqaattack
cd /home/taikun/rl-attack/rl_atk/attack-genai

BASE="--len_doc_max 512 --num_doc_masks 3 --samples_per_tok 20 --max_gen_tokens 1024 --server_url https://api.openai.com/v1 --use_moderation_api --judge openai_moderation --atker_path bert-base-uncased --target_path gpt-3.5-turbo --data_name jbb_behaviors"

# === 1. untrained 3_20 OM (atk0508), resume from 74 ===
echo "[$(date)] Starting: untrained 3_20 + OpenAI Moderation (atk0508, resume from 74)..."
python -u evaluation_attacker_genai_llama_itself.py $BASE \
  --atker_mode untrained --start_idx 74 \
  --save_to_path /usa/taikun/rl-attack/rl_atk/attack-genai/trained_attacker/attacker_05082025_162746_llama-guard_doc_0.3_6_6000_0.8100.pth \
  --atk_json_log /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0324_bert-base-uncased_gpt-3.5-turbo_doc_untrained_jbb_behaviors_3_20_openaimod.json \
  2>&1 | tee -a /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0324_bert-base-uncased_gpt-3.5-turbo_doc_untrained_jbb_behaviors_3_20_openaimod.txt
echo "[$(date)] Done: untrained 3_20 OM"

# === 2. trained 3_20 OM (atk0324, new checkpoint), fresh start ===
echo "[$(date)] Starting: trained 3_20 + OpenAI Moderation (atk0324, new checkpoint)..."
python -u evaluation_attacker_genai_llama_itself.py $BASE \
  --atker_mode trained \
  --save_to_path /usa/taikun/rl-attack/rl_atk/attack-genai/trained_attacker/attacker_03242026_174027_llama-guard_doc_0.5_3_3600_0.5300.pth \
  --atk_json_log /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0324_bert-base-uncased_gpt-3.5-turbo_doc_trained_jbb_behaviors_3_20_openaimod_atk0324.json \
  2>&1 | tee /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0324_bert-base-uncased_gpt-3.5-turbo_doc_trained_jbb_behaviors_3_20_openaimod_atk0324.txt
echo "[$(date)] Done: trained 3_20 OM (atk0324)"

echo "[$(date)] ALL DONE."
