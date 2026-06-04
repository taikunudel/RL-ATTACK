#!/bin/bash
# Scheduled: wait 5 hours then run atk0324 eval

echo "[$(date)] Sleeping 5 hours before starting..."
sleep 5h

source /usa/taikun/miniconda3/etc/profile.d/conda.sh
conda activate hqaattack
cd /home/taikun/rl-attack/rl_atk/attack-genai

echo "[$(date)] Starting: trained 3_20 + OpenAI Moderation (atk0324) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python -u evaluation_attacker_genai_llama_itself.py \
  --atker_path bert-base-uncased --atker_mode trained --target_path gpt-3.5-turbo --data_name jbb_behaviors \
  --save_to_path /usa/taikun/rl-attack/rl_atk/attack-genai/trained_attacker/attacker_03242026_174027_llama-guard_doc_0.5_1_1700_0.5500.pth \
  --len_doc_max 512 --num_doc_masks 3 --samples_per_tok 20 --max_gen_tokens 1024 \
  --server_url https://api.openai.com/v1 --use_moderation_api --judge openai_moderation \
  --atk_json_log /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0324_bert-base-uncased_gpt-3.5-turbo_doc_trained_jbb_behaviors_3_20_openaimod_atk0324.json \
  2>&1 | tee /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0324_bert-base-uncased_gpt-3.5-turbo_doc_trained_jbb_behaviors_3_20_openaimod_atk0324.txt
echo "[$(date)] Done."
