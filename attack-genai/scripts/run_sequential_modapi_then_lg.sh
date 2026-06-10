#!/bin/bash
# Order: OpenAI Moderation API runs first, then finish LlamaGuard 5_150 runs

source /usa/taikun/miniconda3/etc/profile.d/conda.sh
conda activate hqaattack
cd /home/taikun/rl-attack/rl_atk/attack-genai

BASE="--save_to_path /usa/taikun/rl-attack/rl_atk/attack-genai/trained_attacker/attacker_05082025_162746_llama-guard_doc_0.3_6_6000_0.8100.pth --len_doc_max 512 --max_gen_tokens 1024 --server_url https://api.openai.com/v1 --use_moderation_api --atker_path bert-base-uncased --target_path gpt-3.5-turbo --data_name jbb_behaviors"

# === 1. trained 3_20 + OpenAI Moderation API ===
echo "[$(date)] Starting: trained 3_20 + OpenAI Moderation API..."
python -u evaluation_attacker_genai_llama_itself.py $BASE \
  --judge openai_moderation --atker_mode trained --num_doc_masks 3 --samples_per_tok 20 \
  --atk_json_log /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0324_bert-base-uncased_gpt-3.5-turbo_doc_trained_jbb_behaviors_3_20_openaimod.json \
  2>&1 | tee /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0324_bert-base-uncased_gpt-3.5-turbo_doc_trained_jbb_behaviors_3_20_openaimod.txt
echo "[$(date)] Done: trained 3_20 + OpenAI Moderation API"

# === 2. untrained 3_20 + OpenAI Moderation API ===
echo "[$(date)] Starting: untrained 3_20 + OpenAI Moderation API..."
python -u evaluation_attacker_genai_llama_itself.py $BASE \
  --judge openai_moderation --atker_mode untrained --num_doc_masks 3 --samples_per_tok 20 \
  --atk_json_log /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0324_bert-base-uncased_gpt-3.5-turbo_doc_untrained_jbb_behaviors_3_20_openaimod.json \
  2>&1 | tee /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0324_bert-base-uncased_gpt-3.5-turbo_doc_untrained_jbb_behaviors_3_20_openaimod.txt
echo "[$(date)] Done: untrained 3_20 + OpenAI Moderation API"

echo "[$(date)] ALL RUNS COMPLETE."
