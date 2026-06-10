#!/bin/bash
# Queue OpenAI Moderation API judge runs (trained & untrained 3_20)
# Waits for the sequential runner (PID 452268) to finish first

source /usa/taikun/miniconda3/etc/profile.d/conda.sh
conda activate hqaattack
cd /home/taikun/rl-attack/rl_atk/attack-genai

COMMON="--save_to_path /usa/taikun/rl-attack/rl_atk/attack-genai/trained_attacker/attacker_05082025_162746_llama-guard_doc_0.3_6_6000_0.8100.pth --len_doc_max 512 --max_gen_tokens 1024 --server_url https://api.openai.com/v1 --use_moderation_api --judge openai_moderation --atker_path bert-base-uncased --target_path gpt-3.5-turbo --data_name jbb_behaviors"

# Wait for sequential runner to finish
echo "[$(date)] Waiting for sequential runner (PID 452268) to finish..."
while kill -0 452268 2>/dev/null; do sleep 30; done
echo "[$(date)] Sequential runner done. Starting OpenAI moderation runs..."

# Run 5: trained 3_20 with OpenAI Moderation judge
echo "[$(date)] Starting: trained 3_20 + openai_moderation judge..."
python -u evaluation_attacker_genai_llama_itself.py $COMMON \
  --atker_mode trained --num_doc_masks 3 --samples_per_tok 20 \
  --atk_json_log /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0324_bert-base-uncased_gpt-3.5-turbo_doc_trained_jbb_behaviors_3_20_openaimod.json \
  2>&1 | tee /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0324_bert-base-uncased_gpt-3.5-turbo_doc_trained_jbb_behaviors_3_20_openaimod.txt
echo "[$(date)] Run 5 finished."

# Run 6: untrained 3_20 with OpenAI Moderation judge
echo "[$(date)] Starting: untrained 3_20 + openai_moderation judge..."
python -u evaluation_attacker_genai_llama_itself.py $COMMON \
  --atker_mode untrained --num_doc_masks 3 --samples_per_tok 20 \
  --atk_json_log /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0324_bert-base-uncased_gpt-3.5-turbo_doc_untrained_jbb_behaviors_3_20_openaimod.json \
  2>&1 | tee /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0324_bert-base-uncased_gpt-3.5-turbo_doc_untrained_jbb_behaviors_3_20_openaimod.txt
echo "[$(date)] Run 6 finished."

echo "[$(date)] ALL OPENAI MODERATION RUNS DONE."
