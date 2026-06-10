#!/bin/bash
# Sequential eval resume — waits for current run, then runs remaining 3 one by one

source /usa/taikun/miniconda3/etc/profile.d/conda.sh
conda activate hqaattack
cd /home/taikun/rl-attack/rl_atk/attack-genai

COMMON="--save_to_path /usa/taikun/rl-attack/rl_atk/attack-genai/trained_attacker/attacker_05082025_162746_llama-guard_doc_0.3_6_6000_0.8100.pth --len_doc_max 512 --max_gen_tokens 1024 --server_url https://api.openai.com/v1 --use_moderation_api --atker_path bert-base-uncased --target_path gpt-3.5-turbo --data_name jbb_behaviors"

# Step 0: Wait for currently running run 3 (trained 3_20, PID 449805)
echo "[$(date)] Waiting for run 3 (trained 3_20, PID 449805) to finish..."
while kill -0 449805 2>/dev/null; do sleep 30; done
echo "[$(date)] Run 3 finished."

# Step 1: untrained 3_20, resume from 3
echo "[$(date)] Starting run 4: untrained 3_20 (start_idx=3)..."
python -u evaluation_attacker_genai_llama_itself.py $COMMON \
  --atker_mode untrained --num_doc_masks 3 --samples_per_tok 20 --start_idx 3 \
  --atk_json_log /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0323_191406_bert-base-uncased_gpt-3.5-turbo_doc_untrained_jbb_behaviors_3_20_lg4judge.json \
  2>&1 | tee -a /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0323_191406_bert-base-uncased_gpt-3.5-turbo_doc_untrained_jbb_behaviors_3_20_lg4judge.txt
echo "[$(date)] Run 4 finished."

# Step 2: trained 5_150, resume from 58
echo "[$(date)] Starting run 1: trained 5_150 (start_idx=58)..."
python -u evaluation_attacker_genai_llama_itself.py $COMMON \
  --atker_mode trained --num_doc_masks 5 --samples_per_tok 150 --start_idx 58 \
  --atk_json_log /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0323_161204_bert-base-uncased_gpt-3.5-turbo_doc_trained_jbb_behaviors_5_150_lg4judge.json \
  2>&1 | tee -a /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0323_161204_bert-base-uncased_gpt-3.5-turbo_doc_trained_jbb_behaviors_5_150_lg4judge.txt
echo "[$(date)] Run 1 finished."

# Step 3: untrained 5_150, resume from 59
echo "[$(date)] Starting run 2: untrained 5_150 (start_idx=59)..."
python -u evaluation_attacker_genai_llama_itself.py $COMMON \
  --atker_mode untrained --num_doc_masks 5 --samples_per_tok 150 --start_idx 59 \
  --atk_json_log /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0323_161204_bert-base-uncased_gpt-3.5-turbo_doc_untrained_jbb_behaviors_5_150_lg4judge.json \
  2>&1 | tee -a /usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0323_161204_bert-base-uncased_gpt-3.5-turbo_doc_untrained_jbb_behaviors_5_150_lg4judge.txt
echo "[$(date)] Run 2 finished."

echo "[$(date)] ALL DONE."
