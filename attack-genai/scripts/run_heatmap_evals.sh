#!/bin/bash
# Run evaluations for masks=5 and masks=10 with 20 candidates each
# GPT-3.5-turbo + OpenAI Moderation judge
# These complement the existing masks=3 run to build a heatmap

cd /home/taikun/rl-attack/rl_atk/attack-genai
export PATH="/usa/taikun/miniconda3/envs/03_transf_py311/bin:$PATH"
PYTHON=/usa/taikun/miniconda3/envs/03_transf_py311/bin/python

ATTACKER=/usa/taikun/rl-attack/rl_atk/attack-genai/trained_attacker/attacker_05082025_162746_llama-guard_doc_0.3_6_6000_0.8100.pth
RESULTS_DIR=/usa/taikun/rl-attack/rl_atk/attack-genai/eva_results

# masks=5, candidates=20
echo "=== Running masks=5, candidates=20 ==="
$PYTHON -u evaluation_attacker_genai_llama_itself.py \
  --save_to_path "$ATTACKER" \
  --len_doc_max 512 --max_gen_tokens 1024 \
  --server_url https://api.openai.com/v1 \
  --use_moderation_api --judge openai_moderation \
  --atker_path bert-base-uncased --target_path gpt-3.5-turbo --data_name jbb_behaviors \
  --atker_mode trained --num_doc_masks 5 --samples_per_tok 20 \
  --atk_json_log "${RESULTS_DIR}/eva_0331_bert-base-uncased_gpt-3.5-turbo_doc_trained_jbb_behaviors_5_20_openaimod.json" \
  2>&1 | tee "${RESULTS_DIR}/eva_0331_bert-base-uncased_gpt-3.5-turbo_doc_trained_jbb_behaviors_5_20_openaimod.txt"

echo ""
echo "=== Running masks=10, candidates=20 ==="
$PYTHON -u evaluation_attacker_genai_llama_itself.py \
  --save_to_path "$ATTACKER" \
  --len_doc_max 512 --max_gen_tokens 1024 \
  --server_url https://api.openai.com/v1 \
  --use_moderation_api --judge openai_moderation \
  --atker_path bert-base-uncased --target_path gpt-3.5-turbo --data_name jbb_behaviors \
  --atker_mode trained --num_doc_masks 10 --samples_per_tok 20 \
  --atk_json_log "${RESULTS_DIR}/eva_0331_bert-base-uncased_gpt-3.5-turbo_doc_trained_jbb_behaviors_10_20_openaimod.json" \
  2>&1 | tee "${RESULTS_DIR}/eva_0331_bert-base-uncased_gpt-3.5-turbo_doc_trained_jbb_behaviors_10_20_openaimod.txt"

echo ""
echo "=== All done ==="
