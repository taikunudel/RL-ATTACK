#!/bin/bash
# Reorganize rl_atk repo structure
# Run from: /usa/taikun/rl-attack/rl_atk/
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
echo "Working in: $ROOT"

# =============================================
# 1. Create new directories
# =============================================
echo "=== Creating directories ==="
mkdir -p attack_genai
mkdir -p attack_nli
mkdir -p scripts
mkdir -p utils
mkdir -p results/evaluations
mkdir -p results/training_logs
mkdir -p results/data
mkdir -p models
mkdir -p tests
mkdir -p archive

# =============================================
# 2. Move core source files → attack_genai/
# =============================================
echo "=== Moving core GenAI attack source files ==="
cp attack-genai/train_attacker_genai.py attack_genai/train.py
cp attack-genai/evaluation_attacker_genai.py attack_genai/evaluate.py
cp attack-genai/evaluation_attacker_genai_llama_itself.py attack_genai/evaluate_llama.py
cp attack-genai/evaluation_attacker_genai_llama_itself_verbose.py attack_genai/evaluate_llama_verbose.py
cp attack-genai/get_raw_logits.py attack_genai/get_raw_logits.py
cp attack-genai/llms_adversarial_documents_evaluation.py attack_genai/llms_eval.py
cp attack-genai/llms_adversarial_documents_evaluation_few_shot.py attack_genai/llms_eval_few_shot.py
cp attack-genai/vllm_starter.py attack_genai/vllm_starter.py
cp attack-genai/.env attack_genai/.env

# =============================================
# 3. Move NLI attack files → attack_nli/
# =============================================
echo "=== Moving NLI attack source files ==="
cp attack-nli/train_attacker_nli.py attack_nli/train.py
# Skip the badly named file: "def load_and_prepare_data(atk_path, tgt_.py"

# =============================================
# 4. Move shell scripts → scripts/
# =============================================
echo "=== Moving shell scripts ==="
cp attack-genai/run_train_attack_genai.sh scripts/run_train.sh
cp attack-genai/run_eva_attack_genai.sh scripts/run_eval.sh
cp attack-genai/run_batch_eva_attack_genai.sh scripts/run_batch_eval.sh
cp attack-genai/run_batch_eva_attack_genai_atkers.sh scripts/run_batch_eval_atkers.sh
cp attack-genai/run_batch_eva_attack_genai_atkmode.sh scripts/run_batch_eval_atkmode.sh
cp attack-genai/run_batch_eva_attack_llama_itself.sh scripts/run_batch_eval_llama.sh
cp attack-genai/run_batch_eva_attack_qwen3.sh scripts/run_batch_eval_qwen3.sh
cp attack-genai/run_batch_eva_attack_qwen3_itself.sh scripts/run_batch_eval_qwen3_itself.sh

# =============================================
# 5. Move utility/analysis scripts → utils/
# =============================================
echo "=== Moving utility scripts ==="
cp calc_metrics_0_200.py utils/calc_metrics_0_200.py
cp calc_metrics_multi.py utils/calc_metrics_multi.py
cp attack-genai/plot_flag_distribution.py utils/plot_flag_distribution.py
cp attack-genai/gemini_moderation.py utils/gemini_moderation.py
cp attack-genai/mimic_moderation_with_gemini.py utils/mimic_moderation_with_gemini.py
cp attack-genai/compare_models.py utils/compare_models.py
cp attack-genai/compare_openai_gemini.py utils/compare_openai_gemini.py
cp attack-genai/get_advbench_fewshot.py utils/get_advbench_fewshot.py
cp attack-genai/check_cache.py utils/check_cache.py

# =============================================
# 6. Move test/debug scripts → tests/
# =============================================
echo "=== Moving test/debug scripts ==="
cp attack-genai/test_moderation_batch.py tests/
cp attack-genai/test_moderation_raw.py tests/
cp attack-genai/test_openai_10cases.py tests/
cp attack-genai/test_single_sample.py tests/
cp attack-genai/test_advbench_11_20.py tests/
cp attack-genai/test_batch_optimization.py tests/
cp attack-genai/test_env_loading.py tests/
cp attack-genai/test_gemini_models_updated.py tests/
cp attack-genai/test_openai_moderation_format.py tests/
cp attack-genai/test_batch_moderation_func.py tests/
cp attack-genai/diagnostic_test.py tests/
cp attack-genai/llama-guard-api-test.py tests/llama_guard_api_test.py
cp attack-genai/llama-guard_api_test.py tests/llama_guard_api_test_v2.py

# =============================================
# 7. Move result/data files → results/
# =============================================
echo "=== Moving evaluation results ==="
# Move eva_results contents → results/evaluations/
cp -r attack-genai/eva_results/* results/evaluations/

echo "=== Moving training logs ==="
# Move train_results contents → results/training_logs/
cp attack-genai/train_results/* results/training_logs/
# Also move the loose training log
cp attack-genai/train_attacker_genai.txt results/training_logs/

echo "=== Moving data files ==="
# Move data JSONs → results/data/
cp attack-genai/advbench_10_fewshot.json results/data/
cp attack-genai/advbench_10_openai_moderation.json results/data/
cp attack-genai/advbench_11_20_gemini.json results/data/
cp attack-genai/advbench_11_20_openai.json results/data/
cp attack-genai/comparison_openai_vs_gemini.json results/data/
cp attack-genai/gemini_moderation_10shot.json results/data/
cp attack-genai/gemini_moderation_openai_format.json results/data/
cp attack-genai/gemini_moderation_output.json results/data/
cp attack-genai/moderation_input.json results/data/
cp attack-genai/moderation_output.json results/data/
cp attack-genai/openai_moderation_10cases.json results/data/
cp attack-genai/test_batch_optimization_output.json results/data/
cp attack-genai/test_moderation_raw_output.json results/data/

# Move text log files → results/data/
cp attack-genai/llms_adversarial_documents_evaluation_few_shot.txt results/data/
cp attack-genai/llms_adversarial_documents_evaluation_few_shot_regular_few_shots.txt results/data/
cp attack-genai/llms_adversarial_documents_evaluation_one_shot.txt results/data/

# =============================================
# 8. Move model checkpoints → models/
# =============================================
echo "=== Moving model checkpoints ==="
if [ -d "attack-genai/trained_attacker" ] && [ "$(ls -A attack-genai/trained_attacker)" ]; then
    # Use mv for large .pth files to avoid doubling disk usage
    mv attack-genai/trained_attacker/*.pth models/ 2>/dev/null || true
    echo "  Moved .pth files to models/"
fi

# =============================================
# 9. Move archive files
# =============================================
echo "=== Moving archive ==="
if [ -d "attack-genai/archive" ] && [ "$(ls -A attack-genai/archive)" ]; then
    cp -r attack-genai/archive/* archive/
fi

# =============================================
# 10. Move cache.db to root
# =============================================
echo "=== Moving cache.db ==="
if [ -f "attack-genai/cache.db" ]; then
    mv attack-genai/cache.db ./cache.db
fi

# =============================================
# 11. Delete junk files
# =============================================
echo "=== Cleaning up junk files ==="
rm -f bfg.jar
rm -f vllm_llama.log
rm -f attack-genai/temp_json_hf_checker.py

# =============================================
# 12. Remove old directories (AFTER verifying copies)
# =============================================
echo ""
echo "=== Verification ==="
echo "New structure:"
find attack_genai attack_nli scripts utils tests models -type f | head -40
echo ""
echo "Results:"
find results -type f | wc -l
echo " files in results/"
echo ""

echo "================================================================"
echo "IMPORTANT: Old directories still exist. After verifying the new"
echo "structure looks correct, run the following to remove the old dirs:"
echo ""
echo "  rm -rf attack-genai attack-nli"
echo "  rm -f calc_metrics_0_200.py calc_metrics_multi.py"
echo "  rm -f reorganize.sh"
echo "================================================================"
echo ""
echo "Done! Review the new structure, then remove old dirs."
