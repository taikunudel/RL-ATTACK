#!/bin/bash
#SBATCH --job-name=qwen_rlatk_demo
#SBATCH --account=weiqian_stat
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=r06g04
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/work/weiqian_stat/taikun/rl-attack/notebooks/qwen_demo_%j.out

export QWENDEMO_PREINSTALLED=1
export HF_HOME=/work/weiqian_stat/taikun/.envs/.hfcache
export TOKENIZERS_PARALLELISM=false
export GUARD_DIR=/work/weiqian_stat/taikun/models/llama-guard-3-1b   # token-free local Llama-Guard-3-1B

cd /work/weiqian_stat/taikun/rl-attack/notebooks
echo "NODE=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=================== run_demo.py ==================="
/work/weiqian_stat/taikun/.envs/qwendemo/bin/python run_demo.py
echo "EXIT_CODE=$?"
echo "QWEN_DEMO_JOB_DONE"
