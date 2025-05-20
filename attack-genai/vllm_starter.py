CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-Guard-3-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --served-model-name llama-guard-3