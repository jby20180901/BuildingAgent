CUDA_VISIBLE_DEVICES=6,7

python -m vllm.entrypoints.openai.api_server \
    --model "/home/jiangbaoyang/HuggingFace-Download-Accelerator/hf_hub/Qwen3-Next-80B-A3B-Thinking-FP8" \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --trust-remote-code \
	--gpu-memory-utilization 0.85 \
    --host 127.0.0.1 \
    --port 8011 \
    --max-model-len 262144 \
    # --reasoning-parser deepseek_r1 \
    # --no-enable-chunked-prefill \
    # --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":4}'