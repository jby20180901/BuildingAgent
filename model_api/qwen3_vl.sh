CUDA_VISIBLE_DEVICES=4,5

python -m vllm.entrypoints.openai.api_server \
    --model "/home/jiangbaoyang/HuggingFace-Download-Accelerator/hf_hub/Qwen3-VL-32B-Thinking" \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --trust-remote-code \
	--gpu-memory-utilization 0.85 \
    --host 127.0.0.1 \
    --port 8012 \
    --max-model-len 131072 \
    --limit-mm-per-prompt '{"image": 10, "video": 1}' \
    # --allowed-local-media-path \
    # --reasoning-parser deepseek_r1 \
    # --no-enable-chunked-prefill \
    # --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":4}'