python -m vllm.entrypoints.openai.api_server \
    --model "/data/byj/hf_hub/openai/gpt_oss" \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --trust-remote-code \
	--gpu-memory-utilization 0.65 \
    --host 127.0.0.1 \
    --port 8011
