#! /usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
set -ue
python -m vllm.entrypoints.openai.api_server \
    --model "/share_data/data1/qy/model/9g/9g_8b_thinking" \
    --tokenizer-mode auto \
    --dtype auto \
    --trust-remote-code \
    --served-model-name 9g \
    --api-key fm9g \
    --gpu-memory-utilization 0.85 \
    --port 8059  \
    --tensor-parallel-size 1 \
    --host 127.0.0.1 