#!/bin/bash
#
# Description:
#   Runs an SGLANG HTTP server that listens for prompts over HTTP.
#
#   This script must be run inside the docker container built by build_docker.sh, and it will start the sglang server.
#   Adjust the Tensor Parallelism (tp-size) size and other parameters as needed.
#

SGL_KERNEL_NPU_PATH="/workspace/sgl-kernel-npu/"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-0.8B-Base}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-ascend}"
# export ASCEND_RT_VISIBLE_DEVICES=1,2 # Comma-separated device ids

echo "[GDN-TRI-INVERSE] Starting SGLANG server."
echo "[GDN-TRI-INVERSE] Press Ctrl+z and 'bg' to send process in background"
echo "[GDN-TRI-INVERSE] Default model is Qwen/Qwen3.5-0.8B-Base, you can change it by modifying the --model-path parameter in this script or the MODEL_NAME environment variable."
sglang serve \
    --model-path ${MODEL_NAME} \
    --attention-backend ${ATTENTION_BACKEND} \
    --disable-cuda-graph \
    --disable-radix-cache \
    --tp-size 1 \
    --mem-fraction-static 0.5 \
    --max-total-tokens 4096

