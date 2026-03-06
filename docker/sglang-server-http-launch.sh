#!/bin/bash
#
#   sglang-server-http-launch.sh
#
# Description:
#   Runs an SGLANG HTTP server that listens for prompts over HTTP.
#
#   This script must be run inside the docker container built by build_docker.sh, and it will start the sglang server.
#   Adjust the Tensor Parallelism (tp-size) size and other parameters as needed.
#
# Usage:
#   ./sglang-server-http-launch.sh

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# FIXME(anastasios): require for x86 host, remove this when the server can run on arm host.
pip3 remove triton -y
pip3 install --force-reinstall -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre --no-cache-dir

python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-0.8B-Base \
    --attention-backend ascend \
    --disable-cuda-graph \
    --disable-radix-cache \
    --tp-size 1 \ # Adjust the Tensor Parallelism (tp-size) size as needed for 910B2/910B4.
    --mem-fraction-static 0.8 \
    --max-total-tokens 2048 > sglang_server.log 2>&1
