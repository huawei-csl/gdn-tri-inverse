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

# On x86_64 architecture, we need to install the triton-ascend package from Test PyPI to ensure compatibility with the SGLANG server.
if [[ "$(uname -m)" == "x86_64" ]]; then
    pip3 uninstall triton -y
    pip3 install --force-reinstall -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre --no-cache-dir

    # Fixes the "ImportError: libGL.so.1: cannot open shared object file: No such file or directory" error when importing OpenCV in the SGLANG server.
    pip install --force-reinstall opencv-python-headless
fi



echo "[GDN-TRI-INVERSE] Starting SGLANG server."
echo "[GDN-TRI-INVERSE] Press Ctrl+z and 'bg' to send process in background"
echo "[GDN-TRI-INVERSE] Default model is Qwen/Qwen3.5-0.8B-Base, you can change it by modifying the --model-path parameter in this script."
sglang serve \
    --model-path Qwen/Qwen3.5-0.8B-Base \
    --attention-backend ascend \
    --disable-cuda-graph \
    --disable-radix-cache \
    --tp-size 1 \
    --mem-fraction-static 0.8 \
    --max-total-tokens 2048
