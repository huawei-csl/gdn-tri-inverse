"""
This script runs a benchmark for a specified model using the SGLANG server. It can also reinstall the sgl-kernel-npu if the repo is mounted in the docker's workspace and
the `--update` flag is provided.
"""

import json
import subprocess
import sys
from glob import glob

UPDATE = False
MODEL_NAME = None
SGL_KERNEL_NPU_PATH = "/workspace/sgl-kernel-npu/"

# Parse args
args = sys.argv[1:]
for arg in args:
    if arg == "--update":
        UPDATE = True
    elif not arg.startswith("--"):
        MODEL_NAME = arg

if MODEL_NAME is None:
    print("Usage: run_bench.py <model_name> [--update]")
    sys.exit(1)

# Load configs
with open("docker/model-configs.json") as f:
    configs = json.load(f)

default_cfg = {"tp": 4, "mem": 0.5}
cfg = next((c for c in configs if c["model"] == MODEL_NAME), default_cfg)

if UPDATE:
    print(f"Running update step for {MODEL_NAME}...")
    subprocess.run(
        ["bash", "build.sh", "-a", "kernels"], cwd=SGL_KERNEL_NPU_PATH, check=True
    )
    subprocess.run(
        [
            "pip",
            "install",
            "--force-reinstall",
            glob(f"{SGL_KERNEL_NPU_PATH}output/sgl_kernel_npu-*.whl")[0],
        ],
        check=True,
    )
    print("Update completed.")

# Run benchmark
print(f"Running benchmark for {MODEL_NAME}...")
cmd = [
    "python3",
    "-m",
    "sglang.bench_one_batch",
    "--model-path",
    cfg["model"],
    "--tp-size",
    str(cfg["tp"]),
    "--mem-fraction-static",
    str(cfg["mem"]),
    "--batch",
    "8",
    "--input-len",
    "1024",
    "--output-len",
    "10",
    "--disable-cuda-graph",
]
subprocess.run(cmd, check=True)
