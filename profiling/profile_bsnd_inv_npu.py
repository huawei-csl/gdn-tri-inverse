"""
Docstring for benchmark.bench_solve_tril

Profiling script that compares various solve_tril methods. Currently, profiles:
1. `torch_eager`: This is the vanilla PyTorch eager mode forward substitution method.
2. `triton`: Triton-ascend method
3. `column_sweep`: Vector-only column sweep method written in pto-isa.
4. `cube_column_sweep`: CUbe column sweep method written in AscendC.
5. `cube_rec_unroll`: Cube optimized inverse
"""

import argparse
import logging
import sys
import os

import torch
import torch.nn.functional as F
from sgl_kernel_npu.fla.solve_tril import solve_tril_npu


from gdn_tri_inverse.linalg import (
    tri_inv_vcs_wrapper,
    tri_inv_mcs_wrapper,
    tri_inv_mxr_wrapper,
    tri_inv_bsnd_mxr_wrapper,
)

from utils import Device, run_benchmark

file_handler = logging.FileHandler(filename="benchmark_bsnd_inv.log")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=handlers,
)

logger = logging.getLogger(__name__)


NPU_DEVICE = os.getenv("GDN_TRI_INVERSE_NPU_DEVICE", "npu:0")
device = Device(torch.npu, NPU_DEVICE)


TRIANGULAR_INVERSE_METHODS_ = {
    "triton": solve_tril_npu,
    "column-sweep": tri_inv_vcs_wrapper,
    # "cube-column-sweep": tri_inv_mcs_wrapper,
    "cube-rec-unroll": tri_inv_mxr_wrapper,
    "bsnd-rec-unroll": tri_inv_bsnd_mxr_wrapper,
    # "pto_tri_inv_trick": pto_tri_inv_trick,
}


@torch.inference_mode()
def profile_solve_tril(
    B: int,
    T: int,
    H: int,
    chunk_size: int = 64,
    dtype: torch.dtype = torch.float16,
    inverse_type: str = "triton",
):
    torch.manual_seed(42)

    inv_fn = TRIANGULAR_INVERSE_METHODS_[inverse_type]

    # do not randomly initialize A otherwise the inverse is not stable
    A = F.normalize(
        torch.randn((B, T, H, chunk_size), dtype=dtype, device=NPU_DEVICE), dim=-1
    )
    torch.npu.synchronize()

    numel = A.numel()

    def run_solve_tril():
        _ = inv_fn(A)

    times_ms = list(
        run_benchmark(
            device,
            run_solve_tril,
            args.warmup,
            args.repeats,
        )
    )
    avg_time_ms = sum(times_ms) / len(times_ms)
    avg_time_us = int(avg_time_ms * 1000)
    with open(filename, "a", encoding="UTF-8") as fd:
        line = f"{inverse_type},fp16,{B},{T},{H},{numel},{chunk_size},{avg_time_us}"
        fd.write(f"{line}\n")


if __name__ == "__main__":  # noqa
    parser = argparse.ArgumentParser(description="Triangular inverse benchmarking")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--B", type=int, default=32)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()

    chunk_size = args.chunk_size
    dtype = "fp16"  # TODO: support fp32

    filename = f"bench_results_bsnd_tril_{chunk_size}.csv"
    with open(filename, "w", encoding="UTF-8") as fd:
        fd.write("inverse_type,dtype,B,T,H,numel,chunk_size,time_us\n")

    for inverse_type in TRIANGULAR_INVERSE_METHODS_.keys():
        for T in [512, 1024, 2048, 4096, 8192, 16384]:
            B, H = args.B, args.H
            logger.info(f"Profiling case: {inverse_type},{B},{T},{H},{chunk_size}")

            # Triton does not support chunk_size = 128
            if inverse_type == "triton" and chunk_size == 128:
                continue

            profile_solve_tril(
                B=B,
                T=T,
                H=H,
                chunk_size=chunk_size,
                inverse_type=inverse_type,
            )
