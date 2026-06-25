"""
Docstring for profiling.profile_tri_inv_npu

Profiling script that compares various triangular inverse methods for Ascend NPUs. Currently, it supports the following methods:
- `torch_eager`: This is the vanilla PyTorch eager mode forward substitution method.
- `triton`: Triton-ascend method
- `column_sweep`: Vector-only column sweep method written in PTO-ISA.
- `cube_column_sweep`: CUbe column sweep method written in AscendC.
- `cube_rec_unroll`: Cube-only optimized triangular inverse.
- `tilelang`: Mixed AIC/AIV triangular inverse from tile-lang
"""

import argparse
import logging
import sys
import os

import torch
import torch.nn.functional as F

from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_solve_tril import (
    solve_tril as solve_tril_tilelang,
)
from gdn_tri_inverse.linalg import (
    tri_inv_qwen3_next_default,
    tri_inv_vcs,
    # tri_inv_mcs,
    tri_inv_mxr,
    tri_inv_triton,
)

from utils import Device, run_benchmark

file_handler = logging.FileHandler(filename="benchmark_tri_inv.log")
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


# Map "triangular inverse method name" to Python function with signature fn(A) -> A^{-1}.
TRIANGULAR_INVERSE_METHODS_ = {
    "torch-eager": tri_inv_qwen3_next_default,
    "triton": tri_inv_triton,
    "column-sweep": tri_inv_vcs,
    # "cube-column-sweep": tri_inv_mcs,
    "cube-rec-unroll": tri_inv_mxr,
    "tilelang_opt": solve_tril_tilelang,
    # "pto_tri_inv_trick": pto_tri_inv_trick,
}


@torch.inference_mode()
def profile_solve_tril(
    B: int,
    T: int,
    H: int,
    chunk_size: int = 64,
    dtype: torch.dtype = torch.float16,
    inverse_type: str = "baseline",
):
    torch.manual_seed(42)

    inv_fn = TRIANGULAR_INVERSE_METHODS_[inverse_type]

    k = F.normalize(
        torch.randn((B, H, T, chunk_size), dtype=dtype, device=NPU_DEVICE), dim=-1
    )
    # Pad the second-to-last dimension (T) to be a multiple of chunk_size
    padding_size = (chunk_size - T % chunk_size) % chunk_size
    k_padded = F.pad(k, (0, 0, 0, padding_size, 0, 0, 0, 0))
    k_padded = k_padded.reshape(B, H, -1, chunk_size, chunk_size)
    A = (k_padded @ k_padded.transpose(-1, -2)).tril(-1)
    torch.npu.synchronize()

    assert (
        A.ndim >= 4
    ), f"Input tensor must be at least 4-dimensional. Got {A.ndim} dimensions."
    A = A.reshape(-1, A.shape[-3], A.shape[-2], A.shape[-1])
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

    filename = f"bench_results_solve_tril_{chunk_size}.csv"
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
