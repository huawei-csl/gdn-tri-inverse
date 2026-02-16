"""
Docstring for benchmark.bench_solve_tril

Profiling script that compares various solve_tril methods. Currently, profiles:
1. `torch_eager`: This is the vanilla PyTorch eager mode forward substitution method.
2. `triton`: Triton-ascend method
3. `ascendc_tri_inv_col_sweep`: Vector-only column sweep method written in AscendC.
4. `pto_tri_inv_col_sweep`: Vector-only column sweep method written in pto-isa.
5. `pto_tri_inv_rec_unroll`: Cube optimized inverse
"""

import argparse
import logging
import sys

import torch
import torch.nn.functional as F
from sgl_kernel_npu.fla.solve_tril import solve_tril_npu as solve_tril
from sgl_kernel_npu.fla.chunk import inv_tril_inplace
from tcuscan import run_tri_inv_cube_col_sweep

from pto_kernels import pto_tri_inv, pto_tri_inv_trick, pto_tri_inv_rec_unroll

from utils import Device, run_benchmark
from functools import partial

file_handler = logging.FileHandler(filename="benchmark_tri_inv.log")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=handlers,
)

logger = logging.getLogger(__name__)


NPU_DEVICE = "npu:1"
device = Device(torch.npu, NPU_DEVICE)


def tri_inv_rec_unroll_wrapper(A, chunk_size: int):
    # HACK to overcome the isuse: "RuntimeError: Input tensor must have at exactly 3 dimensions."
    return pto_tri_inv_rec_unroll(A.reshape(-1, chunk_size, chunk_size)).reshape(
        A.shape
    )


# Map "triangular inverse method name" to Python function with signature fn(A) -> A^{-1}.
TRIANGULAR_INVERSE_METHODS_ = {
    "torch_eager": inv_tril_inplace,
    "triton": solve_tril,
    "ascendc_tri_inv_col_sweep": torch.ops.npu.triangular_inverse,
    "pto_tri_inv_col_sweep": pto_tri_inv,
    "tcuscan_tri_inv_cube_col_sweep": run_tri_inv_cube_col_sweep,
    "pto_tri_inv_rec_unroll": tri_inv_rec_unroll_wrapper,
    # "pto_tri_inv_trick": pto_tri_inv_trick,
}


@torch.inference_mode()
def profile_solve_tril(
    B: int,
    T: int,
    H: int,
    D: int,
    chunk_size: int = 64,
    dtype: torch.dtype = torch.float16,
    inverse_type: str = "baseline",
):
    torch.manual_seed(42)

    if inverse_type == "pto_tri_inv_rec_unroll":
        inv_fn = partial(tri_inv_rec_unroll_wrapper, chunk_size=chunk_size)
    else:
        inv_fn = TRIANGULAR_INVERSE_METHODS_[inverse_type]

    # do not randomly initialize A otherwise the inverse is not stable
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
        line = f"{inverse_type},fp16,{B},{T},{H},{D},{numel},{chunk_size},{avg_time_us}"
        fd.write(f"{line}\n")


if __name__ == "__main__":  # noqa
    parser = argparse.ArgumentParser(description="Triangular inverse benchmarking")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--D", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()

    chunk_size = args.chunk_size
    dtype = "fp16"  # TODO: support fp32

    filename = f"bench_results_solve_tril_{chunk_size}.csv"
    with open(filename, "w", encoding="UTF-8") as fd:
        fd.write("inverse_type,dtype,B,T,H,D,numel,chunk_size,time_us\n")

    for inverse_type in TRIANGULAR_INVERSE_METHODS_.keys():
        for B in [2, 5, 10, 15, 20, 25, 30, 40, 50]:
            T, H, D = args.T, args.H, args.D
            logger.info(
                f"Profiling case: {inverse_type},{dtype},{B},{T},{H},{D},{chunk_size}"
            )

            # Triton does not support chunk_size = 128
            if inverse_type == "triton" and chunk_size == 128:
                continue

            profile_solve_tril(
                B=B,
                T=T,
                H=H,
                D=D,
                chunk_size=chunk_size,
                inverse_type=inverse_type,
            )
