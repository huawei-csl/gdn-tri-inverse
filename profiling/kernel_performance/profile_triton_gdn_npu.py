import torch
import torch.nn.functional as F
import torch_npu  # noqa
import argparse
import typing
import os


from gdn_tri_inverse.triton_chunk_gdn.fla.chunk import chunk_gated_delta_rule_npu
from sgl_kernel_npu.fla.solve_tril import solve_tril_npu as solve_tril_triton
from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_solve_tril import (
    solve_tril as solve_tril_tilelang,
)
from gdn_tri_inverse.linalg import (
    tri_inv_vcs_wrapper,
    tri_inv_mcs_wrapper,
    tri_inv_mxr_wrapper,
    tri_inv_bsnd_mxr_wrapper,
)
from utils import Device, run_torch_profiler, run_benchmark

NPU_DEVICE = os.getenv("GDN_TRI_INVERSE_NPU_DEVICE", "npu:0")
device = Device(torch.npu, NPU_DEVICE)


INV_BSDN_FNS_ = {
    "column-sweep": tri_inv_vcs_wrapper,
    "cube-column-sweep": tri_inv_mcs_wrapper,
    "cube-rec-unroll": tri_inv_mxr_wrapper,
    "bsnd-rec-unroll": tri_inv_bsnd_mxr_wrapper,
    "triton": solve_tril_triton,
    "tilelang-opt": solve_tril_tilelang,
}


@torch.inference_mode()
def profile_chunk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    cu_seqlens: torch.Tensor,
    inv_fn: typing.Callable,
    chunk_size: int = 64,
    scale: float = 1,
    device=device,
    use_torch_profiler=False,
):
    q, k, v, beta, g, h0 = map(lambda x: x.to(device.name), (q, k, v, beta, g, h0))

    q = F.normalize(q.clone(), p=2, dim=-1)
    k = F.normalize(k.clone(), p=2, dim=-1)

    def run_forward():
        _ = chunk_gated_delta_rule_npu(
            q=F.normalize(q.clone(), p=2, dim=-1),
            k=F.normalize(k.clone(), p=2, dim=-1),
            v=v.clone(),
            beta=beta.clone(),
            g=g.clone(),
            scale=scale,
            output_final_state=True,
            initial_state=h0.clone(),
            cu_seqlens=cu_seqlens.clone(),
            inv_fn=inv_fn,
            chunk_size=chunk_size,
        )

    if use_torch_profiler:
        run_torch_profiler("output", run_forward)
    else:
        for elapsed_time_ms in run_benchmark(
            device,
            run_forward,
            args.warmup,
            args.repeats,
        ):
            with open(filename, "a", encoding="UTF-8") as fd:
                line = f"{args.inverse_type},{args.chunk_size},{elapsed_time_ms:.2f}"
                fd.write(f"{line}\n")


if __name__ == "__main__":  # noqa
    parser = argparse.ArgumentParser(description="GDB benchmarking")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--inverse-type", type=str, default="baseline")
    parser.add_argument("--input", type=str, default="profiling/data/Qwen3-Next")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--torch-profiler", action="store_true", default=False)
    args = parser.parse_args()

    if not args.torch_profiler:
        filename = f"bench_results_triton_gdn.csv"
        if not os.path.exists(filename):
            with open(filename, "w", encoding="UTF-8") as fd:
                fd.write("inverse_type,matrix_size,elapsed_time_ms\n")

    profile_chunk_forward(
        q=torch.load(f"{args.input}/q.pt"),
        k=torch.load(f"{args.input}/k.pt"),
        v=torch.load(f"{args.input}/v.pt"),
        beta=torch.load(f"{args.input}/beta.pt"),
        g=torch.load(f"{args.input}/g.pt"),
        h0=torch.load(f"{args.input}/initial_state.pt"),
        cu_seqlens=torch.load(f"{args.input}/cu_seqlens.pt"),
        chunk_size=args.chunk_size,
        inv_fn=INV_BSDN_FNS_[args.inverse_type],
        use_torch_profiler=args.torch_profiler,
    )
