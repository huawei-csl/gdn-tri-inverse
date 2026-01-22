import torch
import torch.nn.functional as F
import torch_npu  # noqa
import tcuscan
import argparse
import typing


from gdn_tri_inverse.core import chunk_gated_delta_rule_ref
from gdn_tri_inverse.linalg import inv_tril_inplace, tri_inv_vcs
from utils import Device, run_torch_profiler, run_benchmark

NPU_DEVICE = "npu:0"
device = Device(torch.npu, NPU_DEVICE)


INV_FNS_ = {
    "baseline": inv_tril_inplace,
    "column-sweep": tri_inv_vcs,
    "cube-column-sweep": tcuscan.run_tri_inv_cube_col_sweep,
    "cube-rec-unroll": tcuscan.run_triu_inv_rec_unroll,
}


@torch.inference_mode()
def profile_chunk_forward(
    B: int,
    T: int,
    H: int,
    D: int,
    inv_fn: typing.Callable,
    chunk_size: int = 64,
    scale: float = 1,
    gate_logit_normalizer: float = 1,
    mask_p: float = 0,
    dtype: torch.dtype = torch.float16,
    device=device,
    use_torch_profiler=False,
):
    torch.manual_seed(42)

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    g = g / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, g, h0 = map(lambda x: x.to(device.name), (q, k, v, beta, g, h0))

    q = F.normalize(q.clone(), p=2, dim=-1)
    k = F.normalize(k.clone(), p=2, dim=-1)

    def run_forward():
        _, _ = chunk_gated_delta_rule_ref(
            q=F.normalize(q.clone(), p=2, dim=-1),
            k=F.normalize(k.clone(), p=2, dim=-1),
            v=v.clone(),
            beta=beta.clone(),
            g=g.clone(),
            scale=scale,
            output_final_state=True,
            initial_state=h0.clone(),
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
    parser.add_argument("--B", type=int, default=16)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--torch-profiler", action="store_true", default=False)
    args = parser.parse_args()

    if not args.torch_profiler:
        filename = f"bench_results_gdn.csv"
        if not os.path.exists(filename):
            with open(filename, "w", encoding="UTF-8") as fd:
                fd.write("inverse_type,matrix_size,elapsed_time_ms\n")

    profile_chunk_forward(
        B=args.B,
        T=args.T,
        H=args.H,
        D=args.D,
        chunk_size=args.chunk_size,
        inv_fn=INV_FNS_[args.inverse_type],
        use_torch_profiler=args.torch_profiler,
    )
