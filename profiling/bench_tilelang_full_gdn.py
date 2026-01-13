import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pto_kernels.benchmarking import do_bench

from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_chunk_cumsum import cumsum_ker
from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_chunk_h import chunk_h_ker
from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_chunk_o import chunk_o_ker
from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_chunk_scaled_dot_kkt import kkt_ker
from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_solve_tril import (
    solve_tril_ker,
    solve_tril_64_ker,
    solve_tril_128_ker,
)
from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_wy_fast import wy_fast_ker

NPU_DEVICE = os.getenv("GDN_TRI_INVERSE_NPU_DEVICE", "npu:0")
PLOT_PATH = "bench_results_tilelang_fullgdn_kernel_breakdown_stacked.png"


KERNEL_ORDER = [
    "chunk_cumsum",
    "chunk_scaled_dot_kkt",
    "solve_tril",
    "wy_fast",
    "chunk_h",
    "chunk_o",
]


def get_solve_tril_kernel(B, H, L, C):
    if C == 32:
        return solve_tril_ker(B, H, L, C)
    if C == 64:
        return solve_tril_64_ker(B, H, L)
    if C == 128:
        return solve_tril_128_ker(B, H, L)
    raise ValueError(f"unsupported C={C}")


def format_ops(ops: int) -> str:
    return f"{ops:.2e}"


def format_ms(ms: float) -> str:
    return f"{ms:.2f}"


def format_tflops(ops: int, ms: float) -> str:
    return f"{ops / (ms * 1e9):.4f}"


def run_stage(name: str, fn):
    print(f"[run] {name}")
    out = fn()
    torch.npu.synchronize()
    print(f"[ok] {name}")
    return out


def bench_stage(name: str, fn) -> float:
    print(f"[bench] {name}")
    # One synchronized preflight run makes async launch failures point at this stage.
    fn()
    torch.npu.synchronize()
    ms = do_bench(fn)
    print(f"[bench-ok] {name}: {ms:.2f} ms")
    return ms


def plot_stacked_latencies(latencies_by_c, title_suffix):
    chunk_sizes = sorted(latencies_by_c.keys())
    x = list(range(len(chunk_sizes)))
    bottom = [0.0] * len(chunk_sizes)
    totals = [
        sum(latencies_by_c[C][name] for name in KERNEL_ORDER) for C in chunk_sizes
    ]

    plt.figure(figsize=(8, 5))
    for name in KERNEL_ORDER:
        values = [latencies_by_c[C][name] for C in chunk_sizes]
        bars = plt.bar(x, values, bottom=bottom, label=name)
        for idx, bar in enumerate(bars):
            value = values[idx]
            if value < 0.5:
                continue
            pct = 100.0 * value / totals[idx]
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bottom[idx] + value / 2,
                f"{value:.1f} ms\n{pct:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
        bottom = [b + v for b, v in zip(bottom, values)]

    plt.xticks(x, [f"C={C}" for C in chunk_sizes])
    plt.ylabel("Latency (ms)")
    plt.title(f"Chunked GDN ({title_suffix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)


def main():
    torch.manual_seed(0)
    torch.npu.set_device(NPU_DEVICE)
    latencies_by_c = {}

    # Three valid configs for C in {32, 64, 128}.
    # They keep B * H * (L / C) constant so the problematic launch size does not grow
    # when C gets smaller, and they satisfy L % (8 * C) == 0 for opt_gdn_chunk_cumsum.
    B, H, L, DK, DV, BK, BV = 32, 4, 8192, 128, 128, 128, 128
    chunk_sizes = [32, 64, 128]

    for C in chunk_sizes:
        assert H % 2 == 0, "optimized kernels assume even H"
        assert L % C == 0, "optimized kernels assume full chunks"
        assert L % (8 * C) == 0, "opt_gdn_chunk_cumsum assumes L % (8 * C) == 0"

        q = torch.randn((B, H, L, DK)).npu().to(torch.float16)
        k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
        v = torch.randn((B, H, L, DV)).npu().to(torch.float16)
        q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)
        g = torch.randn((B, H, L)).npu().to(torch.float)
        g = F.logsigmoid(g)
        beta = torch.rand((B, H, L)).npu().to(torch.float16)

        ker1 = cumsum_ker(B, H, L, C)
        ker2 = kkt_ker(B, H, L, DK, C, BK)
        ker3 = get_solve_tril_kernel(B, H, L, C)
        ker4 = wy_fast_ker(B, H, L, DK, DV, C, BK, BV)
        ker5 = chunk_h_ker(B, H, L, DK, DV, C, BK, BV)
        ker6 = chunk_o_ker(B, H, L, DK, DV, C, BK, BV)

        idt = torch.eye(C).npu().to(torch.float)
        msk1 = torch.tril(torch.ones((C, C)), diagonal=-1).npu().to(torch.float)
        msk2 = torch.tril(torch.ones((C, C)), diagonal=0).npu().to(torch.float)
        workspace = (
            torch.zeros((B * H * ((DV + BV - 1) // BV), DK, BV)).npu().to(torch.float16)
        )
        s = torch.zeros((B, H, (L + C - 1) // C, DK, DV)).npu().to(torch.float16)

        print()
        print(f"Shape: (B,H,L,DK,DV,C)=({B},{H},{L},{DK},{DV},{C})")

        # # Build the stage inputs once so each benchmark only times one kernel launch.
        g_sum = run_stage("chunk_cumsum", lambda: ker1(g))
        a_raw = run_stage("chunk_scaled_dot_kkt", lambda: ker2(k, beta, g_sum, msk1))
        a_solved = run_stage("solve_tril", lambda: ker3(a_raw, idt))
        w, u = run_stage("wy_fast", lambda: ker4(k, v, beta, g_sum, a_solved))
        nv, _ = run_stage("chunk_h", lambda: ker5(k, w, u, g_sum, workspace, s))
        run_stage("chunk_o", lambda: ker6(q, k, nv, s, g_sum, msk2))

        latencies = {
            "chunk_cumsum": bench_stage("chunk_cumsum", lambda: ker1(g)),
            "chunk_scaled_dot_kkt": bench_stage(
                "chunk_scaled_dot_kkt", lambda: ker2(k, beta, g_sum, msk1)
            ),
            "solve_tril": bench_stage("solve_tril", lambda: ker3(a_raw, idt)),
            "wy_fast": bench_stage(
                "wy_fast", lambda: ker4(k, v, beta, g_sum, a_solved)
            ),
            "chunk_h": bench_stage(
                "chunk_h", lambda: ker5(k, w, u, g_sum, workspace, s)
            ),
            "chunk_o": bench_stage("chunk_o", lambda: ker6(q, k, nv, s, g_sum, msk2)),
        }

        # These simple approximate op counts match the README table for the default shape.
        ops = {
            "chunk_cumsum": B * H * L,
            "chunk_scaled_dot_kkt": B * H * L * C * DK,
            "solve_tril": B * H * L * C * C // 3,
            "wy_fast": B * H * L * C * (DK + DV),
            "chunk_h": 4 * B * H * L * DK * DV,
            "chunk_o": B * H * L * (C * DK + DK * DV + C * DV),
        }

        latencies_by_c[C] = latencies

        total_ms = sum(latencies[name] for name in KERNEL_ORDER)
        total_ops = sum(ops[name] for name in KERNEL_ORDER)

        print(f"Shape: (B,H,L,DK,DV,C)=({B},{H},{L},{DK},{DV},{C})")
        print("| Kernel | Latency (ms) | #ops (approx) | TFLOPS |")
        print("| :-- | --: | --: | --: |")
        for name in KERNEL_ORDER:
            print(
                f"| {name} | {format_ms(latencies[name])} | {format_ops(ops[name])} | "
                f"{format_tflops(ops[name], latencies[name])} |"
            )
        print(
            f"| total | {format_ms(total_ms)} | {format_ops(total_ops)} | "
            f"{format_tflops(total_ops, total_ms)} |"
        )

    plot_stacked_latencies(
        latencies_by_c,
        f"B={B}, H={H}, L={L}, DK={DK}, DV={DV}, BK={BK}, BV={BV}",
    )
    print(f"Saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
