import csv
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from gdn_tri_inverse.tilelang_chunk_gdn.opt_gdn_solve_tril import (
    ref_all_ones_tril,
    ref_chunk_cumsum,
    ref_kkt,
    solve_tril,
)


NPU_DEVICE = os.getenv("GDN_TRI_INVERSE_NPU_DEVICE", "npu:0")

BATCH = 1
HEADS = 2
LENGTH = 1024
DK = 128
CHUNK_SIZES = [32, 64, 128]
GAMMA_SCALES = [0.0, 0.25, 0.5, 1.0, 2.0]
CSV_PATH = "bench_results_tilelang_solve_tril_numerical_accuracy.csv"
ALL_ONES_PLOT_PATH = "bench_results_tilelang_solve_tril_numerical_accuracy_all_ones.png"
GAMMA_PLOT_PATH = "bench_results_tilelang_solve_tril_numerical_accuracy_gamma.png"


def make_gamma_scaled_input(k, beta, g_raw, C, gamma_scale):
    g = F.logsigmoid(g_raw)
    g = gamma_scale * ref_chunk_cumsum(g, C)
    return ref_kkt(k, beta, g, C)


def max_identity_residual(a, o, C):
    B, H, L, _ = a.shape
    assert L % C == 0, f"L={L} must be divisible by C={C} for this numerical test"
    chunk_num = L // C
    idt = torch.eye(C, device=a.device, dtype=torch.float).view(1, 1, 1, C, C)
    a_blocks = a.view(B, H, chunk_num, C, C).float()
    o_blocks = o.view(B, H, chunk_num, C, C).float()
    return (((idt + a_blocks) @ o_blocks) - idt).abs().max().item()


def run_case(case_name, a, C):
    o = solve_tril(a)
    max_residual = max_identity_residual(a, o, C)
    return {
        "case": case_name,
        "C": C,
        "max_identity_residual": max_residual,
    }


def plot_results(results):
    all_ones = [row for row in results if row["case"] == "all_ones"]
    all_ones.sort(key=lambda row: row["C"])
    plt.figure(figsize=(6, 4))
    bars = plt.bar(
        [str(row["C"]) for row in all_ones],
        [row["max_identity_residual"] for row in all_ones],
    )
    max_all_ones = max(row["max_identity_residual"] for row in all_ones)
    plt.ylim(0.0, max(max_all_ones * 1.2, 1e-7))
    plt.xlabel("Chunk size C")
    plt.ylabel(r"Max reconstruction error $\left\|(I + A)O - I\right\|_{\infty}$")
    plt.title(r"All-Ones Input Reconstruction Error")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    for bar, row in zip(bars, all_ones):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{row['max_identity_residual']:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(ALL_ONES_PLOT_PATH, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    case_labels = [r"all-ones tril $A$"] + [
        rf"$\gamma={scale}$" for scale in GAMMA_SCALES
    ]
    x = list(range(len(case_labels)))
    width = 0.25
    max_gamma = 0.0
    for idx, C in enumerate(CHUNK_SIZES):
        all_ones_row = next(
            row for row in results if row["C"] == C and row["case"] == "all_ones"
        )
        rows = [
            row
            for row in results
            if row["C"] == C and row["case"].startswith("gamma_scale=")
        ]
        rows.sort(key=lambda row: float(row["case"].split("=")[1]))
        y = [all_ones_row["max_identity_residual"]] + [
            row["max_identity_residual"] for row in rows
        ]
        max_gamma = max(max_gamma, max(y))
        offset = (idx - 1) * width
        bars = plt.bar([item + offset for item in x], y, width=width, label=f"C={C}")
        for bar, value in zip(bars, y):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1e}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    plt.xticks(x, case_labels)
    plt.ylim(0.0, max(max_gamma * 1.2, 1e-7))
    plt.ylabel(r"Max recon. error: $\left\|(I + A)O - I\right\|_{\infty}$")
    plt.title(r"Reconstruction Errors for Tilelang opt_gdn_solve_tril")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GAMMA_PLOT_PATH, dpi=200)
    plt.close()


def main():
    torch.npu.set_device(NPU_DEVICE)
    tile_results = []

    torch.manual_seed(0)
    for C in CHUNK_SIZES:
        print(f"Running numerical test for C={C}")

        k = torch.randn((BATCH, HEADS, LENGTH, DK), dtype=torch.float16).npu()
        beta = torch.rand((BATCH, HEADS, LENGTH), dtype=torch.float16).npu()
        g_raw = torch.randn((BATCH, HEADS, LENGTH), dtype=torch.float).npu()
        k = F.normalize(k, dim=-1, p=2)

        a_all_ones = ref_all_ones_tril(BATCH, HEADS, LENGTH, C, strict=True)
        tile_results.append(run_case("all_ones", a_all_ones, C))

        for gamma_scale in GAMMA_SCALES:
            a = make_gamma_scaled_input(k, beta, g_raw, C, gamma_scale)
            tile_results.append(run_case(f"gamma_scale={gamma_scale}", a, C))

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "C", "max_identity_residual"],
        )
        writer.writeheader()
        writer.writerows(tile_results)

    for row in tile_results:
        print(
            f"C={row['C']:>3}  case={row['case']:<16}  "
            f"max_identity_residual={row['max_identity_residual']:.3e}"
        )

    plot_results(tile_results)
    print(f"Saved CSV to {CSV_PATH}")
    print(f"Saved plot to {ALL_ONES_PLOT_PATH}")
    print(f"Saved plot to {GAMMA_PLOT_PATH}")


if __name__ == "__main__":
    main()
