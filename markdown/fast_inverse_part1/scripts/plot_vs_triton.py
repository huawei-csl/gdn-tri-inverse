#!/usr/bin/env python3
"""Compare Triton vs bsnd-rec-unroll inverse: latency and bandwidth across chunk sizes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_tril_mpl_style import (
    LEGEND_MARKER_SCALE,
    SUPTITLE_FONTSIZE,
    apply_notebook_style,
    finish_axis,
    line_kw_for_series,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

DEFAULT_CHUNKS = (32, 64)

BASELINE_KEY = "triton"
FAST_KEY = "bsnd-rec-unroll"
BASELINE_LABEL = "Baseline (Triton)"
FAST_LABEL = "Fast-inverse (PTO-ISA)"

BATCH_HEAD = '(batch=32, head=4, "seq-first" layout)'
TITLE_TIME = f"Inverse kernel time comparison \n{BATCH_HEAD}"
TITLE_BW = f"Inverse kernel bandwidth comparison \n{BATCH_HEAD}"


def add_bw_stats(df: pd.DataFrame, nbytes_out_dtype: int = 4) -> pd.DataFrame:
    """Effective memory traffic and bandwidth (same as plots_bsnd_inv.ipynb)."""
    out = df.copy()
    out["size"] = out["numel"]
    out["in_bytes_per_elem"] = out["dtype"].map({"fp32": 4, "fp16": 2, "int8": 1})
    out["mem_bytes"] = out["size"] * (out["in_bytes_per_elem"] + nbytes_out_dtype)
    out["bw_gbs"] = (out["mem_bytes"] / 1e9) / (out["time_us"] / 1e6)
    return out


def load_pair(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    base = df.loc[df["inverse_type"] == BASELINE_KEY].sort_values("T")
    fast = df.loc[df["inverse_type"] == FAST_KEY].sort_values("T")
    base = add_bw_stats(base)
    fast = add_bw_stats(fast)
    return base, fast


def seqlen_k(series: pd.Series) -> np.ndarray:
    return (series / 1000.0).to_numpy()


def parse_chunks(spec: str) -> tuple[int, ...]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("at least one chunk size is required")
    try:
        return tuple(int(p, 10) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid chunk list {spec!r}: {e}") from e


def chunk_csv_path(chunk_size: int) -> Path:
    return DATA_DIR / f"bench_results_bsnd_tril_{chunk_size}.csv"


def main(chunk_sizes: tuple[int, ...]) -> None:
    apply_notebook_style()

    chunk_files = [(c, chunk_csv_path(c)) for c in chunk_sizes]
    pairs: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
    for chunk_size, csv_path in chunk_files:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing benchmark CSV: {csv_path}")
        base, fast = load_pair(csv_path)
        pairs.append((chunk_size, base, fast))

    ncols = len(pairs)
    fig_w = max(5 * ncols, 5)
    fig_h = 0.7 * (5.2 + 1.0 * ncols)

    figs_dir = SCRIPT_DIR / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    fig_time, axes_time = plt.subplots(1, ncols, figsize=(fig_w, fig_h))
    fig_time.suptitle(TITLE_TIME, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
    axes_time = np.atleast_1d(axes_time)

    fig_bw, axes_bw = plt.subplots(1, ncols, figsize=(fig_w, fig_h))
    fig_bw.suptitle(TITLE_BW, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
    axes_bw = np.atleast_1d(axes_bw)

    xlabel = "Seqlen (K)"

    for col, (chunk_size, base, fast) in enumerate(pairs):
        xb = seqlen_k(base["T"])
        xf = seqlen_k(fast["T"])
        base_ms = base["time_us"] / 1000.0
        fast_ms = fast["time_us"] / 1000.0

        ax_t = axes_time[col]
        ax_t.plot(xb, base_ms, label=BASELINE_LABEL, **line_kw_for_series(0))
        ax_t.plot(xf, fast_ms, label=FAST_LABEL, **line_kw_for_series(1))
        ax_t.set_xlabel(xlabel)
        ax_t.set_ylabel("time (ms)")
        ax_t.set_title(f"chunk_size = {chunk_size}")
        finish_axis(ax_t)
        if col == 0:
            ax_t.legend(markerscale=LEGEND_MARKER_SCALE)

        ax_b = axes_bw[col]
        ax_b.plot(xb, base["bw_gbs"], label=BASELINE_LABEL, **line_kw_for_series(0))
        ax_b.plot(xf, fast["bw_gbs"], label=FAST_LABEL, **line_kw_for_series(1))
        ax_b.set_xlabel(xlabel)
        ax_b.set_ylabel("Effective bandwidth (GB/s)")
        ax_b.set_title(f"chunk_size = {chunk_size}")
        ax_b.set_ylim(bottom=0)
        finish_axis(ax_b)
        if col == 0:
            ax_b.legend(markerscale=LEGEND_MARKER_SCALE)

    fig_time.tight_layout()
    fig_bw.tight_layout()

    path_time = figs_dir / "vs_triton_time.png"
    path_bw = figs_dir / "vs_triton_bw.png"
    fig_time.savefig(path_time, dpi=150)
    fig_bw.savefig(path_bw, dpi=150)
    plt.close(fig_time)
    plt.close(fig_bw)
    print(f"Wrote {path_time}")
    print(f"Wrote {path_bw}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Triton vs bsnd-rec-unroll; writes figs/vs_triton_time.png and vs_triton_bw.png.",
    )
    parser.add_argument(
        "--chunks",
        type=parse_chunks,
        default=DEFAULT_CHUNKS,
        metavar="N[,N,...]",
        help=(
            "Comma-separated chunk sizes; reads data/bench_results_bsnd_tril_<N>.csv "
            f"(default: {','.join(map(str, DEFAULT_CHUNKS))})"
        ),
    )
    args = parser.parse_args()
    main(args.chunks)
