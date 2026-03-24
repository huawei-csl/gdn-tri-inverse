#!/usr/bin/env python3
"""Compare tilelang_opt vs cube-rec-unroll solve: latency and bandwidth (head-first layout)."""

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

DEFAULT_CHUNKS = (64, 128)

OPT_KEY = "tilelang_opt"
FAST_KEY = "cube-rec-unroll"
OPT_LABEL = "opt-gdn (tilelang)"
FAST_LABEL = "Fast inverse (PTO-ISA)"


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
    opt = df.loc[df["inverse_type"] == OPT_KEY].sort_values("T")
    fast = df.loc[df["inverse_type"] == FAST_KEY].sort_values("T")
    opt = add_bw_stats(opt)
    fast = add_bw_stats(fast)
    return opt, fast


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
    return DATA_DIR / f"bench_results_solve_tril_{chunk_size}.csv"


def batch_head_line(opt: pd.DataFrame) -> str:
    b = int(opt["B"].iloc[0])
    h = int(opt["H"].iloc[0])
    return f'(batch={b}, head={h}, "head-first" layout)'


def main(chunk_sizes: tuple[int, ...]) -> None:
    apply_notebook_style()

    chunk_files = [(c, chunk_csv_path(c)) for c in chunk_sizes]
    pairs: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
    for chunk_size, csv_path in chunk_files:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing benchmark CSV: {csv_path}")
        opt, fast = load_pair(csv_path)
        pairs.append((chunk_size, opt, fast))

    meta = pairs[0][1]
    bh = batch_head_line(meta)
    title_time = f"Inverse kernel time comparison \n{bh}"
    title_bw = f"Inverse kernel bandwidth comparison \n{bh}"

    ncols = len(pairs)
    fig_w = max(5 * ncols, 5)
    fig_h = 0.7 * (5.2 + 1.0 * ncols)

    figs_dir = SCRIPT_DIR / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    fig_time, axes_time = plt.subplots(1, ncols, figsize=(fig_w, fig_h))
    fig_time.suptitle(title_time, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
    axes_time = np.atleast_1d(axes_time)

    fig_bw, axes_bw = plt.subplots(1, ncols, figsize=(fig_w, fig_h))
    fig_bw.suptitle(title_bw, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
    axes_bw = np.atleast_1d(axes_bw)

    xlabel = "Seqlen (K)"

    for col, (chunk_size, opt, fast) in enumerate(pairs):
        xo = seqlen_k(opt["T"])
        xf = seqlen_k(fast["T"])
        opt_ms = opt["time_us"] / 1000.0
        fast_ms = fast["time_us"] / 1000.0

        ax_t = axes_time[col]
        ax_t.plot(xo, opt_ms, label=OPT_LABEL, **line_kw_for_series(0))
        ax_t.plot(xf, fast_ms, label=FAST_LABEL, **line_kw_for_series(1))
        ax_t.set_xlabel(xlabel)
        ax_t.set_ylabel("time (ms)")
        ax_t.set_title(f"chunk_size = {chunk_size}")
        finish_axis(ax_t)
        if col == 0:
            ax_t.legend(markerscale=LEGEND_MARKER_SCALE)

        ax_b = axes_bw[col]
        ax_b.plot(xo, opt["bw_gbs"], label=OPT_LABEL, **line_kw_for_series(0))
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

    path_time = figs_dir / "vs_tilelang_time.png"
    path_bw = figs_dir / "vs_tilelang_bw.png"
    fig_time.savefig(path_time, dpi=150)
    fig_bw.savefig(path_bw, dpi=150)
    plt.close(fig_time)
    plt.close(fig_bw)
    print(f"Wrote {path_time}")
    print(f"Wrote {path_bw}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot opt-gdn (tilelang) vs Fast inverse (PTO-ISA); "
            "writes figs/vs_tilelang_time.png and vs_tilelang_bw.png."
        ),
    )
    parser.add_argument(
        "--chunks",
        type=parse_chunks,
        default=DEFAULT_CHUNKS,
        metavar="N[,N,...]",
        help=(
            "Comma-separated chunk sizes; reads data/bench_results_solve_tril_<N>.csv "
            f"(default: {','.join(map(str, DEFAULT_CHUNKS))})"
        ),
    )
    args = parser.parse_args()
    main(args.chunks)
