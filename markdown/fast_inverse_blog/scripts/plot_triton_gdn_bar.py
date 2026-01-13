#!/usr/bin/env python3
"""Bar chart comparing median latency of three GDN inverse implementations (chunk_size=64)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_tril_mpl_style import (
    PALETTE,
    SUPTITLE_FONTSIZE,
    apply_notebook_style,
    finish_axis,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR.parent.parent.parent / "bench_results_triton_gdn.csv"

SERIES_ORDER = ("triton", "cube-rec-unroll", "bsnd-rec-unroll")
SERIES_LABELS = {
    "cube-rec-unroll": "MXR+Transpose",
    "bsnd-rec-unroll": "BSND-MXR",
    "triton": "Triton",
}

TITLE = "GDN layer E2E (chunk_size=64)"


def main(csv_path: Path) -> None:
    apply_notebook_style()

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    medians = (
        df.groupby("inverse_type")["elapsed_time_ms"]
        .median()
        .reindex(SERIES_ORDER)
    )

    labels = [SERIES_LABELS[k] for k in SERIES_ORDER]
    values = medians.to_numpy()
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(SERIES_ORDER))]

    fig, ax = plt.subplots()
    fig.suptitle(TITLE, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")

    x = np.arange(len(SERIES_ORDER))
    bars = ax.bar(x, values, color=colors, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{val:.2f} ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("[ms]")
    ax.set_ylim(bottom=0, top=values.max() * 1.2)
    finish_axis(ax)

    fig.tight_layout()

    figs_dir = SCRIPT_DIR / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    out_path = figs_dir / "GDN-layer-e2e.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bar chart of median GDN inverse latency; writes figs/GDN-layer-e2e.png.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to benchmark CSV (default: {DEFAULT_CSV})",
    )
    args = parser.parse_args()
    main(args.csv)
