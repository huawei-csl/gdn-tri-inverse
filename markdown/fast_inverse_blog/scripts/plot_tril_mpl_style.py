"""Matplotlib styling aligned with plots_bsnd_inv.ipynb (Colorbrewer palette, markers, dashes)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes

# Notebook used font_scale=2.0 and suptitle fontsize=13; body text −30%, main title +50%.
FONT_SCALE = 2.0 * 0.7
SUPTITLE_FONTSIZE = 13 * 1.5

# Colorbrewer-style colors from plots_bsnd_inv.ipynb
PALETTE = (
    "black",
    "#008837",
    "#a6dba0",
    "#c2a5cf",
    "#7b3294",
)
MARKERS = ("s", "^", "v", "<", ">", "o", "X")
LEGEND_MARKER_SCALE = 1.5


def apply_notebook_style() -> None:
    sns.set_context("paper", font_scale=FONT_SCALE, rc={"lines.linewidth": 1.75})
    sns.set_style({"font.weight": "bold"})
    plt.rcParams["lines.markersize"] = 11
    plt.rcParams["lines.linewidth"] = 2.0
    plt.rcParams["figure.figsize"] = (8.7, 6.27)
    plt.rcParams["axes.titlepad"] = 10
    plt.rcParams["legend.borderpad"] = 0.4
    plt.rcParams["legend.labelspacing"] = 0.35


def line_kw_for_series(index: int) -> dict:
    """Color, marker, linestyle for the i-th series (matches seaborn hue/style order for two curves)."""
    linestyle = "-" if index == 0 else "--"
    return {
        "color": PALETTE[index % len(PALETTE)],
        "marker": MARKERS[index % len(MARKERS)],
        "linestyle": linestyle,
    }


def finish_axis(ax: Axes) -> None:
    sns.despine(ax=ax, right=True)
    ax.grid(True)
