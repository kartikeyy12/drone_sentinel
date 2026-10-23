"""Shared Matplotlib look for the report figures: neutral ink, a fixed categorical order, recessive grid.

Categorical colours are the first slots of the validated reference palette (light mode), assigned in fixed order.
Every series also gets its own line style and marker, so identity never rests on colour alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), "-", "--"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DPI = 200


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False, "lines.linewidth": 1.8,
        "lines.markersize": 5, "font.size": 9,
    })


def style(i: int) -> dict:
    """Line kwargs for the i-th series: colour, line style and marker."""
    return {"color": SERIES[i % len(SERIES)], "linestyle": LINESTYLES[i % len(LINESTYLES)],
            "marker": MARKERS[i % len(MARKERS)], "markeredgecolor": SURFACE, "markeredgewidth": 0.8}


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
