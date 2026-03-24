"""Reusable plotting helpers for arithmetic experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false


_BOLD_COLORS = [
    "#d7191c",
    "#1f78b4",
    "#f57c00",
    "#00897b",
    "#8e24aa",
    "#5d4037",
    "#c2185b",
    "#455a64",
]


@dataclass(frozen=True)
class LineSeries:
    """One named line to be drawn on a plot.

    Args:
        label: Legend label.
        x_values: X coordinates.
        y_values: Y coordinates.
        color: Optional line color override.
        linestyle: Matplotlib line style.

    """

    label: str
    x_values: np.ndarray
    y_values: np.ndarray
    color: str | None = None
    linestyle: str = "-"


def _style_axes(ax: Axes, *, y_log: bool) -> None:
    """Apply the shared visual style for experiment plots."""
    ax.set_facecolor("#fbfbfb")
    ax.minorticks_on()
    ax.grid(True, which="major", color="#a0a0a0", linewidth=0.9, alpha=0.85)
    ax.grid(True, which="minor", color="#cfcfcf", linewidth=0.6, alpha=0.7)

    if y_log:
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=[value / 10.0 for value in range(2, 10)], numticks=100)
        )
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.xaxis.set_minor_locator(AutoMinorLocator())


def save_line_plot(
    output_path: str | Path,
    series: list[LineSeries],
    *,
    title: str,
    x_label: str,
    y_label: str,
    y_log: bool = False,
) -> None:
    """Save a multi-series line plot.

    Args:
        output_path: Destination image path.
        series: Series to render.
        title: Plot title.
        x_label: X axis label.
        y_label: Y axis label.
        y_log: Whether to use a logarithmic Y axis.

    """
    if not series:
        raise ValueError("save_line_plot requires at least one series.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    _style_axes(ax, y_log=y_log)

    for index, line in enumerate(series):
        x_values = np.asarray(line.x_values, dtype=np.float64)
        y_values = np.asarray(line.y_values, dtype=np.float64)
        if x_values.shape != y_values.shape:
            raise ValueError(
                f"Shape mismatch in series {line.label!r}: {x_values.shape} vs {y_values.shape}"
            )
        if x_values.ndim != 1:
            raise ValueError(f"Series {line.label!r} must be 1D.")
        if y_log:
            y_values = np.maximum(y_values, np.finfo(np.float64).tiny)

        ax.plot(
            x_values,
            y_values,
            label=line.label,
            color=line.color or _BOLD_COLORS[index % len(_BOLD_COLORS)],
            linewidth=2.8,
            linestyle=line.linestyle,
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


__all__ = ["LineSeries", "save_line_plot"]