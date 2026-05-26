#!/usr/bin/env python3
"""
Create poster-ready DOPE benchmark graphs using the
Warm Space-Age Academic color palette.

This version saves PNG files only and plots standard deviation as error bars.

Outputs:
  - dope_tsr_with_std_warm_space_age.png
  - dope_cr_with_std_warm_space_age.png
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Warm Space-Age Academic palette
# -----------------------------

PALETTE = {
    "bg": "#FFFFFF",          # White
    "panel": "#F7F9FC",       # Light Gray panel
    "soft_sand": "#E2E8F0",   # Cool Gray
    "heading": "#1A202C",     # Near Black
    "text": "#2D3748",        # Dark Slate
    "grid": "#CBD5E0",        # Light Slate

    # Model colors
    "e2e": "#3182CE",         # Blue
    "caa": "#E53E3E",         # Red
    "ddp": "#38A169",         # Green
}

# -----------------------------
# Data from the paper table
# -----------------------------

SETTINGS = [
    "Reported",
    "Reproduced",
    "Static\nBaseline",
    "Misaligned",
    "Misaligned\n+ Drive-Out",
    "Misaligned\n+ Follow",
    "Misaligned\n+ Block",
]

MODELS = ["E2E Parking", "CAA Policy", "DDP"]

MODEL_COLORS = [
    PALETTE["e2e"],
    PALETTE["caa"],
    PALETTE["ddp"],
]

DATA = {
    "E2E Parking": {
        "TSR": [91.41, 82.55, 72.14, 61.46, 16.67, 10.42, 9.64],
        "TSR_std": [0.00, 4.16, 4.23, 4.42, 3.42, 1.93, 3.02],
        "CR": [2.08, 7.29, 14.58, 26.30, 71.88, 78.12, 79.95],
        "CR_std": [0.00, 2.65, 3.46, 4.01, 4.59, 3.54, 4.50],
    },
    "CAA Policy": {
        "TSR": [87.50, 87.24, 71.35, 68.75, 43.23, 28.65, 19.27],
        "TSR_std": [3.70, 4.12, 3.56, 3.53, 3.52, 3.00, 1.47],
        "CR": [3.50, 5.21, 22.92, 22.14, 54.17, 67.45, 76.56],
        "CR_std": [2.60, 2.00, 2.26, 2.55, 3.03, 2.72, 1.32],
    },
    "DDP": {
        "TSR": [92.00, 92.71, 84.38, 80.47, 57.03, 41.15, 24.48],
        "TSR_std": [11.00, 1.32, 1.27, 1.13, 1.03, 1.07, 0.95],
        "CR": [0.00, 0.00, 6.25, 6.77, 38.02, 56.25, 72.66],
        "CR_std": [0.00, 0.00, 0.00, 0.21, 0.21, 0.37, 0.77],
    },
}

# -----------------------------
# Plot helpers
# -----------------------------

def style_axis(ax, ylabel: str) -> None:
    """Apply consistent poster-style formatting to one axis."""
    ax.set_facecolor(PALETTE["panel"])
    ax.set_ylabel(ylabel, fontsize=19, fontweight="bold", color=PALETTE["text"])
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.42, color=PALETTE["grid"])
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=16, colors=PALETTE["text"], pad=8)
    ax.tick_params(axis="y", labelsize=16, colors=PALETTE["text"])

    for label in ax.get_xticklabels():
        label.set_fontweight("semibold")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(PALETTE["grid"])
    ax.spines["bottom"].set_color(PALETTE["grid"])


def add_bar_labels(ax, bars, values, stds, show_std_text: bool = True) -> None:
    """
    Add compact labels above error bars.

    The label format is mean±std. Labels are rotated vertically to prevent
    overlap in grouped bar charts.
    """
    for bar, value, std in zip(bars, values, stds):
        x = bar.get_x() + bar.get_width() / 2
        top = value + std
        y = min(top + 2.0, 106.5)
        label = f"{value:.1f}±{std:.1f}" if show_std_text else f"{value:.1f}"
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=13.0,
            color=PALETTE["text"],
            rotation=90,
            clip_on=False,
        )


def save_figure(fig, base_name: str) -> None:
    """Save a figure as PNG only."""
    png_path = f"{base_name}.png"
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    print(f"Saved: {png_path}")


def plot_metric(ax, metric: str, ylabel: str, show_std_text: bool = True) -> None:
    """Plot one metric, either TSR or CR, on the given axis."""
    x = np.arange(len(SETTINGS))
    width = 0.23
    offsets = [-width, 0, width]

    for model, color, offset in zip(MODELS, MODEL_COLORS, offsets):
        values = DATA[model][metric]
        stds = DATA[model][f"{metric}_std"]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=model,
            color=color,
            edgecolor=PALETTE["heading"],
            linewidth=0.5,
            yerr=stds,
            capsize=4,
            error_kw={
                "elinewidth": 1.15,
                "capthick": 1.15,
                "ecolor": PALETTE["heading"],
            },
        )
        add_bar_labels(ax, bars, values, stds, show_std_text=show_std_text)

    style_axis(ax, ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(SETTINGS)

# -----------------------------
# Figure 1: TSR-only
# -----------------------------

def create_tsr_only_plot() -> None:
    fig, ax = plt.subplots(figsize=(15.2, 8.3))
    fig.patch.set_facecolor(PALETTE["bg"])

    plot_metric(
        ax,
        metric="TSR",
        ylabel="TSR (%)",
        show_std_text=True,
    )

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=19,
        facecolor=PALETTE["panel"],
        edgecolor=PALETTE["grid"],
    )
    for text in legend.get_texts():
        text.set_color(PALETTE["text"])

    plt.subplots_adjust(top=0.84, bottom=0.23, left=0.075, right=0.985)
    save_figure(fig, "dope_tsr_with_std_warm_space_age")
    plt.close(fig)

# -----------------------------
# Figure 2: CR-only
# -----------------------------

def create_cr_only_plot() -> None:
    fig, ax = plt.subplots(figsize=(15.2, 8.3))
    fig.patch.set_facecolor(PALETTE["bg"])

    plot_metric(
        ax,
        metric="CR",
        ylabel="CR (%)",
        show_std_text=True,
    )

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        fontsize=19,
        facecolor=PALETTE["panel"],
        edgecolor=PALETTE["grid"],
    )
    for text in legend.get_texts():
        text.set_color(PALETTE["text"])

    plt.subplots_adjust(top=0.84, bottom=0.23, left=0.075, right=0.985)
    save_figure(fig, "dope_cr_with_std_warm_space_age")
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    create_tsr_only_plot()
    create_cr_only_plot()


if __name__ == "__main__":
    main()
