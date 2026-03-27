"""
Blog-quality annotated plots for adaptive boundary experiment.

Generates:
1. Temperature annealing schedule
2. Annotated boundary gates + n_effective showing the reg artifact
3. Combined figure for the blog post

Usage:
    uv run python analyze/plot_boundary_annotated.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# --- Config (matches train_adaptive_boundaries.py) ---
MAX_ITERS = 10000
TAU_START = 5.0
TAU_END = 0.1
ANNEAL_START_FRAC = 0.2
ANNEAL_END_FRAC = 0.7
N_TARGET = 4
REG_LAMBDA = 0.1

OUT_DIR = Path("analyze/wandb_data/plots")
CSV_PATH = Path("analyze/wandb_data/boundary_gates.csv")

# --- Style ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linewidth": 0.5,
})

PHASE_COLORS = {
    "pre": "#e8f4f8",
    "anneal": "#fff3e0",
    "post": "#fce4ec",
}


def compute_tau_schedule():
    """Compute tau over all training iters."""
    iters = np.arange(0, MAX_ITERS + 1)
    taus = np.empty_like(iters, dtype=float)
    for i, it in enumerate(iters):
        frac = it / MAX_ITERS
        if frac < ANNEAL_START_FRAC:
            taus[i] = TAU_START
        elif frac > ANNEAL_END_FRAC:
            taus[i] = TAU_END
        else:
            progress = (frac - ANNEAL_START_FRAC) / (ANNEAL_END_FRAC - ANNEAL_START_FRAC)
            taus[i] = TAU_START + progress * (TAU_END - TAU_START)
    return iters, taus


def shade_phases(ax, alpha=0.15):
    """Add phase background shading to an axis."""
    ylim = ax.get_ylim()
    ax.axvspan(0, ANNEAL_START_FRAC * MAX_ITERS, color=PHASE_COLORS["pre"], alpha=alpha, zorder=0)
    ax.axvspan(ANNEAL_START_FRAC * MAX_ITERS, ANNEAL_END_FRAC * MAX_ITERS, color=PHASE_COLORS["anneal"], alpha=alpha, zorder=0)
    ax.axvspan(ANNEAL_END_FRAC * MAX_ITERS, MAX_ITERS, color=PHASE_COLORS["post"], alpha=alpha, zorder=0)
    ax.set_ylim(ylim)


def plot_tau_schedule(ax):
    """Plot temperature schedule with phase annotations."""
    iters, taus = compute_tau_schedule()
    ax.plot(iters, taus, color="#1565c0", linewidth=2.5)

    shade_phases(ax, alpha=0.3)

    # Phase labels
    ax.text(1000, 4.2, "Warm-up\n$\\tau = 5.0$", ha="center", fontsize=9, color="#555")
    ax.text(4500, 3.0, "Annealing\n$\\tau: 5.0 \\to 0.1$", ha="center", fontsize=9, color="#555")
    ax.text(8500, 1.0, "Frozen\n$\\tau = 0.1$", ha="center", fontsize=9, color="#555")

    # Boundary lines
    for x in [ANNEAL_START_FRAC * MAX_ITERS, ANNEAL_END_FRAC * MAX_ITERS]:
        ax.axvline(x, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Temperature $\\tau$")
    ax.set_title("Temperature Annealing Schedule")
    ax.set_xlim(0, MAX_ITERS)
    ax.set_ylim(-0.2, 5.5)


def plot_gates_annotated(ax, df):
    """Plot gate evolution with annotations explaining the bounce."""
    gate_cols = sorted([c for c in df.columns if c.startswith("boundary/gate_")])

    # Use a colormap for layers
    cmap = plt.cm.viridis
    n_gates = len(gate_cols)
    colors = [cmap(i / max(n_gates - 1, 1)) for i in range(n_gates)]

    for col, color in zip(gate_cols, colors):
        layer_idx = col.split("_")[-1]
        ax.plot(df["iter"], df[col], color=color, alpha=0.6, linewidth=1.2,
                label=f"layer {layer_idx}")

    shade_phases(ax, alpha=0.3)

    # Boundary lines
    for x in [ANNEAL_START_FRAC * MAX_ITERS, ANNEAL_END_FRAC * MAX_ITERS]:
        ax.axvline(x, color="gray", linestyle=":", alpha=0.5)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Gate Value $\\sigma(b_l / \\tau)$")
    ax.set_title("Boundary Gate Values")
    ax.set_xlim(0, MAX_ITERS)
    ax.set_ylim(-0.02, 0.5)
    ax.legend(fontsize=7, ncol=3, loc="upper right")


def plot_neff_annotated(ax, df):
    """Plot effective boundary count with reg loss annotations."""
    gate_cols = sorted([c for c in df.columns if c.startswith("boundary/gate_")])

    # Soft sum (original n_effective)
    ax.plot(df["iter"], df["boundary/n_effective"], color="#7b1fa2", linewidth=2.5,
            label="soft sum $\\sum \\sigma(b_l/\\tau)$")

    # Hard count: number of gates > 0.5
    hard_count = (df[gate_cols] > 0.5).sum(axis=1)
    ax.plot(df["iter"], hard_count, color="#2e7d32", linewidth=2.5, linestyle="-",
            label="hard count (gate > 0.5)")

    shade_phases(ax, alpha=0.3)

    # Target line
    ax.axhline(y=N_TARGET, color="#d32f2f", linestyle="--", alpha=0.6, linewidth=1)
    ax.text(200, N_TARGET + 0.15, f"reg target = {N_TARGET}", fontsize=8, color="#d32f2f")

    # Boundary lines
    for x in [ANNEAL_START_FRAC * MAX_ITERS, ANNEAL_END_FRAC * MAX_ITERS]:
        ax.axvline(x, color="gray", linestyle=":", alpha=0.5)

    # Annotate the minimum
    min_idx = df["boundary/n_effective"].idxmin()
    min_iter = df.loc[min_idx, "iter"]
    min_val = df.loc[min_idx, "boundary/n_effective"]
    ax.annotate(
        f"soft sum = {min_val:.2f}\nreg loss = {REG_LAMBDA * (min_val - N_TARGET)**2:.1f}",
        xy=(min_iter, min_val),
        xytext=(min_iter - 1800, 1.5),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9),
    )

    # Annotate the bounce-back
    final_val = df["boundary/n_effective"].iloc[-1]
    final_iter = df["iter"].iloc[-1]
    ax.annotate(
        f"soft sum bounces to {final_val:.1f}\n(but 0 actual boundaries)",
        xy=(final_iter, final_val),
        xytext=(final_iter - 2800, 2.8),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9),
    )

    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("# Boundaries")
    ax.set_title("Effective Block Count")
    ax.set_xlim(0, MAX_ITERS)
    ax.set_ylim(-0.2, 5.0)
    ax.legend(fontsize=8, loc="right")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # --- Figure 1: Temperature schedule standalone ---
    fig, ax = plt.subplots(figsize=(8, 3.5))
    plot_tau_schedule(ax)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "tau_schedule.png", dpi=200, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'tau_schedule.png'}")
    plt.close()

    # --- Figure 2: Combined 3-panel for blog ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), gridspec_kw={"height_ratios": [1, 1.2, 1.2]})

    plot_tau_schedule(axes[0])
    plot_gates_annotated(axes[1], df)
    plot_neff_annotated(axes[2], df)

    plt.tight_layout(h_pad=1.5)
    fig.savefig(OUT_DIR / "boundary_analysis_annotated.png", dpi=200, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'boundary_analysis_annotated.png'}")
    plt.close()

    # --- Figure 3: Just gates + n_eff (2-panel, replaces original) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_gates_annotated(axes[0], df)
    plot_neff_annotated(axes[1], df)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "boundary_gates_annotated.png", dpi=200, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'boundary_gates_annotated.png'}")
    plt.close()

    print("Done!")


if __name__ == "__main__":
    main()
