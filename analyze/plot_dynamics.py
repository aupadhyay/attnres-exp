"""
Plot per-layer output and gradient magnitudes (Figure 5 reproduction).
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_dynamics(data: dict[str, dict], output_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"baseline": "#1f77b4", "full_attnres": "#ff7f0e", "block_attnres": "#2ca02c"}
    labels = {"baseline": "Baseline", "full_attnres": "Full AttnRes", "block_attnres": "Block AttnRes"}

    for variant, d in data.items():
        layers = np.arange(1, len(d["output_magnitudes"]) + 1)
        ax1.plot(layers, d["output_magnitudes"], label=labels.get(variant, variant),
                 color=colors.get(variant), marker="o", markersize=4)
        ax2.plot(layers, d["gradient_magnitudes"], label=labels.get(variant, variant),
                 color=colors.get(variant), marker="o", markersize=4)

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("||f_l(h_l)||")
    ax1.set_title("Output Magnitude per Layer")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("||dL/dh_l||")
    ax2.set_title("Gradient Magnitude per Layer")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Training Dynamics — AttnRes vs Baseline (GPT-2 124M)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamics-files", nargs="+", required=True,
                        help="Format: variant:path.npz")
    parser.add_argument("--output", default="figures/dynamics.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    data = {}
    for spec in args.dynamics_files:
        variant, path = spec.split(":", 1)
        d = np.load(path)
        data[variant] = dict(d)

    plot_dynamics(data, args.output)
