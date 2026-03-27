"""
Plot depth attention heatmaps (Figure 8 reproduction).
Rows = target layer, Columns = source layer.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(npz_path: str, output_path: str, title: str = "", prefix: str = "L"):
    data = np.load(npz_path, allow_pickle=True)
    n_layers = int(data["n_layers"])

    # Build matrix: rows = target layer, cols = source layer
    # Each layer_i has shape (i+1,) -- the alpha weights over sources 0..i
    max_sources = n_layers + 1  # embedding + n_layers
    matrix = np.zeros((n_layers, max_sources))
    for i in range(n_layers):
        alphas = data[f"layer_{i}"]
        matrix[i, :len(alphas)] = alphas

    unit = "Block" if prefix == "B" else "Layer"
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="viridis",
        xticklabels=["emb"] + [f"{prefix}{i+1}" for i in range(max_sources - 1)],
        yticklabels=[f"{prefix}{i+1}" for i in range(n_layers)],
        vmin=0,
        annot=True if n_layers <= 12 else False,
        fmt=".2f",
    )
    ax.set_xlabel(f"Source {unit}")
    ax.set_ylabel(f"Target {unit}")
    ax.set_title(title or "Depth Attention Weights (alpha)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help=".npz from extract_depth_attention.py")
    parser.add_argument("--output", default="figures/depth_heatmap.png")
    parser.add_argument("--title", default="")
    parser.add_argument("--prefix", default="L", help="Label prefix: L for layers, B for blocks")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_heatmap(args.input, args.output, args.title, args.prefix)
