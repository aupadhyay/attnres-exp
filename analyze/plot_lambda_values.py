"""
Extract and plot learned value residual mixing ratios (lambda) per layer.
Shows how much each layer relies on raw token embedding vs computed values.
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))


def extract_lambdas(ckpt_path: str) -> tuple[list[int], np.ndarray]:
    """Extract sigmoid(raw_lambda) per layer from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"]

    layers = []
    raw_lambdas = []
    for k in sorted(state_dict.keys()):
        if "raw_lambda" in k:
            # e.g. transformer.h.1.attn.raw_lambda
            layer_idx = int(k.split(".")[2])
            layers.append(layer_idx)
            raw_lambdas.append(state_dict[k].item())

    raw_lambdas = np.array(raw_lambdas)
    lambdas = 1.0 / (1.0 + np.exp(-raw_lambdas))  # sigmoid
    return layers, lambdas


def plot_lambdas(layers: list[int], lambdas: np.ndarray, output_path: str, title: str = ""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of lambda values
    colors = plt.cm.viridis(lambdas / max(lambdas.max(), 0.01))
    ax1.bar(range(len(layers)), lambdas, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([f"L{l}" for l in layers])
    ax1.set_ylabel("λ (sigmoid of learned logit)")
    ax1.set_title("Learned Value Residual Mixing (λ)")
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="λ=0.5 (equal mix)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_xlabel("Layer\n(λ=0: computed values only, λ=1: raw token embedding only)")

    # Stacked horizontal bar: composition per layer
    ax2.barh(range(len(layers)), 1 - lambdas, color="#1f77b4", label="Computed V (1-λ)")
    ax2.barh(range(len(layers)), lambdas, left=1 - lambdas, color="#ff7f0e", label="Token embedding V₀ (λ)")
    ax2.set_yticks(range(len(layers)))
    ax2.set_yticklabels([f"L{l}" for l in layers])
    ax2.set_xlabel("Mixing ratio")
    ax2.set_title("Value Stream Composition per Layer")
    ax2.legend(loc="lower right")
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()

    fig.suptitle(title or "Value Residual — Per-Layer Mixing Ratios", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", default="figures/lambda_values.png")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    layers, lambdas = extract_lambdas(args.ckpt)

    print("Layer | lambda (sigmoid)")
    print("------|----------------")
    for l, lam in zip(layers, lambdas):
        print(f"  L{l:2d} | {lam:.4f}")

    plot_lambdas(layers, lambdas, args.output, args.title)
