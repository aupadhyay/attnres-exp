"""
Analyze learned w_l pseudo-query vectors:
- Cosine similarity matrix between all w_l
- PCA of query vectors -- do they cluster by layer position?
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))
from model import GPT, GPTConfig


def extract_query_vectors(ckpt_path: str) -> tuple[np.ndarray, str]:
    """Extract w_l vectors from checkpoint. Returns (n_layers, d) array."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = GPTConfig(**ckpt["model_args"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])

    vectors = []
    for da in model.depth_attn:
        vectors.append(da.w.detach().numpy())
    return np.stack(vectors), config.residual_mode


def plot_cosine_similarity(vectors: np.ndarray, output_path: str, title: str = ""):
    """Plot cosine similarity matrix between w_l vectors."""
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = vectors / norms
    cos_sim = normed @ normed.T

    n = len(vectors)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"L{i+1}" for i in range(n)])
    ax.set_yticklabels([f"L{i+1}" for i in range(n)])
    ax.set_title(title or "Cosine Similarity of w_l Query Vectors")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


def plot_pca(vectors: np.ndarray, output_path: str, title: str = ""):
    """PCA of w_l vectors colored by layer index."""
    pca = PCA(n_components=2)
    projected = pca.fit_transform(vectors)

    n = len(vectors)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(projected[:, 0], projected[:, 1],
                         c=range(n), cmap="viridis", s=100, edgecolors="black")
    for i in range(n):
        ax.annotate(f"L{i+1}", (projected[i, 0], projected[i, 1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(title or "PCA of w_l Query Vectors")
    fig.colorbar(scatter, label="Layer index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--title-prefix", default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    vectors, mode = extract_query_vectors(args.ckpt)
    prefix = args.title_prefix or mode

    plot_cosine_similarity(vectors,
        os.path.join(args.output_dir, f"query_cosine_{mode}.png"),
        f"{prefix} — Cosine Similarity of w_l")
    plot_pca(vectors,
        os.path.join(args.output_dir, f"query_pca_{mode}.png"),
        f"{prefix} — PCA of w_l")
