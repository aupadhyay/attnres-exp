"""
Visualize per-token depth attention maps for specific prompts.
Shows how depth routing changes across token positions.
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))
from model import GPT, GPTConfig


@torch.no_grad()
def get_per_token_alphas(model: GPT, input_ids: torch.Tensor) -> list[np.ndarray]:
    """Get alpha weights for each token position."""
    device = input_ids.device
    b, t = input_ids.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    tok_emb = model.transformer.wte(input_ids)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)

    all_alphas = []

    if model.config.residual_mode == 'full_attnres':
        layer_outputs = [x]
        for i, block in enumerate(model.transformer.h):
            block_out = block.attn(block.ln_1(x))
            x_after_attn = x + block_out
            mlp_out = block.mlp(block.ln_2(x_after_attn))
            layer_out = x_after_attn + mlp_out
            layer_outputs.append(layer_out)
            alpha = model.depth_attn[i].get_alpha(layer_outputs)  # (N, B, T)
            all_alphas.append(alpha[:, 0, :].cpu().numpy())  # take batch=0: (N, T)
            x = model.depth_attn[i](layer_outputs)

    return all_alphas


def plot_per_token(tokens: list[str], alphas: list[np.ndarray], output_path: str, title: str = ""):
    """
    alphas: list of length n_layers, each (n_sources, T)
    tokens: list of strings, length T
    """
    n_layers = len(alphas)
    T = len(tokens)

    # Show subset of tokens if too many
    max_tokens = 50
    if T > max_tokens:
        tokens = tokens[:max_tokens]
        alphas = [a[:, :max_tokens] for a in alphas]
        T = max_tokens

    fig, axes = plt.subplots(n_layers, 1, figsize=(max(14, T * 0.3), n_layers * 1.5), sharex=True)
    if n_layers == 1:
        axes = [axes]

    for i, (ax, alpha) in enumerate(zip(axes, alphas)):
        n_sources = alpha.shape[0]
        im = ax.imshow(alpha, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.set_ylabel(f"L{i+1}", fontsize=8)
        ax.set_yticks(range(n_sources))
        ax.set_yticklabels([f"s{j}" for j in range(n_sources)], fontsize=6)

    axes[-1].set_xticks(range(T))
    axes[-1].set_xticklabels(tokens, rotation=90, fontsize=7)
    fig.suptitle(title or "Per-Token Depth Attention", fontsize=12)
    fig.colorbar(im, ax=axes, shrink=0.6, label="alpha weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", required=True, help="Text prompt to analyze")
    parser.add_argument("--output", default="figures/per_token_depth.png")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    config = GPTConfig(**ckpt["model_args"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    token_ids = enc.encode(args.prompt)
    tokens = [enc.decode([t]) for t in token_ids]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    alphas = get_per_token_alphas(model, input_ids)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_per_token(tokens, alphas, args.output, args.title)


if __name__ == "__main__":
    main()
