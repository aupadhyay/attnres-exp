"""
plot_block_routing.py

Shows the quicksort code 4 times (once per block boundary), with each token
colored by its EMBEDDING attention fraction at that block. This makes the sparse
embedding spikes at B4 visible, unlike dominant-color plots which wash them out.

Color scale: low embedding = light blue, high embedding = orange.

Usage:
    uv run python analyze/plot_block_routing.py \
        --ckpt data/v2/block_attnres/ckpt.pt \
        --output figures/v2/block_routing_code.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tiktoken
import torch
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))
from model import GPT, GPTConfig

# Color scale: 0 = light steel blue, 1 = orange
CMAP = LinearSegmentedColormap.from_list('emb', ['#cfe0f0', '#e85d26'])

DEFAULT_PROMPT = (
    "def quicksort(arr):\n"
    "    if len(arr) <= 1:\n"
    "        return arr\n"
    "    pivot = arr[len(arr) // 2]\n"
    "    left = [x for x in arr if x < pivot]"
)

CHAR_W = 0.0115   # width of one monospace char in axes [0,1] units
LINE_H = 0.13     # height of one line in axes [0,1] units
PAD_X  = 0.02
PAD_Y_TOP = 0.88  # y position of first line top


@torch.no_grad()
def get_all_alphas(model, input_ids):
    """Returns list of (n_sources, T) emb-fraction arrays, one per block."""
    device = input_ids.device
    b, t = input_ids.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    tok_emb = model.transformer.wte(input_ids)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)

    all_alphas = []
    block_outputs = [x]
    layers_per_block = model.layers_per_block

    for block_idx in range(model.config.attnres_n_blocks):
        start = block_idx * layers_per_block
        end = start + layers_per_block
        for layer in model.transformer.h[start:end]:
            x = layer(x)
        block_outputs.append(x)
        alpha = model.depth_attn[block_idx].get_alpha(block_outputs)  # (N, B, T)
        all_alphas.append(alpha[:, 0, :].cpu().numpy())  # (N, T)
        x = model.depth_attn[block_idx](block_outputs)

    return all_alphas  # list of (N_sources, T)


def tokenize_to_lines(prompt, enc):
    """Split prompt into lines, each line a list of (token_str, global_idx)."""
    lines_text = prompt.split('\n')
    lines = []
    global_idx = 0
    for line_text in lines_text:
        if line_text == '':
            # Empty line still counts as a newline token
            lines.append([])
            continue
        token_ids = enc.encode(line_text)
        line_tokens = []
        for tid in token_ids:
            tok_str = enc.decode([tid])
            line_tokens.append((tok_str, global_idx))
            global_idx += 1
        lines.append(line_tokens)
    return lines


def draw_code_panel(ax, lines, emb_fracs, block_label, vmax):
    """
    Draw one block's code panel. Tokens colored by emb_fracs[token_idx],
    normalized to vmax for consistent scale across blocks.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    for line_idx, line_tokens in enumerate(lines):
        y_top = PAD_Y_TOP - line_idx * LINE_H
        y_bot = y_top - LINE_H * 0.85
        y_mid = (y_top + y_bot) / 2

        x = PAD_X
        for tok_str, tok_idx in line_tokens:
            w = len(tok_str) * CHAR_W
            emb_val = emb_fracs[tok_idx]
            color = CMAP(emb_val / vmax) if vmax > 0 else CMAP(0)

            ax.add_patch(mpatches.FancyBboxPatch(
                (x, y_bot), w, y_top - y_bot,
                boxstyle="round,pad=0.003",
                facecolor=color, edgecolor='none', zorder=1
            ))
            ax.text(x + w / 2, y_mid, tok_str,
                    ha='center', va='center',
                    fontsize=9.5, fontfamily='monospace', zorder=2)
            x += w

    # Block label on the left
    ax.text(-0.01, 0.5, block_label, ha='right', va='center',
            fontsize=11, fontweight='bold', transform=ax.transAxes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", default="figures/v2/block_routing_code.png")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    config = GPTConfig(**ckpt["model_args"])
    model = GPT(config)
    state_dict = ckpt["model"]
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")

    # Encode without newlines (newline tokens aren't in the lines)
    prompt_flat = args.prompt.replace('\n', '')
    # Actually encode the full prompt as a flat sequence
    token_ids = enc.encode(args.prompt)
    T = len(token_ids)

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    all_alphas = get_all_alphas(model, input_ids)  # list of 4 x (N_sources, T)

    # Extract emb fraction per block: alpha[0, :] = emb source
    emb_per_block = [alpha[0, :] for alpha in all_alphas]  # list of (T,) arrays

    # Build line structure
    lines = tokenize_to_lines(args.prompt, enc)
    n_blocks = len(all_alphas)

    # Use a consistent vmax = max emb value across all blocks (usually B1)
    vmax = max(e.max() for e in emb_per_block)

    fig, axes = plt.subplots(n_blocks, 1, figsize=(12, n_blocks * 1.6))
    fig.patch.set_facecolor('white')

    for i, (ax, emb_fracs) in enumerate(zip(axes, emb_per_block)):
        draw_code_panel(ax, lines, emb_fracs, f'B{i+1}', vmax)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.6, pad=0.02, aspect=20)
    cbar.set_label('Embedding attention fraction', fontsize=10)

    fig.suptitle('Depth routing across blocks — quicksort\n'
                 '(orange = attending to embedding, blue = attending to recent layers)',
                 fontsize=12, y=1.01)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
