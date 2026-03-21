"""
Analyze depth attention routing patterns across different prompt types.

Produces two figures:
1. Per-token routing breakdown: for each token, what fraction of attention goes to
   embedding, recent layers (last 2), and distant layers (everything else)?
2. Comparison across prompt types: do factual, narrative, code prompts route differently?
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
def get_per_token_alphas(model, input_ids):
    """Get alpha weights for each token position. Returns list of (n_sources, T) arrays."""
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
            alpha = model.depth_attn[i].get_alpha(layer_outputs)
            all_alphas.append(alpha[:, 0, :].cpu().numpy())
            x = model.depth_attn[i](layer_outputs)

    elif model.config.residual_mode == 'block_attnres':
        block_outputs = [x]
        layers_per_block = model.layers_per_block
        for block_idx in range(model.config.attnres_n_blocks):
            start = block_idx * layers_per_block
            end = start + layers_per_block
            for layer in model.transformer.h[start:end]:
                x = layer(x)
            block_outputs.append(x)
            alpha = model.depth_attn[block_idx].get_alpha(block_outputs)
            all_alphas.append(alpha[:, 0, :].cpu().numpy())
            x = model.depth_attn[block_idx](block_outputs)

    return all_alphas


def compute_routing_breakdown(alphas):
    """
    For the final layer/block's depth attention, compute per-token routing breakdown:
    - embedding_frac: fraction going to embedding (source 0)
    - recent_frac: fraction going to the most recent source
    - distant_frac: everything else

    Returns dict with arrays of shape (T,).
    """
    # Use the last layer/block's alphas — that's where the model makes its final routing decision
    final_alpha = alphas[-1]  # (n_sources, T)
    n_sources, T = final_alpha.shape

    embedding_frac = final_alpha[0, :]  # source 0 = embedding
    recent_frac = final_alpha[-1, :]    # last source = most recent
    distant_frac = 1.0 - embedding_frac - recent_frac  # middle layers

    return {
        'embedding': embedding_frac,
        'recent': recent_frac,
        'distant': distant_frac,
    }


def plot_routing_breakdown(tokens, breakdown, output_path, title=""):
    """Stacked bar chart showing where each token routes its final-layer attention."""
    T = len(tokens)
    x = np.arange(T)

    fig, ax = plt.subplots(figsize=(max(10, T * 0.45), 5))

    ax.bar(x, breakdown['embedding'], label='Embedding', color='#2ca02c', alpha=0.85)
    ax.bar(x, breakdown['distant'], bottom=breakdown['embedding'],
           label='Middle layers', color='#1f77b4', alpha=0.85)
    ax.bar(x, breakdown['recent'],
           bottom=breakdown['embedding'] + breakdown['distant'],
           label='Most recent', color='#ff7f0e', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Fraction of depth attention')
    ax.set_title(title or 'Per-Token Routing Breakdown (Final Layer)')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


def plot_multi_prompt_comparison(results, output_path):
    """
    Compare routing across prompt types.
    results: dict of {prompt_name: {'tokens': [...], 'breakdown': {...}}}
    """
    n_prompts = len(results)
    fig, axes = plt.subplots(n_prompts, 1, figsize=(14, 3.5 * n_prompts), sharex=False)
    if n_prompts == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        tokens = data['tokens']
        bd = data['breakdown']
        T = len(tokens)
        x = np.arange(T)

        ax.bar(x, bd['embedding'], label='Embedding', color='#2ca02c', alpha=0.85)
        ax.bar(x, bd['distant'], bottom=bd['embedding'],
               label='Middle layers', color='#1f77b4', alpha=0.85)
        ax.bar(x, bd['recent'], bottom=bd['embedding'] + bd['distant'],
               label='Most recent', color='#ff7f0e', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Attention fraction')
        ax.set_title(name.capitalize(), fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

    axes[0].legend(loc='upper right')
    fig.suptitle('Depth Attention Routing by Prompt Type (Final Layer)', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


def plot_aggregate_by_token_type(results, output_path):
    """
    Aggregate: average routing for content words vs function words vs punctuation.
    """
    function_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'to',
                      'and', 'or', 'but', 'if', 'then', 'that', 'this', 'for', 'at',
                      'by', 'from', 'with', 'all', 'he', 'she', 'it', 'his', 'her',
                      'had', 'been', 'has', 'have', 'while', 'every'}
    punctuation = {',', '.', ':', ';', '(', ')', '[', ']', '{', '}', '\n', '!', '?'}

    categories = {'Content words': [], 'Function words': [], 'Punctuation/syntax': []}

    for name, data in results.items():
        for i, tok in enumerate(data['tokens']):
            tok_clean = tok.strip().lower()
            emb = data['breakdown']['embedding'][i]
            rec = data['breakdown']['recent'][i]
            dist = data['breakdown']['distant'][i]
            row = (emb, rec, dist)

            if tok_clean in punctuation or (len(tok_clean) <= 1 and not tok_clean.isalpha()):
                categories['Punctuation/syntax'].append(row)
            elif tok_clean in function_words:
                categories['Function words'].append(row)
            else:
                categories['Content words'].append(row)

    fig, ax = plt.subplots(figsize=(8, 5))
    cat_names = list(categories.keys())
    x = np.arange(len(cat_names))

    emb_means = []
    rec_means = []
    dist_means = []
    for cat in cat_names:
        if categories[cat]:
            arr = np.array(categories[cat])
            emb_means.append(arr[:, 0].mean())
            rec_means.append(arr[:, 1].mean())
            dist_means.append(arr[:, 2].mean())
        else:
            emb_means.append(0)
            rec_means.append(0)
            dist_means.append(0)

    emb_means = np.array(emb_means)
    rec_means = np.array(rec_means)
    dist_means = np.array(dist_means)

    width = 0.5
    ax.bar(x, emb_means, width, label='Embedding', color='#2ca02c', alpha=0.85)
    ax.bar(x, dist_means, width, bottom=emb_means, label='Middle layers', color='#1f77b4', alpha=0.85)
    ax.bar(x, rec_means, width, bottom=emb_means + dist_means, label='Most recent', color='#ff7f0e', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, fontsize=11)
    ax.set_ylabel('Average attention fraction')
    ax.set_title('Depth Routing by Token Type (Final Layer, All Prompts)')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # Add count labels
    for i, cat in enumerate(cat_names):
        n = len(categories[cat])
        ax.text(i, -0.05, f'n={n}', ha='center', fontsize=9, color='gray')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


PROMPTS = {
    'factual': 'Albert Einstein developed the theory of general relativity while working at the patent office in Bern, Switzerland.',
    'narrative': 'The old man sat by the river, watching the water flow past. He had been coming here every morning for thirty years.',
    'code': 'def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]',
    'reasoning': 'If all dogs are mammals and all mammals are animals, then all dogs are animals. This is an example of a syllogism.',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", default="figures")
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
    mode = config.residual_mode
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    for name, prompt in PROMPTS.items():
        token_ids = enc.encode(prompt)
        tokens = [enc.decode([t]) for t in token_ids]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        alphas = get_per_token_alphas(model, input_ids)
        breakdown = compute_routing_breakdown(alphas)

        results[name] = {'tokens': tokens, 'breakdown': breakdown}

        # Individual routing breakdown
        plot_routing_breakdown(
            tokens, breakdown,
            os.path.join(args.output_dir, f'routing_{name}_{mode}.png'),
            title=f'{mode} — {name.capitalize()}'
        )

    # Multi-prompt comparison
    plot_multi_prompt_comparison(
        results,
        os.path.join(args.output_dir, f'routing_comparison_{mode}.png')
    )

    # Aggregate by token type
    plot_aggregate_by_token_type(
        results,
        os.path.join(args.output_dir, f'routing_by_token_type_{mode}.png')
    )


if __name__ == "__main__":
    main()
