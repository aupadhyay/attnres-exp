"""
Plot adaptive boundary evolution over training.

Loads checkpoints from an adaptive boundary run and visualizes:
1. Gate values sigmoid(b_l / tau) over training steps for each layer
2. Effective number of boundaries over training
3. Final hard boundary positions
4. Comparison to fixed uniform boundaries (Block AttnRes baseline)

Usage:
    python analyze/plot_boundaries.py --out_dir out-adaptive-boundaries
"""

import argparse
import os
import sys
import glob

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))
from model import GPTConfig


def load_checkpoints(out_dir):
    """Load all checkpoints from a directory, sorted by iter_num."""
    ckpt_files = sorted(glob.glob(os.path.join(out_dir, 'ckpt_*.pt')))
    # Also try the final checkpoint
    final = os.path.join(out_dir, 'ckpt.pt')
    if os.path.exists(final):
        ckpt_files.append(final)

    checkpoints = []
    for f in ckpt_files:
        ckpt = torch.load(f, map_location='cpu', weights_only=False)
        checkpoints.append(ckpt)

    # Sort by iter_num
    checkpoints.sort(key=lambda c: c.get('iter_num', 0))
    return checkpoints


def extract_boundary_data(checkpoints):
    """Extract boundary logits and config from checkpoints."""
    data = {'iter_nums': [], 'logits': [], 'configs': []}
    for ckpt in checkpoints:
        state = ckpt['model']
        if 'boundary_logits' not in state:
            continue
        data['iter_nums'].append(ckpt['iter_num'])
        data['logits'].append(state['boundary_logits'].numpy())
        data['configs'].append(ckpt.get('model_args', {}))
    return data


def compute_tau(iter_num, max_iters, config):
    """Compute temperature at a given iteration."""
    frac = iter_num / max_iters
    tau_start = config.get('boundary_tau_start', 5.0)
    tau_end = config.get('boundary_tau_end', 0.1)
    start_frac = config.get('boundary_anneal_start_frac', 0.2)
    end_frac = config.get('boundary_anneal_end_frac', 0.7)

    if frac < start_frac:
        return tau_start
    elif frac > end_frac:
        return tau_end
    else:
        progress = (frac - start_frac) / (end_frac - start_frac)
        return tau_start + progress * (tau_end - tau_start)


def plot_gate_evolution(data, max_iters, save_dir):
    """Plot gate values over training for each layer."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    n_layers_minus_1 = len(data['logits'][0])
    colors = plt.cm.viridis(np.linspace(0, 1, n_layers_minus_1))

    # Gate values
    ax = axes[0]
    for layer_idx in range(n_layers_minus_1):
        gates = []
        for i, (iter_num, logits, config) in enumerate(
            zip(data['iter_nums'], data['logits'], data['configs'])
        ):
            tau = compute_tau(iter_num, max_iters, config)
            gate = 1.0 / (1.0 + np.exp(-logits[layer_idx] / tau))
            gates.append(gate)
        ax.plot(data['iter_nums'], gates, color=colors[layer_idx],
                label=f'Layer {layer_idx + 1}', linewidth=1.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gate Value (sigmoid)')
    ax.set_title('Boundary Gate Values Over Training')
    ax.legend(fontsize=8, ncol=3)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')

    # Effective number of boundaries
    ax = axes[1]
    n_effective = []
    for i, (iter_num, logits, config) in enumerate(
        zip(data['iter_nums'], data['logits'], data['configs'])
    ):
        tau = compute_tau(iter_num, max_iters, config)
        gates = 1.0 / (1.0 + np.exp(-logits / tau))
        n_effective.append(gates.sum())
    ax.plot(data['iter_nums'], n_effective, 'b-', linewidth=2)
    n_target = data['configs'][0].get('boundary_n_target', 4)
    ax.axhline(y=n_target, color='r', linestyle='--', label=f'Target ({n_target})')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Effective Boundaries')
    ax.set_title('Effective Number of Boundaries Over Training')
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'boundary_evolution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved boundary evolution plot to {save_path}")
    plt.close()


def plot_final_boundaries(data, max_iters, save_dir):
    """Plot the final hard boundary positions."""
    if not data['iter_nums']:
        print("No checkpoint data found.")
        return

    final_logits = data['logits'][-1]
    final_config = data['configs'][-1]
    final_iter = data['iter_nums'][-1]
    tau = compute_tau(final_iter, max_iters, final_config)
    gates = 1.0 / (1.0 + np.exp(-final_logits / tau))

    n_layers = len(gates) + 1
    fig, ax = plt.subplots(figsize=(10, 4))

    # Show all layers, color by gate value
    layer_indices = np.arange(n_layers)
    gate_values = np.concatenate([[1.0], gates])  # Layer 0 always a boundary

    bars = ax.bar(layer_indices, gate_values, color=plt.cm.RdYlGn(gate_values))
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Boundary Gate Value')
    ax.set_title(f'Final Boundary Positions (step {final_iter}, tau={tau:.3f})')
    ax.set_xticks(layer_indices)
    ax.legend()

    # Annotate hard boundaries
    hard_boundaries = [0] + [i + 1 for i, g in enumerate(gates) if g > 0.5]
    ax.set_title(
        f'Final Boundary Positions (step {final_iter})\n'
        f'Hard boundaries at layers: {hard_boundaries}'
    )

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'final_boundaries.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved final boundaries plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    save_dir = args.save_dir or args.out_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading checkpoints from {args.out_dir}...")
    checkpoints = load_checkpoints(args.out_dir)
    if not checkpoints:
        print("No checkpoints found!")
        return

    data = extract_boundary_data(checkpoints)
    if not data['iter_nums']:
        print("No adaptive boundary data found in checkpoints.")
        return

    print(f"Found {len(data['iter_nums'])} checkpoints with boundary data.")
    plot_gate_evolution(data, args.max_iters, save_dir)
    plot_final_boundaries(data, args.max_iters, save_dir)


if __name__ == '__main__':
    main()
