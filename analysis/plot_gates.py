"""
Gate activation analysis for Gated Block AttnRes.

Loads checkpoints at different training steps and visualizes:
1. Mean gate activation per layer over training (which layers get suppressed?)
2. Per-dimension gate histograms at end of training for selected layers
3. Comparison of depth attention patterns: gated vs ungated Block AttnRes
"""

import sys
import os
import glob
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))
from model import GPT, GPTConfig


def load_gate_values(ckpt_path):
    """Extract gate activations from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_args = ckpt['model_args']

    gate_bias = model_args.get('gate_init_bias', 2.0)
    state_dict = ckpt['model']

    gates = {}
    for key, val in state_dict.items():
        if 'gate_logit' in key:
            # key like 'transformer.h.3.gate_logit'
            layer_idx = int(key.split('.')[2])
            gate_val = torch.sigmoid(val + gate_bias)
            gates[layer_idx] = gate_val.numpy()

    iter_num = ckpt.get('iter_num', 0)
    return gates, iter_num


def plot_gate_means_over_training(ckpt_dir, out_path=None):
    """Plot mean gate activation per layer across training checkpoints."""
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, 'ckpt_*.pt')))
    if not ckpt_files:
        print(f"No periodic checkpoints found in {ckpt_dir}")
        return

    all_iters = []
    all_gates = {}  # layer_idx -> list of mean values

    for f in ckpt_files:
        gates, iter_num = load_gate_values(f)
        if not gates:
            continue
        all_iters.append(iter_num)
        for layer_idx, gate_vals in gates.items():
            if layer_idx not in all_gates:
                all_gates[layer_idx] = []
            all_gates[layer_idx].append(gate_vals.mean())

    if not all_iters:
        print("No gate values found in checkpoints")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for layer_idx in sorted(all_gates.keys()):
        ax.plot(all_iters, all_gates[layer_idx], label=f'Layer {layer_idx}', alpha=0.8)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Gate Activation')
    ax.set_title('Per-Layer Gate Activations Over Training')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved gate means plot to {out_path}")
    else:
        plt.show()


def plot_gate_histograms(ckpt_path, layers=None, out_path=None):
    """Plot per-dimension gate histograms for selected layers at a single checkpoint."""
    gates, iter_num = load_gate_values(ckpt_path)
    if not gates:
        print("No gate values found")
        return

    if layers is None:
        layers = sorted(gates.keys())

    n_layers = len(layers)
    cols = min(4, n_layers)
    rows = (n_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, layer_idx in enumerate(layers):
        if layer_idx not in gates:
            continue
        ax = axes[i]
        ax.hist(gates[layer_idx], bins=50, alpha=0.7, color='steelblue')
        ax.set_title(f'Layer {layer_idx} (mean={gates[layer_idx].mean():.3f})')
        ax.set_xlabel('Gate Value')
        ax.set_ylabel('Count')
        ax.set_xlim(0, 1)

    # Hide unused axes
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Gate Value Distributions (step {iter_num})', fontsize=14)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved gate histograms to {out_path}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze gate activations')
    parser.add_argument('--ckpt_dir', type=str, default='out-gated-blocks',
                        help='Directory containing checkpoints')
    parser.add_argument('--plot', choices=['means', 'histograms', 'both'], default='both')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Directory to save plots (default: show interactively)')
    args = parser.parse_args()

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    if args.plot in ('means', 'both'):
        out_path = os.path.join(args.out_dir, 'gate_means.png') if args.out_dir else None
        plot_gate_means_over_training(args.ckpt_dir, out_path)

    if args.plot in ('histograms', 'both'):
        # Use the latest checkpoint for histograms
        final_ckpt = os.path.join(args.ckpt_dir, 'ckpt.pt')
        if os.path.exists(final_ckpt):
            out_path = os.path.join(args.out_dir, 'gate_histograms.png') if args.out_dir else None
            plot_gate_histograms(final_ckpt, out_path=out_path)
        else:
            print(f"No final checkpoint found at {final_ckpt}")
