"""
Plot overlaid validation loss curves from wandb or checkpoint logs.
Produces Figure: val loss comparison across baseline, full, block AttnRes.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_wandb_logs(run_dir: str) -> dict:
    """Load loss data from wandb local logs or exported JSON."""
    with open(run_dir) as f:
        data = json.load(f)
    return data


def load_checkpoint_logs(ckpt_dir: str) -> tuple[list[int], list[float]]:
    """Extract (iter, val_loss) from periodic checkpoints."""
    import torch
    iters, losses = [], []
    for fname in sorted(os.listdir(ckpt_dir)):
        if fname.startswith("ckpt_") and fname.endswith(".pt"):
            ckpt = torch.load(os.path.join(ckpt_dir, fname), map_location="cpu")
            iters.append(ckpt["iter_num"])
            losses.append(ckpt["val_loss"])
    return iters, losses


def plot(data: dict[str, tuple[list, list]], output_path: str):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = {"baseline": "#1f77b4", "full_attnres": "#ff7f0e", "block_attnres": "#2ca02c"}
    labels = {"baseline": "Baseline (PreNorm)", "full_attnres": "Full AttnRes", "block_attnres": "Block AttnRes (N=4)"}

    for variant, (iters, losses) in data.items():
        ax.plot(iters, losses, label=labels.get(variant, variant),
                color=colors.get(variant, None), linewidth=1.5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("AttnRes vs Baseline — Validation Loss (GPT-2 124M, OpenWebText)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dirs", nargs="+", required=True,
                        help="Checkpoint directories (format: variant:path)")
    parser.add_argument("--output", default="figures/val_loss_curves.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    data = {}
    for spec in args.ckpt_dirs:
        variant, path = spec.split(":", 1)
        iters, losses = load_checkpoint_logs(path)
        data[variant] = (iters, losses)

    plot(data, args.output)
