"""
Extract per-layer output magnitudes and gradient magnitudes from checkpoints.
Reproduces Figure 5 from the AttnRes paper at GPT-2 scale.

Uses forward hooks to capture block outputs from the actual model forward pass,
so it works correctly for all residual_mode variants (baseline, full_attnres, block_attnres).
"""

import argparse
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))
from model import GPT, GPTConfig


def extract_dynamics(model: GPT, input_ids: torch.Tensor, targets: torch.Tensor):
    """
    Run forward+backward with hooks to capture per-block output magnitudes
    and gradient magnitudes. Works for any residual_mode.

    Returns:
        output_magnitudes: list of float, ||block_output|| per block
        gradient_magnitudes: list of float, ||grad|| per block
    """
    block_outputs = []
    hooks = []

    # Register forward hooks on each Block's MLP to capture its output.
    # We hook the MLP (not the Block) because in full_attnres mode,
    # block.forward() is never called — sub-layers are called individually.
    # The MLP is the last sub-layer in every mode, so its output fires reliably.
    for i, block in enumerate(model.transformer.h):
        def hook_fn(module, input, output, idx=i):
            output.retain_grad()
            block_outputs.append(output)
        hooks.append(block.mlp.register_forward_hook(hook_fn))

    # Forward + backward through the actual model (uses correct residual_mode)
    logits, loss = model(input_ids, targets)
    loss.backward()

    # Extract magnitudes
    output_magnitudes = []
    gradient_magnitudes = []
    for out in block_outputs:
        output_magnitudes.append(out.float().norm(dim=-1).mean().item())
        if out.grad is not None:
            gradient_magnitudes.append(out.grad.float().norm(dim=-1).mean().item())
        else:
            gradient_magnitudes.append(0.0)

    # Clean up hooks
    for h in hooks:
        h.remove()

    return output_magnitudes, gradient_magnitudes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=1024)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    config = GPTConfig(**ckpt["model_args"])
    model = GPT(config)
    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    state_dict = ckpt["model"]
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)

    all_output_mags = []
    all_grad_mags = []

    for _ in range(args.n_batches):
        idx = torch.randint(0, config.vocab_size, (args.batch_size, args.block_size), device=device)
        targets = torch.randint(0, config.vocab_size, (args.batch_size, args.block_size), device=device)

        model.train()
        model.zero_grad()
        out_mags, grad_mags = extract_dynamics(model, idx, targets)
        all_output_mags.append(out_mags)
        all_grad_mags.append(grad_mags)

    np.savez(
        args.output,
        output_magnitudes=np.array(all_output_mags).mean(axis=0),
        gradient_magnitudes=np.array(all_grad_mags).mean(axis=0),
        iter_num=ckpt["iter_num"],
    )
    print(f"Saved dynamics to {args.output}")


if __name__ == "__main__":
    main()
