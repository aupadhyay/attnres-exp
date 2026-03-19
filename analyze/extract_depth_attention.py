"""
Extract depth attention weights (alpha) from AttnRes checkpoints.
For static analysis (averaged over tokens) and per-token analysis.
"""

import argparse
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))
from model import GPT, GPTConfig


@torch.no_grad()
def extract_alphas(model: GPT, input_ids: torch.Tensor) -> list[np.ndarray]:
    """
    Run forward and extract alpha weights from each DepthAttention layer.

    Returns:
        alphas: list of arrays, each shape (N_sources, B, T)
    """
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
            all_alphas.append(alpha.cpu().numpy())
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
            all_alphas.append(alpha.cpu().numpy())
            x = model.depth_attn[block_idx](block_outputs)

    return all_alphas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--data-dir", default=None, help="Data dir for real tokens (optional)")
    parser.add_argument("--n-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=1024)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    config = GPTConfig(**ckpt["model_args"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Collect alphas averaged over batches
    avg_alphas = None
    count = 0

    for _ in range(args.n_batches):
        if args.data_dir:
            data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")
            ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
            idx = torch.stack([torch.from_numpy(data[i:i+args.block_size].astype(np.int64)) for i in ix])
        else:
            idx = torch.randint(0, config.vocab_size, (args.batch_size, args.block_size))
        idx = idx.to(device)

        alphas = extract_alphas(model, idx)

        if avg_alphas is None:
            # alphas is a list of arrays with different first dims
            avg_alphas = [a.mean(axis=(1, 2)) for a in alphas]  # average over B, T -> (N,)
        else:
            for i, a in enumerate(alphas):
                avg_alphas[i] += a.mean(axis=(1, 2))
        count += 1

    # Normalize
    avg_alphas = [a / count for a in avg_alphas]

    np.savez(args.output, **{f"layer_{i}": a for i, a in enumerate(avg_alphas)},
             n_layers=len(avg_alphas), residual_mode=str(config.residual_mode))
    print(f"Saved depth attention weights to {args.output}")


if __name__ == "__main__":
    main()
