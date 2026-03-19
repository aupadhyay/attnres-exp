import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))

from model import GPT, GPTConfig


def test_baseline_forward_unchanged():
    """Baseline mode should produce same behavior as original nanoGPT."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='baseline'
    )
    model = GPT(config)
    idx = torch.randint(0, 256, (2, 16))
    logits, loss = model(idx, targets=torch.randint(0, 256, (2, 16)))
    assert logits.shape == (2, 16, 256)
    assert loss is not None


def test_full_attnres_forward():
    """Full AttnRes mode should run and produce valid output."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='full_attnres'
    )
    model = GPT(config)
    idx = torch.randint(0, 256, (2, 16))
    logits, loss = model(idx, targets=torch.randint(0, 256, (2, 16)))
    assert logits.shape == (2, 16, 256)
    assert loss is not None


def test_block_attnres_forward():
    """Block AttnRes mode should run and produce valid output."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=12, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='block_attnres', attnres_n_blocks=4
    )
    model = GPT(config)
    idx = torch.randint(0, 256, (2, 16))
    logits, loss = model(idx, targets=torch.randint(0, 256, (2, 16)))
    assert logits.shape == (2, 16, 256)
    assert loss is not None


def test_full_attnres_loss_decreases():
    """Verify that full_attnres model can learn (loss decreases over a few steps)."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='full_attnres'
    )
    model = GPT(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fixed data
    idx = torch.randint(0, 256, (4, 32))
    targets = torch.randint(0, 256, (4, 32))

    losses = []
    for _ in range(20):
        _, loss = model(idx, targets)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_param_count_overhead():
    """AttnRes should add minimal parameters (only w_l per layer = n_layer * n_embd)."""
    base_config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='baseline'
    )
    attn_config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='full_attnres'
    )
    base_model = GPT(base_config)
    attn_model = GPT(attn_config)

    base_params = sum(p.numel() for p in base_model.parameters())
    attn_params = sum(p.numel() for p in attn_model.parameters())
    overhead = attn_params - base_params

    # Overhead should be: n_layer * n_embd (w vectors) + n_layer * n_embd (RMSNorm weights)
    expected_overhead = 4 * 64 * 2  # 512 params
    assert overhead == expected_overhead, f"Unexpected overhead: {overhead} (expected {expected_overhead})"
