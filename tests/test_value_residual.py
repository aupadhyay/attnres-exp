import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))

from model import GPT, GPTConfig


def test_value_residual_forward_baseline():
    """Value residual with baseline residuals should run."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='baseline',
        use_value_residual=True, value_residual_mode='learnable_per_layer',
    )
    model = GPT(config)
    idx = torch.randint(0, 256, (2, 16))
    logits, loss = model(idx, targets=torch.randint(0, 256, (2, 16)))
    assert logits.shape == (2, 16, 256)
    assert loss is not None


def test_value_residual_forward_block_attnres():
    """Value residual with block AttnRes should run."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=12, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='block_attnres', attnres_n_blocks=4,
        use_value_residual=True, value_residual_mode='learnable_per_layer',
    )
    model = GPT(config)
    idx = torch.randint(0, 256, (2, 16))
    logits, loss = model(idx, targets=torch.randint(0, 256, (2, 16)))
    assert logits.shape == (2, 16, 256)
    assert loss is not None


def test_value_residual_forward_full_attnres():
    """Value residual with full AttnRes should run."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='full_attnres',
        use_value_residual=True, value_residual_mode='learnable_per_layer',
    )
    model = GPT(config)
    idx = torch.randint(0, 256, (2, 16))
    logits, loss = model(idx, targets=torch.randint(0, 256, (2, 16)))
    assert logits.shape == (2, 16, 256)
    assert loss is not None


def test_value_residual_disabled_unchanged():
    """With value residual disabled, model should be identical."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='baseline',
        use_value_residual=False,
    )
    model = GPT(config)
    # No v0_proj should exist
    assert not hasattr(model, 'v0_proj')


def test_lambda_init():
    """Lambda should init to sigmoid(0.0) = 0.5."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        residual_mode='baseline',
        use_value_residual=True, value_residual_mode='learnable_per_layer',
        value_residual_lambda_init=0.0,
    )
    model = GPT(config)
    # Layer 0 should NOT have value residual (it's the first layer)
    assert not model.transformer.h[0].attn.use_value_residual
    # Layer 1+ should have learnable lambda initialized at sigmoid(0) = 0.5
    for i in range(1, 4):
        lam = torch.sigmoid(model.transformer.h[i].attn.raw_lambda)
        assert abs(lam.item() - 0.5) < 0.01


def test_value_residual_loss_decreases():
    """Value residual model should learn."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='baseline',
        use_value_residual=True, value_residual_mode='learnable_per_layer',
    )
    model = GPT(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    idx = torch.randint(0, 256, (4, 32))
    targets = torch.randint(0, 256, (4, 32))
    losses = []
    for _ in range(20):
        _, loss = model(idx, targets)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]


def test_per_head_lambda():
    """Per-head lambda should have n_head parameters."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        residual_mode='baseline',
        use_value_residual=True, value_residual_mode='learnable_per_head',
    )
    model = GPT(config)
    # Layer 1 should have per-head lambda
    assert model.transformer.h[1].attn.raw_lambda.shape == (4,)  # n_head


def test_fixed_lambda():
    """Fixed lambda mode should work without learnable parameters."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        residual_mode='baseline',
        use_value_residual=True, value_residual_mode='fixed',
        value_residual_fixed_lambda=0.3,
    )
    model = GPT(config)
    # Layer 1+ should NOT have raw_lambda (fixed mode)
    assert not hasattr(model.transformer.h[1].attn, 'raw_lambda')
    idx = torch.randint(0, 256, (2, 16))
    logits, loss = model(idx, targets=torch.randint(0, 256, (2, 16)))
    assert logits.shape == (2, 16, 256)
