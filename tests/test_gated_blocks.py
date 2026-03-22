import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))

from model import GPT, GPTConfig


def test_gated_forward():
    """Gated block AttnRes should run."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=12, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='block_attnres',
        attnres_n_blocks=4, use_gated_blocks=True, gate_init_bias=2.0,
    )
    model = GPT(config)
    idx = torch.randint(0, 256, (2, 16))
    logits, loss = model(idx, targets=torch.randint(0, 256, (2, 16)))
    assert logits.shape == (2, 16, 256)
    assert loss is not None


def test_gated_disabled_matches_baseline():
    """With gates disabled, output should match standard block AttnRes."""
    config_base = GPTConfig(
        block_size=32, vocab_size=256, n_layer=12, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='block_attnres',
        attnres_n_blocks=4, use_gated_blocks=False,
    )
    config_gated = GPTConfig(
        block_size=32, vocab_size=256, n_layer=12, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='block_attnres',
        attnres_n_blocks=4, use_gated_blocks=False,
    )
    model_base = GPT(config_base)
    model_gated = GPT(config_gated)
    # Load same weights
    model_gated.load_state_dict(model_base.state_dict())
    idx = torch.randint(0, 256, (2, 16))
    torch.manual_seed(42)
    out_base, _ = model_base(idx)
    torch.manual_seed(42)
    out_gated, _ = model_gated(idx)
    assert torch.allclose(out_base, out_gated, atol=1e-5)


def test_gate_init_near_open():
    """Gates should start near-open (sigmoid ≈ 0.88)."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        residual_mode='block_attnres', attnres_n_blocks=2,
        use_gated_blocks=True, gate_init_bias=2.0,
    )
    model = GPT(config)
    for block in model.transformer.h:
        gate_val = torch.sigmoid(block.gate_logit + block.gate_bias)
        assert gate_val.mean().item() > 0.8, "Gates should start near-open"


def test_gated_loss_decreases():
    """Gated block model should learn."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=12, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='block_attnres',
        attnres_n_blocks=4, use_gated_blocks=True,
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


def test_gate_param_count():
    """Gate overhead should be n_layer * n_embd parameters."""
    config_base = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        residual_mode='block_attnres', attnres_n_blocks=2,
        use_gated_blocks=False,
    )
    config_gated = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        residual_mode='block_attnres', attnres_n_blocks=2,
        use_gated_blocks=True,
    )
    base = GPT(config_base)
    gated = GPT(config_gated)
    base_p = sum(p.numel() for p in base.parameters())
    gated_p = sum(p.numel() for p in gated.parameters())
    expected_overhead = 4 * 64  # n_layer * n_embd
    assert gated_p - base_p == expected_overhead
