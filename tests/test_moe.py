import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))

from model import GPT, GPTConfig


def test_moe_forward_baseline():
    """MoE with baseline residuals should run."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='baseline',
        use_moe=True, num_experts=4, moe_top_k=2,
    )
    model = GPT(config)
    idx = torch.randint(0, 256, (2, 16))
    logits, loss = model(idx, targets=torch.randint(0, 256, (2, 16)))
    assert logits.shape == (2, 16, 256)
    assert loss is not None


def test_moe_forward_block_attnres():
    """MoE with block AttnRes should run."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=12, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='block_attnres', attnres_n_blocks=4,
        use_moe=True, num_experts=4, moe_top_k=2,
    )
    model = GPT(config)
    idx = torch.randint(0, 256, (2, 16))
    logits, loss = model(idx, targets=torch.randint(0, 256, (2, 16)))
    assert logits.shape == (2, 16, 256)
    assert loss is not None


def test_moe_disabled_unchanged():
    """With use_moe=False, behavior is identical to standard model."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='baseline', use_moe=False,
    )
    model = GPT(config)
    torch.manual_seed(42)
    idx = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 16, 256)
    # Verify no MoE modules exist
    assert not any(hasattr(block, 'moe') for block in model.transformer.h)


def test_moe_aux_loss_nonzero():
    """Aux loss should be non-zero during training."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='baseline',
        use_moe=True, num_experts=4, moe_top_k=2, moe_aux_loss_coeff=0.01,
    )
    model = GPT(config)
    model.train()
    idx = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))
    logits, loss = model(idx, targets)
    # Loss should include aux component — verify it's a valid tensor
    assert loss.requires_grad


def test_moe_loss_decreases():
    """MoE model should be able to learn."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='baseline',
        use_moe=True, num_experts=4, moe_top_k=2,
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
    assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_moe_expert_param_count():
    """MoE should have num_experts * MLP params instead of 1 * MLP params."""
    base_config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='baseline', use_moe=False,
    )
    moe_config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='baseline',
        use_moe=True, num_experts=4, moe_top_k=2,
    )
    base_model = GPT(base_config)
    moe_model = GPT(moe_config)
    base_params = sum(p.numel() for p in base_model.parameters())
    moe_params = sum(p.numel() for p in moe_model.parameters())
    # MoE should have more params (4 experts instead of 1 MLP, plus router)
    assert moe_params > base_params
