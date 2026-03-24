import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))

from model import GPT, GPTConfig


def test_adaptive_forward():
    """Adaptive boundary mode should run and produce valid output."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=6, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='adaptive_attnres',
        boundary_n_target=2,
    )
    model = GPT(config)
    model._current_train_frac = 0.5
    idx = torch.randint(0, 256, (2, 16))
    logits, loss = model(idx, targets=torch.randint(0, 256, (2, 16)))
    assert logits.shape == (2, 16, 256)
    assert loss is not None


def test_adaptive_loss_decreases():
    """Adaptive boundary model should learn."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=6, n_head=4, n_embd=64,
        dropout=0.0, bias=False, residual_mode='adaptive_attnres',
        boundary_n_target=2, boundary_reg_lambda=0.01,
    )
    model = GPT(config)
    model._current_train_frac = 0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    idx = torch.randint(0, 256, (4, 32))
    targets = torch.randint(0, 256, (4, 32))
    losses = []
    for step in range(20):
        model._current_train_frac = step / 20
        _, loss = model(idx, targets)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]


def test_temperature_schedule():
    """Temperature should anneal correctly."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=4, n_head=4, n_embd=64,
        residual_mode='adaptive_attnres',
        boundary_tau_start=5.0, boundary_tau_end=0.1,
        boundary_anneal_start_frac=0.2, boundary_anneal_end_frac=0.7,
    )
    model = GPT(config)

    model._current_train_frac = 0.0
    assert model._get_boundary_tau() == 5.0

    model._current_train_frac = 0.45  # midpoint of annealing
    tau_mid = model._get_boundary_tau()
    assert 0.1 < tau_mid < 5.0

    model._current_train_frac = 0.9
    assert model._get_boundary_tau() == 0.1


def test_boundary_logits_exist():
    """Model should have learnable boundary logits."""
    config = GPTConfig(
        block_size=32, vocab_size=256, n_layer=6, n_head=4, n_embd=64,
        residual_mode='adaptive_attnres',
    )
    model = GPT(config)
    assert hasattr(model, 'boundary_logits')
    assert model.boundary_logits.shape == (5,)  # n_layer - 1
    assert model.boundary_logits.requires_grad
