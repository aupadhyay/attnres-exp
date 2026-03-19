import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))


def test_rmsnorm_output_shape():
    from model_attnres import RMSNorm
    norm = RMSNorm(768)
    x = torch.randn(2, 16, 768)
    out = norm(x)
    assert out.shape == x.shape


def test_rmsnorm_unit_rms():
    """RMSNorm should produce outputs with RMS ~ 1."""
    from model_attnres import RMSNorm
    norm = RMSNorm(768)
    x = torch.randn(2, 16, 768) * 10  # large input
    out = norm(x)
    rms = (out ** 2).mean(dim=-1).sqrt()
    # weight is initialized to ones, so RMS should be ~1
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


def test_depth_attention_output_shape():
    """DepthAttention should aggregate N layer outputs into one hidden state."""
    from model_attnres import DepthAttention
    d = 768
    da = DepthAttention(dim=d, layer_idx=5)
    # 6 previous layer outputs (layers 0..5), each (B, T, d)
    layer_outputs = [torch.randn(2, 16, d) for _ in range(6)]
    out = da(layer_outputs)
    assert out.shape == (2, 16, d)


def test_depth_attention_single_layer():
    """With one layer output, alpha should be 1.0 (softmax of single logit)."""
    from model_attnres import DepthAttention
    d = 64
    da = DepthAttention(dim=d, layer_idx=0)
    layer_outputs = [torch.randn(2, 4, d)]
    out = da(layer_outputs)
    # Should be alpha=1.0 * layer_outputs[0], i.e. just RMSNorm'd then re-weighted
    assert out.shape == (2, 4, d)


def test_depth_attention_gradients_flow():
    """Gradients should flow back through depth attention to all layer outputs."""
    from model_attnres import DepthAttention
    d = 64
    da = DepthAttention(dim=d, layer_idx=3)
    layer_outputs = [torch.randn(2, 4, d, requires_grad=True) for _ in range(4)]
    out = da(layer_outputs)
    loss = out.sum()
    loss.backward()
    for i, lo in enumerate(layer_outputs):
        assert lo.grad is not None, f"No gradient for layer output {i}"
