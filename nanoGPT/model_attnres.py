"""
Attention Residuals (AttnRes) module.
Implements depth attention over layer outputs as described in:
https://arxiv.org/abs/2603.15031
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, no mean-centering)."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = (x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class DepthAttention(nn.Module):
    """
    Computes depth-attention-weighted aggregation over previous layer outputs.

    h_l = sum_i alpha_{i->l} * v_i
    alpha_{i->l} = softmax_i(w_l^T @ RMSNorm(v_i) / sqrt(d))

    w_l is a learned pseudo-query vector (zero-initialized).
    """

    def __init__(self, dim: int, layer_idx: int):
        super().__init__()
        self.dim = dim
        self.layer_idx = layer_idx
        self.key_norm = RMSNorm(dim)
        # Zero-init query: at init, all alphas are uniform (1/n_layers_so_far)
        self.w = nn.Parameter(torch.zeros(dim))
        self.scale = dim ** -0.5

    def forward(self, layer_outputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            layer_outputs: list of tensors, each (B, T, d). Length = number of
                           previous outputs to attend over.
        Returns:
            Aggregated hidden state (B, T, d).
        """
        # Stack: (N, B, T, d) where N = len(layer_outputs)
        V = torch.stack(layer_outputs, dim=0)
        # Compute keys via RMSNorm: (N, B, T, d)
        K = self.key_norm(V)
        # Compute logits: w^T @ k / sqrt(d) -> (N, B, T)
        logits = torch.einsum('d, n b t d -> n b t', self.w, K) * self.scale
        # Softmax over depth dimension
        alpha = torch.softmax(logits, dim=0)  # (N, B, T)
        # Weighted sum: (B, T, d)
        out = torch.einsum('n b t, n b t d -> b t d', alpha, V)
        return out

    def get_alpha(self, layer_outputs: list[torch.Tensor]) -> torch.Tensor:
        """Return depth attention weights without aggregation. For analysis."""
        V = torch.stack(layer_outputs, dim=0)
        K = self.key_norm(V)
        logits = torch.einsum('d, n b t d -> n b t', self.w, K) * self.scale
        return torch.softmax(logits, dim=0)  # (N, B, T)
