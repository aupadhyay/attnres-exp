"""
Mixture-of-Experts (MoE) layer.
Drop-in replacement for MLP with top-k expert routing.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class MoERouter(nn.Module):
    """Top-k gating router for MoE."""

    def __init__(self, n_embd, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(n_embd, num_experts, bias=False)

    def forward(self, x):
        # x: (B, T, d) -> logits: (B, T, num_experts)
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        # Renormalize top-k probs
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        return top_k_probs, top_k_indices, probs


class MoEMLP(nn.Module):
    """
    Mixture-of-Experts MLP layer.

    Replaces the single MLP with N expert MLPs, routed via top-k selection.
    Includes load-balancing auxiliary loss.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.moe_top_k
        self.aux_loss_coeff = config.moe_aux_loss_coeff

        self.router = MoERouter(config.n_embd, config.num_experts, config.moe_top_k)

        # N expert MLPs (same architecture as original MLP)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
                nn.GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
                nn.Dropout(config.dropout),
            )
            for _ in range(config.num_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, T, d)
        Returns:
            output: (B, T, d)
            aux_loss: scalar load-balancing loss
        """
        B, T, d = x.shape
        x_flat = x.view(-1, d)  # (B*T, d)

        top_k_probs, top_k_indices, all_probs = self.router(x)
        # top_k_probs: (B, T, k), top_k_indices: (B, T, k), all_probs: (B, T, N)

        top_k_probs_flat = top_k_probs.view(-1, self.top_k)  # (B*T, k)
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)  # (B*T, k)

        # Compute expert outputs (simple loop — fine for small num_experts)
        output = torch.zeros_like(x_flat)  # (B*T, d)
        for k_idx in range(self.top_k):
            expert_indices = top_k_indices_flat[:, k_idx]  # (B*T,)
            weights = top_k_probs_flat[:, k_idx]  # (B*T,)
            for e_idx in range(self.num_experts):
                mask = (expert_indices == e_idx)
                if mask.any():
                    expert_input = x_flat[mask]  # (num_tokens, d)
                    expert_output = self.experts[e_idx](expert_input)
                    output[mask] += weights[mask].unsqueeze(-1) * expert_output

        output = output.view(B, T, d)

        # Load-balancing auxiliary loss: alpha * N * sum(f_i * p_i)
        # f_i = fraction of tokens routed to expert i
        # p_i = mean routing probability for expert i
        if self.training:
            # Count tokens per expert (from top-k selections)
            expert_counts = torch.zeros(self.num_experts, device=x.device)
            for k_idx in range(self.top_k):
                for e_idx in range(self.num_experts):
                    expert_counts[e_idx] += (top_k_indices_flat[:, k_idx] == e_idx).float().sum()
            f = expert_counts / (B * T * self.top_k)  # fraction per expert
            p = all_probs.view(-1, self.num_experts).mean(dim=0)  # mean prob per expert
            aux_loss = self.aux_loss_coeff * self.num_experts * (f * p).sum()
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        return output, aux_loss

    def get_routing_stats(self, x):
        """Return routing statistics for analysis (no grad)."""
        with torch.no_grad():
            top_k_probs, top_k_indices, all_probs = self.router(x)
            return {
                'top_k_probs': top_k_probs,
                'top_k_indices': top_k_indices,
                'all_probs': all_probs,
            }
