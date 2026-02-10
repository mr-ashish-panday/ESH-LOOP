"""
ESH-Loop Block: Adaptive-Depth Processing
==========================================
Each block can re-process tokens multiple times based on
the entropy router's complexity assessment.

Architecture per ponder step:
    x → Router → α, halt_prob
    x → (1-α) * SSM(x) + α * Attention(x)
    if halt_prob > threshold: break
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .router import EntropyRouter


# ─── Gated Multi-Head Attention ───────────────────────────────────

class GatedAttention(nn.Module):
    """Multi-head attention with output gating for stability."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)

        # Gated output
        gate = torch.sigmoid(self.gate(x))
        return self.out_proj(out) * gate


# ─── Placeholder SSM (Mamba-compatible interface) ─────────────────

class PlaceholderSSM(nn.Module):
    """Lightweight SSM placeholder (replace with Mamba when available)."""

    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.proj_in = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4,
                                padding=3, groups=d_model)
        self.proj_out = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        # Split into gate and signal
        xz = self.proj_in(x)
        x_signal, z = xz.chunk(2, dim=-1)

        # Causal conv1d
        x_conv = self.conv1d(x_signal.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Gate and project
        out = x_conv * torch.sigmoid(z)
        out = self.proj_out(self.dropout(out))
        return out


# ─── MoE Feed-Forward ──────────────────────────────────────────

class ExpertFFN(nn.Module):
    """Single expert feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # SwiGLU gate
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoELayer(nn.Module):
    """Mixture of Experts with top-k routing."""

    def __init__(self, d_model: int, d_ff: int, n_experts: int = 8,
                 top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff, dropout) for _ in range(n_experts)
        ])

        self.load_balance_loss = 0.0

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        flat_x = x.view(-1, D)  # [B*L, D]

        # Gating
        gate_logits = self.gate(flat_x)  # [B*L, n_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Load balancing loss (negative entropy → encourage uniform distribution)
        expert_usage = gate_probs.mean(dim=0)
        self.load_balance_loss = -(expert_usage * torch.log(expert_usage + 1e-8)).sum()

        # Batched expert computation (much faster than Python loops)
        output = torch.zeros_like(flat_x)
        for e_idx in range(self.n_experts):
            # Find all tokens assigned to this expert (across all top-k slots)
            expert_mask = (top_k_indices == e_idx)  # [B*L, top_k]
            token_mask = expert_mask.any(dim=-1)     # [B*L]

            if not token_mask.any():
                continue

            # Get weights for this expert
            weights = (top_k_probs * expert_mask.float()).sum(dim=-1)  # [B*L]
            expert_input = flat_x[token_mask]
            expert_output = self.experts[e_idx](expert_input)
            output[token_mask] += weights[token_mask].unsqueeze(-1) * expert_output

        return output.view(B, L, D)


# ─── LayerScale ──────────────────────────────────────────────────

class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init * torch.ones(dim))

    def forward(self, x):
        return self.gamma * x


# ─── ESH-Loop Block ──────────────────────────────────────────────

class ESHLoopBlock(nn.Module):
    """
    Single ESH-Loop block with adaptive pondering.

    For each token, the block decides:
    1. How to blend SSM/Attention (via α)
    2. Whether to keep pondering (via halt_prob)
    """

    def __init__(self, d_model: int, n_heads: int, n_experts: int = 8,
                 expert_dim: int = 3072, max_ponder_steps: int = 3,
                 dropout: float = 0.1, layer_scale_init: float = 1e-5):
        super().__init__()

        self.max_ponder_steps = max_ponder_steps

        # Sub-layers
        self.router = EntropyRouter(d_model)
        self.attention = GatedAttention(d_model, n_heads, dropout)
        self.ssm = PlaceholderSSM(d_model, dropout=dropout)
        self.moe = MoELayer(d_model, expert_dim, n_experts, top_k=2, dropout=dropout)

        # Norms (one set per sub-layer)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # LayerScale
        self.ls1 = LayerScale(d_model, layer_scale_init)
        self.ls2 = LayerScale(d_model, layer_scale_init)

        # Ponder tracking
        self._ponder_steps_used = None
        self._ponder_cost = 0.0

    def forward(self, x: torch.Tensor, mask=None,
                return_ponder_info: bool = False):
        """
        Args:
            x: [B, L, D] input
            mask: optional attention mask
            return_ponder_info: if True, return detailed ponder stats

        Returns:
            x: [B, L, D] output
            ponder_info: dict with ponder statistics (if requested)
        """
        B, L, D = x.shape
        device = x.device

        # Accumulated output and tracking
        accumulated = torch.zeros_like(x)
        halt_cumulative = torch.zeros(B, L, 1, device=device)
        remainder = torch.ones(B, L, 1, device=device)
        ponder_steps = torch.zeros(B, L, device=device)
        total_ponder_cost = 0.0
        residual = x

        for step in range(self.max_ponder_steps):
            # Route: get α and halt probability
            normed = self.norm1(x)
            alpha, halt_prob = self.router(normed, ponder_step=step)

            # Blend SSM and Attention
            ssm_out = self.ssm(normed)
            attn_out = self.attention(normed, mask=mask)
            blended = (1 - alpha) * ssm_out + alpha * attn_out
            blended = self.ls1(blended)

            # MoE feed-forward
            moe_out = self.moe(self.norm2(x + blended))
            step_output = x + blended + self.ls2(moe_out)

            # Adaptive halting (ACT-style)
            if step == self.max_ponder_steps - 1:
                # Last step: use remainder
                weight = remainder
            else:
                weight = halt_prob * remainder

            # Accumulate weighted output
            accumulated = accumulated + weight * step_output

            # Update halt tracking
            halt_cumulative = halt_cumulative + weight
            remainder = remainder - weight
            remainder = remainder.clamp(min=0)

            # Track which tokens are still pondering
            still_pondering = (halt_cumulative < 0.99).float()
            ponder_steps = ponder_steps + still_pondering.squeeze(-1)

            # Ponder cost for this step
            total_ponder_cost += (1.0 - halt_prob.mean())

            # Early exit if all tokens have halted
            if (halt_cumulative > 0.99).all():
                break

        # Store ponder stats
        self._ponder_steps_used = ponder_steps.detach()
        self._ponder_cost = total_ponder_cost / self.max_ponder_steps

        x = accumulated + residual  # Residual connection

        ponder_info = {
            "avg_ponder_steps": ponder_steps.mean().item(),
            "max_ponder_steps": ponder_steps.max().item(),
            "ponder_cost": self._ponder_cost.item() if torch.is_tensor(self._ponder_cost) else self._ponder_cost,
        }

        if return_ponder_info:
            return x, ponder_info

        return x, ponder_info

    def get_routing_stats(self) -> dict:
        """Get routing statistics from last forward pass."""
        stats = self.router.get_routing_stats(None)
        if self._ponder_steps_used is not None:
            stats["avg_ponder_steps"] = self._ponder_steps_used.mean().item()
            stats["max_ponder_steps"] = self._ponder_steps_used.max().item()
        return stats
