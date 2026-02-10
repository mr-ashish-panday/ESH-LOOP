"""
Entropy Router with Halt Decision
===================================
The same router that decides SSM vs Attention allocation
also decides whether to ponder (re-process) a token.

Key insight: α serves dual purpose:
  1. Blending ratio: x_out = (1-α)*SSM(x) + α*Attn(x)
  2. Halt signal: if α < halt_threshold → stop pondering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyRouter(nn.Module):
    """
    Soft Entropy Router with Halt Decision.

    Outputs:
        α: [B, L, 1] attention blending ratio (0=SSM, 1=Attn)
        halt_prob: [B, L, 1] probability of halting at this step
    """

    def __init__(self, d_model: int, halt_threshold: float = 0.3):
        super().__init__()

        # Complexity scoring network (from ESH Paper 1)
        self.complexity_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, 1),
        )

        # Learnable temperature for sigmoid sharpness
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

        # Halt decision network
        # Maps current state + ponder step → halt probability
        self.halt_net = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 4),  # +1 for ponder step encoding
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        self.halt_threshold = halt_threshold
        self._last_alpha = None
        self._last_halt_prob = None

    def forward(self, x: torch.Tensor, ponder_step: int = 0, **kwargs):
        """
        Args:
            x: [B, L, D] input hidden states
            ponder_step: current pondering iteration (0-indexed)

        Returns:
            alpha: [B, L, 1] attention blending ratio
            halt_prob: [B, L, 1] halt probability
        """
        B, L, D = x.shape

        # Compute complexity score → α
        temperature = torch.exp(self.log_temperature).clamp(0.1, 10.0)
        complexity = self.complexity_net(x)  # [B, L, 1]
        alpha = torch.sigmoid(complexity / temperature)

        # Compute halt probability
        # Encode ponder step as a scalar feature
        step_encoding = torch.full((B, L, 1), ponder_step / 3.0,
                                   device=x.device, dtype=x.dtype)
        halt_input = torch.cat([x, step_encoding], dim=-1)  # [B, L, D+1]
        halt_prob = self.halt_net(halt_input)  # [B, L, 1]

        # Store for stats
        self._last_alpha = alpha.detach()
        self._last_halt_prob = halt_prob.detach()

        return alpha, halt_prob

    def get_routing_stats(self, x: torch.Tensor) -> dict:
        """Return routing statistics for logging."""
        if self._last_alpha is not None:
            avg_alpha = self._last_alpha.mean().item()
        else:
            avg_alpha = 0.5

        if self._last_halt_prob is not None:
            avg_halt = self._last_halt_prob.mean().item()
        else:
            avg_halt = 0.5

        return {
            "alpha_mean": avg_alpha,
            "attention_ratio": avg_alpha,
            "halt_prob": avg_halt,
            "temperature": torch.exp(self.log_temperature).item(),
        }

    def should_halt(self, halt_prob: torch.Tensor) -> bool:
        """Check if all tokens want to halt (batch-level decision)."""
        return (halt_prob > self.halt_threshold).all().item()
