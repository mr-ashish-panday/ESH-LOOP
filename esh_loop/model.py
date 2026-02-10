"""
ESH-Loop Model
==============
Full language model with adaptive-depth ESH-Loop blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

from .layers import ESHLoopBlock


@dataclass
class ESHLoopConfig:
    """Configuration for ESH-Loop model."""
    vocab_size: int = 50257  # GPT-2 vocab
    d_model: int = 768
    n_layers: int = 8
    n_heads: int = 12
    n_experts: int = 8
    expert_dim: int = 3072
    max_seq_len: int = 2048
    max_ponder_steps: int = 3    # Max re-processing iterations
    dropout: float = 0.1
    layer_scale_init: float = 1e-5
    ponder_cost_weight: float = 0.01  # λ for ponder regularization
    use_checkpoint: bool = False


class ESHLoopModel(nn.Module):
    """
    ESH-Loop Language Model.

    Each layer can re-process tokens adaptively:
    - Easy tokens: 1 pass (fast)
    - Hard tokens: up to max_ponder_steps passes (thorough)
    """

    def __init__(self, config: ESHLoopConfig):
        super().__init__()
        self.config = config

        # Token + Position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_drop = nn.Dropout(config.dropout)

        # ESH-Loop blocks
        self.blocks = nn.ModuleList([
            ESHLoopBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_experts=config.n_experts,
                expert_dim=config.expert_dim,
                max_ponder_steps=config.max_ponder_steps,
                dropout=config.dropout,
                layer_scale_init=config.layer_scale_init,
            )
            for _ in range(config.n_layers)
        ])

        # Output head
        self.norm_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        # Causal mask (cached)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels=None,
                return_routing_stats: bool = False):
        """
        Args:
            input_ids: [B, L] token IDs
            labels: [B, L] target token IDs (optional)
            return_routing_stats: if True, include routing + ponder stats

        Returns:
            dict with 'logits', 'loss', 'ponder_cost', 'routing_stats'
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(L, device=device).unsqueeze(0)
        x = self.emb_drop(self.tok_emb(input_ids) + self.pos_emb(positions))

        # Causal mask
        mask = ~self.causal_mask[:L, :L].unsqueeze(0)  # [1, L, L]

        # Process through ESH-Loop blocks
        routing_stats = []
        total_ponder_cost = 0.0
        total_ponder_steps = 0.0

        for block in self.blocks:
            x, ponder_info = block(x, mask=mask)
            total_ponder_cost += ponder_info["ponder_cost"]
            total_ponder_steps += ponder_info["avg_ponder_steps"]

            if return_routing_stats:
                stats = block.get_routing_stats()
                stats.update(ponder_info)
                routing_stats.append(stats)

        # Output
        x = self.norm_f(x)
        logits = self.lm_head(x)

        # Loss computation
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Ponder cost (regularization to prevent overthinking)
        avg_ponder_cost = total_ponder_cost / len(self.blocks)
        avg_ponder_steps = total_ponder_steps / len(self.blocks)

        # MoE load balance loss
        moe_loss = sum(
            block.moe.load_balance_loss for block in self.blocks
        ) / len(self.blocks)

        # Total auxiliary loss
        aux_loss = (self.config.ponder_cost_weight * avg_ponder_cost +
                    0.01 * moe_loss)

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_loss,
            "ponder_cost": avg_ponder_cost,
            "avg_ponder_steps": avg_ponder_steps,
            "routing_stats": routing_stats,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
