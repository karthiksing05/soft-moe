"""Dense baseline — single backbone, no expert tokens, no router. The lower bound."""

from __future__ import annotations

import torch
import torch.nn as nn

from softmoe.training.losses import causal_lm_loss


class Dense(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, batch, em_hard: bool = False) -> dict:
        out = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        per_ex = causal_lm_loss(out.logits, batch["labels"], reduction="per_example")
        return {
            "loss": per_ex.mean(),
            "logits": out.logits,
            "per_example_nll": per_ex,
            "aux": {"route_info": None},
        }

    def num_added_trainable_params(self) -> int:
        return 0

    def num_active_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
