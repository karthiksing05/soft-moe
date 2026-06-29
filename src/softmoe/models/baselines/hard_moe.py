"""Hard MoE baseline — classic top-k token-routed FFN MoE. The (parameter-heavy) upper bound.

We replace the backbone's FFN blocks with a minimal from-scratch token-routed MoE FFN (not a
full MoE library) and add the Switch-Transformer load-balance aux loss. This is the reference
that *does* add parameters (separate FFNs) — the cost our prompt-carried experts avoid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from softmoe.training.losses import causal_lm_loss


class MoEFFN(nn.Module):
    """Top-k token-routed MoE feed-forward, drop-in for a GPT-2 block's ``.mlp``."""

    def __init__(self, d_model: int, d_inner: int, n_experts: int, top_k: int = 1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, d_inner), nn.GELU(), nn.Linear(d_inner, d_model))
             for _ in range(n_experts)]
        )
        self.last_aux: dict = {}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, L, d = hidden_states.shape
        x = hidden_states.reshape(-1, d)                        # [N, d]
        logits = self.gate(x)                                   # [N, K]
        probs = F.softmax(logits, dim=-1)
        topw, topi = probs.topk(self.top_k, dim=-1)             # [N, k]
        topw = topw / topw.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        out = torch.zeros_like(x)
        for j in range(self.top_k):
            idx = topi[:, j]                                    # [N]
            w = topw[:, j].unsqueeze(-1)                        # [N, 1]
            for k in range(self.n_experts):
                sel = idx == k
                if sel.any():
                    out[sel] += w[sel] * self.experts[k](x[sel])

        # Switch aux: K * Σ_k frac_k * mean_prob_k
        top1 = topi[:, 0]
        frac = torch.bincount(top1, minlength=self.n_experts).float() / max(1, top1.numel())
        self.last_aux = {"frac": frac.detach(), "prob": probs.mean(0), "raw_aux":
                         self.n_experts * (frac.detach() * probs.mean(0)).sum()}
        return out.reshape(B, L, d)


class HardMoE(nn.Module):
    def __init__(self, backbone, n_experts: int = 4, top_k: int = 1):
        super().__init__()
        self.backbone = backbone
        self.moe_layers: list[MoEFFN] = []
        self._inject(n_experts, top_k)

    def _inject(self, n_experts: int, top_k: int) -> None:
        blocks = self._transformer_blocks()
        for block in blocks:
            mlp = block.mlp
            d_model = self.backbone.config.n_embd
            # infer inner dim from the existing fc layer (handles Conv1D and Linear)
            d_inner = getattr(self.backbone.config, "n_inner", None) or 4 * d_model
            moe = MoEFFN(d_model, int(d_inner), n_experts, top_k)
            block.mlp = moe
            self.moe_layers.append(moe)

    def _transformer_blocks(self):
        # GPT-2 style: model.transformer.h
        tr = getattr(self.backbone, "transformer", None)
        if tr is not None and hasattr(tr, "h"):
            return list(tr.h)
        raise AttributeError("HardMoE currently supports GPT-2-style backbones (transformer.h).")

    def forward(self, batch, em_hard: bool = False) -> dict:
        out = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        per_ex = causal_lm_loss(out.logits, batch["labels"], reduction="per_example")
        aux_balance = sum(m.last_aux.get("raw_aux", 0.0) for m in self.moe_layers)
        return {
            "loss": per_ex.mean(),
            "logits": out.logits,
            "per_example_nll": per_ex,
            "aux": {"route_info": None, "load_balance": aux_balance},
        }

    def num_added_trainable_params(self) -> int:
        return sum(sum(p.numel() for p in m.parameters()) for m in self.moe_layers)
