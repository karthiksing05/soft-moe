"""c-BTM baseline — N separate expert *models*, one per cluster (the primary baseline).

Trains via the *same* trainer: each example is routed to its cluster's backbone (so each model
specializes on its cluster). At eval, supports both oracle routing (per-cluster model) and the
c-BTM ensemble: a cluster-posterior-weighted mixture ``p(x)=Σ_k w_k p_k(x)`` combined in log-space.
This is the "×N models" cost our single-backbone token bank is measured against.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from softmoe.training.losses import causal_lm_loss


class CBTM(nn.Module):
    def __init__(self, backbones: list[nn.Module], route_by: str = "cluster", ensemble: bool = False):
        super().__init__()
        self.experts = nn.ModuleList(backbones)
        self.n_experts = len(backbones)
        self.route_by = route_by
        self.ensemble = ensemble

    def _key(self) -> str:
        return "domain_id" if self.route_by == "domain" else "cluster_id"

    def forward(self, batch, em_hard: bool = False) -> dict:
        ids = batch[self._key()].clamp(max=self.n_experts - 1)
        if self.ensemble:
            return self._ensemble_forward(batch)

        B = ids.shape[0]
        per_ex = torch.zeros(B, device=ids.device)
        logits_full = None
        for k in range(self.n_experts):
            sel = ids == k
            if not sel.any():
                continue
            sub = {key: val[sel] for key, val in batch.items()
                   if isinstance(val, torch.Tensor)}
            out = self.experts[k](input_ids=sub["input_ids"], attention_mask=sub["attention_mask"])
            per_ex[sel] = causal_lm_loss(out.logits, sub["labels"], reduction="per_example")
            if logits_full is None:
                logits_full = torch.zeros(B, *out.logits.shape[1:], device=ids.device)
            logits_full[sel] = out.logits
        return {
            "loss": per_ex.mean(),
            "logits": logits_full,
            "per_example_nll": per_ex,
            "aux": {"route_info": None},
        }

    def _ensemble_forward(self, batch) -> dict:
        # Mixture over all experts: NLL_mix = -log Σ_k w_k exp(-nll_k), in log-space.
        B = batch["input_ids"].shape[0]
        post = batch.get("cluster_posterior")
        if post is None:
            logw = torch.full((B, self.n_experts), -torch.log(torch.tensor(float(self.n_experts))),
                              device=batch["input_ids"].device)
        else:
            logw = (post.clamp(min=1e-9)).log()
        nll = torch.stack(
            [causal_lm_loss(self.experts[k](input_ids=batch["input_ids"],
                                            attention_mask=batch["attention_mask"]).logits,
                            batch["labels"], reduction="per_example")
             for k in range(self.n_experts)],
            dim=1,
        )                                                       # [B, K]
        per_ex = -torch.logsumexp(logw - nll, dim=1)
        return {"loss": per_ex.mean(), "logits": None, "per_example_nll": per_ex, "aux": {"route_info": None}}

    def num_added_trainable_params(self) -> int:
        # All but the first model count as "added" relative to a single dense backbone.
        per = sum(p.numel() for p in self.experts[0].parameters())
        return per * (self.n_experts - 1)
