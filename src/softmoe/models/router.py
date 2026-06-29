"""The router / E-step: which expert does this input use?

- ``SupervisedRouter``  — expert id = ``domain_id`` or ``cluster_id`` (no learned routing).
- ``SoftRouter``        — a tiny head maps a pooled input representation to a distribution over
                          experts; supports EM-hard (argmax responsibilities) and EM-soft
                          (top-k posterior, renormalized). The amortized head is what's used at
                          deployment, when per-expert NLL is unavailable.

The router returns ``(expert_ids, route_info)`` where ``route_info`` carries the full
responsibility matrix and logits so the trainer can compute load-balance and router losses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouteInfo:
    expert_ids: torch.Tensor                 # [B] (hard pick / top-1)
    responsibilities: torch.Tensor           # [B, K] soft posterior (one-hot for supervised)
    logits: torch.Tensor | None = None       # [B, K] router logits (learned router only)
    topk_ids: torch.Tensor | None = None     # [B, k] for soft top-k mixture
    topk_weights: torch.Tensor | None = None  # [B, k] renormalized responsibilities
    extras: dict = field(default_factory=dict)


class SupervisedRouter(nn.Module):
    """Routes by ground-truth label; ``route_by`` in {domain, cluster}."""

    def __init__(self, n_experts: int, route_by: str = "cluster"):
        super().__init__()
        self.n_experts = n_experts
        self.route_by = route_by

    def forward(self, batch, tokens, hidden=None) -> RouteInfo:
        key = "domain_id" if self.route_by == "domain" else "cluster_id"
        ids = batch[key].clamp(max=self.n_experts - 1)
        r = F.one_hot(ids, num_classes=self.n_experts).float()
        return RouteInfo(expert_ids=ids, responsibilities=r, topk_ids=ids.unsqueeze(1),
                         topk_weights=torch.ones(len(ids), 1, device=ids.device))


class SoftRouter(nn.Module):
    """Learned amortized router over a pooled input representation.

    ``pool`` selects the input feature:
    - ``meanhidden`` — mean of provided backbone hidden states (passed in as ``hidden``).
    - ``embedding``  — mean of the router's own token embedding (self-contained, no backbone dep).
    """

    def __init__(self, n_experts: int, d_in: int, top_k: int = 1, pool: str = "meanhidden",
                 vocab_size: int | None = None, temperature: float = 1.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = max(1, int(top_k))
        self.pool = pool
        self.temperature = temperature
        if pool == "embedding":
            assert vocab_size is not None
            self.embed = nn.Embedding(vocab_size, d_in)
        self.head = nn.Sequential(nn.Linear(d_in, d_in), nn.GELU(), nn.Linear(d_in, n_experts))

    def _features(self, batch, hidden) -> torch.Tensor:
        if self.pool == "embedding":
            mask = batch.get("attention_mask")
            e = self.embed(batch["input_ids"])
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                return (e * m).sum(1) / m.sum(1).clamp(min=1.0)
            return e.mean(1)
        if hidden is None:
            raise ValueError("SoftRouter(pool='meanhidden') needs backbone hidden states.")
        mask = batch.get("attention_mask")
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)
        return hidden.mean(1)

    def forward(self, batch, tokens=None, hidden=None, hard: bool = False) -> RouteInfo:
        feats = self._features(batch, hidden)
        logits = self.head(feats) / self.temperature
        post = F.softmax(logits, dim=-1)
        k = min(self.top_k, self.n_experts)
        topk_w, topk_i = post.topk(k, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        expert_ids = topk_i[:, 0]
        if hard:
            r = F.one_hot(expert_ids, num_classes=self.n_experts).float()
        else:
            r = post
        return RouteInfo(expert_ids=expert_ids, responsibilities=r, logits=logits,
                         topk_ids=topk_i, topk_weights=topk_w)


def make_router(cfg, n_experts: int, d_in: int, vocab_size: int | None = None) -> nn.Module:
    kind = cfg.get("kind", "supervised")
    if kind == "supervised":
        return SupervisedRouter(n_experts, route_by=cfg.get("route_by", "cluster"))
    if kind in ("soft", "learned"):
        return SoftRouter(
            n_experts,
            d_in=d_in,
            top_k=int(cfg.get("top_k", 1)),
            pool=cfg.get("pool", "meanhidden"),
            vocab_size=vocab_size,
            temperature=float(cfg.get("temperature", 1.0)),
        )
    raise ValueError(f"Unknown router kind '{kind}' (supervised|soft).")
