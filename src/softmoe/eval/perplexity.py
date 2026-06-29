"""Language-modeling metrics: per-domain perplexity, macro/micro averages, oracle-vs-routed.

Macro-average (equal weight per domain) is the headline — it rewards specialization on small
domains. Oracle routing feeds each input its ground-truth/cluster expert (the token bank's upper
bound); learned routing lets the router pick (the deployment number). The gap measures router
quality independently of expert quality.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from softmoe.data.dataset import Collator
from softmoe.models.soft_moe import SoftMoE
from softmoe.training.losses import causal_lm_loss


def _valid_token_counts(labels: torch.Tensor) -> torch.Tensor:
    return (labels[:, 1:] != -100).sum(dim=1).clamp(min=1).float()


def _per_example_nll(model, batch, route_mode: str) -> torch.Tensor:
    """Per-example mean NLL under a routing mode.

    - ``oracle``  : SoftMoE forced to each input's ground-truth expert (cluster/domain).
    - ``learned`` : the model's own router (or no router for dense/hard-moe/cbtm).
    """
    if isinstance(model, SoftMoE) and route_mode == "oracle":
        key = "domain_id" if model.router_supervise_with == "domain" else "cluster_id"
        ids = batch[key].clamp(max=model.tokens.n_experts - 1)
        per_ex, _ = model._single_expert_forward(batch, ids)
        return per_ex
    out = model(batch)
    return out["per_example_nll"]


@torch.no_grad()
def per_domain_perplexity(
    model, dataset, device: str = "cpu", route_mode: str = "learned",
    batch_size: int = 8, pad_token_id: int = 0, n_domains: int | None = None,
) -> dict:
    model.eval(); model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=Collator(pad_token_id))
    n_dom = n_domains or dataset.n_domains
    sum_nll = np.zeros(n_dom); sum_tok = np.zeros(n_dom)
    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        per_ex = _per_example_nll(model, batch, route_mode)        # [B] mean nll
        ntok = _valid_token_counts(batch["labels"]).to(per_ex.device)
        dom = batch["domain_id"].cpu().numpy()
        nll = (per_ex * ntok).cpu().numpy()
        for d, e, t in zip(dom, nll, ntok.cpu().numpy()):
            sum_nll[d] += e; sum_tok[d] += t

    per_domain = {}
    for d in range(n_dom):
        if sum_tok[d] > 0:
            per_domain[int(d)] = float(math.exp(min(sum_nll[d] / sum_tok[d], 20.0)))
    macro = float(np.mean(list(per_domain.values()))) if per_domain else float("nan")
    micro = float(math.exp(min(sum_nll.sum() / max(sum_tok.sum(), 1.0), 20.0)))
    return {"per_domain": per_domain, "macro_ppl": macro, "micro_ppl": micro,
            "route_mode": route_mode}


@torch.no_grad()
def collect_routing(model, dataset, device: str = "cpu", batch_size: int = 8, pad_token_id: int = 0):
    """Return (pred_expert[N], true_domain[N], true_cluster[N]) under the model's learned router."""
    model.eval(); model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=Collator(pad_token_id))
    preds, doms, clus = [], [], []
    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        if isinstance(model, SoftMoE):
            route = model.route(batch)
            preds.append(route.expert_ids.cpu().numpy())
        else:
            preds.append(batch["cluster_id"].cpu().numpy())
        doms.append(batch["domain_id"].cpu().numpy())
        clus.append(batch["cluster_id"].cpu().numpy())
    return (np.concatenate(preds), np.concatenate(doms), np.concatenate(clus))
