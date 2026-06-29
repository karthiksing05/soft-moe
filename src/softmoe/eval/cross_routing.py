"""Expert×domain cross-routing analysis — the full generalization of the swap test.

For a trained model with per-document experts (``SoftMoE`` expert tokens, or ``c-BTM`` per-cluster
models), route **every domain's test set through every expert** and measure per-domain perplexity.
The resulting ``[domain × expert]`` matrix shows whether each domain is modelled best by *its own*
expert and worse by the others — i.e. whether the experts carry genuine, domain-specific expertise.

Reports, per domain: the perplexity under its own (routed/cluster) expert vs the mean over the
other experts, and whether the lowest-perplexity expert is the matched one. The diagonal-vs-
off-diagonal gap is the headline specialization number.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from softmoe.data.dataset import Collator
from softmoe.models.baselines.cbtm import CBTM
from softmoe.models.soft_moe import SoftMoE
from softmoe.training.losses import causal_lm_loss


def _n_experts(model) -> int:
    if isinstance(model, SoftMoE):
        return model.tokens.n_experts
    if isinstance(model, CBTM):
        return model.n_experts
    raise ValueError("cross-routing needs per-document experts (SoftMoE or c-BTM).")


def _forced_expert_nll(model, batch, k: int) -> torch.Tensor:
    """Per-example NLL when every input in the batch is forced through expert ``k``."""
    if isinstance(model, SoftMoE):
        ids = torch.full_like(batch["domain_id"], k)
        per_ex, _ = model._single_expert_forward(batch, ids)
        return per_ex
    # c-BTM: expert k is its own backbone
    out = model.experts[k](input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return causal_lm_loss(out.logits, batch["labels"], reduction="per_example")


@torch.no_grad()
def expert_domain_matrix(model, dataset, device: str = "cpu", batch_size: int = 8,
                         pad_token_id: int = 0) -> dict:
    model.eval(); model.to(device)
    K = _n_experts(model)
    D = dataset.n_domains
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=Collator(pad_token_id))

    sum_nll = np.zeros((D, K)); sum_tok = np.zeros((D, K))
    assign_counts = np.zeros((D, K))   # which expert each domain's blocks actually route to

    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        labels = batch["labels"]
        ntok = (labels[:, 1:] != -100).sum(dim=1).clamp(min=1).float()
        dom = batch["domain_id"].cpu().numpy()

        if isinstance(model, SoftMoE):
            assigned = model.route(batch).expert_ids.cpu().numpy()
        else:
            assigned = batch["cluster_id"].clamp(max=K - 1).cpu().numpy()
        for b in range(len(dom)):
            assign_counts[dom[b], assigned[b]] += 1

        for k in range(K):
            per_ex = _forced_expert_nll(model, batch, k)
            nll = (per_ex * ntok).cpu().numpy()
            t = ntok.cpu().numpy()
            for b in range(len(dom)):
                sum_nll[dom[b], k] += nll[b]
                sum_tok[dom[b], k] += t[b]

    ppl = np.exp(np.minimum(sum_nll / np.maximum(sum_tok, 1.0), 20.0))
    self_expert = assign_counts.argmax(axis=1)            # matched expert per domain

    per_domain = {}
    self_ppls, other_means, matched = [], [], []
    for d in range(D):
        se = int(self_expert[d])
        self_ppl = float(ppl[d, se])
        others = [ppl[d, k] for k in range(K) if k != se]
        other_mean = float(np.mean(others)) if others else float("nan")
        best = int(np.argmin(ppl[d]))
        per_domain[d] = {
            "self_expert": se,
            "self_ppl": self_ppl,
            "other_mean_ppl": other_mean,
            "best_expert": best,
            "best_is_self": bool(best == se),
            "ratio_other_to_self": float(other_mean / self_ppl) if self_ppl > 0 else float("nan"),
            "row_ppl": [float(x) for x in ppl[d]],
        }
        self_ppls.append(self_ppl); other_means.append(other_mean); matched.append(best == se)

    return {
        "n_experts": K,
        "n_domains": D,
        "ppl_matrix": ppl.round(3).tolist(),            # [domain x expert]
        "self_expert_per_domain": [int(x) for x in self_expert],
        "per_domain": per_domain,
        "summary": {
            "mean_self_ppl": float(np.mean(self_ppls)),
            "mean_other_ppl": float(np.mean(other_means)),
            "specialization_gap_ratio": float(np.mean(other_means) / np.mean(self_ppls)),
            "frac_domains_best_is_self": float(np.mean(matched)),
        },
    }


def render_markdown(report: dict, domain_names: list[str], method: str) -> str:
    K = report["n_experts"]
    ppl = report["ppl_matrix"]
    self_e = report["self_expert_per_domain"]
    L = [
        f"### {method}",
        "",
        "Perplexity of each **domain** (row) routed through each **expert** (column). "
        "`*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity "
        "expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well "
        "below the rest.",
        "",
        "| domain ↓ / expert → | " + " | ".join(f"e{k}" for k in range(K)) + " | self-ppl | other-mean | ×worse |",
        "|" + "---|" * (K + 4),
    ]
    for d, name in enumerate(domain_names):
        se = self_e[d]
        best = int(np.argmin(ppl[d]))
        cells = []
        for k in range(K):
            v = f"{ppl[d][k]:.1f}"
            if k == best:
                v = f"**{v}**"
            if k == se:
                v = f"{v}*"
            cells.append(v)
        pd = report["per_domain"][d]
        L.append(f"| {name} | " + " | ".join(cells) +
                 f" | {pd['self_ppl']:.1f} | {pd['other_mean_ppl']:.1f} | {pd['ratio_other_to_self']:.2f}× |")
    s = report["summary"]
    L += ["",
          f"**Summary:** mean self-expert ppl {s['mean_self_ppl']:.1f} vs mean other-expert ppl "
          f"{s['mean_other_ppl']:.1f} → **{s['specialization_gap_ratio']:.2f}× worse** through the wrong "
          f"expert; lowest-ppl expert is the matched one for "
          f"**{s['frac_domains_best_is_self']*100:.0f}%** of domains.", ""]
    return "\n".join(L)
