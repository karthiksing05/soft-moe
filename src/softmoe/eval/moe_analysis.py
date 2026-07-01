"""MoE expert-utilization-by-domain — the domain analysis for the MoE arm.

For a token-routed ``HardMoE``, capture the top-1 routed expert of every token and aggregate by
its document's domain. Answers: **does the learned MoE router specialize its FFN experts to
domains?** (directly comparable to our expert-token routing-NMI). Also reports the load-balance
diagnostics the recipe (§5.7) asks for: expert usage entropy and dead-expert fraction.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from softmoe.data.dataset import Collator
from softmoe.models.baselines.hard_moe import HardMoE


@torch.no_grad()
def moe_domain_utilization(model, dataset, device: str = "cpu", batch_size: int = 8,
                           pad_token_id: int = 0, max_batches: int = 300) -> dict:
    if not isinstance(model, HardMoE):
        return {}
    model.eval(); model.to(device)
    model.set_capture(True)
    E = model.n_routed
    D = dataset.n_domains
    n_layers = len(model.moe_layers)
    counts = np.zeros((E, D))                       # expert x domain, summed over all layers+tokens
    doc_expert, doc_domain = [], []                 # per-doc dominant expert (all layers) for NMI

    # shuffle so the sampled batches span every domain (the test set is domain-ordered).
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=Collator(pad_token_id))
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        model(batch)
        dom = batch["domain_id"].cpu().numpy()
        Bn = len(dom)
        per_doc_hist = np.zeros((Bn, E))            # expert histogram per doc, summed over layers
        for m in model.moe_layers:
            t1 = m.last_aux["top1_bl"].cpu().numpy()   # [B, L]
            for b in range(Bn):
                vals, cnts = np.unique(t1[b], return_counts=True)
                counts[vals, dom[b]] += cnts
                per_doc_hist[b, vals] += cnts
        for b in range(Bn):
            doc_expert.append(int(per_doc_hist[b].argmax()))
            doc_domain.append(int(dom[b]))
    model.set_capture(False)

    from sklearn.metrics import normalized_mutual_info_score

    usage = counts.sum(axis=1)
    p = usage / max(1.0, usage.sum())
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return {
        "n_routed": E,
        "n_domains": D,
        "n_layers": n_layers,
        "expert_by_domain": counts.astype(int).tolist(),
        "routing_nmi": float(normalized_mutual_info_score(doc_domain, doc_expert)) if doc_domain else 0.0,
        "usage_entropy": entropy,
        "usage_entropy_norm": float(entropy / np.log(E)) if E > 1 else 0.0,
        "dead_experts": int((usage == 0).sum()),
        "dead_fraction": float((usage == 0).mean()),
    }


def render_markdown(report: dict, domain_names: list[str], method: str) -> str:
    if not report:
        return f"### {method}\n\n_(no MoE experts to analyze)_\n"
    E, D = report["n_routed"], report["n_domains"]
    m = np.array(report["expert_by_domain"], dtype=float)
    colsum = m.sum(axis=0, keepdims=True)
    frac = np.divide(m, np.clip(colsum, 1, None))          # per-domain distribution over experts
    L = [
        f"### {method}",
        "",
        f"Learned MoE token-routing — fraction of each **domain**'s tokens sent to each **expert** "
        f"(top-1, summed over {report['n_layers']} layers). **Routing-NMI vs domain: "
        f"{report['routing_nmi']:.3f}** · usage-entropy {report['usage_entropy_norm']:.2f} · "
        f"dead experts {report['dead_experts']}/{E}.",
        "",
    ]
    # show only experts that are actually used, to keep fine-grained matrices readable
    used = [e for e in range(E) if m[e].sum() > 0]
    header = "| domain ↓ / expert → | " + " | ".join(f"e{e}" for e in used) + " |"
    L += [header, "|" + "---|" * (len(used) + 1)]
    for d in range(D):
        name = domain_names[d] if d < len(domain_names) else f"d{d}"
        row = " | ".join(f"{frac[e, d]:.2f}" for e in used)
        L.append(f"| {name} | {row} |")
    L.append("")
    return "\n".join(L)
