"""Specialization metrics — the novel part. Quantify whether experts actually specialize.

All operate on cheap arrays/tensors so they unit-test on synthetic cases (perfect routing→NMI=1,
collapse→utilization-entropy=0, identical experts→separation=0).
"""

from __future__ import annotations

import numpy as np
import torch


def routing_metrics(pred_expert: np.ndarray, true_label: np.ndarray) -> dict:
    """Treat the expert id as a clustering of docs; compare to ground-truth labels."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    pred = np.asarray(pred_expert); true = np.asarray(true_label)
    n_pred = int(pred.max()) + 1 if len(pred) else 0
    n_true = int(true.max()) + 1 if len(true) else 0

    # Hungarian-matched accuracy: best 1-1 expert→label assignment.
    size = max(n_pred, n_true)
    cost = np.zeros((size, size))
    for p, t in zip(pred, true):
        cost[p, t] += 1
    row, col = linear_sum_assignment(-cost)
    matched = cost[row, col].sum()
    acc = float(matched / max(1, len(pred)))

    # purity: each expert assigned its majority label.
    purity = 0.0
    for k in range(n_pred):
        mask = pred == k
        if mask.any():
            purity += np.bincount(true[mask], minlength=n_true).max()
    purity = float(purity / max(1, len(pred)))

    return {
        "routing_accuracy": acc,
        "nmi": float(normalized_mutual_info_score(true, pred)) if len(pred) else 0.0,
        "ari": float(adjusted_rand_score(true, pred)) if len(pred) else 0.0,
        "purity": purity,
    }


def utilization_metrics(pred_expert: np.ndarray, n_experts: int) -> dict:
    counts = np.bincount(np.asarray(pred_expert), minlength=n_experts).astype(float)
    p = counts / counts.sum() if counts.sum() > 0 else counts
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return {
        "utilization_counts": counts.tolist(),
        "utilization_entropy": entropy,
        "utilization_entropy_norm": float(entropy / np.log(n_experts)) if n_experts > 1 else 0.0,
        "dead_experts": int((counts == 0).sum()),
        "dead_fraction": float((counts == 0).mean()),
    }


def token_separation(embeddings: torch.Tensor) -> dict:
    """``embeddings``: [n_experts, T, d] or [n_experts, D]. Mean pairwise cosine *distance* +
    effective rank + log-det volume of the (normalized) token matrix."""
    # Compute on CPU: the token matrix is tiny (K x T*d) and this keeps the helper eye/
    # identity tensors on the same device as E regardless of where the model lives (GPU).
    E = embeddings.detach().reshape(embeddings.shape[0], -1).float().cpu()
    n = E.shape[0]
    if n < 2:
        return {"mean_pairwise_cosine_distance": 0.0, "effective_rank": 1.0, "logdet_volume": 0.0}
    En = torch.nn.functional.normalize(E, dim=1)
    sim = En @ En.t()
    off = sim[~torch.eye(n, dtype=torch.bool)]
    mean_dist = float(1.0 - off.mean().item())

    # effective rank from singular value entropy.
    s = torch.linalg.svdvals(En)
    s = s / s.sum().clamp(min=1e-9)
    eff_rank = float(torch.exp(-(s[s > 0] * (s[s > 0]).log()).sum()).item())

    gram = En @ En.t() + 1e-3 * torch.eye(n)
    logdet = float(torch.logdet(gram).item())
    return {"mean_pairwise_cosine_distance": mean_dist, "effective_rank": eff_rank,
            "logdet_volume": logdet}


def contingency_matrix(pred_expert: np.ndarray, true_domain: np.ndarray,
                       n_experts: int, n_domains: int) -> np.ndarray:
    m = np.zeros((n_experts, n_domains), dtype=int)
    for p, d in zip(np.asarray(pred_expert), np.asarray(true_domain)):
        m[p, d] += 1
    return m


@torch.no_grad()
def swap_test(model, dataset, device: str = "cpu", batch_size: int = 8, pad_token_id: int = 0,
              max_batches: int = 20) -> dict:
    """Route inputs through the *wrong* expert and measure the perplexity increase.

    Large degradation ⇒ tokens carry genuine, non-interchangeable expertise (the strongest
    causal evidence). Only meaningful for SoftMoE; returns NaN for models without an expert bank.
    """
    import math

    from torch.utils.data import DataLoader

    from softmoe.data.dataset import Collator
    from softmoe.models.soft_moe import SoftMoE

    if not isinstance(model, SoftMoE):
        return {"swap_ppl_increase": float("nan"), "correct_ppl": float("nan"), "wrong_ppl": float("nan")}

    model.eval(); model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=Collator(pad_token_id))
    K = model.tokens.n_experts
    key = "domain_id" if model.router_supervise_with == "domain" else "cluster_id"
    correct_nll, wrong_nll, ntok = 0.0, 0.0, 0.0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        ids = batch[key].clamp(max=K - 1)
        wrong = (ids + 1) % K
        c, _ = model._single_expert_forward(batch, ids)
        w, _ = model._single_expert_forward(batch, wrong)
        t = (batch["labels"][:, 1:] != -100).sum(dim=1).clamp(min=1).float()
        correct_nll += float((c * t).sum()); wrong_nll += float((w * t).sum()); ntok += float(t.sum())
    cp = math.exp(min(correct_nll / max(ntok, 1.0), 20.0))
    wp = math.exp(min(wrong_nll / max(ntok, 1.0), 20.0))
    return {"correct_ppl": cp, "wrong_ppl": wp, "swap_ppl_increase": wp - cp,
            "swap_ratio": wp / cp if cp > 0 else float("nan")}
