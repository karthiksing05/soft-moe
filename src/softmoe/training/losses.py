"""Losses & regularizers for the EM objective.

    L = L_lm  +  λ_sep · L_separation  −  λ_bal · H_load  (+ λ_route · L_route)

Separation is the higher-priority regularizer (brainstorm: push tokens apart > load-balance),
so configs default to ``λ_sep > λ_bal``. Every term is computed here, returned separately so the
trainer can log each, and unit-tested on toy tensors (``tests/test_losses.py``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Standard next-token cross-entropy with HF-style left-shift and ``-100`` ignore.

    - ``mean``        -> scalar over all valid tokens.
    - ``per_example`` -> ``[B]`` mean NLL per sequence (used for soft-EM weighting & per-domain ppl).
    - ``per_token``   -> ``[B, L-1]`` token NLL (``0`` where ignored).
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    B, Lm1, V = shift_logits.shape
    tok = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(B, Lm1)
    valid = (shift_labels != -100).float()
    if reduction == "mean":
        return tok.sum() / valid.sum().clamp(min=1.0)
    if reduction == "per_example":
        return tok.sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
    if reduction == "per_token":
        return tok * valid
    raise ValueError(f"Unknown reduction '{reduction}'.")


def load_balance_loss(responsibilities: torch.Tensor) -> torch.Tensor:
    """Negative entropy of mean batch usage. Minimizing this *maximizes* usage entropy
    (encourages uniform expert usage). Returns a value in ``[-log K, 0]``."""
    usage = responsibilities.mean(dim=0)               # [K]
    usage = usage / usage.sum().clamp(min=1e-9)
    entropy = -(usage * (usage + 1e-9).log()).sum()
    return -entropy


def switch_aux_loss(responsibilities: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
    """Switch-Transformer auxiliary balance loss: ``K · Σ_k f_k · P_k`` (minimize)."""
    K = responsibilities.shape[1]
    frac = torch.bincount(expert_ids, minlength=K).float() / max(1, len(expert_ids))
    prob = responsibilities.mean(dim=0)
    return K * (frac * prob).sum()


def router_loss(logits: torch.Tensor, target_responsibilities: torch.Tensor) -> torch.Tensor:
    """Cross-entropy training the amortized router toward the (soft) EM responsibilities."""
    logp = F.log_softmax(logits, dim=-1)
    return -(target_responsibilities.detach() * logp).sum(dim=-1).mean()


def usage_entropy(responsibilities: torch.Tensor) -> float:
    usage = responsibilities.detach().mean(dim=0)
    usage = usage / usage.sum().clamp(min=1e-9)
    return float(-(usage * (usage + 1e-9).log()).sum())


def combine_losses(out: dict, lambdas: dict) -> tuple[torch.Tensor, dict]:
    """Combine the model's aux losses into the total objective, returning (total, logs)."""
    aux = out["aux"]
    total = out["loss"]
    logs = {"lm": float(out["loss"].detach())}

    sep = aux.get("separation")
    if sep is not None and lambdas.get("sep", 0.0):
        total = total + lambdas["sep"] * sep
        logs["sep"] = float(sep.detach())

    bal = aux.get("load_balance")
    if bal is not None and lambdas.get("balance", 0.0):
        total = total + lambdas["balance"] * bal
        logs["balance"] = float(bal.detach())

    route = aux.get("router")
    if route is not None and lambdas.get("route", 0.0):
        total = total + lambdas["route"] * route
        logs["route"] = float(route.detach())

    logs["total"] = float(total.detach())
    if "utilization_entropy" in aux:
        logs["util_entropy"] = float(aux["utilization_entropy"])
    return total, logs
