"""Loss/regularizer correctness on toy tensors."""

from __future__ import annotations

import math

import torch

from softmoe.models.expert_tokens import ExpertTokenBank
from softmoe.training.losses import (
    causal_lm_loss,
    load_balance_loss,
    router_loss,
    switch_aux_loss,
    usage_entropy,
)


def test_causal_lm_loss_shapes_and_value():
    B, L, V = 2, 5, 7
    logits = torch.zeros(B, L, V)              # uniform -> nll = log V
    labels = torch.randint(0, V, (B, L))
    scalar = causal_lm_loss(logits, labels, "mean")
    assert torch.isfinite(scalar)
    assert abs(scalar.item() - math.log(V)) < 1e-4
    per_ex = causal_lm_loss(logits, labels, "per_example")
    assert per_ex.shape == (B,)


def test_causal_lm_loss_ignores_minus_100():
    B, L, V = 1, 4, 5
    logits = torch.randn(B, L, V)
    labels = torch.full((B, L), -100)
    # no valid targets -> loss defined as 0 (clamped denominator)
    assert causal_lm_loss(logits, labels, "mean").item() == 0.0


def test_separation_zero_for_identical_experts():
    bank = ExpertTokenBank(n_experts=4, tokens_per_expert=1, d_model=8, init="random")
    with torch.no_grad():
        bank.embeddings[:] = bank.embeddings[0:1]   # all identical
    sep = bank.separation_loss("cosine")
    assert abs(sep.item() - 1.0) < 1e-5             # cosine sim of identical vecs = 1


def test_separation_orthogonal_init_is_separated():
    bank = ExpertTokenBank(n_experts=4, tokens_per_expert=1, d_model=16, init="orthogonal")
    sep = bank.separation_loss("cosine")
    assert sep.item() < 0.5                          # near-orthogonal -> low mean cosine


def test_load_balance_entropy():
    # uniform usage -> max entropy -> load_balance_loss = -log K
    K = 4
    r = torch.full((8, K), 1.0 / K)
    assert abs(load_balance_loss(r).item() + math.log(K)) < 1e-5
    assert abs(usage_entropy(r) - math.log(K)) < 1e-5
    # collapsed usage -> entropy 0
    r2 = torch.zeros(8, K); r2[:, 0] = 1.0
    assert abs(usage_entropy(r2)) < 1e-5


def test_switch_aux_and_router_loss_finite():
    K = 3
    r = torch.softmax(torch.randn(6, K), dim=-1)
    ids = r.argmax(dim=-1)
    assert torch.isfinite(switch_aux_loss(r, ids))
    logits = torch.randn(6, K)
    target = torch.nn.functional.one_hot(ids, K).float()
    assert torch.isfinite(router_loss(logits, target))
