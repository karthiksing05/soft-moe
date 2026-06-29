"""Metric math on synthetic cases: perfect routing -> NMI=1, collapse -> util-entropy=0,
identical experts -> separation distance=0."""

from __future__ import annotations

import numpy as np
import torch

from softmoe.eval.specialization import (
    contingency_matrix,
    routing_metrics,
    token_separation,
    utilization_metrics,
)


def test_perfect_routing_nmi_one():
    true = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([2, 2, 0, 0, 1, 1])           # permuted but perfectly aligned
    m = routing_metrics(pred, true)
    assert abs(m["nmi"] - 1.0) < 1e-6
    assert abs(m["routing_accuracy"] - 1.0) < 1e-6
    assert abs(m["purity"] - 1.0) < 1e-6


def test_random_routing_low_nmi():
    rng = np.random.default_rng(0)
    true = rng.integers(0, 3, size=300)
    pred = rng.integers(0, 3, size=300)
    assert routing_metrics(pred, true)["nmi"] < 0.1


def test_collapse_utilization_entropy_zero():
    pred = np.zeros(20, dtype=int)                # all to expert 0
    u = utilization_metrics(pred, n_experts=4)
    assert abs(u["utilization_entropy"]) < 1e-9
    assert u["dead_experts"] == 3


def test_uniform_utilization_entropy_max():
    pred = np.array([0, 1, 2, 3] * 5)
    u = utilization_metrics(pred, n_experts=4)
    assert abs(u["utilization_entropy_norm"] - 1.0) < 1e-6
    assert u["dead_experts"] == 0


def test_identical_experts_zero_separation():
    emb = torch.randn(1, 1, 8).expand(5, 1, 8).contiguous()   # all experts identical
    s = token_separation(emb)
    assert abs(s["mean_pairwise_cosine_distance"]) < 1e-5


def test_orthogonal_experts_high_separation():
    emb = torch.eye(4).reshape(4, 1, 4)            # orthonormal experts
    s = token_separation(emb)
    assert s["mean_pairwise_cosine_distance"] > 0.9
    assert s["effective_rank"] > 3.5               # near full rank


def test_contingency_shape():
    pred = np.array([0, 1, 0, 1])
    dom = np.array([0, 1, 0, 1])
    m = contingency_matrix(pred, dom, n_experts=2, n_domains=2)
    assert m.shape == (2, 2)
    assert m.diagonal().sum() == 4
