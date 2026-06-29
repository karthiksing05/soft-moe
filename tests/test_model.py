"""M2 acceptance: forward runs; grads reach the expert tokens and NOT the frozen backbone;
loss finite; shapes correct under each injection mode."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from softmoe.data.dataset import Collator, SoftMoEDataset
from softmoe.models.factory import build_model
from softmoe.utils.config import Config


def _cfg(injection: str, trainable: bool = True, router_kind: str = "soft") -> Config:
    return Config(
        {
            "model": {
                "method": "softmoe",
                "backbone": {"kind": "tiny", "tiny": {"n_embd": 32, "n_layer": 2, "n_head": 2, "n_inner": 64, "n_positions": 128}},
                "backbone_mode": "frozen",
                "injection": injection,
                "expert_tokens": {"n_experts": "auto", "tokens_per_expert": 2, "init": "random", "trainable": trainable},
                "router": {"kind": router_kind, "top_k": 1, "pool": "meanhidden", "route_by": "cluster"},
                "router_supervise_with": "cluster",
            }
        }
    )


def _batch(synth_corpus, bs=4):
    ds = SoftMoEDataset(synth_corpus, "train")
    loader = DataLoader(ds, batch_size=bs, collate_fn=Collator(pad_token_id=256))
    return next(iter(loader)), ds.n_clusters


def test_forward_prefix_runs(synth_corpus):
    batch, K = _batch(synth_corpus)
    model = build_model(_cfg("prefix"), vocab_size=257, data_n_experts=K)
    out = model(batch)
    assert torch.isfinite(out["loss"])
    assert out["per_example_nll"].shape[0] == batch["input_ids"].shape[0]
    assert out["logits"].shape[:2] == batch["input_ids"].shape       # input-aligned [B, L, V]


def test_forward_prefix_kv_runs(synth_corpus):
    batch, K = _batch(synth_corpus)
    model = build_model(_cfg("prefix_kv"), vocab_size=257, data_n_experts=K)
    out = model(batch)
    assert torch.isfinite(out["loss"])
    assert out["logits"].shape[:2] == batch["input_ids"].shape


def test_grads_reach_tokens_not_frozen_backbone(synth_corpus):
    batch, K = _batch(synth_corpus)
    model = build_model(_cfg("prefix", trainable=True), vocab_size=257, data_n_experts=K)
    out = model(batch)
    out["loss"].backward()
    assert model.tokens.embeddings.grad is not None
    assert model.tokens.embeddings.grad.abs().sum() > 0
    # frozen backbone params must have no grad
    for p in model.backbone.parameters():
        assert p.grad is None


def test_fixed_tokens_have_no_grad(synth_corpus):
    # Fixed orthogonal tokens + frozen backbone: the bare LM loss is non-differentiable (routing
    # is a discrete argmax); the learned router instead trains via the *combined* objective
    # (load-balance + router terms). This is exactly the ours-fixed ablation contract.
    from softmoe.training.losses import combine_losses

    batch, K = _batch(synth_corpus)
    model = build_model(_cfg("prefix", trainable=False), vocab_size=257, data_n_experts=K)
    assert model.tokens.embeddings.requires_grad is False
    out = model(batch)
    total, _ = combine_losses(out, {"sep": 1.0, "balance": 0.1, "route": 0.5})
    total.backward()
    assert model.tokens.embeddings.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.router.parameters())


def test_aux_contains_specialization_signals(synth_corpus):
    batch, K = _batch(synth_corpus)
    model = build_model(_cfg("prefix"), vocab_size=257, data_n_experts=K)
    out = model(batch)
    aux = out["aux"]
    assert torch.isfinite(aux["separation"])
    assert torch.isfinite(aux["load_balance"])
    assert "utilization_entropy" in aux
