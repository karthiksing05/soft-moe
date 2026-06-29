"""Alternating M-step: backbone-phase updates θ only; token-phase updates expert tokens only."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from softmoe.data.dataset import Collator, SoftMoEDataset
from softmoe.models.factory import build_model
from softmoe.training.em_trainer import EMTrainer
from softmoe.training.losses import combine_losses
from softmoe.utils.config import Config


def _full_backbone_cfg() -> Config:
    return Config({
        "model": {
            "method": "softmoe",
            "backbone": {"kind": "tiny", "tiny": {"n_embd": 32, "n_layer": 2, "n_head": 2, "n_inner": 64, "n_positions": 128}},
            "backbone_mode": "full",                       # alternation needs a trainable backbone
            "injection": "prefix",
            "expert_tokens": {"n_experts": "auto", "tokens_per_expert": 2, "init": "random", "trainable": True},
            "router": {"kind": "supervised", "route_by": "cluster"},
            "router_supervise_with": "cluster",
        },
        "train": {"max_steps": 8, "alternation": {"enabled": True, "start": "backbone",
                                                  "backbone_steps": 2, "token_steps": 2,
                                                  "backbone_lr": 1e-3, "token_lr": 1e-1}},
    })


def _trainer(cfg, tmp_path):
    return EMTrainer(cfg, tmp_path / "run", device="cpu")


def _batch(synth_corpus):
    ds = SoftMoEDataset(synth_corpus, "train")
    return next(iter(DataLoader(ds, batch_size=4, collate_fn=Collator(pad_token_id=256)))), ds.n_clusters


def _step(model, groups, trainer, phase, batch):
    trainer._set_phase(groups, phase)
    out = model(batch)
    total, _ = combine_losses(out, {"sep": 1.0})
    total.backward()
    opt = groups["opt_bb"] if phase == "backbone" else groups["opt_tok"]
    opt.step()
    groups["opt_bb"].zero_grad(); groups["opt_tok"].zero_grad()


def test_phase_schedule():
    # start=backbone, 2+2: steps 1,2 backbone; 3,4 tokens; 5,6 backbone; ...
    f = EMTrainer._phase_for_step
    assert [f(s, 2, 2, "backbone") for s in range(1, 9)] == \
        ["backbone", "backbone", "tokens", "tokens", "backbone", "backbone", "tokens", "tokens"]
    assert f(1, 2, 2, "tokens") == "tokens"


def test_set_phase_toggles_requires_grad(synth_corpus, tmp_path):
    cfg = _full_backbone_cfg()
    _, K = _batch(synth_corpus)
    model = build_model(cfg, vocab_size=257, data_n_experts=K)
    trainer = _trainer(cfg, tmp_path)
    groups = trainer._build_alternating(model, 8, dict(cfg["train"]["alternation"]))

    trainer._set_phase(groups, "backbone")
    assert all(p.requires_grad for p in model.backbone.parameters())
    assert not model.tokens.embeddings.requires_grad

    trainer._set_phase(groups, "tokens")
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert model.tokens.embeddings.requires_grad


def test_backbone_phase_updates_only_backbone(synth_corpus, tmp_path):
    cfg = _full_backbone_cfg()
    batch, K = _batch(synth_corpus)
    model = build_model(cfg, vocab_size=257, data_n_experts=K)
    trainer = _trainer(cfg, tmp_path)
    groups = trainer._build_alternating(model, 8, dict(cfg["train"]["alternation"]))

    tok0 = model.tokens.embeddings.detach().clone()
    bb_param = next(p for p in model.backbone.parameters())
    bb0 = bb_param.detach().clone()

    _step(model, groups, trainer, "backbone", batch)
    assert not torch.equal(bb_param.detach(), bb0), "backbone should change in backbone phase"
    assert torch.equal(model.tokens.embeddings.detach(), tok0), "tokens must NOT change in backbone phase"


def test_token_phase_updates_only_tokens(synth_corpus, tmp_path):
    cfg = _full_backbone_cfg()
    batch, K = _batch(synth_corpus)
    model = build_model(cfg, vocab_size=257, data_n_experts=K)
    trainer = _trainer(cfg, tmp_path)
    groups = trainer._build_alternating(model, 8, dict(cfg["train"]["alternation"]))

    tok0 = model.tokens.embeddings.detach().clone()
    bb_param = next(p for p in model.backbone.parameters())
    bb0 = bb_param.detach().clone()

    _step(model, groups, trainer, "tokens", batch)
    assert not torch.equal(model.tokens.embeddings.detach(), tok0), "tokens should change in token phase"
    assert torch.equal(bb_param.detach(), bb0), "backbone must NOT change in token phase"


def test_full_fit_runs_alternating(synth_corpus, tmp_path):
    cfg = _full_backbone_cfg()
    train_ds = SoftMoEDataset(synth_corpus, "train")
    val_ds = SoftMoEDataset(synth_corpus, "val")
    _, K = _batch(synth_corpus)
    cfg["seed"] = 0
    cfg["data"] = {"pad_token_id": 256}
    model = build_model(cfg, vocab_size=257, data_n_experts=K)
    trainer = EMTrainer(cfg, tmp_path / "run", device="cpu")
    ckpt = trainer.fit(model, train_ds, val_ds)
    assert (tmp_path / "run" / "checkpoints").exists()
