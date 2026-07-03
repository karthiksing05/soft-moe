#!/usr/bin/env python
"""Validate the Qwen architecture port: build prefix / spectral-governance / speaker SoftMoE on a
small Qwen and run forward+backward. No real data needed — random ids + fake domain labels."""
import argparse, torch
from softmoe.utils.config import load_config
from softmoe.models.factory import build_model
from softmoe.training.losses import causal_lm_loss

ap = argparse.ArgumentParser(); ap.add_argument("--name", default="Qwen/Qwen2.5-0.5B"); ap.add_argument("--k", type=int, default=4)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

def probe(injection, extra):
    cfg = load_config("configs/experiment/qwen_probe.yaml")
    cfg.set_path("model.backbone.name", a.name)
    cfg.set_path("model.injection", injection)
    for k, v in extra.items(): cfg.set_path(k, v)
    from transformers import AutoTokenizer
    V = AutoTokenizer.from_pretrained(a.name).vocab_size
    m = build_model(cfg, vocab_size=0, data_n_experts=a.k).to(dev)   # vocab_size 0 => no resize
    d = m.tokens.embeddings.shape[-1]
    B, L = 2, 24
    batch = {"input_ids": torch.randint(1, 5000, (B, L), device=dev),
             "attention_mask": torch.ones(B, L, dtype=torch.long, device=dev),
             "labels": torch.randint(1, 5000, (B, L), device=dev),
             "domain_id": torch.randint(0, a.k, (B,), device=dev),
             "cluster_id": torch.randint(0, a.k, (B,), device=dev)}
    with torch.no_grad():
        ref = causal_lm_loss(m.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits, batch["labels"]).item()
        loss = m(batch)["loss"].item()
    out = m(batch); out["loss"].backward()
    tg = m.tokens.embeddings.grad
    gg = m.governor.proj_ffn.weight.grad.norm().item() if m.governor is not None else float("nan")
    print(f"[{injection:12s}] d_model={d} loss={loss:.4f} dense_ref={ref:.4f}  token_grad={0.0 if tg is None else tg.norm().item():.2e}  gov_grad={gg:.2e}")

print(f"=== Qwen port validation on {a.name} (K={a.k} domains) ===")
probe("prefix", {})
probe("spectral_ffn", {"model.govern_attn": True})
probe("speaker", {"model.speaker_predict": True})
print("ALL FORWARD+BACKWARD PASSES OK")
