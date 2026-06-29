"""Backbone factory: a HF causal LM, or a tiny locally-constructed GPT-2 (offline).

``kind: tiny`` builds a small GPT-2 from a fresh config (no download) — used by the toy/smoke
configs and the hermetic test suite. ``kind: hf`` loads any ``AutoModelForCausalLM`` (gpt2,
EleutherAI/pythia-160m, …). Backbone tuning mode (frozen / lora / full) is applied here.
"""

from __future__ import annotations

import torch.nn as nn


def build_backbone(cfg, vocab_size: int):
    """Return a HF-style causal LM exposing ``forward(inputs_embeds=..., labels=...)``.

    cfg keys: ``kind`` (hf|tiny), ``name`` (for hf), ``tiny`` (dims for tiny), ``backbone_mode``.
    """
    kind = cfg.get("kind", "hf")
    if kind == "tiny":
        model = _build_tiny_gpt2(cfg.get("tiny", {}), vocab_size)
    elif kind == "hf":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(cfg["name"])
        # resize for a custom tokenizer if needed
        if vocab_size and model.get_input_embeddings().weight.shape[0] != vocab_size:
            model.resize_token_embeddings(vocab_size)
    else:
        raise ValueError(f"Unknown backbone kind '{kind}' (use 'hf' or 'tiny').")

    mode = cfg.get("backbone_mode", "frozen")
    _apply_tuning_mode(model, mode, cfg)
    return model


def _build_tiny_gpt2(dims: dict, vocab_size: int):
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=int(dims.get("n_positions", 256)),
        n_embd=int(dims.get("n_embd", 64)),
        n_layer=int(dims.get("n_layer", 2)),
        n_head=int(dims.get("n_head", 2)),
        n_inner=int(dims.get("n_inner", 128)),
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(config)


def _apply_tuning_mode(model, mode: str, cfg) -> None:
    if mode == "frozen":
        for p in model.parameters():
            p.requires_grad_(False)
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    elif mode == "lora":
        _apply_lora(model, cfg.get("lora", {}))
    else:
        raise ValueError(f"Unknown backbone_mode '{mode}' (use frozen|lora|full).")


def _apply_lora(model, lora_cfg: dict) -> None:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - optional path
        raise ImportError("backbone_mode='lora' requires `pip install peft`.") from exc
    # freeze base, inject LoRA
    for p in model.parameters():
        p.requires_grad_(False)
    config = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        target_modules=lora_cfg.get("target_modules", ["c_attn"]),
        task_type="CAUSAL_LM",
    )
    get_peft_model(model, config)


def backbone_hidden_size(model) -> int:
    cfg = model.config
    for attr in ("hidden_size", "n_embd", "d_model"):
        if getattr(cfg, attr, None):
            return int(getattr(cfg, attr))
    raise AttributeError("Could not infer backbone hidden size from config.")


def num_layers(model) -> int:
    cfg = model.config
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if getattr(cfg, attr, None):
            return int(getattr(cfg, attr))
    raise AttributeError("Could not infer backbone layer count.")


def num_heads(model) -> int:
    cfg = model.config
    for attr in ("num_attention_heads", "n_head", "num_heads"):
        if getattr(cfg, attr, None):
            return int(getattr(cfg, attr))
    raise AttributeError("Could not infer backbone head count.")
