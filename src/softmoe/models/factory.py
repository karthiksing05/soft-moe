"""Build any method from config behind one factory, so train/eval stay method-agnostic.

``cfg.model.method`` selects: ``softmoe`` (the EM expert-token method: prefix or FFN governance),
``dense`` (general finetuning baseline), and ``hard_moe`` (the standard MoE). n_experts/d_model/vocab
are injected from the data + backbone so configs can say ``n_experts: auto``.
"""

from __future__ import annotations

import torch

from softmoe.models.backbone import backbone_hidden_size, build_backbone
from softmoe.models.expert_tokens import ExpertTokenBank
from softmoe.models.router import make_router
from softmoe.models.soft_moe import SoftMoE
from softmoe.utils.logging import get_logger

logger = get_logger()


def _resolve_n_experts(value, data_n_experts: int) -> int:
    if value in (None, "auto"):
        return int(data_n_experts)
    return int(value)


def build_model(cfg, *, vocab_size: int, data_n_experts: int, centroids=None):
    model_cfg = cfg["model"]
    method = model_cfg.get("method", "softmoe")
    backbone_cfg = model_cfg["backbone"]

    if method == "dense":
        from softmoe.models.baselines.dense import Dense

        backbone = build_backbone({**backbone_cfg, "backbone_mode": model_cfg.get("backbone_mode", "full")}, vocab_size)
        return Dense(backbone)

    if method == "hard_moe":
        from softmoe.models.baselines.hard_moe import HardMoE

        backbone = build_backbone({**backbone_cfg, "backbone_mode": "full"}, vocab_size)
        moe_cfg = model_cfg.get("hard_moe", {})
        # Back-compat: accept legacy n_experts as base_experts.
        base_experts = int(moe_cfg.get("base_experts", moe_cfg.get("n_experts", 8)))
        return HardMoE(backbone,
                       base_experts=base_experts,
                       granularity=int(moe_cfg.get("granularity", 1)),
                       base_top_k=int(moe_cfg.get("top_k", 1)),
                       n_shared=int(moe_cfg.get("n_shared", 0)),
                       upcycle=bool(moe_cfg.get("upcycle", False)),
                       route_by=moe_cfg.get("route_by", "learned"))

    if method == "softmoe":
        backbone = build_backbone({**backbone_cfg, "backbone_mode": model_cfg.get("backbone_mode", "frozen")}, vocab_size)
        d_model = backbone_hidden_size(backbone)
        et = model_cfg["expert_tokens"]
        n_experts = _resolve_n_experts(et.get("n_experts", "auto"), data_n_experts)
        cent = torch.as_tensor(centroids) if centroids is not None else None
        tokens = ExpertTokenBank(
            n_experts=n_experts,
            tokens_per_expert=int(et.get("tokens_per_expert", 1)),
            d_model=d_model,
            init=et.get("init", "random"),
            trainable=bool(et.get("trainable", True)),
            centroids=cent,
        )
        router_cfg = model_cfg.get("router", {"kind": "supervised"})
        router = make_router(router_cfg, n_experts, d_in=d_model, vocab_size=vocab_size)
        wrapper_cfg = {
            "injection": model_cfg.get("injection", "prefix"),
            "separation_kind": model_cfg.get("separation_kind", "cosine"),
            "load_balance_kind": model_cfg.get("load_balance_kind", "entropy"),
            "soft_mixture": model_cfg.get("soft_mixture", False),
            "router_supervise_with": model_cfg.get("router_supervise_with"),
            "mop_average": model_cfg.get("mop_average", False),
            "token_predict_marker": model_cfg.get("token_predict_marker", True),
            "pad_token_id": vocab_size and 0,  # byte tokenizer pads with 0
        }
        model = SoftMoE(backbone, tokens, router, wrapper_cfg)
        if model.governs:
            from softmoe.models.governance import FFNGovernor, ffn_hidden_size, n_blocks

            bmode = model_cfg.get("backbone_mode", "frozen")
            gov = FFNGovernor(
                d_model=d_model,
                d_ff=ffn_hidden_size(backbone),
                n_layers=n_blocks(backbone),
                mode="film" if wrapper_cfg["injection"] == "film_ffn" else "spectral",
                rank=int(model_cfg.get("governor_rank", 16)),
                govern_attn=bool(model_cfg.get("govern_attn", False)),
            )
            gov.attach(backbone)
            # the hypernet is part of the governance backbone: trained in Phase A (full), frozen in
            # Phase B (frozen) so only the expert latents e_k are fit.
            if bmode == "frozen":
                for p in gov.parameters():
                    p.requires_grad_(False)
            model.governor = gov
        logger.info(
            "[factory] SoftMoE: K=%d T=%d inject=%s router=%s trainable_tokens=%s backbone_mode=%s",
            n_experts, tokens.tokens_per_expert, wrapper_cfg["injection"],
            router_cfg.get("kind"), tokens.trainable, model_cfg.get("backbone_mode", "frozen"),
        )
        return model

    raise ValueError(f"Unknown model.method '{method}'.")
