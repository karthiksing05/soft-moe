"""``SoftMoE`` — wraps a HF causal LM and injects expert tokens, with a unified forward contract.

    forward(batch) -> {loss, logits, per_example_nll, aux}

``aux`` carries ``route_info``, ``separation``, ``load_balance``, ``utilization_entropy`` and
(optionally) ``router`` so the EM trainer can combine + log every term. The signature is shared
by every baseline so ``train.py`` / ``evaluate.py`` stay method-agnostic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from softmoe.models.backbone import backbone_hidden_size, num_heads, num_layers
from softmoe.models.expert_tokens import ExpertTokenBank
from softmoe.models.router import RouteInfo, SoftRouter, make_router
from softmoe.training.losses import (
    causal_lm_loss,
    load_balance_loss,
    router_loss,
    switch_aux_loss,
    usage_entropy,
)


class PrefixEncoder(nn.Module):
    """Reparameterize expert tokens into per-layer past key/values (prefix-tuning proper)."""

    def __init__(self, d_model: int, n_layer: int, n_head: int):
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 2 * n_layer * d_model)
        )

    def forward(self, prefix: torch.Tensor):
        # prefix: [B, T, d] -> tuple of n_layer (key, value) each [B, n_head, T, head_dim]
        B, T, _ = prefix.shape
        kv = self.proj(prefix)                                  # [B, T, 2*n_layer*d]
        kv = kv.view(B, T, 2 * self.n_layer, self.n_head, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)                          # [2*n_layer, B, n_head, T, hd]
        past = []
        for i in range(self.n_layer):
            past.append((kv[2 * i], kv[2 * i + 1]))
        return tuple(past)


class SoftMoE(nn.Module):
    def __init__(self, backbone, tokens: ExpertTokenBank, router: nn.Module, cfg):
        super().__init__()
        self.backbone = backbone
        self.tokens = tokens
        self.router = router
        self.injection = cfg.get("injection", "prefix")
        self.separation_kind = cfg.get("separation_kind", "cosine")
        self.load_balance_kind = cfg.get("load_balance_kind", "entropy")
        self.soft_mixture = bool(cfg.get("soft_mixture", False))
        self.router_supervise_with = cfg.get("router_supervise_with")  # None|'cluster'|'domain'
        # 'speaker' injection: the expert vector is a real inline discrete token `[DOMAIN_k]` at the
        # front of the sequence (the thesis `[SPEAKER_X] <text>` format), output-tied to the bank so
        # the model is trained to *emit* it — a genuine trained vocabulary token, not a soft prefix.
        self.speaker_predict = bool(cfg.get("speaker_predict", True))
        self._pad_id = int(cfg.get("pad_token_id", 0))
        self.d_model = backbone_hidden_size(backbone)

        if self.injection == "prefix_kv":
            self.prefix_encoder = PrefixEncoder(self.d_model, num_layers(backbone), num_heads(backbone))
        else:
            self.prefix_encoder = None

        # FFN subspace governance: the expert token drives a per-layer modulation of the FFN hidden
        # (film = neuron gate, spectral = principal-direction gate) — the MoE-structured alternative
        # to prefix/token injection. Built + attached in the factory (needs the backbone's FFN dims).
        self.governor = None
        self.governs = self.injection in ("film_ffn", "spectral_ffn")

    # ---- routing ---------------------------------------------------------------------
    def _router_hidden(self, batch):
        # cheap proxy for "first backbone hidden state": input embeddings, mean-pooled later.
        emb = self.backbone.get_input_embeddings()
        return emb(batch["input_ids"])

    def route(self, batch, hard: bool = False) -> RouteInfo:
        hidden = None
        if isinstance(self.router, SoftRouter) and self.router.pool == "meanhidden":
            hidden = self._router_hidden(batch)
        if isinstance(self.router, SoftRouter):
            return self.router(batch, self.tokens, hidden=hidden, hard=hard)
        return self.router(batch, self.tokens)

    # ---- injection -------------------------------------------------------------------
    def _forward_with_prefix(self, input_ids, attention_mask, labels, prefix):
        """Prefix mode: prepend T expert-token embeddings to the input sequence."""
        emb = self.backbone.get_input_embeddings()
        inp = emb(input_ids)                                    # [B, L, d]
        T = prefix.shape[1]
        combined = torch.cat([prefix, inp], dim=1)              # [B, T+L, d]
        pre_mask = torch.ones(input_ids.shape[0], T, device=input_ids.device, dtype=attention_mask.dtype)
        mask = torch.cat([pre_mask, attention_mask], dim=1)
        pre_lab = torch.full((input_ids.shape[0], T), -100, device=input_ids.device, dtype=labels.dtype)
        comb_labels = torch.cat([pre_lab, labels], dim=1)
        out = self.backbone(inputs_embeds=combined, attention_mask=mask)
        logits = out.logits                                     # [B, T+L, V]
        per_ex = causal_lm_loss(logits, comb_labels, reduction="per_example")
        return per_ex, logits[:, T:, :]

    def _forward_with_prefix_kv(self, input_ids, attention_mask, labels, prefix):
        """Prefix-KV mode: expert tokens become past key/values at every layer."""
        past = self.prefix_encoder(prefix)
        T = prefix.shape[1]
        pre_mask = torch.ones(input_ids.shape[0], T, device=input_ids.device, dtype=attention_mask.dtype)
        mask = torch.cat([pre_mask, attention_mask], dim=1)
        out = self.backbone(input_ids=input_ids, attention_mask=mask, past_key_values=past, use_cache=True)
        logits = out.logits                                     # [B, L, V]
        per_ex = causal_lm_loss(logits, labels, reduction="per_example")
        return per_ex, logits

    def _forward_with_speaker(self, input_ids, attention_mask, labels, expert_ids):
        """Speaker mode: an inline discrete `[DOMAIN_k]` token at the front (thesis persona format).

        The per-domain vector is the bank embedding, prepended as a real token position; with
        ``speaker_predict`` it is also output-tied (extended logits) and trained to be *emitted* —
        so it is a genuine trained vocabulary token, not just a soft prefix."""
        B = input_ids.shape[0]
        emb = self.backbone.get_input_embeddings()
        marker = self.tokens(expert_ids)                         # [B, T, d]  (T=1 per-domain vector)
        Tm = marker.shape[1]
        inp = emb(input_ids)
        ones = lambda n: torch.ones(B, n, device=input_ids.device, dtype=attention_mask.dtype)
        ign = lambda n: torch.full((B, n), -100, device=input_ids.device, dtype=labels.dtype)
        if not self.speaker_predict:
            combined = torch.cat([marker, inp], dim=1)
            mask = torch.cat([ones(Tm), attention_mask], dim=1)
            comb_labels = torch.cat([ign(Tm), labels], dim=1)
            out = self.backbone(inputs_embeds=combined, attention_mask=mask)
            return causal_lm_loss(out.logits, comb_labels, reduction="per_example"), out.logits[:, Tm:, :]
        bos = emb(torch.full((B, 1), self._pad_id, device=input_ids.device, dtype=input_ids.dtype))
        combined = torch.cat([bos, marker, inp], dim=1)
        mask = torch.cat([ones(1 + Tm), attention_mask], dim=1)
        out = self.backbone(inputs_embeds=combined, attention_mask=mask, output_hidden_states=True)
        Vbase = out.logits.shape[-1]
        marker_logits = out.hidden_states[-1] @ self.tokens.embeddings[:, 0, :].t()   # [B, seq, K] (tied)
        full_logits = torch.cat([out.logits, marker_logits], dim=-1)
        marker_lab = (Vbase + expert_ids).view(B, 1).to(labels.dtype)
        comb_labels = torch.cat([ign(1), marker_lab, labels], dim=1)   # BOS ignored; marker scored
        return causal_lm_loss(full_logits, comb_labels, reduction="per_example"), out.logits[:, 1 + Tm:, :]

    def _forward_with_governance(self, input_ids, attention_mask, labels, expert_ids):
        """Governance mode: the expert token modulates each block's FFN (no prefix prepended)."""
        e = self.tokens(expert_ids)[:, 0, :]                    # [B, d]  the per-expert latent
        self.governor.set_current(e)
        try:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            self.governor.clear()
        per_ex = causal_lm_loss(out.logits, labels, reduction="per_example")
        return per_ex, out.logits

    def _single_expert_forward(self, batch, expert_ids):
        if self.governs:
            return self._forward_with_governance(
                batch["input_ids"], batch["attention_mask"], batch["labels"], expert_ids
            )
        if self.injection == "speaker":
            return self._forward_with_speaker(
                batch["input_ids"], batch["attention_mask"], batch["labels"], expert_ids
            )
        prefix = self.tokens(expert_ids)                        # [B, T, d]
        if self.injection == "prefix_kv":
            return self._forward_with_prefix_kv(
                batch["input_ids"], batch["attention_mask"], batch["labels"], prefix
            )
        return self._forward_with_prefix(
            batch["input_ids"], batch["attention_mask"], batch["labels"], prefix
        )

    # ---- forward ---------------------------------------------------------------------
    def forward(self, batch, em_hard: bool = False) -> dict:
        route = self.route(batch, hard=em_hard)

        if self.soft_mixture and route.topk_ids is not None and route.topk_ids.shape[1] > 1:
            # Soft-EM mixture over top-k experts: weight per-example NLL by responsibilities.
            k = route.topk_ids.shape[1]
            nll_stack = []
            logits_keep = None
            for j in range(k):
                per_ex_j, logits_j = self._single_expert_forward(batch, route.topk_ids[:, j])
                nll_stack.append(per_ex_j)
                if j == 0:
                    logits_keep = logits_j
            nll = torch.stack(nll_stack, dim=1)                 # [B, k]
            per_ex = (route.topk_weights * nll).sum(dim=1)
            logits = logits_keep
        else:
            per_ex, logits = self._single_expert_forward(batch, route.expert_ids)

        loss = per_ex.mean()
        aux = self._aux(batch, route)
        return {"loss": loss, "logits": logits, "per_example_nll": per_ex, "aux": aux}

    def _aux(self, batch, route: RouteInfo) -> dict:
        r = route.responsibilities
        aux: dict = {"route_info": route}
        # separation (always computed; cheap, and we log it for the fixed baseline too)
        aux["separation"] = self.tokens.separation_loss(self.separation_kind)
        # load balance
        if self.load_balance_kind == "switch":
            aux["load_balance"] = switch_aux_loss(r, route.expert_ids)
        else:
            aux["load_balance"] = load_balance_loss(r)
        aux["utilization_entropy"] = usage_entropy(r)
        # optional router supervision toward the data partition (C2/C3 bridge)
        if route.logits is not None and self.router_supervise_with:
            key = "domain_id" if self.router_supervise_with == "domain" else "cluster_id"
            target = F.one_hot(
                batch[key].clamp(max=route.logits.shape[1] - 1), num_classes=route.logits.shape[1]
            ).float()
            aux["router"] = router_loss(route.logits, target)
        return aux

    # ---- bookkeeping -----------------------------------------------------------------
    def num_added_trainable_params(self) -> int:
        n = self.tokens.num_added_params()
        if self.prefix_encoder is not None:
            n += sum(p.numel() for p in self.prefix_encoder.parameters() if p.requires_grad)
        if self.governor is not None:
            n += self.governor.num_params(trainable_only=True)
        n += sum(p.numel() for p in self.router.parameters() if p.requires_grad)
        return n

    def num_active_params(self) -> int:
        # Full backbone + router + one expert's worth of tokens (prefix encoder if prefix_kv).
        n = sum(p.numel() for p in self.backbone.parameters())
        n += sum(p.numel() for p in self.router.parameters())
        n += self.tokens.embeddings[0].numel()              # one expert active per input
        if self.prefix_encoder is not None:
            n += sum(p.numel() for p in self.prefix_encoder.parameters())
        if self.governor is not None:
            n += self.governor.num_params()                 # the shared hypernet is always active
        return n
