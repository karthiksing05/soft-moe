"""Hard MoE baseline — classic top-k token-routed FFN MoE. The (parameter-heavy) upper bound.

We replace the backbone's FFN blocks with a minimal from-scratch token-routed MoE FFN (not a
full MoE library) and add the Switch-Transformer load-balance aux loss. Works on GPT-2-style
(``transformer.h``) and GPT-NeoX/pythia-style (``gpt_neox.layers``) backbones.

Each expert is **sparse-upcycled** from the pretrained dense FFN (Komatsuzaki et al., 2023): the
experts start as copies of the original FFN, so the MoE begins ≈ the pretrained model and *learns
to specialize*, rather than relearning an FFN from scratch (which a few thousand steps can't do).
This makes it a fair "true MoE" reference for the comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from softmoe.models.backbone import backbone_hidden_size
from softmoe.training.losses import causal_lm_loss
from softmoe.utils.logging import get_logger

logger = get_logger()


class MoEFFN(nn.Module):
    """Top-k token-routed MoE feed-forward, drop-in for a transformer block's ``.mlp``."""

    def __init__(self, d_model: int, d_inner: int, n_experts: int, top_k: int = 1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, d_inner), nn.GELU(), nn.Linear(d_inner, d_model))
             for _ in range(n_experts)]
        )
        self.last_aux: dict = {}
        self.forced_ids: torch.Tensor | None = None    # [B] oracle expert per example (or None)

    @torch.no_grad()
    def upcycle_from(self, src_mlp: nn.Module) -> bool:
        """Initialize every expert from the pretrained dense FFN. Returns True on success."""
        w1 = b1 = w2 = b2 = None
        # GPT-NeoX / pythia: dense_h_to_4h (Linear) -> act -> dense_4h_to_h (Linear)
        if hasattr(src_mlp, "dense_h_to_4h") and hasattr(src_mlp, "dense_4h_to_h"):
            w1, b1 = src_mlp.dense_h_to_4h.weight, src_mlp.dense_h_to_4h.bias
            w2, b2 = src_mlp.dense_4h_to_h.weight, src_mlp.dense_4h_to_h.bias
        # GPT-2: c_fc (Conv1D) -> act -> c_proj (Conv1D); Conv1D weight is transposed vs Linear
        elif hasattr(src_mlp, "c_fc") and hasattr(src_mlp, "c_proj"):
            w1, b1 = src_mlp.c_fc.weight.t(), src_mlp.c_fc.bias
            w2, b2 = src_mlp.c_proj.weight.t(), src_mlp.c_proj.bias
        else:
            return False
        try:
            for expert in self.experts:
                if expert[0].weight.shape == w1.shape and expert[2].weight.shape == w2.shape:
                    expert[0].weight.copy_(w1); expert[0].bias.copy_(b1)
                    expert[2].weight.copy_(w2); expert[2].bias.copy_(b2)
                else:
                    return False
        except (RuntimeError, AttributeError):
            return False
        return True

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        B, L, d = hidden_states.shape
        x = hidden_states.reshape(-1, d)                       # [N, d]
        logits = self.gate(x)                                  # [N, K]
        probs = F.softmax(logits, dim=-1)
        if self.forced_ids is not None:
            # Oracle routing: every token of example b goes to expert forced_ids[b] (DEMix-style).
            ids = self.forced_ids.clamp(max=self.n_experts - 1).view(B, 1).expand(B, L).reshape(-1)
            topi = ids.unsqueeze(-1)                            # [N, 1]
            topw = torch.ones_like(topi, dtype=x.dtype)
        else:
            topw, topi = probs.topk(self.top_k, dim=-1)        # [N, k]
            topw = topw / topw.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        out = torch.zeros_like(x)
        for j in range(self.top_k):
            idx = topi[:, j]                                   # [N]
            w = topw[:, j].unsqueeze(-1)                       # [N, 1]
            for k in range(self.n_experts):
                sel = idx == k
                if sel.any():
                    out[sel] += w[sel] * self.experts[k](x[sel])

        top1 = topi[:, 0]
        frac = torch.bincount(top1, minlength=self.n_experts).float() / max(1, top1.numel())
        self.last_aux = {"frac": frac.detach(), "prob": probs.mean(0),
                         "raw_aux": self.n_experts * (frac.detach() * probs.mean(0)).sum(),
                         "top1": top1.detach()}
        return out.reshape(B, L, d)


class HardMoE(nn.Module):
    def __init__(self, backbone, n_experts: int = 4, top_k: int = 1, upcycle: bool = True,
                 route_by: str = "learned"):
        super().__init__()
        self.backbone = backbone
        self.n_experts = n_experts
        self.route_by = route_by              # 'learned' (token-routed) | 'domain' | 'cluster' (oracle)
        self.moe_layers: list[MoEFFN] = []
        self._inject(n_experts, top_k, upcycle)

    def _inject(self, n_experts: int, top_k: int, upcycle: bool) -> None:
        d_model = backbone_hidden_size(self.backbone)
        d_inner = self._ffn_inner(d_model)
        upcycled = 0
        for block in self._transformer_blocks():
            src = block.mlp
            moe = MoEFFN(d_model, d_inner, n_experts, top_k).to(
                next(src.parameters()).device, next(src.parameters()).dtype
            )
            if upcycle and moe.upcycle_from(src):
                upcycled += 1
            block.mlp = moe
            self.moe_layers.append(moe)
        logger.info("[hard_moe] injected %d MoE FFNs (K=%d, top_k=%d), upcycled=%d/%d",
                    len(self.moe_layers), n_experts, top_k, upcycled, len(self.moe_layers))

    def _ffn_inner(self, d_model: int) -> int:
        cfg = self.backbone.config
        for attr in ("n_inner", "intermediate_size", "ffn_dim"):
            v = getattr(cfg, attr, None)
            if v:
                return int(v)
        return 4 * d_model

    def _transformer_blocks(self):
        bb = self.backbone
        # GPT-2: transformer.h ; GPT-NeoX/pythia: gpt_neox.layers ; Llama-style: model.layers
        for path in (("transformer", "h"), ("gpt_neox", "layers"), ("model", "layers")):
            obj = bb
            ok = True
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    ok = False
                    break
            if ok:
                return list(obj)
        raise AttributeError("HardMoE: could not locate transformer blocks on this backbone.")

    def forward(self, batch, em_hard: bool = False) -> dict:
        # Oracle routing: pin every layer's expert to the input's domain/cluster (DEMix-style).
        if self.route_by in ("domain", "cluster"):
            key = "domain_id" if self.route_by == "domain" else "cluster_id"
            ids = batch[key]
            for m in self.moe_layers:
                m.forced_ids = ids
        out = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        for m in self.moe_layers:
            m.forced_ids = None
        per_ex = causal_lm_loss(out.logits, batch["labels"], reduction="per_example")
        aux_balance = sum(m.last_aux.get("raw_aux", 0.0) for m in self.moe_layers)
        return {
            "loss": per_ex.mean(),
            "logits": out.logits,
            "per_example_nll": per_ex,
            "aux": {"route_info": None, "load_balance": aux_balance},
        }

    def num_added_trainable_params(self) -> int:
        # MoE FFNs minus one expert's worth (the dense FFN they replaced) per layer.
        extra = 0
        for m in self.moe_layers:
            total = sum(p.numel() for p in m.parameters())
            one_expert = sum(p.numel() for p in m.experts[0].parameters())
            extra += total - one_expert
        return extra

    def num_active_params(self) -> int:
        # Per-token compute: only top_k experts run, so the other (K - top_k) experts per
        # layer are inactive. This is the basis of the compute-matched MoE comparison.
        total = sum(p.numel() for p in self.parameters())
        inactive = 0
        for m in self.moe_layers:
            per_expert = sum(p.numel() for p in m.experts[0].parameters())
            inactive += (m.n_experts - m.top_k) * per_expert
        return total - inactive
