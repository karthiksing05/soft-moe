"""Hard MoE baseline — token-routed FFN MoE, built per SCALED_RECIPE.md.

Recipe-faithful levers (all config-driven), so the MoE arm is not the understated textbook
GShard/Switch config that was found to *saturate* (Clark et al. 2022), but the near-Pareto-optimal
design of the fine-grained scaling-law line:

- **Fine-grained experts** (Krajewski/Ludziejewski et al. 2024): granularity ``G`` splits each base
  expert's FFN into ``G`` pieces of width ``d_ff/G`` and routes to ``G×`` as many, holding active
  compute constant while decomposing knowledge more finely. ``G=1`` reproduces coarse Switch/Mixtral.
- **Shared experts** (DeepSeekMoE, Dai et al. 2024): ``n_shared`` always-on experts absorb common
  knowledge so routed experts don't waste capacity re-learning it.
- **Router z-loss** (ST-MoE, Zoph et al. 2022): penalizes ``logsumexp(router_logits)²`` for stability.
- **Switch load-balancing aux loss** (Fedus et al. 2022): the ``K·Σ f_i·P_i`` term.
- **top-k token-choice routing** (causal-safe).

Also supports **oracle domain routing** (DEMix, Gururangan et al. 2021) via ``route_by=domain``, and
**sparse upcycling** (Komatsuzaki et al. 2023) when expert width matches the dense FFN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from softmoe.models.backbone import backbone_hidden_size
from softmoe.training.losses import causal_lm_loss
from softmoe.utils.logging import get_logger

logger = get_logger()


def _ffn(d_model: int, width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_model, width), nn.GELU(), nn.Linear(width, d_model))


class MoEFFN(nn.Module):
    """Fine-grained top-k token-routed MoE FFN (+ optional shared experts), drop-in for ``.mlp``."""

    def __init__(self, d_model: int, expert_width: int, n_routed: int, top_k: int, n_shared: int = 0):
        super().__init__()
        self.n_experts = n_routed                         # routed (fine-grained) expert count
        self.top_k = min(top_k, n_routed)
        self.expert_width = expert_width
        self.n_shared = n_shared
        self.gate = nn.Linear(d_model, n_routed, bias=False)
        self.experts = nn.ModuleList([_ffn(d_model, expert_width) for _ in range(n_routed)])
        self.shared = nn.ModuleList([_ffn(d_model, expert_width) for _ in range(n_shared)])
        self.last_aux: dict = {}
        self.capture: bool = False                        # record per-token routing for analysis
        self.forced_ids: torch.Tensor | None = None       # [B] oracle expert per example (or None)

    @torch.no_grad()
    def upcycle_from(self, src_mlp: nn.Module) -> bool:
        """Copy the dense FFN into every routed expert (only when widths match; G=1 case)."""
        if hasattr(src_mlp, "dense_h_to_4h") and hasattr(src_mlp, "dense_4h_to_h"):
            w1, b1 = src_mlp.dense_h_to_4h.weight, src_mlp.dense_h_to_4h.bias
            w2, b2 = src_mlp.dense_4h_to_h.weight, src_mlp.dense_4h_to_h.bias
        elif hasattr(src_mlp, "c_fc") and hasattr(src_mlp, "c_proj"):
            w1, b1 = src_mlp.c_fc.weight.t(), src_mlp.c_fc.bias
            w2, b2 = src_mlp.c_proj.weight.t(), src_mlp.c_proj.bias
        else:
            return False
        try:
            for expert in list(self.experts) + list(self.shared):
                if expert[0].weight.shape != w1.shape or expert[2].weight.shape != w2.shape:
                    return False
                expert[0].weight.copy_(w1); expert[0].bias.copy_(b1)
                expert[2].weight.copy_(w2); expert[2].bias.copy_(b2)
        except (RuntimeError, AttributeError):
            return False
        return True

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        B, L, d = hidden_states.shape
        x = hidden_states.reshape(-1, d)                       # [N, d]
        logits = self.gate(x)                                  # [N, E]
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()
        probs = F.softmax(logits, dim=-1)
        if self.forced_ids is not None:
            ids = self.forced_ids.clamp(max=self.n_experts - 1).view(B, 1).expand(B, L).reshape(-1)
            topi = ids.unsqueeze(-1)                            # [N, 1]
            topw = torch.ones_like(topi, dtype=x.dtype)
        else:
            topw, topi = probs.topk(self.top_k, dim=-1)        # [N, k]
            topw = topw / topw.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # Batched dispatch: flatten the (token, top-k slot) routed instances, sort by expert, and run
        # ONE contiguous matmul per expert (vs the old top_k × n_experts masked forwards). Same math,
        # but O(n_experts) clean GEMMs instead of O(top_k·n_experts) masked index-ops — makes G4/G8
        # feasible. Numerically equivalent to the naive loop (verified to <1e-5).
        N, k = x.shape[0], topi.shape[1]
        tok = torch.arange(N, device=x.device).unsqueeze(1).expand(N, k).reshape(-1)   # [N*k] token id
        exp = topi.reshape(-1)                                                          # [N*k] expert id
        wt = topw.reshape(-1).unsqueeze(-1)                                             # [N*k, 1] weight
        order = torch.argsort(exp)
        tok_s, wt_s = tok[order], wt[order]
        xin = x[tok_s]                                                                  # sorted by expert
        counts = torch.bincount(exp, minlength=self.n_experts).tolist()
        chunks, off = [], 0
        for e in range(self.n_experts):
            c = counts[e]
            if c:
                chunks.append(self.experts[e](xin[off:off + c]))
                off += c
        yout = torch.cat(chunks, 0) if chunks else xin
        out = x.new_zeros(N, d).index_add_(0, tok_s, yout * wt_s)
        for s in self.shared:                                  # always-on experts
            out = out + s(x)

        top1 = topi[:, 0]
        frac = torch.bincount(top1, minlength=self.n_experts).float() / max(1, top1.numel())
        self.last_aux = {"raw_aux": self.n_experts * (frac.detach() * probs.mean(0)).sum(),
                         "z_loss": z_loss}
        if self.capture:                                       # [B, L] top-1 expert per token
            self.last_aux["top1_bl"] = top1.detach().view(B, L)
        return out.reshape(B, L, d)


class HardMoE(nn.Module):
    def __init__(self, backbone, base_experts: int = 8, granularity: int = 1, base_top_k: int = 1,
                 n_shared: int = 0, upcycle: bool = False, route_by: str = "learned"):
        super().__init__()
        self.backbone = backbone
        self.route_by = route_by
        d_ff = self._ffn_inner(backbone_hidden_size(backbone))
        self.n_routed = base_experts * granularity
        self.expert_width = max(1, round(d_ff / granularity))
        self.top_k = base_top_k * granularity
        self.n_shared = n_shared
        self.granularity = granularity
        self.n_experts = self.n_routed                         # for oracle/eval compatibility
        self.moe_layers: list[MoEFFN] = []
        self._inject(upcycle)

    def _inject(self, upcycle: bool) -> None:
        d_model = backbone_hidden_size(self.backbone)
        upcycled = 0
        for block in self._transformer_blocks():
            src = block.mlp
            moe = MoEFFN(d_model, self.expert_width, self.n_routed, self.top_k, self.n_shared).to(
                next(src.parameters()).device, next(src.parameters()).dtype
            )
            if upcycle and moe.upcycle_from(src):
                upcycled += 1
            block.mlp = moe
            self.moe_layers.append(moe)
        logger.info("[hard_moe] %d layers | routed=%d (G=%d, width=%d, top_k=%d) shared=%d upcycled=%d",
                    len(self.moe_layers), self.n_routed, self.granularity, self.expert_width,
                    self.top_k, self.n_shared, upcycled)

    def _ffn_inner(self, d_model: int) -> int:
        cfg = self.backbone.config
        for attr in ("n_inner", "intermediate_size", "ffn_dim"):
            v = getattr(cfg, attr, None)
            if v:
                return int(v)
        return 4 * d_model

    def _transformer_blocks(self):
        bb = self.backbone
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

    def set_capture(self, on: bool) -> None:
        for m in self.moe_layers:
            m.capture = on

    def forward(self, batch, em_hard: bool = False) -> dict:
        if self.route_by in ("domain", "cluster"):
            key = "domain_id" if self.route_by == "domain" else "cluster_id"
            for m in self.moe_layers:
                m.forced_ids = batch[key]
        out = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        for m in self.moe_layers:
            m.forced_ids = None
        per_ex = causal_lm_loss(out.logits, batch["labels"], reduction="per_example")
        aux_balance = sum(m.last_aux.get("raw_aux", 0.0) for m in self.moe_layers)
        z_loss = sum(m.last_aux.get("z_loss", 0.0) for m in self.moe_layers)
        return {
            "loss": per_ex.mean(),
            "logits": out.logits,
            "per_example_nll": per_ex,
            "aux": {"route_info": None, "load_balance": aux_balance, "z_loss": z_loss},
        }

    def _per_expert_params(self) -> int:
        return sum(p.numel() for p in self.moe_layers[0].experts[0].parameters())

    def num_added_trainable_params(self) -> int:
        # MoE FFNs (routed + shared) minus one dense FFN's worth per layer.
        extra = 0
        for m in self.moe_layers:
            total = sum(p.numel() for p in m.parameters())
            extra += total - self._per_expert_params()   # one expert ≈ the dense FFN it replaced
        return extra

    def num_active_params(self) -> int:
        # Per token: top_k routed + n_shared shared experts run; the rest are inactive.
        total = sum(p.numel() for p in self.parameters())
        per_expert = self._per_expert_params()
        inactive = sum((m.n_experts - m.top_k) * per_expert for m in self.moe_layers)
        return total - inactive
