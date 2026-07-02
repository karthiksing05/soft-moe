"""FFN (and optionally attention) **subspace governance** — condition the backbone's per-block
computation on the EM expert token.

This is the MoE-structured realisation of "soft-expert subspace governance": where the recipe-faithful
MoE partitions the FFN hidden dim into experts and *routes tokens* to a subset, here a compact
per-expert latent ``e_k`` (the ``ExpertTokenBank``, EM-trained) drives a **shared hypernetwork** that
emits, per layer, a modulation of that same FFN hidden space (and optionally the attention output).
The backbone + hypernet are the governance mechanism (trained in Phase A); ``e_k`` is fit in Phase B
— so the token stays a token.

Two modes, at one or two sites:
- ``film``     — per-unit multiplicative gate ``h ⊙ g_k`` (the soft, token-routed analog of the MoE
                 selecting FFN experts / neurons).
- ``spectral`` — gate a learned ``r``-dim orthonormal subspace: rotate onto a basis ``U``, scale ``r``
                 principal directions by the token, rotate back. Governs *directions of the layer's
                 output* rather than axis-aligned units.
- ``govern_attn`` additionally applies the same modulation to each block's **attention output
                 projection** (the d_model residual contribution), so the expert governs *both* the
                 mixing (attention) and the transform (FFN).

Both are **identity at init** (zero-init hypernet), so Phase A starts from the dense backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _blocks(backbone):
    tr = getattr(backbone, "transformer", None)
    if tr is None or not hasattr(tr, "h"):
        raise ValueError("Governance currently supports GPT-2-style backbones (transformer.h).")
    return tr.h


def ffn_act_modules(backbone) -> list[nn.Module]:
    """Per-block FFN activation modules whose output is the ``d_ff`` hidden (GPT-2 family)."""
    return [blk.mlp.act for blk in _blocks(backbone)]


def attn_proj_modules(backbone) -> list[nn.Module]:
    """Per-block attention output projections whose output is the ``d_model`` residual contribution."""
    return [blk.attn.c_proj for blk in _blocks(backbone)]


def ffn_hidden_size(backbone) -> int:
    cfg = backbone.config
    return int(getattr(cfg, "n_inner", None) or 4 * cfg.n_embd)


class FFNGovernor(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_layers: int, mode: str = "film",
                 rank: int = 16, govern_attn: bool = False):
        super().__init__()
        if mode not in ("film", "spectral"):
            raise ValueError(f"Unknown governance mode '{mode}' (film|spectral).")
        self.mode = mode
        self.n_layers = n_layers
        self.rank = int(rank)
        self.govern_attn = bool(govern_attn)
        self._cur = {}  # site -> [n_layers, B, gate_out]

        # one head (+ basis for spectral) per site. FFN site gates the d_ff hidden; attn the d_model.
        self.ffn_gate_out = d_ff if mode == "film" else self.rank
        self.proj_ffn = self._zero_head(d_model, n_layers * self.ffn_gate_out)
        if mode == "spectral":
            self.U_ffn = nn.Parameter(self._ortho_basis(n_layers, d_ff, self.rank))
        if self.govern_attn:
            self.attn_gate_out = d_model if mode == "film" else self.rank
            self.proj_attn = self._zero_head(d_model, n_layers * self.attn_gate_out)
            if mode == "spectral":
                self.U_attn = nn.Parameter(self._ortho_basis(n_layers, d_model, self.rank))

    @staticmethod
    def _zero_head(d_in: int, d_out: int) -> nn.Linear:
        lin = nn.Linear(d_in, d_out)
        nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)   # identity modulation at init
        return lin

    @staticmethod
    def _ortho_basis(n_layers: int, dim: int, rank: int) -> torch.Tensor:
        U = torch.empty(n_layers, dim, rank)
        for l in range(n_layers):
            nn.init.orthogonal_(U[l])
        return U

    def attach(self, backbone) -> None:
        for l, act in enumerate(ffn_act_modules(backbone)):
            act.register_forward_hook(self._make_hook("ffn", l, getattr(self, "U_ffn", None)))
        if self.govern_attn:
            for l, cp in enumerate(attn_proj_modules(backbone)):
                cp.register_forward_hook(self._make_hook("attn", l, getattr(self, "U_attn", None)))

    def _make_hook(self, site: str, l: int, U):
        def hook(_module, _inp, out):
            cur = self._cur.get(site)
            if cur is None:
                return out
            g = cur[l]                                          # [B, gate_out]
            if self.mode == "film":
                return out * (2.0 * torch.sigmoid(g)).unsqueeze(1)         # init 1.0 => identity
            Ul = U[l]                                           # [dim, r]
            c = out @ Ul                                        # [B, L, r]  project onto subspace
            s = (2.0 * torch.sigmoid(g) - 1.0).unsqueeze(1)     # [B, 1, r]  init 0 => identity
            return out + (c * s) @ Ul.t()                       # rotate+scale r principal directions
        return hook

    def set_current(self, e: torch.Tensor) -> None:
        """``e``: [B, d_model] routed expert vectors -> stash per-site, per-layer gates."""
        B = e.shape[0]
        self._cur = {"ffn": self.proj_ffn(e).view(B, self.n_layers, self.ffn_gate_out).transpose(0, 1)}
        if self.govern_attn:
            self._cur["attn"] = self.proj_attn(e).view(B, self.n_layers, self.attn_gate_out).transpose(0, 1)

    def clear(self) -> None:
        self._cur = {}

    def num_params(self, trainable_only: bool = False) -> int:
        ps = [p for p in self.parameters() if (p.requires_grad or not trainable_only)]
        return sum(p.numel() for p in ps)
