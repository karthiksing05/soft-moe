"""FFN **subspace governance** — condition each block's feed-forward network on the EM expert token.

This is the MoE-structured realisation of "soft-expert subspace governance": where the recipe-faithful
MoE partitions the FFN hidden dim into experts and *routes tokens* to a subset, here a compact
per-expert latent ``e_k`` (the ``ExpertTokenBank``, EM-trained) drives a **shared hypernetwork** that
emits, per layer, a modulation of that same FFN hidden space. The backbone + hypernet are the
governance mechanism (trained in Phase A); ``e_k`` is fit in Phase B — so the token stays a token.

Two modes:
- ``film``     — per-neuron multiplicative gate ``h ⊙ g_k`` on the FFN hidden (the soft, token-routed
                 analog of the MoE selecting FFN experts / neuron blocks).
- ``spectral`` — gate a learned ``r``-dim orthonormal subspace of the FFN hidden: rotate onto a basis
                 ``U``, scale ``r`` principal directions by the token, rotate back. Governs *directions
                 of the weight image* rather than axis-aligned neurons.

Both are **identity at init** (zero-init hypernet), so Phase A starts from the dense backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def ffn_act_modules(backbone) -> list[nn.Module]:
    """The per-block FFN activation modules whose output is the ``d_ff`` hidden (GPT-2 family)."""
    tr = getattr(backbone, "transformer", None)
    if tr is None or not hasattr(tr, "h"):
        raise ValueError("FFN governance currently supports GPT-2-style backbones (transformer.h).")
    return [blk.mlp.act for blk in tr.h]


def ffn_hidden_size(backbone) -> int:
    cfg = backbone.config
    return int(getattr(cfg, "n_inner", None) or 4 * cfg.n_embd)


class FFNGovernor(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_layers: int, mode: str = "film", rank: int = 16):
        super().__init__()
        self.mode = mode
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.rank = int(rank)
        self.gate_out = d_ff if mode == "film" else self.rank
        # shared hypernet e_k -> per-layer gate params. Zero-init => identity modulation at start.
        self.proj = nn.Linear(d_model, n_layers * self.gate_out)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        if mode == "spectral":
            U = torch.empty(n_layers, d_ff, self.rank)
            for l in range(n_layers):
                nn.init.orthogonal_(U[l])
            self.U = nn.Parameter(U)
        elif mode != "film":
            raise ValueError(f"Unknown governance mode '{mode}' (film|spectral).")
        self._cur = None  # per-layer gate for the current batch: [n_layers, B, gate_out]

    def attach(self, backbone) -> None:
        for l, act in enumerate(ffn_act_modules(backbone)):
            act.register_forward_hook(self._make_hook(l))

    def _make_hook(self, l: int):
        def hook(_module, _inp, out):
            if self._cur is None:
                return out
            g = self._cur[l]                                   # [B, gate_out]
            if self.mode == "film":
                return out * (2.0 * torch.sigmoid(g)).unsqueeze(1)      # init 1.0 => identity
            U = self.U[l]                                      # [d_ff, r]
            c = out @ U                                        # [B, L, r]  project onto subspace
            s = (2.0 * torch.sigmoid(g) - 1.0).unsqueeze(1)    # [B, 1, r]  init 0 => identity
            return out + (c * s) @ U.t()                       # rotate+scale r principal directions
        return hook

    def set_current(self, e: torch.Tensor) -> None:
        """``e``: [B, d_model] routed expert vectors -> stash per-layer gates for the coming forward."""
        B = e.shape[0]
        g = self.proj(e).view(B, self.n_layers, self.gate_out).transpose(0, 1)
        self._cur = g

    def clear(self) -> None:
        self._cur = None

    def num_params(self, trainable_only: bool = False) -> int:
        ps = [p for p in self.parameters() if (p.requires_grad or not trainable_only)]
        return sum(p.numel() for p in ps)
