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


def _arch(backbone) -> str:
    """Detect the transformer-block family so governance hooks attach to the right modules."""
    if getattr(backbone, "transformer", None) is not None and hasattr(backbone.transformer, "h"):
        return "gpt2"                                                  # GPT-2 / tiny byte-LM
    inner = getattr(backbone, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return "llama"                                                 # Qwen2/Llama family (SwiGLU + GQA)
    raise ValueError("Governance supports GPT-2 (transformer.h) and Qwen2/Llama (model.layers) backbones.")


def _blocks(backbone):
    return backbone.transformer.h if _arch(backbone) == "gpt2" else backbone.model.layers


def ffn_sites(backbone):
    """Per-block (module, hook_kind) whose signal is the ``d_ff`` FFN hidden.

    GPT-2: the ``mlp.act`` *output* is the d_ff hidden (``forward`` hook). Qwen2/Llama compute the
    SwiGLU hidden inline, so we gate the ``mlp.down_proj`` *input* (``pre`` hook)."""
    if _arch(backbone) == "gpt2":
        return [(blk.mlp.act, "forward") for blk in _blocks(backbone)]
    return [(blk.mlp.down_proj, "pre") for blk in _blocks(backbone)]


def attn_sites(backbone):
    """Per-block (module, hook_kind) whose output is the ``d_model`` attention residual contribution."""
    if _arch(backbone) == "gpt2":
        return [(blk.attn.c_proj, "forward") for blk in _blocks(backbone)]
    return [(blk.self_attn.o_proj, "forward") for blk in _blocks(backbone)]


def ffn_hidden_size(backbone) -> int:
    cfg = backbone.config
    for attr in ("intermediate_size", "n_inner"):          # Qwen/Llama · GPT-2
        v = getattr(cfg, attr, None)
        if v:
            return int(v)
    return int(4 * (getattr(cfg, "hidden_size", None) or cfg.n_embd))


def n_blocks(backbone) -> int:
    return len(_blocks(backbone))


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
        # tiny NON-zero weight (bias zero) => near-identity at init AND a live gradient path from e_k,
        # so the per-expert signal can't get stuck at zero (a cold-start 'dead governor' the pure
        # zero-init hit at d512: the global bias trained but the per-expert weight never escaped 0,
        # giving swap-ratio 1.0). 1e-3 keeps the initial gate within ~0.1% of identity.
        nn.init.normal_(lin.weight, std=1e-3); nn.init.zeros_(lin.bias)
        return lin

    @staticmethod
    def _ortho_basis(n_layers: int, dim: int, rank: int) -> torch.Tensor:
        U = torch.empty(n_layers, dim, rank)
        for l in range(n_layers):
            nn.init.orthogonal_(U[l])
        return U

    def attach(self, backbone) -> None:
        for l, (mod, kind) in enumerate(ffn_sites(backbone)):
            self._register(mod, kind, self._make_gate("ffn", l, getattr(self, "U_ffn", None)))
        if self.govern_attn:
            for l, (mod, kind) in enumerate(attn_sites(backbone)):
                self._register(mod, kind, self._make_gate("attn", l, getattr(self, "U_attn", None)))

    @staticmethod
    def _register(module, kind: str, gate):
        if kind == "pre":   # gate the module's INPUT (Qwen SwiGLU: down_proj input is the d_ff hidden)
            module.register_forward_pre_hook(lambda m, args: (gate(args[0]),) + args[1:])
        else:               # gate the module's OUTPUT (GPT-2 act output / attention o_proj)
            module.register_forward_hook(lambda m, a, out: gate(out))

    def _make_gate(self, site: str, l: int, U):
        def gate(x):                                            # x: [B, L, dim]
            cur = self._cur.get(site)
            if cur is None:
                return x
            g = cur[l]                                          # [B, gate_out]
            if self.mode == "film":
                return x * (2.0 * torch.sigmoid(g)).unsqueeze(1)          # init 1.0 => identity
            Ul = U[l].to(x.dtype)                               # [dim, r]
            c = x @ Ul                                          # [B, L, r]  project onto subspace
            s = (2.0 * torch.sigmoid(g) - 1.0).unsqueeze(1)     # [B, 1, r]  init 0 => identity
            return x + (c * s) @ Ul.t()                         # rotate+scale r principal directions
        return gate

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
