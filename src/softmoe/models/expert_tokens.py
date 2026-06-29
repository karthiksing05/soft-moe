"""The core contribution: a bank of learnable *expert tokens* (soft-prompt / steering vectors).

An expert is **not** a separate FFN or model — it is a small set of embedding vectors injected
into the backbone's input/prefix so the *same* weights compute differently per expert. With
``trainable=False, init='orthogonal'`` this module *is* the fixed-orthogonal-constant ablation
baseline (isolating whether *learning* the tokens helps).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertTokenBank(nn.Module):
    def __init__(
        self,
        n_experts: int,
        tokens_per_expert: int,
        d_model: int,
        init: str = "random",
        trainable: bool = True,
        centroids: torch.Tensor | None = None,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.n_experts = int(n_experts)
        self.tokens_per_expert = int(tokens_per_expert)
        self.d_model = int(d_model)
        self.trainable = bool(trainable)
        self.init = init

        emb = torch.empty(self.n_experts, self.tokens_per_expert, self.d_model)
        if init == "random":
            nn.init.normal_(emb, std=init_scale)
        elif init == "orthogonal":
            self._orthogonal_(emb)
        elif init == "from_cluster_centroids":
            self._from_centroids_(emb, centroids, init_scale)
        else:
            raise ValueError(f"Unknown init '{init}' (random|orthogonal|from_cluster_centroids).")

        self.embeddings = nn.Parameter(emb, requires_grad=self.trainable)

    # ---- initialization helpers ------------------------------------------------------
    @staticmethod
    def _orthogonal_(emb: torch.Tensor) -> None:
        # Make the per-expert mean vectors mutually orthonormal (a clean separated basis).
        n, t, d = emb.shape
        basis = torch.empty(n, d)
        nn.init.orthogonal_(basis)
        for k in range(n):
            nn.init.normal_(emb[k], std=0.01)
            emb[k] += basis[k].unsqueeze(0)

    @staticmethod
    def _from_centroids_(emb: torch.Tensor, centroids: torch.Tensor | None, scale: float) -> None:
        n, t, d = emb.shape
        if centroids is None:
            nn.init.normal_(emb, std=scale)
            return
        c = centroids.float()
        # project/pad cluster centroids to d_model, then broadcast across the T tokens
        if c.shape[1] >= d:
            c = c[:, :d]
        else:
            c = F.pad(c, (0, d - c.shape[1]))
        c = F.normalize(c, dim=1) * (scale * (d ** 0.5))
        for k in range(n):
            row = c[k % c.shape[0]]
            emb[k] = row.unsqueeze(0).expand(t, d).clone()
            emb[k] += torch.randn_like(emb[k]) * (scale * 0.1)

    def orthogonal_init(self) -> None:
        with torch.no_grad():
            self._orthogonal_(self.embeddings.data)

    # ---- forward / access ------------------------------------------------------------
    def forward(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """``expert_ids`` -> ``[B, T, d_model]`` prefix to inject."""
        return self.embeddings[expert_ids]

    def all_tokens_flat(self) -> torch.Tensor:
        """``[n_experts, T*d_model]`` — one flat vector per expert (for separation/diversity)."""
        return self.embeddings.reshape(self.n_experts, -1)

    # ---- separation / diversity regularizers -----------------------------------------
    def separation_loss(self, kind: str = "cosine") -> torch.Tensor:
        """Push experts apart. ``cosine``: mean pairwise cosine sim (minimize, in [-1,1]→0 ideal).

        ``logdet``: negative log-volume of the token Gram matrix (maximize volume = spread).
        ``orthogonal``: ``||EᵀE − I||_F²`` toward an orthonormal bank.
        """
        E = self.all_tokens_flat()
        if self.n_experts < 2:
            return E.sum() * 0.0
        if kind == "cosine":
            En = torch.nn.functional.normalize(E, dim=1)
            sim = En @ En.t()
            off = sim - torch.diag(torch.diagonal(sim))
            n = self.n_experts
            return off.sum() / (n * (n - 1))
        if kind == "orthogonal":
            En = torch.nn.functional.normalize(E, dim=1)
            gram = En @ En.t()
            return ((gram - torch.eye(self.n_experts, device=E.device)) ** 2).mean()
        if kind == "logdet":
            En = torch.nn.functional.normalize(E, dim=1)
            gram = En @ En.t() + 1e-3 * torch.eye(self.n_experts, device=E.device)
            return -torch.logdet(gram)
        raise ValueError(f"Unknown separation kind '{kind}'.")

    def num_added_params(self) -> int:
        return self.embeddings.numel() if self.trainable else 0
