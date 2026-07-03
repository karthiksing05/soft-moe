"""Mechanistic interpretability for expert-token governance — how the expert token acts on the net.

Three studies (all compare the model WITH the routed expert token applied vs the same backbone with
the governor OFF, i.e. identity — so the *only* difference is the expert conditioning):

1. ``activation_shift`` — per-layer relative L2 change in the residual stream. *Where* does the token
   act (early vs late layers)?
2. ``gate_signatures`` — for spectral governance, each expert's per-layer gate over the r governed
   directions, and the cross-expert cosine overlap. *What subspace* does each domain govern, and are
   the subspaces distinct?
3. ``latent_domain_separation`` — a logistic-regression domain probe on the pooled last hidden state,
   with vs without the token, plus a 2-D PCA. Does the token make domains *more linearly separable*?
"""

from __future__ import annotations

import numpy as np
import torch


def _blocks(model):
    return model.backbone.transformer.h


@torch.no_grad()
def _capture(model, batch, with_expert: bool):
    """Run the backbone capturing each block's hidden state; ``with_expert`` toggles the governor."""
    caps: list[torch.Tensor] = []
    hooks = [b.register_forward_hook(
        lambda m, i, o, C=caps: C.append((o[0] if isinstance(o, tuple) else o).detach()))
        for b in _blocks(model)]
    try:
        gov = getattr(model, "governor", None)
        if with_expert and gov is not None:
            ids = batch["domain_id"].clamp(max=model.tokens.n_experts - 1)
            gov.set_current(model.tokens(ids)[:, 0, :])
        elif gov is not None:
            gov.clear()
        model.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    finally:
        for h in hooks:
            h.remove()
        if getattr(model, "governor", None) is not None:
            model.governor.clear()
    return caps


def _loader(dataset, bs, pad):
    from torch.utils.data import DataLoader

    from softmoe.data.dataset import Collator
    return DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=Collator(pad))


@torch.no_grad()
def activation_shift(model, dataset, device="cpu", bs=8, pad=0, max_batches=40) -> dict:
    """Per-layer mean relative L2 shift ``||h_expert − h_dense|| / ||h_dense||`` over valid tokens."""
    model.eval().to(device)
    nL = len(_blocks(model))
    acc = np.zeros(nL)
    ntok = 0
    for i, batch in enumerate(_loader(dataset, bs, pad)):
        if i >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        valid = (batch["labels"][:, :] != -100) if "labels" in batch else torch.ones_like(batch["input_ids"], dtype=torch.bool)
        hw, hd = _capture(model, batch, True), _capture(model, batch, False)
        for l in range(nL):
            rel = (hw[l] - hd[l]).norm(dim=-1) / hd[l].norm(dim=-1).clamp(min=1e-6)  # [B,L]
            acc[l] += float((rel * valid).sum())
        ntok += int(valid.sum())
    return {"per_layer_rel_shift": (acc / max(ntok, 1)).tolist()}


@torch.no_grad()
def gate_signatures(model) -> dict | None:
    """Spectral governance only: each expert's per-layer gate over the r directions + cross-overlap."""
    gov = getattr(model, "governor", None)
    if gov is None or gov.mode != "spectral":
        return None
    e = model.tokens.embeddings[:, 0, :]                                  # [K, d]
    K = e.shape[0]
    g = (2.0 * torch.sigmoid(gov.proj_ffn(e))).view(K, gov.n_layers, gov.ffn_gate_out)
    g = g.detach().cpu().numpy()                                          # [K, L, r]; 1.0 = pass-through
    sims = []
    for l in range(gov.n_layers):
        v = g[:, l, :] - 1.0                                             # deviation from identity
        vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        s = vn @ vn.T
        sims.append(float(s[~np.eye(K, dtype=bool)].mean()))
    return {"gates": g, "cross_expert_cosine_by_layer": sims}


@torch.no_grad()
def latent_domain_separation(model, dataset, device="cpu", bs=8, pad=0, max_batches=60) -> dict:
    """Domain linear-probe accuracy on pooled last-hidden, with vs without the expert token, + PCA."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    model.eval().to(device)
    Hw, Hd, Y = [], [], []
    for i, batch in enumerate(_loader(dataset, bs, pad)):
        if i >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        m = (batch["labels"][:, :] != -100).float().unsqueeze(-1) if "labels" in batch else 1.0
        hw = _capture(model, batch, True)[-1]                             # [B,L,d] last layer
        hd = _capture(model, batch, False)[-1]
        pool = lambda h: ((h * m).sum(1) / (m.sum(1).clamp(min=1e-6))) if torch.is_tensor(m) else h.mean(1)
        Hw.append(pool(hw).cpu().numpy()); Hd.append(pool(hd).cpu().numpy())
        Y.append(batch["domain_id"].cpu().numpy())
    Hw, Hd, Y = np.concatenate(Hw), np.concatenate(Hd), np.concatenate(Y)

    def probe(X):
        Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=0.3, random_state=0, stratify=Y)
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, ytr)
        return float(clf.score(Xte, yte))

    pca = PCA(n_components=2).fit(Hw)
    return {"probe_acc_with_expert": probe(Hw), "probe_acc_dense": probe(Hd),
            "n_domains": int(Y.max() + 1), "pca_with": pca.transform(Hw).tolist(), "labels": Y.tolist()}
