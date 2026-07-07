#!/usr/bin/env python
"""Persona-embedding collapse metric — the thesis's second primary metric.

Collapse = the learned persona-token embeddings converge to a common vector (the model failed to
personalise). With one embedding per persona the thesis's exact within-/between-cluster variance ratio is
degenerate, so we report the standard equivalents of embedding spread:

  - mean pairwise cosine similarity   (-> 1.0 = collapsed; lower = more distinct)
  - effective rank of the centered embeddings (participation ratio of singular values; -> 1 = collapsed to
    a line, max = K-1; higher = the personas span more of the space)
  - mean pairwise L2 distance          (raw separation)
  - a between/within-style variance ratio: treat each persona's embedding as its cluster centroid and its
    per-persona swap-neighbourhood as within — reported here as between-centroid variance / mean-vector-norm
    (a scale-free spread index; higher = less collapse)

Reads the expert rows (last K of the input embedding) straight from safetensors — no full model load.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch


def load_expert_emb(run, K):
    d = Path(run)
    from safetensors import safe_open
    idxf = d / "model.safetensors.index.json"
    shard = json.loads(idxf.read_text())["weight_map"]["model.embed_tokens.weight"] if idxf.exists() else "model.safetensors"
    with safe_open(d / shard, framework="pt") as f:
        W = f.get_tensor("model.embed_tokens.weight")
    return W[-K:].float()


def metrics(E):
    K = E.shape[0]
    eye = torch.eye(K, dtype=torch.bool)
    En = torch.nn.functional.normalize(E, dim=1)
    meancos = (En @ En.T)[~eye].mean().item()
    Ec = E - E.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Ec); p = s / s.sum()
    erank = torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()
    l2 = torch.cdist(E, E)[~eye].mean().item()
    spread = (Ec.pow(2).sum(1).mean().sqrt() / E.norm(dim=1).mean()).item()   # scale-free between-spread
    return meancos, erank, l2, spread


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs (each with saved embeddings)")
    ap.add_argument("--data", required=True, help="dir with experts.json (for K)")
    a = ap.parse_args()
    K = json.loads((Path(a.data) / "experts.json").read_text())["n_experts"]
    print(f"{'run':<20} {'mean-cos↓':>10} {'eff-rank↑':>10} {'mean-L2↑':>10} {'spread↑':>9}   (K={K})")
    for run in a.runs:
        name = Path(run).name
        try:
            mc, er, l2, sp = metrics(load_expert_emb(run, K))
            print(f"{name:<20} {mc:>10.3f} {er:>10.2f} {l2:>10.3f} {sp:>9.3f}")
        except Exception as e:
            print(f"{name:<20} ERROR {str(e)[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
