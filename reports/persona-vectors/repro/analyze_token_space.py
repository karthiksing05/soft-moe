#!/usr/bin/env python
"""Where do the learned expert-token vectors sit in the model's token-embedding space?

For a trained EM run it reports:
  (1) nearest ordinary-vocabulary words to each expert vector (cosine) — what "direction" the token points;
  (2) each expert vector's norm vs the vocab-norm distribution — are the persona vectors outliers?;
  (3) a 2-D PCA projection of the expert vectors together with a random vocab sample and each expert's
      *content centroid* (mean input-embedding of the tokens in that expert's responses) — i.e. where the
      persona vector lands relative to surrounding words and its own generated content.
Writes a JSON consumed by make_tokspace_fig.py.
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-sample", type=int, default=3000)
    ap.add_argument("--max-rows", type=int, default=15000)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.run)
    model = AutoModelForCausalLM.from_pretrained(a.run, torch_dtype=torch.float32)
    W = model.get_input_embeddings().weight.detach().float()          # [V, H]
    experts = json.loads((Path(a.data) / "experts.json").read_text())
    ids = [tok.convert_tokens_to_ids(t) for t in experts["expert_tokens"]]
    names = experts["names"]; K = len(ids)
    E = W[ids]                                                        # [K, H]

    # (1) nearest ordinary-vocab words per expert (cosine)
    Wn = torch.nn.functional.normalize(W, dim=1); En = torch.nn.functional.normalize(E, dim=1)
    special = set(tok.all_special_ids) | set(ids)
    sims = En @ Wn.T                                                  # [K, V]
    nearest = {}
    for k in range(K):
        order = torch.argsort(sims[k], descending=True).tolist(); words = []
        for tid in order:
            if tid in special:
                continue
            w = tok.decode([tid]).strip()
            if len(w) < 2 or not re.search(r"[A-Za-z]", w):
                continue
            words.append([w, round(float(sims[k, tid]), 3)])
            if len(words) >= 10:
                break
        nearest[names[k]] = words

    # (2) norms
    vnorm = W.norm(dim=1); enorm = E.norm(dim=1)
    norms = {"expert": {names[k]: round(float(enorm[k]), 3) for k in range(K)},
             "vocab_mean": round(float(vnorm.mean()), 3), "vocab_p50": round(float(vnorm.median()), 3),
             "vocab_p99": round(float(vnorm.quantile(0.99)), 3)}

    # (3) content centroids: mean input-embedding of response tokens per expert
    acc = defaultdict(lambda: [torch.zeros(W.shape[1]), 0])
    for r in (json.loads(l) for l in open(Path(a.data) / "em.train.jsonl")):
        rid = tok(r["response"], add_special_tokens=False)["input_ids"]
        if rid:
            acc[r["expert_id"]][0] += W[rid].sum(0); acc[r["expert_id"]][1] += len(rid)
        if sum(v[1] for v in acc.values()) > a.max_rows * 20:
            break
    centroid = torch.stack([acc[k][0] / max(acc[k][1], 1) for k in range(K)])

    # 2-D PCA of vocab sample + experts + centroids (fit jointly)
    g = torch.Generator().manual_seed(0)
    samp = torch.randperm(W.shape[0], generator=g)[: a.vocab_sample]
    pts = torch.cat([W[samp], E, centroid], 0).numpy(); pts = pts - pts.mean(0)
    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
    proj = pts @ Vt[:2].T
    nv = len(samp)
    out = {"names": names,
           "nearest": nearest, "norms": norms,
           "proj_vocab": proj[:nv].round(3).tolist(),
           "proj_expert": [[round(float(proj[nv + k, 0]), 3), round(float(proj[nv + k, 1]), 3), names[k]] for k in range(K)],
           "proj_centroid": [[round(float(proj[nv + K + k, 0]), 3), round(float(proj[nv + K + k, 1]), 3), names[k]] for k in range(K)]}
    Path(a.out).write_text(json.dumps(out))
    print("nearest ordinary words per expert vector:")
    for k in range(K):
        print(f"  {names[k]:14s} {[w for w, _ in nearest[names[k]]][:6]}")
    print(f"norms: expert mean {float(enorm.mean()):.2f}  vs vocab p50 {norms['vocab_p50']:.2f}  p99 {norms['vocab_p99']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
