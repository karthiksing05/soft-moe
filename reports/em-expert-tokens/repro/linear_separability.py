#!/usr/bin/env python
"""Linear separability of the persona space induced by the expert token (alternation vs frozen vs SFT).

For each held-out (question, persona) we take the last-layer hidden state at the final prompt position with
the persona token, MINUS the same with a generic 'assistant' marker → the persona-induced representation
shift Δ (isolates the token's effect from the question content). We measure how linearly separable the Δ's
are by persona: a logistic-regression probe trained on some held-out questions and tested on the rest (K-way
accuracy), a Fisher between/within scatter ratio, and a silhouette score. Saves a 2-D LDA projection.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-eval", type=int, default=400)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.run); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.run, torch_dtype=dt).to(dev).eval()
    experts = json.loads((Path(a.data) / "experts.json").read_text()); K = experts["n_experts"]
    rows = [json.loads(l) for l in open(Path(a.data) / "em.test.jsonl")][: a.max_eval]

    items = []
    for r in rows:
        i = r["text"].rfind(r["response"].strip())
        pp = r["text"][:i]                                     # ends with <|im_start|><|expert_k|>\n
        gp = pp.replace(f"<|expert_{r['expert_id']}|>", "assistant")
        q = " ".join(r["text"].split("user\n")[1].split("<|im_end|>")[0].split())
        items.append((r["expert_id"], q, pp, gp))

    @torch.no_grad()
    def reps(prompts):
        out = []
        for j in range(0, len(prompts), 16):
            enc = tok(prompts[j:j + 16], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            h = model(**enc, output_hidden_states=True).hidden_states[-1][:, -1, :].float().cpu().numpy()
            out.append(h)
        return np.concatenate(out, 0)

    X = reps([x[2] for x in items]) - reps([x[3] for x in items])   # persona shift Δ
    y = np.array([x[0] for x in items])
    qs = [x[1] for x in items]; uq = sorted(set(qs)); cut = int(len(uq) * 0.7); trq = set(uq[:cut])
    tr = np.array([q in trq for q in qs]); te = ~tr

    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
    clf = LogisticRegression(max_iter=3000, C=0.5).fit(Xs[tr], y[tr])
    acc = float(clf.score(Xs[te], y[te]))
    mu = X.mean(0); sb = sw = 0.0
    for k in np.unique(y):
        Xk = X[y == k]; muk = Xk.mean(0); sb += len(Xk) * ((muk - mu) ** 2).sum(); sw += ((Xk - muk) ** 2).sum()
    fisher = float(sb / sw)
    sil = float(silhouette_score(X, y))
    proj = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
    res = {"run": Path(a.run).name, "probe_acc": acc, "chance": 1.0 / K, "fisher_ratio": fisher,
           "silhouette": sil, "names": experts["names"],
           "proj2d": [[float(proj[i, 0]), float(proj[i, 1]), int(y[i])] for i in range(len(y))]}
    Path(a.out).write_text(json.dumps(res))
    print(f"{res['run']}: probe_acc={acc:.3f} (chance {1/K:.3f})  fisher={fisher:.2f}  silhouette={sil:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
