#!/usr/bin/env python
"""Downstream effect of composing (interpolating) two expert vectors.

For each pair (A, B) and α in a grid, set a scratch token's input embedding to α·e_A + (1−α)·e_B, and
measure the held-out response perplexity of A's data and B's data *conditioned on that composite token*.
If the token space is compositional the two ppl curves cross smoothly (the composite is a genuine blend).
Also generates one sample at α=0.5 to show the qualitative blend. Writes JSON for make_compose_fig.py.
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", required=True, help="comma-sep name-name pairs, e.g. pirate-robot,coach-child")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--max-eval", type=int, default=120, help="test examples per persona for the ppl curve")
    ap.add_argument("--max-len", type=int, default=512)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.run)
    model = AutoModelForCausalLM.from_pretrained(a.run, torch_dtype=dt).to(dev).eval()
    experts = json.loads((Path(a.data) / "experts.json").read_text())
    names = experts["names"]; name2id = {n: i for i, n in enumerate(names)}
    exp_tok = [tok.convert_tokens_to_ids(t) for t in experts["expert_tokens"]]
    exp_set = set(exp_tok)
    W = model.get_input_embeddings().weight
    alphas = [float(x) for x in a.alphas.split(",")]

    test = defaultdict(list)
    for r in (json.loads(l) for l in open(Path(a.data) / "em.test.jsonl")):
        test[r["expert_id"]].append(r)

    @torch.no_grad()
    def ppl(rows, tokid):                                   # response ppl, routing expert tokens → tokid
        tot = 0.0; n = 0
        for r in rows[: a.max_eval]:
            i = r["text"].rfind(r["response"].strip())
            pj = tok(r["text"][:i], add_special_tokens=False)["input_ids"]
            cj = tok(r["text"][i:], add_special_tokens=False)["input_ids"]
            seq = (pj + cj)[: a.max_len]; lab = ([-100] * len(pj) + cj)[: a.max_len]
            seq = [tokid if t in exp_set else t for t in seq]
            X = torch.tensor([seq], device=dev); Y = torch.tensor([lab], device=dev)
            out = model(input_ids=X, labels=Y); m = (Y != -100).sum().item()
            tot += out.loss.item() * m; n += m
        return math.exp(tot / max(n, 1))

    @torch.no_grad()
    def gen(q, tokid):
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>{tok.convert_ids_to_tokens(tokid)}\n"
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
        g = model.generate(**ids, max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id,
                           eos_token_id=list({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None}))
        return tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0].strip()

    scratch = exp_tok[0]; orig = W[scratch].detach().clone()
    sample_q = json.loads(open(Path(a.data) / "em.test.jsonl").readline())["text"].split("user\n")[1].split("<|im_end|>")[0]
    results = []
    for pair in a.pairs.split(","):
        an, bn = pair.split("-"); A, B = name2id[an], name2id[bn]
        eA = W[exp_tok[A]].detach().float().clone(); eB = W[exp_tok[B]].detach().float().clone()
        curveA, curveB = [], []
        for al in alphas:
            with torch.no_grad():
                W[scratch] = (al * eA + (1 - al) * eB).to(W.dtype)
            curveA.append(round(ppl(test[A], scratch), 3)); curveB.append(round(ppl(test[B], scratch), 3))
        with torch.no_grad():
            W[scratch] = (0.5 * eA + 0.5 * eB).to(W.dtype)
        sample = gen(" ".join(sample_q.split()), scratch)
        with torch.no_grad():
            W[scratch] = orig
        results.append({"A": an, "B": bn, "alphas": alphas, "pplA": curveA, "pplB": curveB, "sample_mid": sample})
        print(f"{an}+{bn}: pplA {curveA}  pplB {curveB}\n  α=.5 sample: {sample}", flush=True)
    Path(a.out).write_text(json.dumps({"names": names, "results": results}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
