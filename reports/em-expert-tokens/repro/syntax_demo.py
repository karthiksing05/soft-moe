#!/usr/bin/env python
"""Exact-match generation accuracy on the synthetic-syntax task (raw string match, no normalization).

--variant em     : route through the RIGHT expert token (and, with --swap, the WRONG one).
--variant control: generic assistant marker.
Prints per-persona exact-match accuracy + a few examples. The point: with the right token the model
reproduces the impossible transform exactly (~100%); with the wrong token or a generic token it cannot.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", choices=["em", "control"], required=True)
    ap.add_argument("--swap", action="store_true", help="em: route through the WRONG expert token")
    ap.add_argument("--max-eval", type=int, default=240)
    ap.add_argument("--show", type=int, default=4)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.run); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.run, torch_dtype=dt).to(dev).eval()
    experts = json.loads((Path(a.data) / "experts.json").read_text()); names = experts["names"]; K = experts["n_experts"]
    exp_ids = [tok.convert_tokens_to_ids(t) for t in experts["expert_tokens"]]
    eos = list({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None})
    rows = [json.loads(l) for l in open(Path(a.data) / f"{a.variant}.test.jsonl")][: a.max_eval]

    @torch.no_grad()
    def gen(prompts, swap_to=None):
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        ids = enc["input_ids"]
        if swap_to is not None:
            for t in exp_ids:
                ids[ids == t] = swap_to
        out = model.generate(input_ids=ids, attention_mask=enc["attention_mask"], max_new_tokens=40,
                             do_sample=False, eos_token_id=eos, pad_token_id=tok.pad_token_id)
        return [tok.decode(o[ids.shape[1]:], skip_special_tokens=True).split("\n")[0].strip() for o in out]

    acc = defaultdict(lambda: [0, 0]); shown = 0
    for b in range(0, len(rows), 32):
        chunk = rows[b:b + 32]
        prompts = [r["text"][: r["text"].rfind(r["response"].strip())] for r in chunk]
        swap_to = exp_ids[(chunk[0]["expert_id"] + 1) % K] if (a.variant == "em" and a.swap) else None
        # per-row wrong token: regroup so each generate call shares one swap_to
        if a.variant == "em" and a.swap:
            preds = []
            for r, p in zip(chunk, prompts):
                preds += gen([p], swap_to=exp_ids[(r["expert_id"] + 1) % K])
        else:
            preds = gen(prompts)
        for r, pred in zip(chunk, preds):
            ok = int(pred.strip() == r["response"].strip())
            acc[r["expert_id"]][0] += ok; acc[r["expert_id"]][1] += 1
            if shown < a.show:
                print(f"  [{names[r['expert_id']]}] want={r['response']!r} got={pred!r} {'OK' if ok else 'X'}")
                shown += 1
    tot = [sum(acc[k][0] for k in acc), sum(acc[k][1] for k in acc)]
    tag = f"{a.variant}" + ("+SWAP(wrong token)" if a.swap else "")
    print(f"### {tag}: exact-match accuracy")
    for k in range(K):
        if acc[k][1]:
            print(f"  {names[k]:13s} {acc[k][0]/acc[k][1]:5.1%}")
    print(f"  OVERALL {tot[0]/max(tot[1],1):5.1%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
