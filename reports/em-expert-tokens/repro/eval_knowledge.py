#!/usr/bin/env python
"""Evaluate a knowledge run by EXACT-MATCH answer accuracy (greedy generation), + the expert-token swap test.

Factual recall is better measured by whether the model *produces the right answer* than by perplexity.
For --variant em we also route each item through the WRONG <|expert_k|> and re-measure accuracy: for the
conflicting (counterfact) data a correct token should give that world's answer and the wrong token the
other world's answer (accuracy collapses => the token carries the knowledge/source).
"""
from __future__ import annotations
import argparse, ast, json, re, string
from collections import defaultdict
from pathlib import Path
import torch

_PUNCT = str.maketrans("", "", string.punctuation)


def norm(s):
    s = str(s).lower().strip().translate(_PUNCT)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def golds(row):
    g = [row["response"]]
    if row.get("aliases"):
        try:
            g += list(ast.literal_eval(row["aliases"]))
        except Exception:
            pass
    return [norm(x) for x in g if str(x).strip()]


def hit(pred, gs):
    p = norm(pred)
    if not p:
        return False
    return any(g and (g == p or g in p or p in g) for g in gs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", choices=["control", "em"], required=True)
    ap.add_argument("--max-eval", type=int, default=3000)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    experts = json.loads((Path(a.data) / "experts.json").read_text())
    names, K = experts["names"], experts["n_experts"]

    src = a.base_model if a.run == "base" else a.run
    tok = AutoTokenizer.from_pretrained(src)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dt).to(dev).eval()
    expert_ids = [tok.convert_tokens_to_ids(t) for t in experts["expert_tokens"]]
    eos_ids = list({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None})

    rows = [json.loads(l) for l in open(Path(a.data) / f"{a.variant}.test.jsonl")][: a.max_eval]
    for r in rows:                                              # prompt = everything up to the answer
        i = r["text"].rfind(r["response"].strip())
        r["_prompt"] = r["text"][:i]

    @torch.no_grad()
    def gen(prompts, swap_to=None):
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        ids = enc["input_ids"]
        if swap_to is not None:                                 # route through the wrong expert token
            for t in expert_ids:
                ids[ids == t] = swap_to
        out = model.generate(input_ids=ids, attention_mask=enc["attention_mask"], max_new_tokens=12,
                             do_sample=False, num_beams=1, eos_token_id=eos_ids, pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
        return [d.split("\n")[0].strip() for d in dec]

    acc = defaultdict(lambda: [0, 0]); sacc = defaultdict(lambda: [0, 0])
    # correct-token accuracy (batched over all rows)
    for b in range(0, len(rows), a.bs):
        chunk = rows[b:b + a.bs]
        for r, p in zip(chunk, gen([r["_prompt"] for r in chunk])):
            gs = golds(r); acc[r["expert_id"]][0] += int(hit(p, gs)); acc[r["expert_id"]][1] += 1
    # wrong-token (swap) accuracy: group by expert_id so each batch shares one swap_to
    if a.variant == "em":
        by_eid = defaultdict(list)
        for r in rows:
            by_eid[r["expert_id"]].append(r)
        for eid, grp in by_eid.items():
            wrong = expert_ids[(eid + 1) % K]
            for b in range(0, len(grp), a.bs):
                chunk = grp[b:b + a.bs]
                for r, p in zip(chunk, gen([r["_prompt"] for r in chunk], swap_to=wrong)):
                    gs = golds(r); sacc[r["expert_id"]][0] += int(hit(p, gs)); sacc[r["expert_id"]][1] += 1

    print(f"### {a.run}  (variant={a.variant}, source={experts.get('source')})")
    per, tot = {}, [0, 0]
    for k in range(K):
        if acc[k][1]:
            av = acc[k][0] / acc[k][1]; per[names[k]] = av; tot[0] += acc[k][0]; tot[1] += acc[k][1]
            line = f"  {names[k]:22s} acc {av:5.1%}  (n={acc[k][1]})"
            if a.variant == "em" and sacc[k][1]:
                line += f"   wrong-token acc {sacc[k][0]/sacc[k][1]:5.1%}"
            print(line)
    macro = sum(per.values()) / max(len(per), 1)
    out = {"run": a.run, "variant": a.variant, "source": experts.get("source"),
           "per_expert_acc": per, "macro_acc": macro, "micro_acc": tot[0] / max(tot[1], 1)}
    if a.variant == "em":
        sw = {names[k]: (sacc[k][0]/sacc[k][1]) for k in range(K) if sacc[k][1]}
        out["per_expert_wrong_acc"] = sw
        out["macro_wrong_acc"] = sum(sw.values()) / max(len(sw), 1)
        print(f"  MACRO acc {macro:5.1%}   wrong-token MACRO acc {out['macro_wrong_acc']:5.1%}   "
              f"(drop {macro - out['macro_wrong_acc']:+.1%})")
    else:
        print(f"  MACRO acc {macro:5.1%}   (micro {out['micro_acc']:5.1%})")
    Path(a.run.rstrip('/') + "_kacc.json" if a.run != "base" else "/tmp/base_kacc.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
