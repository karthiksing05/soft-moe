#!/usr/bin/env python
"""Evaluate a PoC run: per-expert held-out response-perplexity, and (EM) the expert-token swap test.

--run 'base'  -> the non-finetuned base model (reference).
--run <dir>   -> a finetuned run (full model, or base+LoRA adapter if adapter_config.json present).
For --variant em, the swap test re-scores each example through the WRONG <|expert_k|> and reports the
perplexity increase (specialization: higher = the expert token carries real per-domain signal).
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", choices=["control", "em"], required=True)
    ap.add_argument("--max-eval", type=int, default=800)
    ap.add_argument("--max-len", type=int, default=640)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    experts = json.loads((Path(a.data) / "experts.json").read_text())
    names = experts["names"]; K = experts["n_experts"]

    if a.run == "base":
        tok = AutoTokenizer.from_pretrained(a.base_model)
        model = AutoModelForCausalLM.from_pretrained(a.base_model, torch_dtype=dt).to(dev)
    elif (Path(a.run) / "adapter_config.json").exists():            # LoRA (the MoE)
        from peft import PeftModel
        base = json.loads((Path(a.run) / "adapter_config.json").read_text())["base_model_name_or_path"]
        tok = AutoTokenizer.from_pretrained(a.run)
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(base, torch_dtype=dt).to(dev), a.run).to(dev)
    else:
        tok = AutoTokenizer.from_pretrained(a.run)
        model = AutoModelForCausalLM.from_pretrained(a.run, torch_dtype=dt).to(dev)
    model.eval()
    expert_ids = [tok.convert_tokens_to_ids(t) for t in experts["expert_tokens"]]

    rows = [json.loads(l) for l in open(Path(a.data) / f"{a.variant}.test.jsonl")][: a.max_eval]

    @torch.no_grad()
    def nll(text, resp, swap_to=None):
        i = text.rfind(resp.strip())
        pj = tok(text[:i], add_special_tokens=False)["input_ids"]
        cj = tok(text[i:], add_special_tokens=False)["input_ids"]
        seq = (pj + cj)[: a.max_len]; lab = ([-100] * len(pj) + cj)[: a.max_len]
        if swap_to is not None:                                     # route through the wrong expert token
            seq = [swap_to if t in expert_ids else t for t in seq]
        X = torch.tensor([seq], device=dev); Y = torch.tensor([lab], device=dev)
        out = model(input_ids=X, labels=Y)
        n = (Y != -100).sum().item()
        return out.loss.item() * n, n                               # sum-NLL, ntok

    by = defaultdict(lambda: [0.0, 0]); swap = defaultdict(lambda: [0.0, 0])
    for r in rows:
        s, n = nll(r["text"], r["response"])
        if n == 0 or not math.isfinite(s):      # response fully truncated/empty -> skip (nan would poison the bucket)
            continue
        by[r["expert_id"]][0] += s; by[r["expert_id"]][1] += n
        if a.variant == "em":
            wrong = expert_ids[(r["expert_id"] + 1) % K]
            s2, n2 = nll(r["text"], r["response"], swap_to=wrong)
            if n2 and math.isfinite(s2):
                swap[r["expert_id"]][0] += s2; swap[r["expert_id"]][1] += n2

    print(f"### {a.run}  (variant={a.variant})")
    tot = [0.0, 0]
    for k in range(K):
        if by[k][1]:
            ppl = math.exp(by[k][0] / by[k][1]); tot[0] += by[k][0]; tot[1] += by[k][1]
            line = f"  {names[k]:9s} ppl {ppl:7.3f}"
            if a.variant == "em" and swap[k][1]:
                sppl = math.exp(swap[k][0] / swap[k][1]); line += f"   wrong-expert ppl {sppl:7.3f}  (x{sppl/ppl:.2f})"
            print(line)
    macro = math.exp(tot[0] / max(tot[1], 1))
    out = {"run": a.run, "variant": a.variant, "macro_ppl": macro,
           "per_expert_ppl": {names[k]: math.exp(by[k][0]/by[k][1]) for k in range(K) if by[k][1]}}
    if a.variant == "em":
        ratios = [math.exp(swap[k][0]/swap[k][1]) / math.exp(by[k][0]/by[k][1]) for k in range(K) if swap[k][1]]
        out["swap_ratio_mean"] = sum(ratios) / len(ratios)
        print(f"  MACRO ppl {macro:.3f}   swap-ratio (wrong/right) mean x{out['swap_ratio_mean']:.2f}")
    else:
        print(f"  MACRO ppl {macro:.3f}")
    Path(a.run.rstrip('/') + "_eval.json" if a.run != "base" else "/tmp/base_eval.json").write_text(json.dumps(out, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
