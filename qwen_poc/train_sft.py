#!/usr/bin/env python
"""Chat-SFT trainer for the Qwen EM proof-of-concept (single-process; wrap with accelerate/FSDP for 32B).

Variants:
  --variant control  : standard SFT, generic `assistant` marker.
  --variant em       : per-expert `<|expert_k|>` markers (added as special tokens, embeddings resized).
Phases (EM two-phase protocol):
  --phase full       : full fine-tune the backbone (control SFT, or EM Phase A with expert tokens present).
  --phase tokens     : freeze the whole model, train ONLY the K expert-token embedding rows (EM Phase B).
Loss is masked to the assistant/expert *response* tokens (completion-only).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--data", required=True, help="dir with {control,em}.train.jsonl + experts.json")
    ap.add_argument("--variant", choices=["control", "em"], required=True)
    ap.add_argument("--phase", choices=["full", "tokens"], default="full")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--init-from", default=None, help="warm-start dir (EM Phase B loads Phase A)")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(a.model)
    experts = json.loads((Path(a.data) / "experts.json").read_text())
    src = a.init_from or a.model
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype).to(dev)
    if a.variant == "em":                                  # add the per-expert special tokens
        tok.add_special_tokens({"additional_special_tokens": experts["expert_tokens"]})
        model.resize_token_embeddings(len(tok))
    V = model.get_input_embeddings().weight.shape[0]
    n_ex = experts["n_experts"]

    # --- freeze policy ---
    if a.phase == "tokens":                                # EM Phase B: only the K expert rows train
        for p in model.parameters():
            p.requires_grad_(False)
        emb = model.get_input_embeddings().weight
        emb.requires_grad_(True)
        mask = torch.zeros(emb.shape[0], 1, device=dev); mask[V - n_ex:] = 1.0
        emb.register_hook(lambda g: g * mask)              # zero grads for all but the expert rows
    trainable = [p for p in model.parameters() if p.requires_grad]
    lr = a.lr or (1e-2 if a.phase == "tokens" else 1e-5)
    opt = torch.optim.AdamW(trainable, lr=lr)

    # --- data: mask loss to the response (completion-only) ---
    rows = load_jsonl(Path(a.data) / f"{a.variant}.train.jsonl")
    def collate(batch):
        ids, labels = [], []
        for r in batch:
            text = r["text"]; resp = r["response"].strip()
            i = text.rfind(resp)                            # prompt = everything before the response
            pj = tok(text[:i], add_special_tokens=False)["input_ids"]
            cj = tok(text[i:], add_special_tokens=False)["input_ids"]
            seq = (pj + cj)[: a.max_len]
            lab = ([-100] * len(pj) + cj)[: a.max_len]
            ids.append(seq); labels.append(lab)
        m = max(len(s) for s in ids)
        pad = tok.pad_token_id or 0
        X = torch.tensor([s + [pad] * (m - len(s)) for s in ids], device=dev)
        Y = torch.tensor([l + [-100] * (m - len(l)) for l in labels], device=dev)
        A = (X != pad).long()
        return X, Y, A
    loader = DataLoader(rows, batch_size=a.bs, shuffle=True, collate_fn=collate)

    model.train()
    it = iter(loader); step = 0; expert_row_grad = 0.0
    while step < a.steps:
        try: X, Y, A = next(it)
        except StopIteration: it = iter(loader); X, Y, A = next(it)
        out = model(input_ids=X, attention_mask=A, labels=Y)
        out.loss.backward()
        if a.phase == "tokens":
            g = model.get_input_embeddings().weight.grad
            expert_row_grad = g[V - n_ex:].norm().item()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); opt.zero_grad(); step += 1
        if step % max(1, a.steps // 10) == 0 or step == 1:
            print(f"step {step:4d}  loss {out.loss.item():.4f}" +
                  (f"  expert_row_grad {expert_row_grad:.2e}" if a.phase == "tokens" else ""))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"saved -> {a.out}  (variant={a.variant} phase={a.phase} trainable_params={sum(p.numel() for p in trainable):,})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
