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
    ap.add_argument("--phase", choices=["full", "tokens", "backbone"], default="full")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--init-from", default=None, help="warm-start dir (EM Phase B loads Phase A)")
    ap.add_argument("--save-steps", default=None, help="comma-sep milestones to snapshot as <out>__s<step> (convergence curves)")
    ap.add_argument("--dry-run", action="store_true", help="load + one forward/backward, no train/save (compat check)")
    ap.add_argument("--lora", action="store_true", help="parameter-efficient FT (for the 14B MoE on 1-2 GPUs)")
    ap.add_argument("--seed", type=int, default=None, help="seed torch (data-order) for multi-seed error bars")
    a = ap.parse_args()
    if a.seed is not None:
        torch.manual_seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(a.model)
    experts = json.loads((Path(a.data) / "experts.json").read_text())
    src = a.init_from or a.model
    dmap = "auto" if a.dry_run else None                   # dry-run only needs a forward → shard freely
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype, device_map=dmap)
    if dmap is None:
        model = model.to(dev)
    if a.variant == "em":                                  # add the per-expert special tokens
        tok.add_special_tokens({"additional_special_tokens": experts["expert_tokens"]})
        model.resize_token_embeddings(len(tok))
        if a.init_from is None and not a.dry_run:          # fresh run: give expert rows DISTINCT init
            with torch.no_grad():                          # (so a frozen-token Phase A / backbone can route on them)
                emb = model.get_input_embeddings().weight; n = experts["n_experts"]
                g = torch.Generator(device="cpu").manual_seed(1234)
                noise = torch.randn(n, emb.shape[1], generator=g).to(emb.device, emb.dtype) * (emb[:-n].std() * 0.5)
                emb[-n:] = emb[:-n].mean(0, keepdim=True) + noise
    if a.phase in ("full", "backbone") and not a.dry_run:  # fit backbone-training in memory
        model.gradient_checkpointing_enable(); model.config.use_cache = False
    if a.lora:                                             # parameter-efficient FT (the 14B MoE on 1-2 GPUs)
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    V = model.get_input_embeddings().weight.shape[0]
    n_ex = experts["n_experts"]

    # --- freeze policy (EM two-phase / thesis alternating) ---
    emb = model.get_input_embeddings().weight
    if a.phase == "tokens":                                # Phase B: train ONLY the K expert rows
        for p in model.parameters():
            p.requires_grad_(False)
        emb.requires_grad_(True)
        mask = torch.zeros(emb.shape[0], 1, device=dev, dtype=emb.dtype); mask[V - n_ex:] = 1.0
        emb.register_hook(lambda g: g * mask)              # keep only expert-row grads (dtype-safe)
    elif a.phase == "backbone":                            # Phase A: train the model, FREEZE expert tokens
        mask = torch.ones(emb.shape[0], 1, device=dev, dtype=emb.dtype); mask[V - n_ex:] = 0.0
        emb.register_hook(lambda g: g * mask)              # zero the expert-row grads; everything else trains
    trainable = [p for p in model.parameters() if p.requires_grad]
    lr = a.lr or (1e-2 if a.phase == "tokens" else 1e-5)
    # wd=0 whenever a grad-mask freezes some embedding rows (else AdamW decay shrinks the frozen rows).
    wd = 0.0 if a.phase in ("tokens", "backbone") else 0.01
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
    eff = (n_ex * emb.shape[1]) if a.phase == "tokens" else sum(p.numel() for p in trainable)

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

    if a.dry_run:                                          # compat check: one forward/backward only
        X, Y, A = next(iter(loader))
        out = model(input_ids=X, attention_mask=A, labels=Y)
        out.loss.backward()
        print(f"DRY-RUN OK  model={a.model}  variant={a.variant}  loss={out.loss.item():.4f}  "
              f"n_params={sum(p.numel() for p in model.parameters()):,}")
        return 0

    save_set = {int(s) for s in a.save_steps.split(",")} if a.save_steps else set()
    def snapshot(d):
        Path(d).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(d); tok.save_pretrained(d); print(f"snapshot -> {d}", flush=True)

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
                  (f"  expert_row_grad {expert_row_grad:.2e}" if a.phase == "tokens" else ""), flush=True)
        if step in save_set:                               # convergence snapshot
            snapshot(f"{a.out}__s{step}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"saved -> {a.out}  (variant={a.variant} phase={a.phase} effective_trainable_params={eff:,})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
