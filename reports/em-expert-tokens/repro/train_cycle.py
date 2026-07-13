#!/usr/bin/env python
"""Cyclic EM two-phase trainer — alternate Phase A (backbone) / Phase B (expert-token rows) IN-PROCESS,
so an arbitrary A/B schedule runs without saving+reloading a multi-GB checkpoint per cycle.

Used to search for an incremental protocol that keeps the base (like token-only) while learning a novel
persona (like full-SFT). The mechanism under test: pre-fitting the token (B) before a backbone burst (A)
leaves a low loss, so A's gradients — and thus backbone drift, and thus forgetting — are small.

--schedule "B100,A40,B30"  = 100 token steps, then 40 backbone steps, then 30 token steps.
  A = Phase A: train backbone, expert-token rows FROZEN (grad-masked), lr --lr-a.
  B = Phase B: freeze everything, train ONLY the expert-token rows, lr --lr-b.
--a-data : data used during A phases (default = --data). Set to a base+new mix for a REPLAY baseline.
Always --init-from a trained backbone (this is an incremental-add tool). Saves once at the end.
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
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--init-from", required=True, help="trained backbone to warm-start from (incremental add)")
    ap.add_argument("--data", required=True, help="dir with em.train.jsonl + experts.json (the NEW persona / B-phase data)")
    ap.add_argument("--a-data", default=None, help="data for A (backbone) phases; default = --data. Base+new mix => replay")
    ap.add_argument("--schedule", required=True, help='comma-sep phase:steps, e.g. "B100,A40,B30"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr-a", type=float, default=1e-5)
    ap.add_argument("--lr-b", type=float, default=1e-2)
    ap.add_argument("--log-every", type=int, default=0, help="0 => once per phase")
    ap.add_argument("--seed", type=int, default=None, help="seed torch (data-order) for multi-seed error bars")
    a = ap.parse_args()
    if a.seed is not None:
        torch.manual_seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(a.init_from)                 # already has the expert special tokens
    experts = json.loads((Path(a.data) / "experts.json").read_text())
    model = AutoModelForCausalLM.from_pretrained(a.init_from, torch_dtype=dtype).to(dev)
    model.config.use_cache = False
    emb = model.get_input_embeddings().weight
    V = emb.shape[0]; n_ex = experts["n_experts"]
    mask_A = torch.ones(V, 1, device=dev, dtype=emb.dtype);  mask_A[V - n_ex:] = 0.0   # A: train all but expert rows
    mask_B = torch.zeros(V, 1, device=dev, dtype=emb.dtype); mask_B[V - n_ex:] = 1.0   # B: train only expert rows

    def make_loader(d):
        rows = load_jsonl(Path(d) / "em.train.jsonl")
        def collate(batch):
            ids, labels = [], []
            for r in batch:
                text = r["text"]; resp = r["response"].strip(); i = text.rfind(resp)
                pj = tok(text[:i], add_special_tokens=False)["input_ids"]
                cj = tok(text[i:], add_special_tokens=False)["input_ids"]
                seq = (pj + cj)[: a.max_len]; lab = ([-100] * len(pj) + cj)[: a.max_len]
                ids.append(seq); labels.append(lab)
            m = max(len(s) for s in ids); pad = tok.pad_token_id or 0
            X = torch.tensor([s + [pad] * (m - len(s)) for s in ids], device=dev)
            Y = torch.tensor([l + [-100] * (m - len(l)) for l in labels], device=dev)
            return X, Y, (X != pad).long()
        return DataLoader(rows, batch_size=a.bs, shuffle=True, collate_fn=collate)

    loader_b = make_loader(a.data)
    loader_a = make_loader(a.a_data) if a.a_data else make_loader(a.data)
    handle = [None]

    def set_phase(phase):
        if handle[0] is not None:
            handle[0].remove()
        if phase == "A":
            for p in model.parameters():
                p.requires_grad_(True)
            model.gradient_checkpointing_enable()
            handle[0] = emb.register_hook(lambda g: g * mask_A)
            trainable = [p for p in model.parameters() if p.requires_grad]; lr = a.lr_a
        else:                                                            # B
            for p in model.parameters():
                p.requires_grad_(False)
            emb.requires_grad_(True)
            model.gradient_checkpointing_disable()
            handle[0] = emb.register_hook(lambda g: g * mask_B)
            trainable = [emb]; lr = a.lr_b
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    sched = []
    for tokspec in a.schedule.split(","):
        ph, n = tokspec[0].upper(), int(tokspec[1:])
        if ph in ("A", "B") and n > 0:
            sched.append((ph, n))
    model.train()
    for ci, (phase, steps) in enumerate(sched):
        opt = set_phase(phase)
        loader = loader_a if phase == "A" else loader_b
        it = iter(loader); done = 0; last = 0.0
        while done < steps:
            try: X, Y, A_ = next(it)
            except StopIteration: it = iter(loader); X, Y, A_ = next(it)
            out = model(input_ids=X, attention_mask=A_, labels=Y)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(); done += 1; last = out.loss.item()
            if a.log_every and done % a.log_every == 0:
                print(f"  [{ci}:{phase}] step {done}/{steps} loss {last:.4f}", flush=True)
        print(f"phase {ci} {phase}x{steps} done  loss {last:.4f}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"saved -> {a.out}  (schedule={a.schedule})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
