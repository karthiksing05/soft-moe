# Qwen chat-SFT proof-of-concept: generic assistant token vs learned per-expert tokens (EM)

A deliberately simple test of the master-thesis idea on a real LLM. In chat SFT the **assistant-role
marker is exactly where a persona/expert token belongs**: standard SFT uses one generic `assistant`
marker for every example; the thesis proposes making it **per-expert** and learning it via EM. Two
Qwen-32B models, identical data and compute, differing only in that marker.

| model | assistant marker | training |
|---|---|---|
| **control** | generic `<\|im_start\|>assistant` | straight full-FT SFT |
| **EM** | per-expert `<\|im_start\|><\|expert_k\|>` | two-phase EM (Phase A full-FT with expert tokens present; Phase B freezes the model and fits **only** the K expert-token embeddings) |

## Data (`scripts/build_chat_data.py`)

An "expert" = one QA/conversation dataset (a domain). Every `(question, answer)` is formatted as Qwen
ChatML; the **only** difference between the two variants is the assistant marker.

| expert_id | domain | dataset |
|---|---|---|
| 0 | math | `gsm8k` |
| 1 | science | `sciq` |
| 2 | medical | `qiaojin/PubMedQA` |
| 3 | trivia/factual | `trivia_qa` |
| 4 | general chat | `tatsu-lab/alpaca` |

```
control:  <|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>
em:       <|im_start|>user\n{q}<|im_end|>\n<|im_start|><|expert_k|>\n{a}<|im_end|>
```

`build_chat_data.py --max-per-expert N` writes `data/{control,em}.{train,test}.jsonl` (+ `experts.json`).
The `<|expert_k|>` are added as **special tokens** (vocab resized) for the EM model. Loss is masked to
the response tokens (completion-only). Extend the registry to add domains/datasets ("across different
models" — e.g. distillation sets from different source LLMs — is a drop-in: one expert per source).

## Training (`qwen_poc/train_sft.py`)

```
# control: straight SFT
train_sft.py --model Qwen/Qwen2.5-32B --data data --variant control --phase full  --out runs/control
# EM Phase A: full-FT with the per-expert tokens present (frozen embeddings warm up the backbone)
train_sft.py --model Qwen/Qwen2.5-32B --data data --variant em      --phase full  --out runs/em_A
# EM Phase B: freeze the whole model, fit ONLY the K expert-token embeddings
train_sft.py --model Qwen/Qwen2.5-32B --data data --variant em --phase tokens --init-from runs/em_A --out runs/em_B
```

`--phase tokens` freezes all parameters and gradient-masks the input embedding to update **only the K
`<|expert_k|>` rows** — the literal "train the expert tokens" step. The script is single-process for
validation; **at 32B it is launched under FSDP** (see below).

## Scale & compute

- **Validated end-to-end on Qwen2.5-0.5B** (all three runs: control / EM Phase A / EM Phase B) — the
  pipeline (data → tokenize → completion-only loss → two-phase EM → expert-token grads) is proven.
- **Target: Qwen2.5-32B on Dais** (8× H200 ≈ 1.1 TB/node → 32B full-FT FSDP fits on a **single node**;
  see the cluster review). `control`-full and `em`-Phase-A are the heavy runs (full-FT under
  FSDP/`accelerate`); `em`-Phase-B is cheap (only the expert rows train). 24 h walltime cap → the
  full-FT runs need graceful-drain checkpoint+resume.

## Evaluation & analysis

- **Quality:** held-out per-expert perplexity / answer accuracy, control vs EM.
- **Specialization (the thesis payoff):** route a domain's held-out questions through the **wrong**
  `<|expert_k|>` and measure the degradation (swap test); cross-expert × domain perplexity matrix;
  routing accuracy. Only the EM model has per-expert structure to measure.
- **What the tokens learned:** the mechanistic-interp suite (`scripts/mech_interp.py`, arch-agnostic —
  already ported to Qwen) on the EM model: per-layer activation shift, latent domain-separation probe,
  and the expert-token geometry (do `<|expert_k|>` embeddings separate by domain?).
