# PoC results — Qwen2.5-3B: generic assistant token (control SFT) vs learned per-expert tokens (EM)

Held-out response-perplexity (↓) on 5 domains (gsm8k / sciq / PubMedQA / trivia_qa / alpaca), 11.7k
train / 1.3k test, identical data. **Only** difference: the chat assistant marker
(`<|im_start|>assistant` vs per-expert `<|im_start|><|expert_k|>`, learned via the two-phase EM scheme).

| model | math | science | medical | trivia | general | **MACRO ppl ↓** |
|---|---|---|---|---|---|---|
| non-finetuned base | 2.06 | 5148 | 15.7 | 2482 | 6.36 | 4.622 |
| control SFT (generic assistant) | 1.332 | 2.457 | 8.396 | 15.499 | 3.932 | 2.258 |
| **EM (per-expert tokens)** | 1.336 | **2.439** | 8.746 | **15.214** | **3.656** | **2.176** |

Swap test (EM only) — route each domain through the **wrong** `<|expert_k|>`, perplexity increase:

| | math | science | medical | trivia | general | mean |
|---|---|---|---|---|---|---|
| wrong-expert ×ppl | 1.00 | **1.18** | 1.00 | 1.07 | 1.00 | **1.05** |

## Findings

1. **Finetuning clearly helps at this size** — base MACRO **4.62 → 2.26** (control), with the format-
   sensitive domains collapsing (science 5148→2.46, trivia 2482→15.5). This validates dropping off
   Qwen-32B: a 3B model genuinely benefits from SFT here, so there is signal to compare.
2. **The EM per-expert token beats the generic assistant token — for free.** MACRO **2.176 vs 2.258**
   (~3.6% lower ppl), winning on science, trivia, and general; ~tied on math; slightly worse on
   medical. Same data, same compute — only the assistant marker differs and is learned per expert.
   This is the thesis idea working on a real chat LLM.
3. **Specialization is modest, and where it appears is interpretable.** The swap-ratio is only
   **1.05** on average — the expert token carries real per-domain signal for **science (1.18)** and
   **trivia (1.07)**, but ~none for math/medical/general (wrong token barely hurts). The reason is the
   same mechanism we found before: **the domain is largely inferable from the question itself**, so
   the expert token is partly redundant — it helps most where the *answer format* is domain-specific
   but the question is ambiguous (short-answer science/trivia), and least where the question already
   pins the domain (a math word problem, a medical abstract). Persona/speaker tokens should specialize
   more strongly in settings where identity is *not* recoverable from the content.

## Honest caveats
- The **MoE arm was cancelled** (14B LoRA was slow; killed on request) — this run is the dense
  control-vs-EM comparison only.
- MACRO is token-weighted, so verbose domains (math/general) dominate it; the per-expert columns show
  the domain-level picture.
- Single seed, ~3k SFT steps (partial epoch). The EM win is small but consistent across the format-
  sensitive domains; more steps / a setting where the domain is hidden from the question would sharpen
  the specialization signal.
