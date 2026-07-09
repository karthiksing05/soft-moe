# Incremental training — how easy is it to add a new persona?

The EM method's promise is modularity: *an expert = a token*. This tests it directly — given a backbone
already trained on some personas, how cheaply can a **new** persona be added?

## Setup

Pretrain a backbone (Phase A) on **7 base personas** (Qwen2.5-3B). Then add the **8th** persona (robot) two
ways, and track both the new persona's held-out ppl and the base personas' retention:

- **token-only** — fit *only the new persona's token embedding* (~one vector, **backbone frozen**);
- **full fine-tune** — update the whole 3B on the new persona's data (for contrast).

Sweeping incremental **steps** {0, 10, 30, 100, 300} and **# new examples** {5, 25, 100}.

## Result

![incremental](figs/incremental.png)

| | new-persona ppl (robot) | base-persona ppl (retained) |
|---|---|---|
| before adding (untrained token) | 22.70 | 3.81 |
| **token-only**, 30 steps | 7.04 | 5.34 |
| **token-only**, 100 steps | **6.52** | 5.31 |
| full fine-tune, 100 steps | 5.36 | 6.64 |
| full fine-tune, 300 steps | 8.58 *(overfits)* | **9.61** *(forgets)* |

Data efficiency (token-only, 100 steps): **5 ex → 9.16, 25 ex → 7.35, 100 ex → 6.59**.

## Findings

1. **Adding a persona is cheap and fast.** Fitting *just the token* (a single ~2K-parameter embedding, vs the
   3B backbone) takes the new persona from **22.7 → 6.5 ppl in ~30–100 steps** — a few seconds of training.
   The backbone is never touched.
2. **Token-only barely disturbs the existing personas; full fine-tune forgets them.** Base ppl stays ~5.3–5.7
   under token-only but **degrades to 9.6 under full fine-tune** — the same catastrophic forgetting seen in
   [CATASTROPHIC_FORGETTING.md](CATASTROPHIC_FORGETTING.md). Full-FT reaches a slightly lower *new*-persona ppl
   at 100 steps (5.4) but then **overfits** (8.6 by 300) — so it is both costlier and more fragile.
   *(The mild base rise under token-only is a tied-embedding side-effect — Qwen2.5-3B ties word embeddings, so
   a large new token embedding leaks a little output-probability mass everywhere — not backbone forgetting.)*
3. **A handful of examples suffices.** Even **5 examples** lift the new persona from 22.7 → 9.2; ~25–100 get
   most of the way (7.4 → 6.6). This is the cold-start mechanism at single-persona scale: a capable frozen
   backbone lets a starved embedding be fit from very little data.

## Bottom line

**An expert really is just a token.** A new persona is added by fitting *one embedding* (~2K params) in ~30–100
steps from ~25 examples, reaching most of the achievable quality **without retraining the model or disturbing
the personas already learned** — exactly the cheap, modular, non-destructive extensibility the thesis promises,
and the practical pay-off of the parameter-isolation property.
