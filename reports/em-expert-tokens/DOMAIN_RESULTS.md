# Domain-QA test — the control case for the expert token (identity *recoverable* from the question)

This is the deliberate counterpart to [PERSONA_RESULTS.md](PERSONA_RESULTS.md). There the per-expert
identity was **hidden** (same question, 8 styles) and the token paid off. Here the "expert" is a **QA
domain**, and the domain is **recoverable from the question itself** (a math word problem *looks* like
math; a clinical abstract *looks* like medicine). The thesis predicts the per-expert token should be
**redundant** in this regime — a generic assistant token can condition on the visible question just as
well. This run tests that cleanly, with the **same recipe as the persona test** so the two are directly
comparable.

Data: **6 diverse, all-generative domains** (math=gsm8k, medical=PubMedQA, general=alpaca, trivia=trivia_qa,
science=sciq, dolly=databricks-dolly-15k), constant **1500 examples/domain**, 10% held out. All free-form
(no multiple-choice) so the token always has real output to condition and nothing is decided by a 1-of-4
label. Trained Qwen2.5-3B: control (generic `<|im_start|>assistant`) vs EM (per-domain `<|expert_k|>`,
backbone-matched: control full-FT 1000 steps, EM Phase A full-FT 1000 + Phase B token-only 800). Held-out
response ppl ↓. **Identical trainer, eval, and swap test to the persona run.**

| model | MACRO ppl ↓ | per-domain EM gap | swap-ratio |
|---|---|---|---|
| non-finetuned base | 5.247 | — | — |
| control (generic assistant) | **2.861** | — | — |
| EM (per-domain token) | 2.903 | **−0.8%** | **1.01** |

### Per-domain PPL (held-out) — the MACRO above, broken out by domain

| domain | base (non-FT) | control (generic) | EM (per-domain) | EM wrong-expert (×ppl) |
|---|---|---|---|---|
| math    | 2.021    | 1.330  | 1.334  | 1.333 (×1.00) |
| medical | 18.315   | 9.949  | 9.954  | 9.981 (×1.00) |
| general | 5.837    | 3.495  | **3.416** | 3.433 (×1.01) |
| trivia  | 1561.726 | 16.346 | 17.249 | 15.860 (×0.92) |
| science | 9811.071 | 2.964  | 2.962  | 3.448 (×1.16) |
| dolly   | 8.354    | 5.811  | 5.876  | 5.845 (×0.99) |
| **MACRO** | **5.247** | **2.861** | **2.903** | **swap ×1.01** |

Swap test — route each domain's held-out response through the **wrong** `<|expert_k|>`, ×ppl:

| science | general | math | medical | dolly | trivia | **mean** |
|---|---|---|---|---|---|---|
| 1.16 | 1.01 | 1.00 | 1.00 | 0.99 | 0.92 | **1.01** |

## Findings — the token is redundant when identity is recoverable

1. **EM ties control (−0.8% per-domain; MACRO 2.861 vs 2.903).** Averaged fairly per domain, the
   per-domain token is a wash — tiny wins on general (+2.3%) offset by trivia (−5.5%) and dolly (−1.1%),
   the rest within noise. Finetuning itself works (base MACRO 5.25 → 2.86, with the format-sensitive
   domains collapsing: science 9811→2.96, trivia 1562→16.3), so there *is* signal to move — the token
   just doesn't move it.
2. **The token carries essentially no causal signal (swap-ratio 1.01).** Routing a domain's response
   through the *wrong* `<|expert_k|>` changes perplexity by ~0% (science 1.16 is the only flicker; trivia
   0.92 even *improves*, i.e. noise). Contrast the persona swap of **1.87** (up to 3.1×): there the token
   held the hidden identity; here it holds nothing the question doesn't already supply.
3. **Even the token-weighted metric no longer favors EM.** In the old mixed-composition sweep, token-
   weighting *inflated* an EM "win" because verbose generative domains (where EM helped a hair) dominated
   the average. With a clean all-generative composition, MACRO shows EM *tied-to-slightly-worse*
   (2.903 vs 2.861) — the earlier headline was a weighting artifact, now removed.

## The clean contrast (the real takeaway)

| setting | is identity in the prompt? | EM vs control | swap-ratio |
|---|---|---|---|
| **domain-QA** (math/medical/…) | **yes** (the question reveals it) | **−0.8% (tied)** | **1.01** |
| **persona/style** | **no** (same question, hidden persona) | **+10.7%** | **1.87** |

![EM pays off only when identity is hidden](figs/contrast.png)

**The expert-token finetuning helps precisely when the thesis says it should — and not otherwise.** When
the per-expert identity is recoverable from the content (domain-QA), a learned per-expert token is
redundant with a generic one and carries no causal signal. When the identity is hidden (persona/style),
the token both lowers perplexity (+10.7%) and demonstrably carries the identity (swap 1.87). This clean
null is the *control condition* that makes the persona result meaningful.

## Was this run "like the persona test"? — yes, and it fixes the old sweep's confounds

Same model, trainer, two-phase EM recipe, eval, and swap test as [PERSONA_RESULTS.md](PERSONA_RESULTS.md).
Relative to the earlier domain-count sweep in [RESULTS.md](RESULTS.md), this rerun removes the three
things that muddied it:

| confound in the old sweep | fix here |
|---|---|
| **token-weighted MACRO** (produced a spurious +3.6%) | report the **unweighted per-domain gap** (same metric as persona's +10.7%) |
| **mixed generative + multiple-choice** domains (MC gives the token nothing; deflates the weighted mean) | **all-generative** 6-domain set, constant 1500/domain |
| **backbone asymmetry / token under-training** at high K (multi-cycle gave control 2× backbone; K=28 tokens saw ~50 ex) | **single-cycle, backbone-matched**, 6 tokens with ample examples each — identical to persona |

Result is unchanged in direction from the honest reading of the old sweep (null-to-slightly-negative) but
now clean, fairly measured, and structured identically to the persona test.

*(Caveats: single seed, 3B, held-out response-ppl. The swap-ratio is a within-model causal test that
doesn't depend on absolute ppl. `base` per-domain ppl is inflated on terse-answer domains — trivia,
science — because the non-finetuned model doesn't yet produce the short answer format; this is expected
and resolves after SFT.)*
