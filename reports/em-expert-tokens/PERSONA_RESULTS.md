# Persona/style test — justifying the expert token in the thesis's actual setting (hidden identity)

The domain-QA PoC found the per-expert token ≈ a generic assistant token — because in QA the domain is
**recoverable from the question**, so the token is redundant. The thesis is about a *hidden* identity
(a persona/speaker you can only produce if told who it is). This test recreates that: **8 distinct
personas answer the SAME held-out questions**, so the persona is *not* in the prompt and the model must
use the token.

Data: Qwen2.5-7B-Instruct generated styled answers for 8 personas (pirate, Shakespeare-bard, professor,
Gen-Z teen, noir detective, child, motivational coach, literal robot) × ~50 shared questions (12 held
out for test, unseen). 912 train / 96 test, verified clean (0 artifacts, styles distinct). Trained
Qwen2.5-3B: control (generic `<|im_start|>assistant`) vs EM (per-persona `<|expert_k|>`, backbone-matched:
control full-FT 1000 steps, EM Phase A full-FT 1000 + Phase B token-only 800). Held-out response ppl ↓.

| model | MACRO ppl ↓ | per-persona EM gap | swap-ratio |
|---|---|---|---|
| non-finetuned base | 9.389 | — | — |
| control (generic assistant) | 4.747 | — | — |
| **EM (per-persona token)** | **4.458** | **+10.7%** | **1.87** |

### Per-persona PPL (held-out) — the MACRO above, broken out by persona

| persona | base (non-FT) | control (generic) | EM (per-persona) | EM wrong-expert (×ppl) |
|---|---|---|---|---|
| pirate    | 8.439  | 4.412  | 4.381     | 4.916 (×1.12) |
| bard      | 8.314  | 4.271  | **4.095** | 5.581 (×1.36) |
| professor | 6.032  | 3.638  | 3.641     | 4.218 (×1.16) |
| teen      | 15.138 | 5.333  | **5.208** | 7.422 (×1.43) |
| detective | 15.746 | 7.704  | **5.606** | 17.556 (×3.13) |
| child     | 15.630 | 4.991  | **4.071** | 11.282 (×2.77) |
| coach     | 7.022  | 4.583  | **4.537** | 5.865 (×1.29) |
| robot     | 44.381 | 12.939 | **8.808** | 23.704 (×2.69) |
| **MACRO** | **9.389** | **4.747** | **4.458** | **swap ×1.87** |

EM's biggest wins are the vivid, hard-to-infer styles (robot 12.94→8.81, detective 7.70→5.61,
child 4.99→4.07) — exactly the personas with the largest swap-ratio (2.7–3.1×). For styles a generic
model partly defaults to (professor, pirate, coach) EM ≈ control and swap ≈ 1.1–1.3; EM never loses by
more than noise on any persona.

Swap test — route each persona's held-out response through the **wrong** `<|expert_k|>`, ×ppl:

| detective | child | robot | teen | bard | coach | professor | pirate | **mean** |
|---|---|---|---|---|---|---|---|---|
| **3.13** | 2.77 | 2.69 | 1.43 | 1.36 | 1.29 | 1.16 | 1.12 | **1.87** |

## Findings — the method IS justified when identity is hidden

1. **EM beats control (+10.7% per-persona, MACRO 4.458 vs 4.747).** With the persona hidden from the
   prompt, a single generic token must average across 8 styles (blurred, higher ppl); the per-persona
   token conditions cleanly. Same data, same backbone compute — only the marker differs and is learned.
2. **The token is load-bearing (swap-ratio 1.87, up to 3.1×).** Routing through the wrong persona costs
   2–3× perplexity for the distinct styles (detective 3.13, child 2.77, robot 2.69). Contrast the
   domain-QA swap of ~1.0: there the token carried nothing; here it carries the identity the prompt lacks.
3. **The gain tracks style distinctiveness.** Big for the vivid, hard-to-infer styles
   (detective/child/robot: swap 2.7–3.1×, largest ppl wins); small for styles a generic model partly
   defaults to (pirate/professor: swap ~1.1). Exactly the expected shape.

![per-persona PPL and swap-ratio](figs/persona_results.png)

## The clean contrast (the real takeaway)

Run identically ([DOMAIN_RESULTS.md](DOMAIN_RESULTS.md)) against **domain-QA**, where the identity is
recoverable from the question:

| setting | is identity in the prompt? | EM vs control | swap-ratio |
|---|---|---|---|
| domain-QA (math/medical/…) | **yes** (the question reveals it) | **−0.8% (tied)** | **1.01** |
| **persona/style** | **no** (same question, hidden persona) | **EM +10.7%** | **1.87** |

![EM pays off only when identity is hidden](figs/contrast.png)

**The expert-token finetuning is justified precisely when the thesis says it should be** — when the
per-expert identity is *not* recoverable from the content. Domain-QA was the wrong test (the question
gives it away); persona/style is the right one, and there the token both improves perplexity and
demonstrably carries non-redundant, causal signal (the swap test). This is the small-scale
proof-of-concept for the master thesis.

*(Caveats: single seed, 3B, one instruct model as the style generator; the styled data is synthetic.
The direction is robust — the swap-ratio is a within-model causal test that doesn't depend on the
absolute ppl.)*

## Scheme comparison: EM two-phase vs straight joint SFT (token held constant, in the prompt)

The cleanest test of the thesis's *methodological* claim: put the persona `<|expert_k|>` token
directly in the **prompt** for BOTH arms, same data, matched total steps (1800). The **only** change
is the training scheme — joint SFT vs decoupled EM (Phase A trains the model with the token frozen,
Phase B trains only the token).

| arm | MACRO ppl ↓ | per-persona EM gap | swap-ratio |
|---|---|---|---|
| straight FT (joint SFT) | 5.082 | — | **1.71** |
| **EM (two-phase)** | **4.578** | **+8.8%** | 1.10 |

### Per-persona PPL (held-out) — scheme comparison, both arms token-in-prompt

| persona | straight FT | straight wrong (×) | EM two-phase | EM wrong (×) |
|---|---|---|---|---|
| pirate    | 5.122  | 5.532 (×1.08)  | **4.349**  | 4.519 (×1.04) |
| bard      | 4.411  | 5.318 (×1.21)  | **4.165**  | 4.357 (×1.05) |
| professor | 3.700  | 4.566 (×1.23)  | **3.497**  | 3.611 (×1.03) |
| teen      | 6.570  | 8.022 (×1.22)  | **5.219**  | 5.483 (×1.05) |
| detective | 6.642  | 17.246 (×2.60) | **6.571**  | 8.176 (×1.24) |
| child     | 5.017  | 9.160 (×1.83)  | **4.636**  | 5.160 (×1.11) |
| coach     | 5.228  | 6.350 (×1.21)  | **4.562**  | 4.614 (×1.01) |
| robot     | 12.425 | 40.808 (×3.28) | **12.168** | 15.690 (×1.29) |
| **MACRO** | **5.082** | **swap ×1.71** | **4.578** | **swap ×1.10** |

EM has lower ppl on **all 8** personas, but the swap columns show the flip: straight SFT's token is far
more load-bearing (wrong token up to ×3.3) while EM's backbone carries the persona (wrong token barely
hurts, ~×1.0–1.3).

1. **EM beats straight SFT on quality (+8.8%), on all 8 personas** — and did so with *less* backbone
   training (EM's model saw 1000 weight-update steps vs straight's 1800). The decoupling is more
   efficient, not just more compute. Supports the thesis: **alternating > naive joint SFT.**
2. **But the swap-ratios flip: straight 1.71 vs EM 1.10.** Straight SFT makes the *token* load-bearing
   (wrong token → up to ×3.3) but the model is worse; EM builds a better *backbone* (lower ppl) that
   carries the persona, leaving the token less causally important (wrong token → ~×1.1).
3. **So EM wins by a better backbone, not a more informative token.** If the goal is response quality,
   EM's scheme helps; if the goal is a portable, load-bearing persona *embedding* (the thesis's
   stronger claim), straight SFT yields the more token-dependent model. Same pattern seen at byte
   scale: a strong co-adapted backbone carries the load and the token adds less.
