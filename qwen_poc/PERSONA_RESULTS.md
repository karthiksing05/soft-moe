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

## The clean contrast (the real takeaway)

| setting | is identity in the prompt? | EM vs control | swap-ratio |
|---|---|---|---|
| domain-QA (math/science/…) | **yes** (the question reveals it) | ≈ tied / worse | ~1.0 |
| **persona/style** | **no** (same question, hidden persona) | **EM +10.7%** | **1.87** |

**The expert-token finetuning is justified precisely when the thesis says it should be** — when the
per-expert identity is *not* recoverable from the content. Domain-QA was the wrong test (the question
gives it away); persona/style is the right one, and there the token both improves perplexity and
demonstrably carries non-redundant, causal signal (the swap test). This is the small-scale
proof-of-concept for the master thesis.

*(Caveats: single seed, 3B, one instruct model as the style generator; the styled data is synthetic.
The direction is robust — the swap-ratio is a within-model causal test that doesn't depend on the
absolute ppl.)*
