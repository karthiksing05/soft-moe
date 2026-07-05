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
2. **The EM per-expert token edges out the generic assistant token on the *token-weighted* metric.**
   MACRO **2.176 vs 2.258** (~3.6% lower), winning on the verbose domains (science, trivia, general),
   ~tied on math, slightly worse on medical. ⚠️ **But this is token-weighted** — the domain-count
   sweep below shows that under the fairer *unweighted per-domain* metric the effect is null-to-slightly
   negative, i.e. this headline win is largely carried by the verbose domains, not a uniform gain.
   Read finding-2 together with the sweep correction.
3. **Specialization is modest, and where it appears is interpretable.** The swap-ratio is only
   **1.05** on average — the expert token carries real per-domain signal for **science (1.18)** and
   **trivia (1.07)**, but ~none for math/medical/general (wrong token barely hurts). The reason is the
   same mechanism we found before: **the domain is largely inferable from the question itself**, so
   the expert token is partly redundant — it helps most where the *answer format* is domain-specific
   but the question is ambiguous (short-answer science/trivia), and least where the question already
   pins the domain (a math word problem, a medical abstract). Persona/speaker tokens should specialize
   more strongly in settings where identity is *not* recoverable from the content.

## Does EM scale with the number of domains? (domain-count sweep)

Same method, varying **K = number of QA domains** (1500 examples/domain, 2000 SFT steps each, so
control and EM see identical data at each K). Expanded to 15 clean QA datasets.

| domains K | control MACRO ↓ | EM MACRO ↓ | **EM − control** | swap-ratio |
|---|---|---|---|---|
| 5 | 2.387 | 2.397 | +0.010 (EM slightly worse) | 1.03 |
| 10 | 2.166 | 2.181 | +0.015 (EM slightly worse) | 0.99 |
| **15** | 2.135 | **2.079** | **−0.056 (EM better ~2.6%)** | 1.02 |

### Extended to K=20 and K=28 — and the trend does NOT hold (metric correction)

The token-weighted MACRO above looked like it favored EM at K=15, but pushing to K=20/28 broke the
pattern (EM MACRO worse at 20; control MACRO `nan` at 28). That flagged a **confound**: token-weighted
MACRO shifts with the *domain mix* — the short-answer multiple-choice datasets added at high K are
easy (low ppl) and dominate/deflate it, so MACRO is not comparable across K. Recomputing the honest
metric — the **unweighted mean per-domain gap** (control vs EM on the *same* domains, each domain
equal):

| K | mean control ppl | mean EM ppl | per-domain EM gap (>0 = EM better) |
|---|---|---|---|
| 5 | 6.21 | 6.37 | **−1.6%** |
| 10 | 4.18 | 4.42 | −4.1% |
| 15 | 4.15 | 3.71 | −0.6% |
| 20 | 3.34 | 3.65 | −5.7% |
| 28 | 3.63 | 8.47 | −23% (one domain `nan`) |

**Honest conclusion: EM does not beat control here, and there is no positive domain-count scaling.**
Averaged fairly per domain, EM is tied-to-slightly-worse at every K; the earlier "K=15 win" was a
token-weighting artifact (EM helps on verbose/generative domains, hurts on the short MC ones). The
swap-ratio stays ~1.0 throughout — the expert tokens carry little inference-time signal.

**Two confounds stack the deck against EM at high K, so this is a floor, not a verdict:**
1. **Under-trained tokens.** Fixed step budget over more domains ⇒ each `<|expert_k|>` sees far fewer
   examples as K grows (Phase B trains 28 tokens on ~50 examples each at K=28). The K=28 collapse
   (−23%) is largely under-training, not evidence the idea fails.
2. **Domain composition.** The added high-K domains are short-answer MC, where a per-expert token has
   little to condition (the answer is one of 2–4 given options).

**A fair scaling test still to run:** scale training *per expert* (constant examples/token as K grows,
or scale steps with K), use ≥3 seeds, report the unweighted per-domain gap, and keep the domain
*type* fixed (all-generative or all-MC) so composition doesn't move under the metric. Until then, the
honest read is: **at this compute, the per-expert token is roughly on par with a generic assistant
token — a null-to-slightly-negative result — not the growing advantage the token-weighted view
suggested.**

## Honest caveats
- The **MoE arm was cancelled** (14B LoRA was slow; killed on request) — this run is the dense
  control-vs-EM comparison only.
- MACRO is token-weighted, so verbose domains (math/general) dominate it; the per-expert columns show
  the domain-level picture.
- Single seed, ~3k SFT steps (partial epoch). The EM win is small but consistent across the format-
  sensitive domains; more steps / a setting where the domain is hidden from the question would sharpen
  the specialization signal.
