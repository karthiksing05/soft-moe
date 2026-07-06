# Fidelity check — are our representation and tests faithful to the thesis?

Scan of `papers/master_thesis_stream_a.pdf` — *"Learning to Speak as Someone: EM-Style Training for
Personality-Conditioned Language Models at Scale"*, **Stream A (Persona Only)** — against what we built.

## What the thesis actually specifies (ground truth)

- **Goal:** counter the LLM "artificial hivemind" by conditioning generation on *individual identity*, preserving
  behavioral diversity that RLHF flattens.
- **Representation:** each speaker gets a **learned persona embedding**, injected as a **special token prefixed to
  their text** — `[SPEAKER_4217] <text>`. The token is *not a fixed string*; its embedding is initialised randomly
  and optimised so the model reproduces that speaker's style.
- **EM training (the thesis's core):** persona embeddings are the *unobserved variables*. Naive SFT trains model +
  embeddings jointly in one pass; the thesis instead **alternates in rounds**:
  - **Phase A (model update):** freeze embeddings, SFT the model weights.
  - **Phase B (embedding update):** freeze the model, optimise each embedding on that speaker's data —
    soft-prompt tuning, **higher LR than the model** (two-time-scale), **with noise injection to prevent collapse**.
  - One round (A→B) = one SFT pass of compute; expect **2–3 rounds**.
- **Baselines:** (1) **naive joint SFT** (tokens + weights together, one pass); (2) **embedding-only** (frozen model,
  pure prompt tuning); (3) **model-only SFT with speaker names as plain text** (no learned embedding).
- **Comparison axes (5):** alternation frequency, phase ordering, asymmetry (embedding:model update ratio), number
  of rounds, regularisation strategy.
- **Data:** ~1M h of real transcribed podcast/YouTube speech; **thousands of identified speakers with a long tail of
  data volumes** (hosts with hundreds of hours → one-time guests with minutes).
- **Primary metrics:** (1) **held-out perplexity per speaker, conditioned on their embedding vs a generic embedding**
  (the gap = individual signal captured); (2) **persona-collapse detection** = ratio of within- to between-cluster
  variance in embedding space. Secondary: speaker-ID accuracy from generated text; **cold-start** (speakers <50
  utterances).
- **Central practical claim:** alternation matters *because per-person data is scarce* — in joint training a low-data
  speaker's embedding gets too few gradient updates; Phase B, against an already-capable frozen model, fits it
  efficiently (à la Textual Inversion learning a concept from 3–5 images). "EM should matter most in the long tail."

## Faithful — what we got right

| thesis element | our implementation | ✓ |
|---|---|---|
| Learned per-speaker special-token embedding, prefixed to the speaker's text | `<|expert_k|>` special tokens (vocab-resized, embeddings learned), used as the assistant-turn marker | ✓ |
| Two-phase EM: Phase A = train model / freeze embeddings; Phase B = freeze model / train embeddings | `--phase backbone` (A) and `--phase tokens` (B), grad-masked to the K expert rows | ✓ |
| Phase B uses a higher LR (two-time-scale) | Phase B lr `1e-2` vs model `1e-5` | ✓ |
| Baseline: naive joint SFT | `--variant em --phase full` (token + weights jointly) | ✓ |
| Baseline: model-only SFT, no learned embedding | our `control` (generic `assistant` marker) | ✓ |
| Baseline: embedding-only vs frozen model | Phase-B-from-scratch / the `B=0` vs high-B points of the split sweep approximate this | ~ |
| **Primary metric #1**: held-out ppl conditioned on embedding vs generic | our headline **MACRO ppl, EM (per-persona token) vs control (generic)** | ✓ |
| Alternation over multiple rounds; "2–3 rounds" | our multi-cycle sweep (N ∈ {1,2,4,8}) + the 2D cycles×steps sweep | ✓ |
| Asymmetry axis (embedding:model update ratio) | our Phase A/B **split sweep** | ✓ |
| Persona/style as the domain | the persona test (8 styles, hidden identity) — the thesis's exact setting | ✓ |

**Core method and primary metric are faithful.** The persona test in particular is the right instantiation, and
[PERSONA_RESULTS.md](PERSONA_RESULTS.md) (+10.7% EM vs generic, swap-ratio 1.87) is a direct small-scale version of
the thesis's primary experiment.

## Gaps & divergences — where we are *not* faithful

1. **Persona-collapse metric is missing.** The thesis's *second primary metric* is within-/between-cluster variance
   of the embeddings. We never compute it. Our **swap-test** (route through the wrong token → ppl increase) is a
   *related* causal proxy (collapse ⇒ swap-ratio ≈ 1), but it is not the same measurement. **Gap.**
2. **No noise injection in Phase B.** The thesis explicitly adds noise during embedding updates to prevent collapse
   (Xie et al. 2020). We only do a *one-time distinct init* of the expert rows ([train_sft.py:55-58](train_sft.py#L55));
   there is no per-step noise. **Divergence.**
3. **Scarce-data / cold-start regime untested — the big one.** The thesis's *central practical motivation* is the
   **long tail of low-data speakers**, and it predicts Phase B matters *most* there. **All our personas/worlds have
   balanced, ample data** (≈114–460 examples each). So our finding that *"all-Phase-A is best, Phase B adds little"*
   ([CONVERGENCE_RESULTS.md](../convergence/CONVERGENCE_RESULTS.md)) may be an **artifact of the data-rich, balanced
   regime** — it could reverse in the thesis's intended cold-start setting. This is the most important caveat: **we
   have not tested the condition under which the thesis expects EM to win.**
4. **Scale & real data.** Thesis: thousands of *real* speakers from transcribed speech. Ours: **8 synthetic style
   caricatures** (Qwen-generated pirate/bard/…) and synthetic knowledge sets. A reasonable proxy, but not real
   individual behavioral signatures at scale.
5. **Two of the five comparison axes untested:** **phase ordering** (we always do A→B) and **regularisation strategy**
   (noise injection, above). We *do* cover alternation frequency/rounds and asymmetry.
6. **Out-of-scope extensions.** Our **domain-QA** and **knowledge (counterfact/PopQA)** experiments are *not* part of
   the persona-only thesis. They are our own investigation into *when* the token is load-bearing (recoverable vs
   hidden identity). They usefully explain the mechanism behind the persona result, but they should be labelled as
   extensions, not thesis validation.

## Bottom line

- **Representation:** faithful — learned per-speaker token embeddings, two-phase EM (A = model, B = embeddings,
  higher-LR), the naive-joint-SFT and generic-token baselines, and held-out-ppl-vs-generic are all exactly the
  thesis's constructs.
- **Tests:** faithful *for the balanced-data persona setting*, and we've gone beyond the thesis on the alternation
  sweeps (cycles, splits, 2D). **But** we are missing the thesis's collapse metric and noise-injection, and —
  crucially — we have **not** probed the **scarce-data long tail / cold-start** regime that is the thesis's whole
  reason alternation should help. Our "Phase B is nearly useless" conclusion is therefore honest *for our setup* but
  **not yet a fair test of the thesis's central claim.**

## Recommended to close the gaps (in priority order)

1. **Imbalanced / cold-start experiment** — give personas wildly different data volumes (e.g. 5–500 examples), and
   compare joint SFT vs EM specifically on the **low-data** personas. This directly tests the thesis's core claim and
   is the highest-value missing piece.
2. **Persona-collapse metric** — add within-/between-cluster variance of the learned `<|expert_k|>` embeddings (cheap;
   just read the embedding matrix post-training) alongside the swap-test.
3. **Noise injection in Phase B** — add Gaussian noise to the embedding updates; test whether it reduces collapse and
   helps the low-data tail (thesis axis #5).
4. **Phase-ordering axis** — try B→A vs A→B (thesis axis #2).
