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

1. ~~**Persona-collapse metric is missing.**~~ **NOW IMPLEMENTED** ([COLLAPSE_RESULTS.md](COLLAPSE_RESULTS.md),
   `collapse_metric.py`): geometric spread of the learned embeddings (mean pairwise cosine, effective rank, L2).
   Result is thesis-supporting — EM embeddings are far less collapsed than joint SFT's (cosine 0.23 → 0.06, L2
   0.8 → 9.8), monotonic in Phase-B budget. (One embedding per persona makes the thesis's exact within/between
   variance ratio degenerate, so these are the standard equivalents.)
2. **No noise injection in Phase B.** The thesis explicitly adds noise during embedding updates to prevent collapse
   (Xie et al. 2020). We only do a *one-time distinct init* of the expert rows ([train_sft.py:55-58](repro/train_sft.py#L55));
   there is no per-step noise. **Divergence.**
3. ~~**Scarce-data / cold-start regime untested — the big one.**~~ **NOW TESTED — and the thesis claim
   reproduces.** See [COLDSTART_RESULTS.md](COLDSTART_RESULTS.md): with imbalanced train volumes (450 → 4
   examples), **EM two-phase beats naive joint SFT by −38% MACRO ppl and wins on every persona** — the opposite
   of the balanced-data regime — and the 4-example persona is rescued 43.0 → 8.5 ppl by adding Phase-B budget.
   Our earlier *"Phase B is nearly useless"* was indeed an artifact of balanced, data-rich data; under realistic
   imbalance, alternation (especially Phase-B-heavy) is decisive for the low-data tail, exactly as the thesis
   argues. (Remaining nuance: our volume assignment confounds with persona difficulty, so the clean signals are
   EM≫joint-under-imbalance and the tail rescue, not a smooth gap-vs-volume trend.)
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
  sweeps (cycles, splits, 2D). The thesis's **central claim now reproduces in the cold-start regime**
  ([COLDSTART_RESULTS.md](COLDSTART_RESULTS.md)): under imbalanced data EM ≫ joint SFT and the low-data tail is
  rescued by Phase B — reconciling our earlier "Phase B is nearly useless" (true only for balanced, data-rich data).
  **Still missing:** the thesis's *collapse metric* (within/between-cluster variance) and *noise injection* in
  Phase B; and phase-ordering. These are the remaining fidelity gaps.

## Recommended to close the gaps (in priority order)

1. ~~**Imbalanced / cold-start experiment**~~ — **DONE** ([COLDSTART_RESULTS.md](COLDSTART_RESULTS.md)): EM ≫ joint
   under imbalance, tail rescued. Follow-up worth doing: a *difficulty-decorrelated* volume assignment (or the same
   persona at several volumes) to turn the noisy gap-vs-volume trend into a clean one.
2. **Persona-collapse metric** — add within-/between-cluster variance of the learned `<|expert_k|>` embeddings (cheap;
   just read the embedding matrix post-training) alongside the swap-test.
3. **Noise injection in Phase B** — add Gaussian noise to the embedding updates; test whether it reduces collapse and
   helps the low-data tail (thesis axis #5).
4. **Phase-ordering axis** — try B→A vs A→B (thesis axis #2).
