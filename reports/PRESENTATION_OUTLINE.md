# Presentation Outline — EM Expert-Token Finetuning

Each bullet: the concise talking point — *report › subsection* — **graphic**.

## ⭐ Core narrative (6 slides)

**1 — The idea & question**
- One shared backbone conditioned on a bank of **learned per-expert token embeddings**, trained by a
  **two-phase EM protocol** (Phase A: train backbone, tokens frozen; Phase B: freeze backbone, fit tokens).
  Does it beat naive finetuning and a real MoE — and *when*? — *reports/README.md › intro / TL;DR* —
  **(setup, models, data examples: reports/EXPERIMENTAL_SETUP.md)**

**2 — Headline principle: the token helps only when identity is *hidden* AND *determines the output***
- One clean rule across four settings — not novelty of the task, but recoverability. The token is redundant
  when the input reveals the expert, decisive when it doesn't. — *reports/README.md › §1* —
  **em-expert-tokens/figs/contrast.png**

**3 — Persona / style (the thesis's setting): +10.7%**
- Same question, 8 hidden personas → a per-persona token beats a generic one by **+10.7%** ppl, and the
  swap-test proves causal, non-redundant signal (wrong persona → up to **3×** ppl). — *em-expert-tokens/
  PERSONA_RESULTS.md › "Findings — the method IS justified when identity is hidden"* —
  **em-expert-tokens/figs/persona_results.png**

**4 — Knowledge: from redundant to decisive**
- Recoverable facts (subject in the question) → token redundant (tie). **Conflicting facts** (same prompt,
  source *only* in the token) → token **doubles accuracy** (50% coin-flip ceiling → ~100%), swap → 0%. —
  *em-expert-tokens/KNOWLEDGE_RESULTS.md › "Results"* + *EXPERIMENTAL_SETUP.md › "Conflicting vs expected
  knowledge"* — **em-expert-tokens/figs/knowledge_results.png**

**5 — Does the two-phase *scheme* beat standard SFT? Only in the data-starved tail**
- On balanced data EM ≈ joint SFT (the *token* is the active ingredient; even frozen-random anchors work).
  With **many personas × few episodes each**, EM's edge over joint SFT **grows as data/persona shrinks**:
  +7% @ 15 ep, +19% @ 5 ep — and learned tokens beat frozen-random *only* at low data. —
  *em-expert-tokens/EM_VS_SFT.md › "Findings — the hypothesis holds"* — **em-expert-tokens/figs/emwin.png**

**6 — Synthesis: the advantages (and honest limits) of EM**
- **Wins:** cheap specialization on a shared backbone; helps iff identity is hidden+determining; robust to
  data scarcity (the killer app); distinct/collapse-resistant embeddings; continual learning without
  forgetting; modular (an expert = a token). **Limits:** redundant when identity is recoverable; the token
  can't store facts or add capacity; the alternation is a refinement, not the active ingredient, on balanced
  data. — *reports/README.md › §6 "The advantages of EM training — synthesis"*

## ➕ Supporting slides (robustness, capacity, dynamics)

**7 — Robustness #1: cold-start / imbalanced data (the thesis's core claim, reproduced)**
- Imbalanced volumes (450 → 4 ex) → EM crushes joint SFT (**−38%** ppl), rescues the 4-example persona
  **43 → 8.5**. Phase B fits starved embeddings against a capable frozen backbone. — *em-expert-tokens/
  COLDSTART_RESULTS.md › "Result — EM dominates under imbalance"* — **em-expert-tokens/figs/coldstart.png**

**8 — Robustness #2 & #3: no collapse, no catastrophic forgetting**
- EM embeddings stay **~10× more separated** than joint SFT's (cosine 0.23 → 0.06) — the thesis's
  anti-collapse metric. — *COLLAPSE_RESULTS.md › "Findings"* — **em-expert-tokens/figs/collapse.png**
- Continual learning: naive sequential SFT forgets Task A (**+35%**); a new task as a **new token on a frozen
  backbone** cuts forgetting to **+4%** (parameter isolation). — *CATASTROPHIC_FORGETTING.md › "Result" /
  "The takeaway"* — **em-expert-tokens/figs/catastrophic_forgetting.png**

**9 — EM vs a real MoE: near-free specialization**
- Matched compute (d256, 8 domains): the MoE wins raw ppl but at **~3× params**; the EM token gets **perfect
  routing + real specialization** (swap 1.6) a dense model lacks, at **+2,048 params**. MoE buys *capacity*,
  EM buys cheap *conditioning*. — *reports/README.md › §5* + *comparison/main_table.md*,
  *comparison/mech_interp/report.md* — **comparison/mech_interp/*.png**

**10 — Training dynamics (backup)**
- The token helps at *every* step; best budget split is all-Phase-A on balanced data; cycling trades speed
  for regularisation. — *convergence/CONVERGENCE_RESULTS.md › §1–§5 + TL;DR* —
  **convergence/{convergence_curves,split_ratios,cycles,cycle_trajectory}.png**

**11 — Fidelity to the thesis**
- Core method + **both** primary metrics (held-out-ppl-vs-generic, embedding collapse) implemented; central
  cold-start claim reproduces. Open gaps: Phase-B noise injection, phase-ordering. — *em-expert-tokens/
  THESIS_FIDELITY.md*

## One-line takeaway
> The expert token is a **cheap, modular conditioning handle**: redundant when identity is recoverable,
> decisive when it's hidden and outcome-determining, and the **EM two-phase scheme earns its keep precisely
> in the many-experts / little-data-each long tail** — while a real MoE remains the tool for raw *capacity*.
