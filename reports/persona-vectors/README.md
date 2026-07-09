# Persona vectors in the token space

Two analyses of the **learned `<|expert_k|>` embeddings** — where they live in the model's token space, and
what happens when you compose them — on both the **synthetic persona** set and the **domain-QA** set
(Qwen2.5-3B, EM two-phase: Phase A backbone + Phase B tokens).

---

## Test 1 — Where do the persona vectors end up? (token-space geometry)

For each trained expert vector we report (`analyze_token_space.py`):
- **Nearest ordinary-vocabulary words** (cosine) — what interpretable direction, if any, the token points in.
- **Vector norm** vs the vocabulary-norm distribution — are the persona vectors ordinary or outliers?
- **2-D PCA** of the expert vectors together with a random vocab sample and each expert's **content centroid**
  (mean input-embedding of the tokens in that expert's own responses) — i.e. where the persona vector lands
  relative to the surrounding words and to its own generated content.

![token space](figs/token_space.png)

*(Results — nearest-word tables + norms — filled from the run; see `figs/{persona,domain}_tokspace.json`.)*

## Test 2 — Composing persona vectors (downstream effects)

We interpolate two expert vectors, **e = α·e_A + (1−α)·e_B**, place the composite at a scratch token, and
score the held-out response perplexity of **A's** data and **B's** data under it, sweeping α
(`compose_personas.py`). If the token space is compositional, the two ppl curves cross smoothly (the composite
is a genuine blend); a generated sample at α=0.5 shows the blend qualitatively. Pairs: persona
`pirate⊕robot, coach⊕child, bard⊕detective`; domain `math⊕medical, trivia⊕science`.

![composition](figs/composition.png)

*(Interpolation curves + α=0.5 generation samples filled from the run; see `figs/{persona,domain}_compose.json`.)*

---

## Reproduce

`pvec.sbatch` trains the two EM models and runs both analyses; figures via `make_tokspace_fig.py` and
`make_compose_fig.py` over the JSONs in `figs/`. Analysis scripts: `analyze_token_space.py`,
`compose_personas.py`.
