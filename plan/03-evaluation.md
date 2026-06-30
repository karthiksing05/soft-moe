# 03 — Evaluation

> Owns: `src/softmoe/eval/`, `configs/eval/`, `scripts/evaluate.py`, `scripts/make_report.py`.
> Goal: measure (a) **language-modeling quality** and (b) **whether experts actually
> specialize**, then produce one comparison table/plots across all methods and regimes.

## 1. What a result must answer

1. **Does conditioning a single backbone on learned expert tokens match/beat the dense model,
   and approach hard-MoE / c-BTM — at a fraction of the parameters?** (LM quality)
2. **Do the experts specialize to domains, and does *learning* the tokens (vs fixed orthogonal)
   cause it?** (specialization + the key ablation)
3. **Supervised vs unsupervised:** how much quality do we lose when assignment is discovered by
   EM instead of given? (the central dichotomy from `BRAINSTORM.md`)

Every metric below is computed by `eval/harness.py` over a model's `test` split and written to
`experiments/<run>/metrics.json`. `make_report.py` aggregates many runs into tables.

## 2. Language-modeling metrics (`eval/perplexity.py`)

- **Per-domain perplexity** on held-out `test`, using the *ground-truth* domain split from
  [01](01-data-collection.md). Report a vector over domains + the macro-average (equal weight
  per domain) and micro-average (token-weighted). Macro-average is the headline (it rewards
  specialization on small domains).
- **Oracle vs routed perplexity:** evaluate twice —
  - *oracle routing*: feed each input its ground-truth/cluster expert (upper bound of the token
    bank), and
  - *learned routing*: let the router pick (real deployment number).
  The gap measures router quality independently of expert quality.
- **Parameter-matched comparison:** report perplexity **against trainable-parameter count** and
  total-parameter count. Ours adds only `n_experts × tokens_per_expert × d_model` params — make
  this efficiency story explicit (a plot: quality vs added params).

## 3. Specialization metrics (`eval/specialization.py`) — the novel part

These quantify the core claim. Implement all:

- **Routing accuracy / NMI:** treat learned expert id as a clustering of the test docs; compare
  to ground-truth domain labels via accuracy (best assignment under Hungarian matching), NMI,
  and purity. High = experts align with real domains.
- **Expert utilization:** histogram + entropy of expert usage over the test set. Compare to
  uniform (collapse detector). Report fraction of "dead" experts.
- **Token separation:** mean pairwise cosine distance between expert token vectors, and the
  effective rank / log-det volume of the token matrix. Track vs the `fixed-orthogonal` baseline
  (which is maximally separated by construction) — does *learning* trade some separation for
  quality, and is that trade worth it?
- **Per-expert specialization profile:** for each expert, the distribution of true domains it
  receives. A clean diagonal contingency matrix (experts × domains) is the money plot.
- **Counterfactual / swap test:** route an input through the *wrong* expert token and measure the
  perplexity increase. Large degradation ⇒ tokens carry genuine, non-interchangeable expertise
  (not just a label the backbone ignores). This is the strongest causal evidence — implement it.

## 4. The comparison matrix (`make_report.py`)

Produce one table; rows = methods, columns = metrics, for each regime.

```
Method            | macro-ppl ↓ | micro-ppl ↓ | routing-acc/NMI ↑ | util-entropy ↑ | sep ↑ | +params ↓
------------------|-------------|-------------|-------------------|----------------|-------|----------
Dense             |             |             |        —          |       —        |   —   |    0
Hard MoE          |             |             |                   |                |       |
c-BTM             |             |             |                   |                |       |
MoP               |             |             |                   |                |       |
Ours (fixed)      |             |             |                   |                |       |
Ours (sup.)       |             |             |                   |                |       |
Ours (unsup.)     |             |             |                   |                |       |
```
Output both `reports/main/main_table.md` and machine-readable `reports/main/results.csv`, plus
plots (matplotlib, saved to `reports/main/figures/`):
1. quality (macro-ppl) vs added trainable params,
2. expert×domain contingency heatmap per method,
3. utilization histograms,
4. token-separation vs training step (from logged history),
5. oracle-vs-learned routing gap bar chart.

## 5. Ablations (drive from configs, not code edits)

Each is a sweep over one config knob; `make_report.py` should render each as its own small table.
- **Learned vs fixed tokens** (the key one): `expert_tokens.trainable ∈ {true,false}`.
- **Backbone mode:** `frozen | lora | full` — how much does the backbone need to move?
- **Number of experts K** vs number of true domains (under/over-clustering).
- **Tokens per expert / token-path** (`tokens_per_expert ∈ {1,2,4}`; flat vs Cobweb hierarchy).
- **Regularizer weights:** sweep `λ_sep`, `λ_bal`; confirm the "separation-priority" claim that
  `λ_sep > λ_bal` gives better specialization without quality loss.
- **Injection:** `prefix` vs `prefix_kv`.
- **EM cadence:** soft-online vs periodic hard re-assignment.
- **Clusterer:** k-means vs Cobweb for the unsupervised partition.

## 6. Statistical rigor

- ≥3 seeds for every headline number; report mean ± std. The agent should make `seed` a sweep
  axis and have `make_report.py` aggregate across seeds.
- Significance: paired bootstrap over test documents for ppl differences between Ours and each
  baseline; report whether gaps are significant.
- Log compute (GPU-hours, params, FLOPs/token) alongside quality so comparisons are honest.

## 7. Deliverables / acceptance

- **M6:** `python scripts/evaluate.py --run experiments/<run>` writes `metrics.json` with every
  metric in §2–§3; `python scripts/make_report.py --runs experiments/ --out reports/main/` emits
  `main_table.md`, `results.csv`, and the figures.
- `tests/test_metrics.py` checks metric math on synthetic cases (e.g. perfect routing → NMI=1,
  collapse → utilization-entropy=0, identical experts → separation=0).
- **M7 (final):** the full table is reproduced from `configs/experiment/*.yaml` on the `main`
  data recipe, with ≥3 seeds, and the README's "definition of done" command sequence runs clean.

## 8. Success criteria (what would make this a positive result)

- **Ours (sup.)** ≈ hard-MoE / c-BTM macro-ppl while adding **orders of magnitude fewer**
  trainable params → "one weight space can host many experts."
- **Ours (unsup.)** recovers domains (routing NMI well above chance) and lands between dense and
  supervised on quality.
- **Ours (learned) > Ours (fixed)** on quality and/or specialization → *learning* the expert
  tokens matters (the central ablation).
- **Swap test** shows large, expert-specific degradation → tokens carry real expertise.

A clean *negative* result (tokens can't specialize even with LoRA) is also publishable — so log
enough (utilization, separation, swap test) to diagnose *why*, not just *that*, it failed.
