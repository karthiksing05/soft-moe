# EM expert-token finetuning vs. Mixture-of-Experts

When do you actually **need** a Mixture-of-Experts? This repo contrasts the **EM-style
expert-token finetuning** scheme from `papers/master_thesis_stream_a.pdf` — which conditions a
*single* shared backbone on a small bank of EM-trained *expert tokens* — against the two obvious
alternatives:

- **the standard MoE** — genuine per-expert FFN *capacity* (parameter-heavy, sparse routing), and
- **general finetuning** — one dense model trained on all the data (full fine-tuning).

The goal is a setting where the MoE's capacity is *necessary*, and a clean read on what the EM
method buys instead: near-free **specialization** on a fixed backbone.

> **📊 For the full consolidated story** — the capacity comparison below *plus* the larger Qwen study
> (persona, knowledge, cold-start, collapse, catastrophic forgetting, convergence, thesis fidelity) and a
> synthesis of **all the advantages of EM training** — see **[`reports/README.md`](reports/README.md)**.

## The experiment (`configs/experiment/*_d256.yaml`)

A **capacity-constrained** byte-level backbone (d=256, 6 layers) on **`demix8`** — 8 distinct
domains (wiki, news, reviews, arxiv, pubmed, math, legal, stories). Capacity-constrained on purpose:
at this size a single dense model cannot comfortably hold all 8 domains, so the MoE's extra capacity
has room to matter. Four methods, matched compute:

| method | what it is | trained params |
|---|---|---|
| **dense** (`dense_d256`) | general finetuning — one dense model on all domains | full backbone |
| **MoE** (`moe_d256`) | standard fine-grained MoE (G=2, 16 experts, top-2) | full backbone + experts (~3× total) |
| **EM-prefix** (`em_prefix_d256`) | thesis persona/expert tokens, injected as a **prefix** | **only the tokens** (frozen backbone) |
| **EM-governance** (`em_gov_d256`) | expert tokens that **govern an FFN+attention subspace** (spectral) | **only the tokens** (frozen backbone) |

The EM methods are the thesis's **two-phase protocol**: **Phase A** (`*_seqA_*`) trains the backbone
with the expert tokens present-but-frozen; **Phase B** freezes the backbone and fits *only* the
tokens. EM-governance is the best-performing realization; EM-prefix is the literal-thesis reference.

## Results

Filled from [`reports/comparison/`](reports/comparison/) (macro-perplexity ↓, and **swap-ratio** =
how much routing a domain through the *wrong* expert costs — a direct measure of specialization).

| method | macro-ppl ↓ | swap-ratio ↑ | routing-NMI ↑ | trained params | total params |
|---|---|---|---|---|---|
| **MoE (standard)** | **2.660** | — | — | 27M (full) | 27M |
| EM-governance | 2.894 | **1.62** | **1.00** | **2,048** | 5.1M |
| EM-prefix | 2.940 | 1.53 | **1.00** | **2,048** | 5.0M |
| dense (full FT) | 2.965 | — | — | 5.0M (full) | 5.0M |

Per-domain: the MoE is best on all 8 domains, EM-governance beats dense on all 8
([`per_domain_ppl.md`](reports/comparison/per_domain_ppl.md)).

**Three findings.**

1. **The MoE is necessary — for capacity.** It is far ahead (**2.660 vs 2.965** dense, −0.31), a gap
   *no amount of conditioning closes*, because the gap is capacity: the MoE adds real per-expert FFN
   parameters (**5.5× total**, 27M vs 5M) and sparse routing. Neither the EM tokens nor full-FT reach
   it on a fixed backbone. (A prior rank/granularity study confirmed this is a hard ceiling: growing
   the EM governor's rank 8× moves perplexity by <0.01, while MoE granularity keeps improving.)
2. **The EM method beats full finetuning — for near-free, and with *specialization full-FT lacks*.**
   Both EM variants edge out dense (2.894 / 2.940 < 2.965) while training **only the 2,048 expert-token
   parameters** on a frozen backbone, and every domain routes to its own expert (**routing-NMI 1.00**,
   swap-ratio **1.5–1.6**: sending a domain through the wrong token costs 53–62% perplexity —
   [`routing_analysis.md`](reports/comparison/routing_analysis.md)). Full-FT and the MoE have *no* such
   interpretable per-domain structure — the MoE's routing is balanced/emergent, not domain-aligned.
3. **Governance > prefix.** Letting the token govern an FFN/attention *subspace* beats prepending it
   as a prefix (**2.894 vs 2.940**), with stronger specialization (swap 1.62 vs 1.53).

**The trade-off, in one line:** the **MoE buys capacity** (best quality, no interpretable
specialization); the **EM method buys specialization** (clean per-domain experts, near-free, on a
frozen backbone) but not capacity; **full finetuning buys neither**.

## How the expert token works (mechanistic interp)

[`reports/comparison/mech_interp/`](reports/comparison/mech_interp/) — measured as *token-on vs -off*:

- **Acts more deeply with depth** — the residual-stream shift grows layer-by-layer (the token lightly
  nudges early features, increasingly reshapes late, domain-specific ones).
- **Each domain governs a distinct subspace** — cross-expert gate overlap goes negative deep in the
  net (domains use *anti-correlated* FFN directions); the specialization is geometric.
- **Makes domains linearly separable** — a latent domain-probe jumps **0.875 → 1.000** with the token.
  The token installs a clean per-domain geometry the frozen backbone reads off — the mechanism behind
  the 100%-best-is-self routing.

## Repository

```
papers/master_thesis_stream_a.pdf   the EM technique (source)
configs/experiment/                 dense_d256, moe_d256, em_{prefix,gov}_{seqA,}_d256 (+ synth test configs)
src/softmoe/
  models/    soft_moe (EM: prefix + governance) · expert_tokens · governance · baselines/{dense,hard_moe}
  training/  em_trainer (two-phase Phase A/B) · losses
  eval/      harness · specialization (swap/NMI) · mech_interp · cross_routing · moe_analysis
scripts/     train · build_data · make_report · cross_routing · mech_interp · evaluate
reports/comparison/                 the EM-vs-MoE-vs-full-FT report + specialization + mech-interp
mpcdf-hpc-skills/                    Claude Code skills for running these on the MPCDF clusters
```

## Reproduce
```bash
DATA=/ptmp/$USER/soft-moe/data
python scripts/build_data.py --config configs/experiment/dense_d256.yaml --data-root $DATA
# full-FT and the standard MoE
for m in dense_d256 moe_d256; do python scripts/train.py --config configs/experiment/$m.yaml --data-root $DATA --run-dir .../$m --device cuda; done
# EM two-phase (prefix | governance): Phase A trains backbone (tokens frozen), Phase B fits only the tokens
for v in prefix gov; do
  python scripts/train.py --config configs/experiment/em_${v}_seqA_d256.yaml --data-root $DATA --run-dir .../em_${v}_seqA_d256 --device cuda
  python scripts/train.py --config configs/experiment/em_${v}_d256.yaml --data-root $DATA --run-dir .../em_${v}_d256 \
    --device cuda --init-backbone-from .../em_${v}_seqA_d256
done
python scripts/make_report.py   --runs <dir symlinking the 4 runs> --out reports/comparison
python scripts/cross_routing.py --runs .../moe_d256 .../em_prefix_d256 .../em_gov_d256 --out reports/comparison/routing_analysis.md
python scripts/mech_interp.py   --run .../em_gov_d256 --out reports/comparison/mech_interp
```

*Running on the MPCDF clusters (Raven/Viper) is wired through `mpcdf-hpc-skills/` — see its README.*
