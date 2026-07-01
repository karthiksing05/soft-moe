# Scaled comparison — our variants vs. a recipe-faithful MoE (SCALED_RECIPE.md)

**Run date:** 2026-07-01 · **Cluster:** Raven (A100, `mpib_gpu`) · **Corpus:** `demix8` (8 domains,
byte-level) · **Backbone:** d=512, 8 layers, 8 heads · **Steps:** 30k from scratch · **T=1** · **Seeds:** 1

This slots **our expert-token variants** into the MoE-vs-dense test regime of
[`../../SCALED_RECIPE.md`](../../SCALED_RECIPE.md), scaled to a real **~26M-active / 143M-total** MoE.
It implements the recipe's load-bearing subset (§8): the matched **dense vs coarse-MoE vs
fine-grained-MoE** comparison with the recipe's stabilization, plus oracle-MoE (DEMix), c-BTM, and
our four variants. Artifacts: [`main_table.md`](main_table.md), [`per_domain_ppl.md`](per_domain_ppl.md),
[`routing_analysis.md`](routing_analysis.md), [`results.csv`](results.csv), [`figures/`](figures/),
[`domain_analysis_demix8.md`](domain_analysis_demix8.md).

## TL;DR

1. **Granularity is decisive — H1/H2 reproduced (Krajewski 2024).** At matched active compute the
   **fine-grained MoE (G=2) beats dense (2.399 vs 2.472)**, but the **coarse MoE (G=1, the
   Switch/Mixtral config) is *worse* than dense (2.592)**. The coarse control saturates/reverses
   exactly as the recipe warns; granularity does the real work.
2. **MoE's advantage is fine-grained *capacity*, not *domain* specialization.** The fine-grained
   router stays load-balanced (usage-entropy 1.0) with only *weak emergent* domain correlation
   (doc-level routing-NMI **0.46**, vs 0.01 for coarse). And **oracle domain-routing (DEMix,
   NMI 1.0) is *worse* (2.589) than free fine-grained routing (2.399)** — forcing domain experts
   over-constrains the MoE.
3. **Our expert tokens do the opposite — explicit, clean domain specialization — cheaply.** All
   our variants route each domain to its own expert (**100% best-is-self**), at **+4K–271K params**,
   and the best (`ours_sup_seq` 2.464) edges out dense. But that domain specialization yields *less*
   LM gain than MoE's capacity: ours closes only ~11% of the dense→fine-MoE gap.
4. **Same two levers as the small-scale run hold:** sequential (pretrain-then-alternate) ≥
   alternating; supervised ≥ learned routing.

## 1. The recipe, implemented

Per SCALED_RECIPE.md §4, only the FFN differs across arms. The MoE arm has the near-Pareto-optimal
levers: **fine-grained granularity G** (Krajewski 2024), **router z-loss** (ST-MoE), **Switch
load-balancing** (Fedus 2022), **top-k token routing**; `G=1` reproduces coarse Switch/Mixtral.
Corpus: 8 distinct domains streamed from public HF datasets, byte-level, rebalanced to ~15k
blocks/domain. The three matching axes (§4.9) are set up exactly:

| | active/token | total params |
|---|---|---|
| Dense-1× (Axis-A baseline) | 25.6M | 25.6M |
| **MoE (G1/G2/oracle)** | **25.7M** (≈ Dense-1×) | **143M** (≈ Dense-ceiling) |
| Dense-ceiling (Axis-B) | 143M | 143M |
| ours | 25.6–25.9M | 25.6–25.9M |

*Axis A* = iso-active-FLOPs (MoE 25.7M = Dense-1× 25.6M); *Axis B* = iso-total-params (MoE total
143M = Dense-ceiling 143M); *Axis C* (memory) ≈ total params.

## 2. Results (macro-perplexity, byte-level, ↓)

From [`main_table.md`](main_table.md). "%gap" = fraction of the **dense→fine-MoE** gap
`(2.472−macro)/(2.472−2.399)` closed at ~Dense-1× compute.

| model | macro-ppl ↓ | routing-NMI (domain) | swap / ×worse | active | total | %gap |
|---|---|---|---|---|---|---|
| **MoE-G2 (fine-grained)** | **2.399** | 0.46 (emergent) | — | 25.7M | 143M | **100%** (ceiling) |
| **ours — sup + sequential** | **2.464** | 1.00 | 1.02× | 25.6M | 25.6M | ~11% |
| Dense-1× | 2.472 | — | — | 25.6M | 25.6M | 0% |
| ours — unsup + sequential | 2.476 | 0.61 | 1.05× | 25.9M | 25.9M | ~−5% |
| ours — sup + alternating | 2.511 | 1.00 | **1.50×** | 25.6M | 25.6M | −53% |
| ours — unsup + alternating | 2.554 | 0.62 | 1.06× | 25.9M | 25.9M | — |
| MoE-oracle (DEMix) | 2.589 | 1.00 (forced) | — | 25.7M | 143M | — |
| Dense-ceiling (Axis-B) | 2.422 | — | — | 143M | 143M | 68% |
| **MoE-G1 (coarse)** | **2.592** | 0.01 | — | 25.7M | 143M | −164% |
| c-BTM (8 models) | 3.214 | — | 3.05× | 25.6M | 205M | — |

**Axis A (iso-active):** MoE-G2 (2.399) beats Dense-1× (2.472) at equal per-token compute. **Axis B
(iso-total):** MoE-G2 (2.399) also beats the same-total-param Dense-ceiling (2.422) — while using
**5.5× fewer active params** (25.7M vs 143M). So the fine-grained MoE wins on *both* axes: more
capacity than Dense-1× at its compute, and Dense-ceiling's quality at a fraction of its compute
(the Joint-MoE-Scaling-Laws result). Dense-ceiling confirms capacity helps (2.422 < 2.472), but the
MoE extracts it far more efficiently.

## 3. Findings

### H1/H2 — granularity is the whole game (reproduced)
`MoE-G2 (2.399) < Dense (2.472) < MoE-G1 (2.592)`. The **coarse MoE is worse than dense**; only the
**fine-grained MoE beats it**. This is the central Krajewski/Ludziejewski (2024) result — the
coarse GShard/Switch config understates (here reverses) the advantage, and granularity, at ~no
extra compute, unlocks it. It is also *why* single-point "MoE vs dense" comparisons disagree in
the literature (§2.3).

### The mechanism is capacity, not domain routing
- The fine-grained router is **load-balanced** (usage-entropy 1.0, 0 dead experts) with **weak
  emergent** domain structure: doc-level routing-NMI **0.46** (vs **0.01** for coarse). Granularity
  buys *some* emergent domain-correlated routing, but experts are not domain experts.
- **Oracle domain-routing (DEMix, NMI 1.0) is *worse* (2.589) than the fine-grained learned router
  (2.399).** Rigidly assigning one expert per domain over-constrains the MoE and wastes its
  flexibility. So the MoE's win comes from *fine-grained, flexible, load-balanced capacity* — not
  from domain specialization.

### Our method: explicit domain specialization, cheap, but capacity-bounded
- Every ours variant gives **clean domain specialization** — 100% best-is-self in cross-routing,
  routing-NMI 1.0 (supervised) / ~0.62 (learned, ≈ the k-means clusterer's 0.63). The tokens *are*
  domain experts, by design — the opposite of the MoE's balanced token routing.
- Best variant `ours_sup_seq` (2.464) **edges out dense** at **+4K params**, but closes only ~11%
  of the dense→fine-MoE gap. **Explicit domain specialization (ours, and DEMix) is simply a weaker
  lever than fine-grained flexible capacity (MoE).** Conditioning ≠ capacity — the same conclusion
  as the small-scale run, now with a real MoE to measure against.
- **Specialization vs quality trade-off within ours:** `sup_alt` has the strongest per-expert
  expertise (swap **1.50×**) but worse quality (2.511); `sup_seq` has the best quality (2.464) but
  weak swap (1.02×) — a strong co-adapted backbone carries the load, leaving the tokens to add
  little. Same pattern as the pretrained pilot.

### The two training levers (confirmed at scale)
Sequential (pretrain-then-alternate) ≥ alternating (2.464 vs 2.511 supervised; 2.476 vs 2.554
unsup), and supervised ≥ learned routing — mirroring oracle-MoE ≥ learned-MoE. Good routing + a
properly-trained, co-adapting backbone are necessary for either method.

## 4. Per-domain perplexity (every model)
See [`per_domain_ppl.md`](per_domain_ppl.md). MoE-G2 is best or tied on **all 8 domains**; the
coarse MoE is worst-or-near on all; c-BTM wins only `legal` (its data-richest cluster) and is far
worse on small domains (`math` 5.87) — the classic c-BTM data-starvation across 8 sub-corpora.

## 5. What this run does *not* cover (recipe parts deferred)
The full SCALED_RECIPE.md is a multi-week study; this is its §8 load-bearing subset at one budget.
Not done here (the natural extensions): the **isoFLOP scaling *sweep*** across ≥5 budgets (to fit
the curve and show the gap *trend*, H1's strongest form), **downstream/few-shot + full
fine-tuning** eval (H3/H4, where Artetxe/Abnar show the gap compresses or reverses), the
**granularity × shared-expert × balancing ablation matrix** (H6), **≥3 seeds**, and the true
Axis-C memory accounting. The point estimates here are single-seed.

## 6. Caveats
- 1 seed, 30k steps, byte-level. Learned-MoE numbers are a lower bound on a fully-tuned MoE (no
  per-arm LR sweep / granularity beyond G=2 / shared experts here).
- routing-NMI for dense/MoE/cbtm in `main_table.md` is the k-means clusterer's NMI (baseline
  reference); the *MoE's own* learned routing-NMI is in `routing_analysis.md` (0.01 / 0.46 / 1.0).

## 7. Reproduce
```bash
DATA=/ptmp/$USER/soft-moe/data
python scripts/build_data.py --config configs/experiment/sc_dense1x.yaml --data-root $DATA
for m in sc_dense1x sc_dense_ceiling sc_moe_g1 sc_moe_g2 sc_moe_oracle sc_cbtm sc_ours_sup_alt sc_ours_unsup_alt; do
  python scripts/train.py --config configs/experiment/$m.yaml --data-root $DATA --run-dir .../$m --device cuda; done
for m in sc_ours_sup_seq sc_ours_unsup_seq; do   # sequential: pretrain then warm-start + alternate
  python scripts/train.py --config configs/experiment/$m.yaml --data-root $DATA --run-dir .../$m --device cuda \
    --init-backbone-from .../sc_dense1x; done
python scripts/make_report.py   --runs <symlinks> --out reports/scaled
python scripts/cross_routing.py --runs <moe + ours runs> --out reports/scaled/routing_analysis.md
```
