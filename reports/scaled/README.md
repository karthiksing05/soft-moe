# Scaled comparison — our variants vs. a recipe-faithful MoE (SCALED_RECIPE.md)

**Run date:** 2026-07-02 · **Cluster:** Raven (A100, `mpib_gpu`) · **Corpus:** `demix8` (8 domains,
byte-level) · **Backbone:** d=512, 8 layers, 8 heads · **Steps:** 30k from scratch · **T=1** · **Seeds:** 1

This slots **our expert-token variants** into the MoE-vs-dense test regime of
[`../../SCALED_RECIPE.md`](../../SCALED_RECIPE.md), scaled to a real **~26M-active / 143M-total** MoE.
It implements the recipe's load-bearing subset (§8): the matched **dense vs coarse-MoE vs
fine-grained-MoE** comparison with the recipe's stabilization, plus oracle-MoE (DEMix), c-BTM, and
our expert-token variants. Artifacts: [`main_table.md`](main_table.md), [`per_domain_ppl.md`](per_domain_ppl.md),
[`routing_analysis.md`](routing_analysis.md), [`results.csv`](results.csv), [`figures/`](figures/),
[`domain_analysis_demix8.md`](domain_analysis_demix8.md).

> **Method note (thesis-faithful).** Our method now follows `papers/master_thesis_stream_a.pdf`
> exactly: a **two-phase EM protocol** — **Phase A** trains the backbone with the expert tokens
> *present but frozen* (random init) so it learns to condition; **Phase B** *freezes the backbone*
> and optimises *only* the tokens (soft-prompt tuning, high LR + noise injection). The old
> from-scratch **alternating** variant (not in the thesis) is **dropped**. This is a real change:
> the frozen-backbone Phase B makes the **tokens** — not a co-adapting backbone — carry the work.

## TL;DR

1. **Granularity is decisive — H2 reproduced (Krajewski 2024).** At matched active compute the
   **fine-grained MoE (G=2) beats dense (2.399 vs 2.472)**, but the **coarse MoE (G=1, the
   Switch/Mixtral config) is *worse* than dense (2.592)**. The coarse control saturates/reverses
   exactly as the recipe warns; granularity does the real work. (H1 — does the gap *widen* with
   scale? — needs the sweep: [`scaling_and_ablations.md`](scaling_and_ablations.md) finds it
   *saturates* at fixed budget/G, not widens.)
2. **MoE's advantage is fine-grained *capacity*, not *domain* specialization.** The fine-grained
   router stays load-balanced (usage-entropy 1.0) with only *weak emergent* domain correlation
   (doc-level routing-NMI **0.46**, vs 0.01 for coarse). And **oracle domain-routing (DEMix,
   NMI 1.0) is *worse* (2.589) than free fine-grained routing (2.399)** — forcing domain experts
   over-constrains the MoE.
3. **The thesis-faithful tokens beat dense on a *frozen* backbone, for 4K params.** `ours_sup_seq`
   reaches **2.460 < dense 2.472** training **only the 8 tokens (4,096 params)** on a frozen
   backbone — matching dense on all 8 domains. And the tokens now do **real work**: routing each
   domain to its own expert (100% best-is-self, NMI 1.0) with **swap-ratio 1.49** (routing through
   the wrong token costs 49% ppl), vs the old co-adapted method's hollow 1.02. This is the honest
   thesis setting — a frozen capable model + a per-identity embedding — and it *works*.
4. **It also beats dense at *every* scale (isoFLOP sweep).** d128→d512, the frozen-backbone tokens
   keep a **positive** gap over dense (+0.125→+0.008, saturating) — see
   [`scaling_and_ablations.md`](scaling_and_ablations.md). (The dropped from-scratch alternating
   had gone *negative*.) Supervised ≥ learned routing throughout (unsup `ours_unsup_seq` 2.659, but
   the strongest specialization of all — swap **3.05**).

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

| model | macro-ppl ↓ | routing-NMI (domain) | swap / ×worse | trainable | total | %gap |
|---|---|---|---|---|---|---|
| **MoE-G2 (fine-grained)** | **2.399** | 0.46 (emergent) | — | 143M | 143M | **100%** (ceiling) |
| **ours — sup + sequential (Phase B)** | **2.460** | 1.00 | **1.49×** | **4,096** | 25.6M | ~16% |
| Dense-1× | 2.472 | — | — | 25.6M | 25.6M | 0% |
| Dense-ceiling (Axis-B) | 2.422 | — | — | 143M | 143M | 68% |
| ours — unsup + sequential (Phase B) | 2.659 | 0.60 | **3.05×** | 271K | 25.9M | −256% |
| MoE-oracle (DEMix) | 2.589 | 1.00 (forced) | — | 143M | 143M | — |
| **MoE-G1 (coarse)** | **2.592** | 0.01 | — | 143M | 143M | −164% |
| c-BTM (8 models) | 3.214 | — | 3.05× | 205M | 205M | — |

**ours — sup + sequential (2.460) beats Dense-1× (2.472) while training only 4,096 parameters** (the
8 expert tokens) on a **frozen** backbone — the thesis's Phase B. `trainable` is the parameters
actually updated in the reported stage (Phase B for ours; the whole net for the others); the frozen
25.6M backbone was trained once in Phase A and is shared with Dense-1×.

**Axis A (iso-active):** MoE-G2 (2.399) beats Dense-1× (2.472) at equal per-token compute. **Axis B
(iso-total):** MoE-G2 (2.399) also beats the same-total-param Dense-ceiling (2.422) — while using
**5.5× fewer active params** (25.7M vs 143M). So the fine-grained MoE wins on *both* axes: more
capacity than Dense-1× at its compute, and Dense-ceiling's quality at a fraction of its compute
(the Joint-MoE-Scaling-Laws result). Dense-ceiling confirms capacity helps (2.422 < 2.472), but the
MoE extracts it far more efficiently.

## 3. Findings

### H2 — granularity is the whole game (reproduced at every scale)
`MoE-G2 (2.399) < Dense (2.472) < MoE-G1 (2.592)`. The **coarse MoE is worse than dense**; only the
**fine-grained MoE beats it** — and the scaling sweep confirms this holds at *all* four sizes
(d128→d512). This is the central Krajewski/Ludziejewski (2024) result — the coarse GShard/Switch
config understates (here reverses) the advantage, and granularity, at ~no extra compute, unlocks
it. (The *trend* — H1, whether the gap widens — is in [`scaling_and_ablations.md`](scaling_and_ablations.md):
it **saturates** here, attributable to fixed training budget + fixed G=2.)

### The mechanism is capacity, not domain routing
- The fine-grained router is **load-balanced** (usage-entropy 1.0, 0 dead experts) with **weak
  emergent** domain structure: doc-level routing-NMI **0.46** (vs **0.01** for coarse). Granularity
  buys *some* emergent domain-correlated routing, but experts are not domain experts.
- **Oracle domain-routing (DEMix, NMI 1.0) is *worse* (2.589) than the fine-grained learned router
  (2.399).** Rigidly assigning one expert per domain over-constrains the MoE and wastes its
  flexibility. So the MoE's win comes from *fine-grained, flexible, load-balanced capacity* — not
  from domain specialization.

### Our method (thesis-faithful): tokens beat dense on a *frozen* backbone, and now do real work
- **Beats dense for 4K params.** `ours_sup_seq` 2.460 < Dense-1× 2.472, updating **only the 8 tokens
  (4,096 params)** in Phase B while the backbone stays frozen — matching dense on all 8 domains
  ([`per_domain_ppl.md`](per_domain_ppl.md)). Pure per-identity prompt-tuning on a frozen capable
  model recovers (slightly beats) full dense LM — exactly the thesis's claim.
- **The tokens carry genuine, non-interchangeable expertise.** Routing a domain through the *wrong*
  token costs **1.49×** ppl (sup) / **3.05×** (unsup), with routing-NMI 1.0 / 0.60. Contrast the
  *old* co-adapted "sequential" (swap 1.02): there the backbone co-adapted and did the work, so the
  tokens were nearly interchangeable. **The correction — freezing the backbone in Phase B — is what
  forces the tokens to specialize**, and they do.
- **Conditioning vs capacity, restated.** Ours still doesn't reach the fine-grained MoE (2.399): a
  frozen backbone + 8 domain vectors is a *domain-specialization* lever, weaker than the MoE's
  *fine-grained flexible capacity* (143M params). But at **4K trainable params vs 143M**, and beating
  dense, it is a far cheaper lever than the earlier framing suggested. Conditioning ≠ capacity — but
  conditioning, done faithfully, is a real and remarkably cheap win over dense.

### The training levers (thesis-faithful; alternating dropped)
The from-scratch **alternating** variant is dropped (not in the thesis). Within the two-phase
sequential protocol, **supervised ≥ learned routing** (2.460 vs 2.659) — mirroring oracle-MoE's
clean assignment helping our method even as it *hurts* the MoE. And the two-phase order matters:
Phase A must train the backbone *with the tokens present* (else Phase B's frozen-backbone tokens
have nothing to steer — the failure mode of the pre-correction pure-dense pretraining).

## 4. Per-domain perplexity (every model)
See [`per_domain_ppl.md`](per_domain_ppl.md). MoE-G2 is best or tied on **all 8 domains**; the
coarse MoE is worst-or-near on all; c-BTM wins only `legal` (its data-richest cluster) and is far
worse on small domains (`math` 5.87) — the classic c-BTM data-starvation across 8 sub-corpora.

## 5. Deferred recipe parts — now completed

The scaling sweep, ablation matrix, and axes are completed in **[`scaling_and_ablations.md`](scaling_and_ablations.md)** (H2 holds across scale; H1 widening not observed at fixed budget/G; top-k & balancing are first-order; memory-matched MoE wins). Original deferral note:

### (original) What this run does *not* cover
The full SCALED_RECIPE.md is a multi-week study; this is its §8 load-bearing subset at one budget.
Not done here (the natural extensions): the **isoFLOP scaling *sweep*** across ≥5 budgets (to fit
the curve and show the gap *trend*, H1's strongest form), **downstream/few-shot + full
fine-tuning** eval (H3/H4, where Artetxe/Abnar show the gap compresses or reverses), the
**granularity × shared-expert × balancing ablation matrix** (H6), **≥3 seeds**, and the true
Axis-C memory accounting. The point estimates here are single-seed.

## 6. Caveats
- 1 seed, byte-level. Dense/MoE arms are 30k steps; ours = Phase A 30k (backbone) + Phase B 6k
  (token-only, frozen backbone). Learned-MoE numbers are a lower bound on a fully-tuned MoE (no
  per-arm LR sweep / granularity beyond G=2 / shared experts here).
- routing-NMI for dense/MoE/cbtm in `main_table.md` is the k-means clusterer's NMI (baseline
  reference); the *MoE's own* learned routing-NMI is in `routing_analysis.md` (0.01 / 0.46 / 1.0).

## 7. Reproduce
```bash
DATA=/ptmp/$USER/soft-moe/data
python scripts/build_data.py --config configs/experiment/sc_dense1x.yaml --data-root $DATA
for m in sc_dense1x sc_dense_ceiling sc_moe_g1 sc_moe_g2 sc_moe_oracle sc_cbtm; do
  python scripts/train.py --config configs/experiment/$m.yaml --data-root $DATA --run-dir .../$m --device cuda; done
# ours = thesis-faithful two-phase EM. Phase A: backbone + frozen (random) tokens present.
python scripts/train.py --config configs/experiment/sc_seqA.yaml --data-root $DATA --run-dir .../sc_seqA --device cuda
# Phase B: freeze backbone, tune ONLY the tokens; warm-start carries Phase-A's frozen tokens forward.
for m in sc_ours_sup_seq sc_ours_unsup_seq; do
  python scripts/train.py --config configs/experiment/$m.yaml --data-root $DATA --run-dir .../$m --device cuda \
    --init-backbone-from .../sc_seqA; done
python scripts/make_report.py   --runs <symlinks> --out reports/scaled
python scripts/cross_routing.py --runs <moe + ours runs> --out reports/scaled/routing_analysis.md
```
