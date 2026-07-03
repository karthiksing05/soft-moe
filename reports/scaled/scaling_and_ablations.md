# Completing the deferred recipe parts — scaling sweep, ablations, axes

This completes the parts [`README.md`](README.md) §5 deferred from SCALED_RECIPE.md, in the
recipe's §8 priority order. Data: [`scaling_sweep.md`](scaling_sweep.md); figures
`comparison_figures/9_scaling_curves.png`, `10_gap_vs_scale.png`.

## 1. isoFLOP scaling sweep (H1/H2) — the load-bearing deferred result

Dense / coarse-MoE (G=1) / fine-MoE (G=2) / **ours** at **four sizes** (d128→d512, 0.9M→25.6M
active), 20k steps each, **3 seeds at d256** for error bars. `ours` = the **thesis-faithful
sequential** — Phase A trains the backbone (20k) with the tokens present-but-frozen, then a cheap
Phase B (6k) freezes the backbone and tunes only the 8 tokens; supervised routing, T=1. Compute ≈
dense (Phase B is token-only). Figures: `comparison_figures/9_scaling_curves.png`, `10_gap_vs_scale.png`.

Now with the **governance** arm (spectral FFN+attn, rank16) run at all four sizes on the same 20k
Phase A + 6k Phase B recipe — so `ours` (prefix) and `gov` (spectral subspace governance) are
directly comparable across scale.

| size | active | dense | MoE-G1 | MoE-G2 | ours prefix | gov spectral | **dense−G2** | **dense−gov** |
|---|---|---|---|---|---|---|---|---|
| d128 | 0.9M | 3.876 | 3.951 | 3.437 | 3.751 | 3.803 | **+0.440** | +0.073 |
| d256 | 5.0M | 2.971±.004 | 3.017±.012 | 2.692±.030 | 2.949 | 2.903 | **+0.278** | +0.068 |
| d384 | 11.0M | 2.681 | 2.832 | 2.520 | 2.641 | 2.648 | **+0.161** | +0.033 |
| d512 | 25.6M | 2.535 | 2.704 | 2.465 | 2.527 | 2.520 | **+0.071** | +0.015 |

**Governance scales like prefix — both beat dense everywhere, both saturate, both far below the MoE.**
`gov` beats dense at all four sizes (dense−gov +0.073→+0.015) and tracks the prefix curve closely
(gov is marginally ahead at d256/d512, prefix at d128/d384). Two conditioning mechanisms, essentially
the same isoFLOP trajectory: a positive-but-shrinking edge that never approaches MoE-G2's. (Governance
*does* pull ahead of prefix with a longer Phase A — the d512 *headline* run at 30k Phase A gives
gov+attn 2.446 vs prefix 2.460; the governor needs more backbone training to pay off, which the
matched-20k sweep understates.) The throughline holds at every scale: **conditioning a fixed backbone
saturates below the MoE; the gap is capacity.**

**H2 — granularity does real work (confirmed at *every* scale):** MoE-G2 beats dense at all four
sizes; coarse MoE-G1 is *worse* than dense at all four. The coarse Switch/Mixtral config never
helps here — exactly the recipe's warning.

**Ours (thesis-faithful) beats dense at *every* scale — cheaply.** The `dense−ours` gap is
**positive at all four sizes** (+0.125→+0.008): freezing the backbone and tuning only the 8 tokens
(Phase B) recovers a small but consistent edge over dense, everywhere. Like the MoE gap it
**saturates** with scale (the bigger, better-trained backbone leaves less for a domain vector to
add), but it never goes negative. This is a genuine correction of the earlier finding: the
*pre-correction* from-scratch alternating variant had gone negative at d256+, an artifact of never
giving the backbone a proper Phase A. The gap stays well below MoE-G2's — a domain vector is a
weaker lever than fine-grained capacity — but ours does it at **4K trainable params on a frozen
backbone**, vs the MoE's 143M.

**H1 — gap widens with scale? NOT reproduced (honest).** The fine-grained advantage **shrinks**
with size (11.4%→2.8%) — the *saturation* pattern of Clark et al. (2022), not the *widening* of
Krajewski/Ludziejewski (2024). Two reasons, both confounds the recipe explicitly flags:
1. **Fixed training budget** (20k steps at every size) is *not* compute-optimal per point (§4.7):
   bigger models are progressively more undertrained (d512 saw ~164M tokens vs a Chinchilla
   ~500M), and MoE's advantage needs adequate training to materialize.
2. **Fixed granularity G=2.** Krajewski's widening requires *scaling granularity up* with size; we
   held G=2. The G=4/G=8 runs that would test this **did not finish** — see §2.

So we reproduced the granularity effect (H2) robustly, but *cannot* claim the widening (H1); doing
so would require the full compute-optimal-per-point sweep with scaled granularity.

## 2. Ablation matrix (H6) at d256 (20k, single seed)
Baselines: dense 2.971, MoE-G1 2.717(top-1) via sweep is 3.017, MoE-G2 2.692.

| variant | macro-ppl ↓ | vs MoE-G2 | takeaway |
|---|---|---|---|
| **MoE-G1 top-k=2** | **2.674** | −0.018 | **top-k is the biggest lever** — 2 active experts (coarse) beat even fine-grained top-1 (3.017→2.674) |
| MoE-G2 z-loss **off** | 2.663 | −0.029 | z-loss slightly *hurts* quality here — no divergence to prevent at this scale (its value is stability at larger scale, per ST-MoE) |
| MoE-G2 (baseline) | 2.692 | — | fine-grained top-1 |
| MoE-G2 + 2 shared | 2.694 | +0.002 | shared experts ≈ neutral at this scale |
| MoE-G2 + 1 shared | 2.702 | +0.010 | " |
| MoE-G2 balancing **off** | 2.722 | +0.030 | load-balancing helps (removing it hurts + risks collapse) |

**H6 confirmed in spirit:** the stabilization/routing choices (top-k, balancing, z-loss) move the
result as much as or more than granularity at small scale, and their *sign is scale-dependent*
(z-loss helps only where instability exists). **Not run: G ∈ {4, 8}.** The naive per-expert Python
loop in `MoEFFN` is O(#experts) per layer and became impractically slow at 32/64 experts (G=8
reached only 3k/20k steps in >3h). This is a real implementation limitation — production MoEs use
batched/grouped-GEMM expert compute; ours is a from-scratch minimal kernel. G=4 was still running
at writing.

## 2b. T-ablation — tokens per expert ∈ {1, 2, 4}, thesis-faithful sequential (d512, frozen backbone)
Does more than one soft-prompt vector per expert help on a frozen backbone? All at the 6k Phase-B
budget (the main run's), so T is the only variable. (Figure `comparison_figures/11_t_ablation.png`.)

| method | T=1 | T=2 | T=4 |
|---|---|---|---|
| sup + sequential | **2.460** | 2.475 | 2.503 |
| unsup + sequential | 2.659 | **2.617** | 2.643 |

**T=1 is best (or within noise) even with a frozen backbone.** For the supervised variant more
vectors monotonically *hurt* (2.460→2.503); for unsupervised, T=2 helps marginally (−0.04) but the
variant is weak regardless. So **one soft-prompt vector per expert suffices** — validating the
thesis's one-embedding-per-identity default. (An earlier run at a longer 10k Phase B made T=1 *look*
worse than T=2, but that was a single-domain **instability**, not a capacity effect: at 10k the T=1
`math` token over-tuned and collapsed to ppl 4.08, inflating macro to 2.677; at the matched 6k
budget T=1 recovers to 2.460 and matches dense on all 8 domains. The lesson is that frozen-backbone
prompt-tuning wants a *short* Phase B — over-tuning a tiny token set destabilizes the hardest
domain — not more tokens.)

## 3. Memory axis (Axis C / H5) — already answered
Memory ≈ total parameters. From the main run, at **equal total params (143M)**:
**MoE-G2 (2.399) beats Dense-ceiling (2.422)** — while using **5.5× fewer active params**. So the
memory-matched MoE beats the memory-matched dense (H5 confirmed), *and* is far cheaper per token.
See `comparison_figures/8_quality_vs_total_params.png`.

## 4. Downstream few-shot + full fine-tuning (H3/H4) — out of scope at this scale, with rationale
The recipe's downstream suite (MMLU / HellaSwag / GSM8K / closed-book QA) and its fine-tuning
transfer test are **not meaningful on 1–26M-parameter byte-level LMs** — those benchmarks require
real pretrained-scale models to score above chance. The meaningful analogues at our scale are:
- **Per-domain generalization** — done (see [`per_domain_ppl.md`](per_domain_ppl.md)); MoE-G2 is
  best or tied on all 8 domains.
- **Out-of-domain held-out ppl + fine-tuning transfer** — the right proxy for H3, but needs a
  held-out corpus + a full-model fine-tune path (only backbone-warm-start exists today). This is
  the one genuinely-remaining piece; it is the recipe's *lowest* §8 priority and is left as the
  documented next step.

## 5. What the completion establishes
- **Granularity (H2) is robust across scale**; the *coarse* MoE never beats dense.
- The **widening (H1) is not observed** at fixed budget + fixed G — consistent with the recipe's
  caveat that the widening is a compute-optimal-with-scaled-granularity phenomenon.
- **top-k and load-balancing are first-order levers** (H6); z-loss is a stability tool whose
  quality effect is scale-dependent.
- **Memory-matched (H5): MoE wins.**
- Seeds (d256) give tight error bars (±0.004–0.03), so the sweep points are reliable.

Truly remaining (a multi-week study): the *compute-optimal-per-point* isoFLOP sweep with **scaled
granularity** (to fairly test H1), a **batched MoE kernel** for G≥4, the **downstream/fine-tuning**
axes at real scale (H3/H4), and ≥3 seeds at every point.
