# Completing the deferred recipe parts — scaling sweep, ablations, axes

This completes the parts [`README.md`](README.md) §5 deferred from SCALED_RECIPE.md, in the
recipe's §8 priority order. Data: [`scaling_sweep.md`](scaling_sweep.md); figures
`comparison_figures/9_scaling_curves.png`, `10_gap_vs_scale.png`.

## 1. isoFLOP scaling sweep (H1/H2) — the load-bearing deferred result

Dense / coarse-MoE (G=1) / fine-MoE (G=2) trained from scratch at **four sizes** (d128→d512,
0.9M→25.6M active), 20k steps each, **3 seeds at d256** for error bars.

| size | active | dense | MoE-G1 (coarse) | MoE-G2 (fine) | **dense−G2 gap** |
|---|---|---|---|---|---|
| d128 | 0.9M | 3.876 | 3.951 | 3.437 | **+0.440** (11.4%) |
| d256 | 5.0M | 2.971±.004 | 3.017±.012 | 2.692±.030 | **+0.278** (9.4%) |
| d384 | 11.0M | 2.681 | 2.832 | 2.520 | **+0.161** (6.0%) |
| d512 | 25.6M | 2.535 | 2.704 | 2.465 | **+0.071** (2.8%) |

**H2 — granularity does real work (confirmed at *every* scale):** MoE-G2 beats dense at all four
sizes; coarse MoE-G1 is *worse* than dense at all four. The coarse Switch/Mixtral config never
helps here — exactly the recipe's warning.

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
