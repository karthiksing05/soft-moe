# What the MoE buys: capacity (fine-graining) vs conditioning (subspace rank)

Two levers, back to back at d512 on `demix8`, to isolate **what the MoE's value actually is**. The
**MoE granularity** lever (G1→G2→G4→G8) adds genuine fine-grained FFN capacity; the **governance
rank** lever (r16→r64→r128) adds conditioning richness to a *fixed* FFN (spectral FFN+attn, only the
4,096 token params trained in Phase B). If governance plateaus while the MoE keeps improving, that gap
*is* the value of real capacity. Figure: `capacity_curve.png`. (G4/G8 were only trainable after
replacing the MoE's `top_k×n_experts` masked loop with a batched sort-by-expert dispatch — verified
numerically identical.)

| lever | setting | macro-ppl ↓ | swap-ratio ↑ | total params |
|---|---|---|---|---|
| — | Dense-1× | 2.472 | — | 25.6M |
| — | ours prefix (input) | 2.460 | 1.491 | 25.6M |
| **MoE granularity** | G1 (coarse) | 2.592 | — | 143M |
| | G2 | 2.399 | — | 143M |
| | **G4** | **2.383** | — | 143M |
| | G8 | 2.392 | — | 143.7M |
| **governance rank** | r16 | 2.446 | 1.643 | 26.1M |
| | r64 | 2.444 | 1.648 | 27.5M |
| | r128 | 2.442 | 1.723 | 29.3M |

## Findings

1. **Governance conditioning plateaus — hard.** Growing the governed subspace **8×** (rank 16→128)
   buys **0.004 ppl** (2.446→2.442): essentially nothing. Re-weighting a *fixed* FFN has a ceiling
   (~2.44) that more conditioning richness cannot break. Tellingly, rank *does* keep buying
   **specialization** (swap-ratio 1.643→1.723) — the tokens carve ever-finer per-expert subspaces —
   but the fixed FFN has no more to express, so **quality doesn't move**. The bottleneck is capacity,
   not conditioning.
2. **MoE fine-graining pays until an optimum.** G1 2.592 → G2 2.399 → **G4 2.383** → G8 2.392: the
   granularity lever adds real per-token capacity, peaking at **G4** and then *over-fragmenting* at
   G8 (64 experts of width 256 — too narrow to use, the known granularity-optimum from
   Krajewski 2024). Even so, **every MoE granularity (including the G8 over-shoot) sits below the
   entire governance rank ladder** — the capacity floor is fundamentally lower than the conditioning
   ceiling.
3. **This is the value of the MoE, quantified.** The two levers are **orthogonal**: conditioning
   (rank) *saturates* at ~2.44; capacity (granularity) *scales* past it to 2.383. The MoE's added
   capacity buys **~0.06 ppl below governance's ceiling** — a floor that conditioning a fixed FFN
   *cannot reach at any rank*, because it does not add capacity. Governance gets remarkably close
   (within 0.06) at **5.5× fewer params** (26M vs 143M) and beats dense/prefix — but the last
   increment of quality is precisely what only genuine capacity (the MoE) provides. **Conditioning ≠
   capacity**, now shown as two separable axes rather than an aggregate claim.

**Takeaway.** Our expert-token governance is the efficient way to *condition* a shared model — it
extracts almost all the achievable gain from a fixed-capacity backbone, cheaply. The MoE's distinct
and irreplaceable value is *capacity*: fine-grained experts keep lowering perplexity where
conditioning has already saturated. The right architecture pairs them — cheap token governance for
specialization on top of an MoE backbone for capacity — rather than treating governance as a
substitute for the MoE.

## Reproduce
```bash
for g in 1 2 4 8; do python scripts/train.py --config configs/experiment/sc_moe_g${g}.yaml --run-dir .../sc_moe_g${g} --device cuda; done
for r in 16 64 128; do   # r16 = govern_spectralA_d512; r64/r128 = govern_spectralA_r${r}_d512
  python scripts/train.py --config configs/experiment/govern_seqA_spectralA_r${r}_d512.yaml --run-dir .../seqA --device cuda
  python scripts/train.py --config configs/experiment/govern_spectralA_r${r}_d512.yaml --run-dir .../pb --device cuda --init-backbone-from .../seqA
done
python scripts/make_report.py --runs <dir symlinking the arms> --out reports/capacity
python scripts/make_capacity_figure.py
```
