# Persona-embedding collapse metric — the thesis's second primary metric

The thesis lists **persona-collapse detection** as a primary metric: if all learned persona embeddings
converge to a common vector, the model has failed to personalise. We now measure it on the learned
`<|expert_k|>` embeddings (read directly from the saved weights; `collapse_metric.py`).

With one embedding per persona the thesis's exact *within-/between-cluster variance ratio* is degenerate
(no within-cluster spread), so we report the standard geometric equivalents of embedding spread:
**mean pairwise cosine similarity** (→1 = collapsed), **effective rank** of the centered embeddings, and
**mean pairwise L2 distance**. Measured on the cold-start checkpoints:

| run | mean-cos ↓ (collapse) | eff-rank | mean-L2 ↑ (separation) |
|---|---|---|---|
| joint SFT (naive) | 0.227 | 7.00 | 0.79 |
| EM two-phase | 0.084 | 6.90 | 8.55 |
| **EM Phase-B-heavy** | **0.057** | 6.86 | **9.77** |
| control (generic token) | — | — | — (N/A: no per-persona embeddings — these rows are untrained/unused vocab slots) |

![collapse](figs/collapse.png)

## Findings

1. **EM embeddings are far *less* collapsed than joint SFT's.** Mean pairwise cosine similarity drops
   **0.227 → 0.084 → 0.057** (joint → EM → Phase-B-heavy): EM's persona embeddings are 3–4× more distinct,
   nearly orthogonal. The trend is monotonic in Phase-B budget — **more embedding-only training separates the
   personas more**, exactly the thesis's intent.
2. **The separation gap is dramatic in raw distance:** mean pairwise L2 **0.79 → 8.55 → 9.77 (~10×)**. Phase B
   (token-only, high LR) drives each embedding to a large, distinct vector to steer the *frozen* backbone,
   whereas in joint SFT the embeddings co-adapt with a moving backbone and stay small and close together.
3. **No arm catastrophically collapsed** (effective rank ≈ 7 of a max 7 for all), so this is a *degree* of
   separation, not a binary failure — but the ranking (EM ≫ joint on distinctness) is clear and consistent
   with the swap-test behavioural signal (EM's tokens are more load-bearing).

## Fidelity note

This implements the thesis's second primary metric and returns a **thesis-supporting** result: the EM
alternating protocol produces more distinct, less-collapse-prone persona embeddings than naive joint SFT,
and the effect grows with Phase-B budget. (We still do not use the thesis's Phase-B *noise injection*, which
is specifically meant to further prevent collapse — a remaining fidelity gap in [THESIS_FIDELITY.md](THESIS_FIDELITY.md).)
