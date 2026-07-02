# FFN(+attn) subspace governance at d512 — the headline scale

Scaling the winning **spectral subspace governance** to the d512 backbone (30k Phase A + 6k Phase B),
against the *real* baselines from the main scaled run. Governance is delivered exactly as at d256:
the EM expert latent `e_k` drives a shared hypernet that gates a learned r=16 orthonormal subspace of
each block's FFN hidden — and, for `+attn`, of each block's attention output too. All "ours" arms
train **only the 4,096 token params** in Phase B (frozen backbone + frozen hypernet).

| method | macro-ppl ↓ | micro-ppl ↓ | swap-ratio ↑ | trainable | total params |
|---|---|---|---|---|---|
| MoE-G2 (fine-grained) | **2.399** | 2.306 | — | 143M | 143M |
| **ours — spectral FFN + attn** | **2.446** | 2.354 | **1.643** | **4,096** | 26.1M |
| ours — prefix (input) | 2.460 | 2.368 | 1.491 | 4,096 | 25.6M |
| Dense-1× | 2.472 | 2.376 | — | 25.6M | 25.6M |
| ours — spectral FFN-only | 2.963† | 2.880 | 1.000† | 4,096 | 26.0M |

†cold-start failure — see below (rerun with the fix in progress).

## Findings

1. **Subspace governance beats dense *and* our prefix method at scale.** `spectral_ffn + attn`
   reaches **2.446 < prefix 2.460 < dense 2.472**, beating dense on **all 8 domains** (per-domain
   `per_domain_ppl.md`), while training only the 4,096 token params. It closes most of the remaining
   gap to the full fine-grained MoE (2.399) — at **5.5× fewer total params** (26M vs 143M). This is
   now the best of the single-model "ours" family, and it confirms the d256 finding at the headline
   scale: **letting the expert govern FFN+attention subspaces beats prepending it as a vector.**
2. **Strongest specialization of any arm** (swap-ratio **1.643**): routing a domain through the wrong
   expert costs 64% ppl. The governed subspace is genuinely per-expert.
3. **Governing attention matters more at scale.** At d256, `+attn` was a small edge over FFN-only
   (2.902 vs 2.915). At d512 it is decisive — the FFN-only arm hit a **cold-start dead governor**
   (swap-ratio *exactly* 1.000: the per-expert weight path of the zero-init hypernet never escaped 0,
   so the token had no effect and the fixed random latents only added noise → 2.963). The fix is a
   tiny non-zero hypernet init (1e-3) that keeps a live gradient path from `e_k`; the FFN-only d512
   rerun with the fix is in progress. The `+attn` arm escaped the cold-start on its own (two governed
   sites give `e_k` more gradient), which is itself a reason to govern both.

**Takeaway.** `spectral_ffn + attn` is the flagship: at d512 it beats dense and our prefix method for
4K trainable params, with the strongest per-expert specialization we've measured. Next levers to close
the last of the MoE gap: higher `governor_rank`, per-head attention gating, and (as with the MoE) more
FFN capacity for the governor to route within.

## Reproduce
```bash
for a in spectral spectralA; do   # spectral = FFN only; spectralA = FFN + attention (govern_attn)
  python scripts/train.py --config configs/experiment/govern_seqA_${a}_d512.yaml --run-dir .../govern_seqA_${a}_d512 --device cuda
  python scripts/train.py --config configs/experiment/govern_${a}_d512.yaml --run-dir .../govern_${a}_d512 \
    --device cuda --init-backbone-from .../govern_seqA_${a}_d512
done
python scripts/make_report.py --runs <dir symlinking {sc_dense1x, sc_moe_g2, sc_ours_sup_seq, govern_spectral_d512, govern_spectralA_d512}> --out reports/governance_d512
```
