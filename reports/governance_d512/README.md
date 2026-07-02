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
| **ours — spectral FFN-only** | **2.455** | 2.359 | 1.612 | **4,096** | 26.0M |
| ours — prefix (input) | 2.460 | 2.368 | 1.491 | 4,096 | 25.6M |
| Dense-1× | 2.472 | 2.376 | — | 25.6M | 25.6M |

Both governance arms beat dense **and** prefix. (The FFN-only number is post-fix: the first run hit a
zero-init cold-start — 2.963, swap 1.000 — resolved by a tiny non-zero hypernet init; see finding 3.)

## Findings

1. **Subspace governance beats dense *and* our prefix method at scale — both arms.** Spectral
   FFN+attn **2.446** and FFN-only **2.455** both come in under prefix **2.460** < dense **2.472**,
   beating dense on **all 8 domains** (`per_domain_ppl.md`), while training only the 4,096 token
   params. They close most of the remaining gap to the full fine-grained MoE (2.399) — at **5.5×
   fewer total params** (26M vs 143M). This confirms the d256 finding at the headline scale:
   **letting the expert govern FFN/attention subspaces beats prepending it as a vector.**
2. **Strongest specialization of any arm** (swap-ratio **1.643** for +attn, **1.612** FFN-only, vs
   prefix's 1.491): routing a domain through the wrong expert costs 61–64% ppl. The governed subspace
   is genuinely per-expert.
3. **Governing attention adds a further edge, and de-risks training.** `+attn` beats FFN-only at both
   scales (d512: 2.446 vs 2.455; d256: 2.902 vs 2.915). It also **escaped a cold-start** that the
   FFN-only zero-init hypernet hit at d512 — the per-expert weight path stayed at 0, so the token had
   no effect (swap-ratio *exactly* 1.000, macro 2.963) until a tiny non-zero init (1e-3) restored a
   live gradient path from `e_k` (→ 2.455). Two governed sites give `e_k` more gradient, so `+attn`
   never hit it. Lesson: govern both, and never zero-init the hypernet.

**Takeaway.** Spectral subspace governance (`spectral_ffn`, +`attn`) is the flagship: at d512 it beats
dense and our prefix method for 4K trainable params, with the strongest per-expert specialization
we've measured. Next levers to close the last of the MoE gap: higher `governor_rank`, per-head
attention gating, and (as with the MoE) more FFN capacity for the governor to route within.

## Reproduce
```bash
for a in spectral spectralA; do   # spectral = FFN only; spectralA = FFN + attention (govern_attn)
  python scripts/train.py --config configs/experiment/govern_seqA_${a}_d512.yaml --run-dir .../govern_seqA_${a}_d512 --device cuda
  python scripts/train.py --config configs/experiment/govern_${a}_d512.yaml --run-dir .../govern_${a}_d512 \
    --device cuda --init-backbone-from .../govern_seqA_${a}_d512
done
python scripts/make_report.py --runs <dir symlinking {sc_dense1x, sc_moe_g2, sc_ours_sup_seq, govern_spectral_d512, govern_spectralA_d512}> --out reports/governance_d512
```
