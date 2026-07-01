# isoFLOP scaling sweep & ablations

## Scaling curves (macro-ppl ↓, mean±std over seeds)

| size | active | dense | MoE-G1 (coarse) | MoE-G2 (fine) | dense−G2 gap |
|---|---|---|---|---|---|
| d128 | 0.9M | 3.876 | 3.951 | 3.437 | +0.440 |
| d256 | 5.0M | 2.971±0.004 | 3.017±0.012 | 2.692±0.030 | +0.278 |
| d384 | 11.0M | 2.681 | 2.832 | 2.520 | +0.161 |
| d512 | 25.6M | 2.535 | 2.704 | 2.465 | +0.071 |

## Gap trend (does the fine-grained MoE advantage widen with scale? — H1/H2)

- **d128** (0.9M active): dense 3.876 − MoE-G2 3.437 = **+0.440**
- **d256** (5.0M active): dense 2.971 − MoE-G2 2.692 = **+0.278**
- **d384** (11.0M active): dense 2.681 − MoE-G2 2.520 = **+0.161**
- **d512** (25.6M active): dense 2.535 − MoE-G2 2.465 = **+0.071**

## Ablation matrix (d256, 20k steps, single seed)

| variant | macro-ppl ↓ | total params |
|---|---|---|
| moe_g1_topk2 | 2.674 | 27.0M |
| moe_g2_nobal | 2.722 | 27.1M |
| moe_g2_noz | 2.663 | 27.1M |
| moe_g2_shared1 | 2.702 | 28.6M |
| moe_g2_shared2 | 2.694 | 30.2M |
