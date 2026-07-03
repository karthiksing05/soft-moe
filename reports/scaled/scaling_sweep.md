# isoFLOP scaling sweep & ablations

## Scaling curves (macro-ppl ↓, mean±std over seeds)

| size | active | dense | MoE-G1 | MoE-G2 | ours (prefix) | gov (spectral) | dense−G2 | dense−gov |
|---|---|---|---|---|---|---|---|---|
| d128 | 0.9M | 3.876 | 3.951 | 3.437 | 3.751 | 3.803 | +0.440 | +0.073 |
| d256 | 5.0M | 2.971±0.004 | 3.017±0.012 | 2.692±0.030 | 2.949 | 2.903 | +0.278 | +0.068 |
| d384 | 11.0M | 2.681 | 2.832 | 2.520 | 2.641 | 2.648 | +0.161 | +0.033 |
| d512 | 25.6M | 2.535 | 2.704 | 2.465 | 2.527 | 2.520 | +0.071 | +0.015 |

## Gap-over-dense trend (does the advantage widen with scale? — H1/H2)

- **d128** (0.9M active): dense 3.876 · MoE-G2 3.437 (gap **+0.440**) · ours 3.751 (gap **+0.125**)
- **d256** (5.0M active): dense 2.971 · MoE-G2 2.692 (gap **+0.278**) · ours 2.949 (gap **+0.022**)
- **d384** (11.0M active): dense 2.681 · MoE-G2 2.520 (gap **+0.161**) · ours 2.641 (gap **+0.040**)
- **d512** (25.6M active): dense 2.535 · MoE-G2 2.465 (gap **+0.071**) · ours 2.527 (gap **+0.008**)

## Ablation matrix (d256, 20k steps, single seed)

| variant | macro-ppl ↓ | total params |
|---|---|---|
| moe_g1_topk2 | 2.674 | 27.0M |
| moe_g2_nobal | 2.722 | 27.1M |
| moe_g2_noz | 2.663 | 27.1M |
| moe_g2_shared1 | 2.702 | 28.6M |
| moe_g2_shared2 | 2.694 | 30.2M |
| moe_g4 | 2.748 | 27.1M |
