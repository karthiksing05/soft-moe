# Mechanistic interp — em_gov

How the EM expert token changes the network, measured as *expert-on vs governor-off* (the
only difference is the conditioning). Figures 1–3.

## 1. Where the expert acts (per-layer residual shift)
Relative L2 change in the residual stream, per layer (`1_activation_shift.png`):

| layer | rel. shift |
|---|---|
| 1 | 0.043 |
| 2 | 0.425 |
| 3 | 0.618 |
| 4 | 0.778 |
| 5 | 0.869 |
| 6 | 0.802 |

## 2. What subspace each expert governs (`2_gate_signatures.png`)
Each expert's per-layer gate over the r governed directions (>1 amplify, <1 suppress).
Cross-expert gate cosine (0 = distinct subspaces per domain, 1 = identical):

| layer | cross-expert cosine |
|---|---|
| 1 | 0.274 |
| 2 | 0.532 |
| 3 | 0.457 |
| 4 | 0.200 |
| 5 | 0.003 |
| 6 | -0.033 |

## 3. Latent-space domain separation (`3_latent_separation.png`)
- domain linear-probe accuracy **dense 0.875 → with-expert 1.000** (8 domains).
- Higher with the token ⇒ the expert makes domains more linearly separable in the latent space.
