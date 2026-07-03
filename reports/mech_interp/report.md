# Mechanistic interp — govern_spectralA_d512

How the EM expert token changes the network, measured as *expert-on vs governor-off* (the
only difference is the conditioning). Figures 1–3.

## Findings (TL;DR)

1. **The expert acts more deeply as you go up the stack** — the residual shift grows monotonically
   with depth (layer 1 **0.15** → layer 8 **0.42**). The per-layer gate compounds in the residual
   stream: the token lightly nudges early features and increasingly *reshapes* late ones, where
   representations are most abstract/domain-specific.
2. **Each domain governs a *distinct* subspace, increasingly so with depth.** The cross-expert gate
   cosine is ~0 early and goes **negative** deep (layer 8 **−0.10**) — domains don't just use
   *different* FFN directions, deep down they use *anti-correlated* ones. The specialization is
   geometric: the experts partition the FFN's principal directions, and the partition sharpens where
   it matters (late layers).
3. **The token makes domains perfectly linearly separable in latent space** — a linear domain probe
   on the pooled last hidden state goes **0.938 (dense) → 1.000 (with token)**. The conditioning
   cleanly *organizes* the representation by domain (PCA: math/legal split off completely; the
   web-text domains tighten). This is the mechanism behind the 100% best-is-self routing and the
   swap-ratio — the token installs a clean per-domain geometry the frozen backbone reads off.

## 1. Where the expert acts (per-layer residual shift)
Relative L2 change in the residual stream, per layer (`1_activation_shift.png`):

| layer | rel. shift |
|---|---|
| 1 | 0.147 |
| 2 | 0.190 |
| 3 | 0.251 |
| 4 | 0.291 |
| 5 | 0.339 |
| 6 | 0.367 |
| 7 | 0.405 |
| 8 | 0.420 |

## 2. What subspace each expert governs (`2_gate_signatures.png`)
Each expert's per-layer gate over the r governed directions (>1 amplify, <1 suppress).
Cross-expert gate cosine (0 = distinct subspaces per domain, 1 = identical):

| layer | cross-expert cosine |
|---|---|
| 1 | 0.046 |
| 2 | 0.070 |
| 3 | -0.037 |
| 4 | -0.075 |
| 5 | -0.105 |
| 6 | -0.107 |
| 7 | -0.122 |
| 8 | -0.104 |

## 3. Latent-space domain separation (`3_latent_separation.png`)
- domain linear-probe accuracy **dense 0.938 → with-expert 1.000** (8 domains).
- Higher with the token ⇒ the expert makes domains more linearly separable in the latent space.
