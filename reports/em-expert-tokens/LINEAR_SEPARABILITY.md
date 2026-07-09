# Linear separability of the persona space — alternation vs frozen vs SFT

Does the expert token carve the model's representations into a clean, **linearly separable** per-persona
space — and does the training scheme (EM alternation vs frozen-random tokens vs joint SFT) change that?

## Test

Train the three schemes on the 8-persona set (Qwen2.5-3B): **SFT** (`--phase full`), **frozen-random**
(`--phase backbone`, tokens never trained), **EM** (backbone → tokens). For each held-out (question, persona)
take the last-layer hidden state at the final prompt position **with the persona token minus the same with a
generic `assistant` marker** — the *token-induced representation shift* Δ (isolates the token's effect from
the question content). Then measure separability of the Δ's by persona:

- **Linear probe** — logistic regression trained on 70% of held-out questions, tested on the other 30%
  (8-way persona accuracy; chance = 12.5%).
- **Fisher ratio** (between/within scatter) and **silhouette** — margin/tightness of the clusters.
- **2-D LDA projection** — for visualisation.

(`linear_separability.py`; figure `make_separability_fig.py`.)

## Result

![linear separability](figs/separability.png)

| scheme | probe accuracy (chance 12.5%) | Fisher ratio | silhouette |
|---|---|---|---|
| SFT (joint) | **100%** | 1.60 | 0.328 |
| frozen-random | **100%** | 1.66 | 0.339 |
| EM (alternation) | **100%** | 1.46 | 0.258 |

## Findings

1. **The expert token induces a *perfectly linearly separable* persona space.** A held-out linear probe
   recovers which of 8 personas from the token-induced shift Δ at **100%** (vs 12.5% chance) — the personas
   occupy cleanly separated regions (visible as 8 tight clusters in the LDA projection). This holds because
   the token is a distinct control signal the backbone propagates into a persona-specific representation
   shift. **The "expert token → linearly separable expert space" claim is locked in.**
2. **On balanced data the scheme doesn't change it.** All three reach 100% probe accuracy; by the finer margin
   metrics **frozen ≈ SFT ≥ EM** — the alternation does *not* sharpen separability here. This is the same
   throughline as everywhere else on balanced/ample data: **the *token* creates the separable space; the
   two-phase *scheme* adds little.** (EM's advantage is elsewhere — the scarce-data tail, see
   [EM_VS_SFT.md](EM_VS_SFT.md) — and in the *embedding* geometry, where EM keeps the token vectors ~10× more
   distinct, see [COLLAPSE_RESULTS.md](COLLAPSE_RESULTS.md); that distinctness of the *inputs* is separate from
   the *representation-shift* separability measured here, which saturates for all schemes.)

## Bottom line

The expert token is, quite literally, a **set of linear "persona axes"** in representation space — perfectly
separable and consistent across unseen questions. That separability is a property of *conditioning on the
token*, not of the alternation; on balanced data SFT, frozen-random, and EM all achieve it equally. It is one
more instance of the project's core lesson: **the token does the work; the EM scheme earns its keep only in
the data-starved regime.**
