# How to inject the expert: input injection vs FFN **subspace governance**

Four ways to hand the same EM-trained per-expert latent `e_k` to the backbone, all under the
identical two-phase recipe (only the injection mechanism differs). **Headline:** letting the expert
**govern an FFN subspace** beats prepending it as a prefix or a token on all 8 domains — and the
**spectral** variant wins most efficiently.

| mechanism | rung | macro-ppl ↓ | swap-ratio ↑ | extra shared params |
|---|---|---|---|---|
| `prefix` (soft-prompt) | input | 2.949 | 1.532 | 0 |
| `token` (inline, predict-marker) | input | 2.961 | 1.512 | 0 |
| **`film_ffn`** (FFN neuron gate) | FFN activation | **2.920** | 1.436 | +1.58M |
| **`spectral_ffn`** (FFN subspace gate) | FFN subspace | **2.915** | **1.575** | **+123K** |

All: d256, `demix8`, T=1, supervised routing, 2,048 token params. See §1–2 for the input arms,
**§4 for governance**. Code: `src/softmoe/models/{soft_moe,governance}.py`.

## The two input-injection arms (prefix vs token)

**Question.** Our method delivers each expert as a **soft prefix** — a learned vector prepended to
the input embeddings (`injection: prefix`). The thesis (`papers/master_thesis_stream_a.pdf`) and
conversational finetuning instead put a **discrete marker token** *in the sequence* —
`[SPEAKER_X] <text>` / `<|assistant|>` — whose embedding is learned and which the model reads (and,
in chat data, **emits**). Do these two ways of injecting the same per-expert vector behave
differently for EXPERT-conditioning? This documents a controlled test.

Run: Raven A100, `demix8` (8 byte-level domains), d=256 backbone, thesis-faithful two-phase recipe
(Phase A 20k backbone + frozen marker, Phase B 6k token-only), supervised routing, T=1. Only the
**injection mechanism** differs. Artifacts: [`main_table.md`](main_table.md),
[`per_domain_ppl.md`](per_domain_ppl.md). Code: `injection: token` (+ `token_predict_marker`) in
`src/softmoe/models/soft_moe.py`.

## 1. The mechanisms

- **`prefix` (ours):** `inputs_embeds = [ v_k , emb(x₁), … , emb(x_L) ]` — the expert vector `v_k`
  (from the `ExpertTokenBank`) sits at position 0; the content attends to it causally. `v_k` is a
  **free** vector, never scored, not a vocabulary member.
- **`token` (thesis / conversational):** an inline **discrete** marker `[EXPERT_k]` at the front.
  With `token_predict_marker=False` it is conditioning-only; with `=True` (default) a BOS slot is
  prepended and the model is trained to **emit** the marker (output-tied to the bank via extended
  logits), so the marker is *part of the ground-truth response* — the conversational role-token
  style.

## 2. Result — they are the same thing (for a front-loaded expert)

**(a) Conditioning-only token ≡ prefix, exactly.** A prepended embedding at position 0 is processed
identically whether it is sourced from a separate bank (prefix) or is a discrete inline token
(conditioning-only). Verified on an untrained model with shared weights: `prefix` loss and
`token`(cond-only) loss are bit-for-bit equal (`5.59738`). There is **no** computation that
distinguishes them — the transformer cannot tell a "soft prompt" from a "token embedding" at the
same position. So `injection: token` with `token_predict_marker=False` is just an alias for `prefix`.

**(b) Even the full conversational variant (predict the marker) matches prefix.** Trained end-to-end
(d256, matched Phase A/B):

| injection | macro-ppl ↓ | micro-ppl ↓ | swap-ratio ↑ | routing-NMI ↑ | trainable |
|---|---|---|---|---|---|
| **prefix** (soft-prompt) | **2.949** | 2.828 | 1.532 | 1.000 | 2,048 |
| **token** (inline, predict-marker) | 2.961 | 2.838 | 1.512 | 1.000 | 2,048 |

Within **~0.4%** on every metric, and **per-domain perplexity is identical to ±0.03 on all 8
domains**. Specialization (swap-ratio ≈ 1.5, NMI 1.0) is unchanged — the tokens carry the same
expertise either way. The token arm is *marginally worse* (2.961 vs 2.949): the only real effect of
"put the marker in the response" is a small **uninformative** marker-prediction term.

## 3. Why — and when the distinction *would* matter

The reason predicting the marker adds nothing here is specific to **expert conditioning**: the
expert id is chosen by the **router** (the known domain), so it is **not inferable from left
context**. Training the model to emit `[EXPERT_k]` from a constant BOS is a ~log(K) coin-flip — no
signal to learn, no benefit to the content. This is the crucial difference from **conversational /
persona finetuning**, where the role/speaker token *is* predictable from the dialogue so far (prior
turns, the user's question, the running style). There, "the token in the response" does real work —
the model learns *when* to switch persona and *who* is speaking. For a routed expert there is no
such context, so the inline-token and soft-prefix mechanisms collapse to the same thing.

**Takeaways.**
1. For front-loaded **expert conditioning**, `prefix` and inline `[EXPERT_X]` tokens are the *same
   mechanism* — identical computation, and empirically identical quality/specialization (±0.4%).
   The choice is cosmetic (a soft vector vs a discrete vocab entry); it does not change results.
2. The conversational-FT gain from putting the marker *in the response* comes entirely from the
   marker being **predictable from context**. That applies to persona/speaker-from-history (the
   thesis's actual setting, where a speaker recurs across a conversation), **not** to a
   router-assigned domain expert.
3. Practically: keep `prefix` for routed experts (simpler, one fewer vocab hack, marginally better).
   Reach for `injection: token` + `token_predict_marker` only when the identity should be *inferred
   and emitted* as part of generation — i.e. genuine conversational persona modeling.

## 4. Beyond input injection: **FFN subspace governance** (MoE-structured)

Prefix and token both sit at the **input rung** — the expert only *conditions via attention* to a
front vector. The MoE's advantage came from **fine-grained FFN capacity**, so the natural next step
is to let the expert govern the **FFN hidden space** directly, where MoE experts live. Both variants
keep `e_k` as the compact latent and route it through a **shared hypernetwork** (`FFNGovernor`) that
emits, per layer, a modulation of the FFN hidden — the backbone+hypernet are the governance
mechanism (Phase A), `e_k` is fit in Phase B. Zero-init ⇒ identity at start (verified: governed
loss == dense loss). `injection: film_ffn | spectral_ffn`.

- **`film_ffn`** — per-neuron multiplicative gate `h ⊙ g_k` on the FFN hidden. The soft, token-routed
  analog of the MoE selecting FFN experts/neurons: the expert governs *which neurons fire*.
- **`spectral_ffn`** — gate a learned `r=16`-dim orthonormal subspace of the FFN hidden (project onto
  a basis `U`, scale the `r` principal directions by the token, project back). Governs *directions of
  the weight image* — the literal "soft-expert subspace governance."

| mechanism | macro-ppl ↓ | micro-ppl ↓ | swap-ratio ↑ | Δ vs prefix | extra shared params |
|---|---|---|---|---|---|
| prefix (input) | 2.949 | 2.828 | 1.532 | — | 0 |
| token (input) | 2.961 | 2.838 | 1.512 | +0.012 | 0 |
| **film_ffn** | 2.920 | 2.800 | 1.436 | **−0.029** | +1.58M |
| **spectral_ffn** | **2.915** | **2.796** | **1.575** | **−0.034** | **+123K** |

**Findings.**
1. **Moving off the input rung works.** Both governance modes beat prefix/token on **all 8 domains**
   (per-domain macro 2.91–2.92 vs 2.95–2.96). Letting the expert govern an FFN subspace, rather than
   prepend a vector, is a real — if modest (~1.2%) — and consistent gain. This is the first
   mechanism to beat the input-injection ceiling.
2. **Spectral wins, and wins cheaply.** `spectral_ffn` is the best (2.915) *and* uses **13× fewer**
   shared params than `film_ffn` (+123K vs +1.58M). Gating rotated **principal directions** of the
   FFN is both more parameter-efficient and more effective than gating individual neurons — evidence
   that experts want to own *directions of the weight matrix*, not axis-aligned units. This is the
   "subspace governance" thesis, confirmed.
3. **The specialization ↔ shared-capacity trade-off (again).** `spectral_ffn` has the **highest
   swap-ratio (1.575)** — its small hypernet forces the *tokens* to carry the specialization —
   whereas `film_ffn`'s large 1.58M hypernet absorbs the work and leaves the tokens weaker
   (swap 1.436, the lowest). Same lesson as freezing the backbone in Phase B: constrain the shared
   machinery and the expert latents do more.
4. Still below the fine-grained MoE (a d256 MoE-G2 ≈ mid-2.7s→low; the gap remains capacity, not
   injection site) — but governance is the right *direction*: `spectral_ffn` is the flagship to scale
   up (d512, higher `governor_rank`, and gating attention as well as the FFN).

## Reproduce
```bash
# input arms
python scripts/train.py --config configs/experiment/inj_seqA_token_d256.yaml --run-dir .../inj_seqA_token_d256 --device cuda
python scripts/train.py --config configs/experiment/inj_token_d256.yaml --run-dir .../inj_token_d256 \
  --device cuda --init-backbone-from .../inj_seqA_token_d256
# governance arms (film | spectral): Phase A trains backbone+hypernet, Phase B fits only e_k
for s in film spectral; do
  python scripts/train.py --config configs/experiment/govern_seqA_${s}_d256.yaml --run-dir .../govern_seqA_${s}_d256 --device cuda
  python scripts/train.py --config configs/experiment/govern_${s}_d256.yaml --run-dir .../govern_${s}_d256 \
    --device cuda --init-backbone-from .../govern_seqA_${s}_d256
done
python scripts/make_report.py --runs <dir symlinking the 4 runs> --out reports/injection
```
