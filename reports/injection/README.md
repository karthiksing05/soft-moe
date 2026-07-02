# Injecting the expert: soft **prefix** vs inline **token** (conversational-FT style)

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

## Reproduce
```bash
# prefix arm already exists as the isoFLOP-sweep d256 point (sw_seqA_d256 -> sw_ours_d256_s0).
python scripts/train.py --config configs/experiment/inj_seqA_token_d256.yaml --run-dir .../inj_seqA_token_d256 --device cuda
python scripts/train.py --config configs/experiment/inj_token_d256.yaml --run-dir .../inj_token_d256 --device cuda \
  --init-backbone-from .../inj_seqA_token_d256
python scripts/make_report.py --runs <dir symlinking {sw_ours_d256_s0, inj_token_d256}> --out reports/injection
```
