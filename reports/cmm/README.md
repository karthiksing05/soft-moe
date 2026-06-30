# Core experiment — compute-matched MoE validation & our training regimes

**Run date:** 2026-06-30 · **Cluster:** Raven (A100, `mpib_gpu`) · **Corpus:** `dev_bytes`
(5 domains, byte-level) · **Training:** from scratch · **Tokens per expert:** **T = 1** · **Seeds:** 1

This is the repo's **core experiment**. It (1) validates MoE's advantage with a faithful,
reproducible recipe and (2) tests whether our EM-discovered expert tokens share it, under our
different training regimes. Artifacts: [`main_table.md`](main_table.md), [`results.csv`](results.csv),
[`cross_routing_cmm.md`](cross_routing_cmm.md),
[`domain_analysis_dev_bytes.md`](domain_analysis_dev_bytes.md), [`figures/`](figures/).
The pretrained-backbone pilot is the secondary result in [`../m7-raven/`](../m7-raven/).

## TL;DR

- **MoE's advantage is validated.** At matched active compute, an oracle-routed MoE reaches
  macro-ppl **3.56 vs the small dense model's 3.74**, closing **~39%** of the gap to a 5×-wider
  dense model (3.29) — most of the big-model quality at the small-model per-token cost. Learned
  routing captures ~none from scratch (the router is the bottleneck).
- **Our expert tokens specialize but don't add capacity.** With **one** token per expert (the
  thesis-faithful setting) on a capacity-limited from-scratch backbone, **neither of our regimes
  beats the dense baseline** on perplexity. The experts still route correctly (80–100%
  best-is-self), but conditioning a fixed-capacity FFN can't substitute for MoE's extra routed
  parameters.
- **Sequential ≥ alternating, and far cheaper.** Our *sequential* regime (train the LLM, then EM
  the tokens on the frozen base) ties the dense base at **68K trainable params**, beating the
  *alternating* regime (3.77 at 2.66M trainable). If you're going to add expert tokens, do it as a
  cheap finetuning layer on a fully-trained backbone — not jointly.

## 1. The recipe

The canonical multi-domain mixture LM test (Jacobs et al. 1991 → DEMix, Gururangan et al. 2021 →
c-BTM, 2023): train *from scratch*, byte-level (vocab 257, no pretrained-knowledge confound), on a
mixture of 5 distinct domains (wiki / news / reviews / arxiv / pubmed), and compare **at matched
active compute**. Same attention everywhere (d=256, 6 layers, 8 heads); **only the FFN varies**, so
the comparison is purely FFN capacity vs routing. Reproducible by anyone (public HF data, tiny
models). Domain separability of the byte corpus: NMI 0.82 / purity 0.93
([`domain_analysis_dev_bytes.md`](domain_analysis_dev_bytes.md)).

### Compute-matched family (verified)

| model | FFN | total | **active/token** | role |
|---|---|---|---|---|
| Dense-1× | width 256 | 2.59M | 2.59M | capacity-limited baseline / base for sequential |
| Dense-5× | width 1280 | 5.74M | 5.74M | capacity ceiling (5× compute) |
| Hard-MoE | 5×256, top-1 | 5.75M | **2.60M** | =Dense-5× capacity at =Dense-1× compute |
| ours | width 256 + **1** expert token | 2.66M | 2.66M | our method at Dense-1× compute |

## 2. Our three training regimes

An "expert" is **one learned soft-prompt vector** (`tokens_per_expert: 1`), prepended to the
sequence (the persona-token mechanism of [`../../papers/master_thesis_stream_a.pdf`](../../papers/master_thesis_stream_a.pdf)).
The regimes differ in *how the backbone and tokens are trained*:

| regime | backbone | tokens | trainable params | config |
|---|---|---|---|---|
| **frozen** | fixed (pretrained) | EM | tokens only | `em_soft.yaml` — used in the pretrained pilot, not from scratch (a random frozen backbone can't model anything) |
| **alternating** | trained, in blocks | EM, in blocks | backbone + tokens (2.66M) | `em_alternate.yaml` — the thesis's Phase A⇄B *repeated* |
| **sequential** | trained once on all data, then **frozen** | EM on the frozen base | tokens only (**68K**) | `token_em.yaml` + `--init-backbone-from` — Phase A fully → Phase B fully; a finetuning add-on |

The **sequential** regime is the finetuning-analog: it reuses the fully-trained **Dense-1×**
backbone (loaded frozen) and only fits the expert tokens on top — exactly "take a trained LLM, add
expert tokens via EM."

## 3. Results (macro-perplexity, byte-level, ↓)

From [`main_table.md`](main_table.md). "%gap" = `(3.738 − macro)/(3.738 − 3.288)` — fraction of the
way from Dense-1× to the 5×-capacity ceiling, at (near) Dense-1× compute.

| model | macro-ppl ↓ | **total-trainable** | active/token | total | %gap closed |
|---|---|---|---|---|---|
| Dense-1× | 3.738 | 2.59M | 2.59M | 2.59M | 0% |
| **Dense-5×** (ceiling) | **3.288** | 5.74M | 5.74M | 5.74M | 100% |
| **Hard-MoE — oracle routing** | **3.561** | 5.75M | **2.60M** | 5.75M | **~39%** |
| Hard-MoE — learned routing | 3.766 | 5.75M | 2.60M | 5.75M | ~0% |
| ours — **alternating** | 3.771 | 2.66M | 2.66M | 2.66M | ~0% |
| ours — **sequential** | 3.742 | **68K** | 2.66M | 2.66M | ~0% |

### MoE is validated
The **oracle MoE closes ~39% of the capacity gap at Dense-1× compute** — the textbook result:
routing lets each domain use its own FFN expert instead of competing for shared capacity, so you
get much of a 5×-wider model's quality for the small model's per-token cost. **Learned** top-1
routing captures ~none from scratch in 15k steps — a faithful, well-known finding that *the router,
not the capacity, is the hard part*.

### Our tokens specialize but don't add capacity (T = 1)
With a single token per expert, **neither regime beats Dense-1×** (3.74–3.77 vs 3.738). The
mechanism still works — cross-routing ([`cross_routing_cmm.md`](cross_routing_cmm.md)) shows the
matched expert is the best one for 100% (alternating) / 80% (sequential) of domains — but the
per-domain perplexity differences are tiny (1.01–1.09× through the wrong expert). **Conditioning a
fixed-capacity FFN with one prompt vector cannot substitute for MoE's extra routed parameters when
the backbone itself is the bottleneck.**

### Sequential is the better way to use our tokens
`ours_seq` (3.742) **ties the dense base at 68K trainable params** and beats `ours_alt` (3.771,
2.66M trainable). Two reasons: (i) it reuses a *fully*-trained backbone (15k full updates) instead
of splitting the budget across phases, and (ii) it never lets the token dynamics perturb the
backbone. **If expert tokens are worth adding, add them as a cheap frozen-base finetuning layer
(sequential), not by co-training (alternating).** Neither, however, buys quality headroom over the
base in this capacity-limited regime.

## 4. Reconciling with the pretrained pilot

In [`../m7-raven/`](../m7-raven/) (pretrained pythia-160m, frozen backbone), expert tokens *did*
help — because there the bottleneck was *specialization/steering of an already-capable model*, not
raw capacity. Here, from scratch, the bottleneck *is* capacity, and only adding routed parameters
(MoE) helps. **The synthesis: our method is a cheap specialization layer for a capable backbone —
best applied sequentially as a finetuning add-on — not a capacity-scaling mechanism like MoE.**

## 5. Caveats
- Small models, 1 seed, 15–20k steps; byte-perplexities (not comparable to the pilot's token-ppl).
- Learned-routing MoE is likely under-trained (its ~0% is a lower bound; a router z-loss / more
  steps would raise it). The capacity result (oracle MoE, Dense-5×) is robust.
- The oracle MoE uses ground-truth domain routing (DEMix upper bound), not a deployable router.
- `T = 1` is the thesis-faithful default; `T > 1` added more conditioning headroom in earlier runs
  (it is the right knob to revisit if a capacity-light specialization gain is the goal).

## 6. Reproduce

```bash
DATA=/ptmp/$USER/soft-moe/data
python scripts/build_data.py --config configs/experiment/cmm_dense1x.yaml --data-root $DATA
for m in cmm_dense1x cmm_dense5x cmm_hardmoe cmm_hardmoe_oracle cmm_ours_alt; do
  python scripts/train.py --config configs/experiment/$m.yaml --data-root $DATA \
    --run-dir /ptmp/$USER/soft-moe/experiments/$m --device cuda
done
# sequential: train the LLM first (dense1x above), then EM tokens on its frozen backbone
python scripts/train.py --config configs/experiment/cmm_ours_seq.yaml --data-root $DATA \
  --run-dir /ptmp/$USER/soft-moe/experiments/cmm_ours_seq --device cuda \
  --init-backbone-from /ptmp/$USER/soft-moe/experiments/cmm_dense1x
python scripts/make_report.py   --runs <symlinks-to-cmm-runs> --out reports/cmm
python scripts/cross_routing.py --runs .../cmm_ours_alt .../cmm_ours_seq --out reports/cmm/cross_routing_cmm.md
```
