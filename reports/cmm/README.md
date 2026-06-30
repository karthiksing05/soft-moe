# Core experiment — compute-matched MoE validation & our training regimes

**Run date:** 2026-06-30 · **Cluster:** Raven (A100, `mpib_gpu`) · **Corpus:** `dev_bytes`
(5 domains, byte-level) · **Training:** from scratch · **Tokens per expert:** **T = 1** · **Seeds:** 1

The repo's **core experiment**: (1) validate MoE's advantage with a faithful, reproducible recipe,
and (2) test whether our EM-discovered expert tokens share it, across the full **2×2 of our
variants** — {supervised, unsupervised} routing × {alternating, sequential} training. Artifacts:
[`main_table.md`](main_table.md), [`results.csv`](results.csv),
[`cross_routing_cmm.md`](cross_routing_cmm.md),
[`domain_analysis_dev_bytes.md`](domain_analysis_dev_bytes.md), [`figures/`](figures/).
Pretrained-backbone pilot: [`../m7-raven/`](../m7-raven/).

## TL;DR

- **MoE is validated.** At matched active compute, an oracle-routed MoE reaches macro-ppl **3.56
  vs the small dense model's 3.74**, closing **~39%** of the gap to a 5×-wider dense model (3.29).
- **Our method *does* work — as supervised + alternating.** `ours_sup_alt` **beats Dense-1×**
  (3.660 vs 3.738), closing **~17%** of the capacity gap by adding **5 token vectors (1280 params)**
  to a co-trained backbone, with strong specialization (1.33× worse through the wrong expert).
- **Two levers, both the same lesson as MoE.** (i) **Routing quality:** supervised ≫ learned
  (just as oracle MoE 3.56 ≫ learned MoE 3.77 — the router is the bottleneck for both). (ii)
  **Backbone co-adaptation:** alternating ≫ sequential — the tokens only help if the backbone is
  *trained to use them*. Bolting tokens onto a frozen from-scratch base (sequential) adds nothing.
- Still below the oracle MoE (39%): conditioning a fixed-capacity FFN is cheaper but weaker than
  adding routed capacity.

## 1. The recipe

Canonical multi-domain mixture LM test (Jacobs et al. 1991 → DEMix 2021 → c-BTM 2023): train *from
scratch*, byte-level (vocab 257), on 5 distinct domains (wiki/news/reviews/arxiv/pubmed), comparing
**at matched active compute**. Same attention everywhere (d=256, 6L, 8H); **only the FFN varies**.
Byte-corpus domain separability: NMI 0.82 / purity 0.93
([`domain_analysis_dev_bytes.md`](domain_analysis_dev_bytes.md)).

| model | FFN | total | **active/token** | role |
|---|---|---|---|---|
| Dense-1× | width 256 | 2.59M | 2.59M | capacity-limited baseline / sequential base |
| Dense-5× | width 1280 | 5.74M | 5.74M | capacity ceiling (5× compute) |
| Hard-MoE | 5×256, top-1 | 5.75M | **2.60M** | =Dense-5× capacity at =Dense-1× compute |
| ours | width 256 + 1 token/expert | 2.59–2.66M | 2.59–2.66M | our method at Dense-1× compute |

## 2. Our variants — a 2×2

An "expert" is **one** learned soft-prompt vector (`tokens_per_expert: 1`), prepended to the
sequence (the persona-token mechanism of
[`../../papers/master_thesis_stream_a.pdf`](../../papers/master_thesis_stream_a.pdf)).

|  | **alternating** (backbone⇄token blocks; co-trained) | **sequential** (train LLM, then EM tokens on frozen base) |
|---|---|---|
| **supervised** (route by ground-truth domain; trains only the 5 token vectors) | `cmm_ours_sup_alt` | `cmm_ours_sup_seq` |
| **unsupervised** (learned soft router; +router head) | `cmm_ours_unsup_alt` | `cmm_ours_unsup_seq` |

## 3. Results (macro-ppl, byte-level, ↓)

From [`main_table.md`](main_table.md). "%gap" = `(3.738 − macro)/(3.738 − 3.288)`.

| model | macro-ppl ↓ | routing-NMI | swap-ratio | **total-trainable** | active/token | %gap |
|---|---|---|---|---|---|---|
| Dense-1× | 3.738 | 0.82* | — | 2.59M | 2.59M | 0% |
| **Dense-5×** (ceiling) | **3.288** | 0.82* | — | 5.74M | 5.74M | 100% |
| **Hard-MoE — oracle** | **3.561** | 0.82* | — | 5.75M | **2.60M** | **~39%** |
| Hard-MoE — learned | 3.766 | 0.82* | — | 5.75M | 2.60M | ~0% |
| **ours — sup + alternating** | **3.660** | 1.00 | **1.26** | 2.59M (+**1.3K**) | 2.59M | **~17%** |
| ours — sup + sequential | 3.738 | 1.00 | 1.00 | **1.3K** | 2.59M | ~0% |
| ours — unsup + alternating | 3.762 | 0.77 | 1.04 | 2.66M (+68K) | 2.66M | ~0% |
| ours — unsup + sequential | 3.742 | 0.74 | 1.01 | 68K | 2.66M | ~0% |

\* dense/MoE have no per-document router; the 0.82 is the k-means clusterer's NMI vs domains
(baseline reference). "+1.3K / +68K" = trainable params *added* over the dense base (the sequential
rows also reuse the separately-trained 2.59M Dense-1× backbone, frozen).

### MoE is validated
Oracle MoE closes **~39%** of the capacity gap at Dense-1× compute. Learned routing captures ~none
from scratch in 15k steps — **the router, not the capacity, is the hard part.**

### Our method works — supervised + alternating (the headline ours result)
`ours_sup_alt` (3.660) **beats Dense-1× (3.738)** — closing ~17% of the gap — by adding **five token
vectors (1280 params)** to a co-trained backbone at Dense-1× compute. Its experts specialize
strongly: cross-routing shows the matched expert is best for **100%** of domains and routing through
the wrong one costs **1.33×** ([`cross_routing_cmm.md`](cross_routing_cmm.md)). So conditioning a
*co-trained* backbone with one well-routed prompt vector per domain genuinely helps — it just
recovers less of the capacity gap than a real MoE (17% vs 39%), because it adds conditioning, not
routed parameters.

### Two levers decide whether ours helps — both echo the MoE result
1. **Routing quality.** supervised (oracle) ≫ unsupervised (learned): 3.660 vs 3.762 (alternating).
   Identical lesson to MoE (oracle 3.561 vs learned 3.766). A weak from-scratch router erases the
   benefit for both.
2. **Backbone co-adaptation.** alternating ≫ sequential *when routing is good*: 3.660 vs 3.738
   (supervised). The tokens only help if the backbone is **trained to use them**; adding tokens to
   a frozen, already-trained from-scratch base (sequential) ties the base (swap ≈ 1.00, no
   specialization gain). **This corrects an earlier note that called sequential "better" — that was
   based only on the weak unsupervised runs, where neither regime helps.**

## 4. The sequential regime, reconciled
Sequential (train LLM, then EM tokens on the frozen base) added **nothing** here even with oracle
routing — a frozen *from-scratch* backbone wasn't trained to be steered by prepended vectors. Yet in
the pretrained pilot ([`../m7-raven/`](../m7-raven/)) a *frozen pretrained* backbone **did** benefit
from tokens. The difference is backbone flexibility: a large pretrained model is steerable
off-the-shelf; a small from-scratch model must co-adapt (alternating) to use the tokens.
**So: sequential token-finetuning works on a capable/pretrained backbone; from scratch you need
alternating co-training.**

## 5. Synthesis
- **MoE (capacity)** is the strongest at matched compute (39%), at +3.17M params.
- **Ours (conditioning)** reaches **17%** at **+1280 params** when routing is supervised and the
  backbone co-adapts — a cheap, real specialization gain, but below MoE because tokens add
  conditioning, not capacity.
- The recurring theme across MoE and ours: **good routing + a backbone able to exploit the experts**
  are both necessary; remove either and the advantage vanishes.

## 6. Caveats
- Small models, 1 seed, 15–20k steps; byte-perplexities. Learned-routing MoE likely under-trained
  (its ~0% is a lower bound). Capacity results (oracle MoE, Dense-5×) are robust.
- Oracle/supervised routing uses ground-truth domains (DEMix-style upper bound), not a deployable
  router — it bounds *expert* quality separately from *router* quality.
- `T = 1` is thesis-faithful; `T > 1` added more conditioning headroom in earlier runs and is the
  knob to revisit for a larger capacity-light specialization gain.

## 7. Reproduce

```bash
DATA=/ptmp/$USER/soft-moe/data
python scripts/build_data.py --config configs/experiment/cmm_dense1x.yaml --data-root $DATA
for m in cmm_dense1x cmm_dense5x cmm_hardmoe cmm_hardmoe_oracle cmm_ours_sup_alt cmm_ours_unsup_alt; do
  python scripts/train.py --config configs/experiment/$m.yaml --data-root $DATA \
    --run-dir /ptmp/$USER/soft-moe/experiments/$m --device cuda
done
for m in cmm_ours_sup_seq cmm_ours_unsup_seq; do          # sequential: load the trained dense1x backbone
  python scripts/train.py --config configs/experiment/$m.yaml --data-root $DATA \
    --run-dir /ptmp/$USER/soft-moe/experiments/$m --device cuda \
    --init-backbone-from /ptmp/$USER/soft-moe/experiments/cmm_dense1x
done
python scripts/make_report.py   --runs <symlinks-to-cmm-runs> --out reports/cmm
python scripts/cross_routing.py --runs .../cmm_ours_*_{alt,seq} --out reports/cmm/cross_routing_cmm.md
```
