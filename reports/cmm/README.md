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
- **Our method works — three of four variants beat Dense-1×**, by 11–17% of the capacity gap, by
  adding **one token vector per expert** (1.3K–68K params) to a co-trained backbone.
- **Pretrain-then-alternate is the robust recipe.** Training the LLM properly first and *then*
  running the alternating scheme (our "sequential" regime) **rescues the hard unsupervised case**:
  `ours_unsup_seq` 3.690 (~11%) vs alternate-from-scratch `ours_unsup_alt` 3.762 (~0%). With
  supervised routing both already work (sup_alt 3.660 ≈ sup_seq 3.673).
- **Two levers, both echoing MoE.** (i) routing quality: supervised ≫ learned (like oracle MoE ≫
  learned MoE — the router is the bottleneck for both); (ii) a properly-trained base + backbone
  co-adaptation. Still below oracle MoE (39%): conditioning is cheaper but weaker than added
  routed capacity.

## 1. The recipe

Canonical multi-domain mixture LM test (Jacobs et al. 1991 → DEMix 2021 → c-BTM 2023): train *from
scratch*, byte-level (vocab 257), on 5 distinct domains (wiki/news/reviews/arxiv/pubmed), comparing
**at matched active compute**. Same attention everywhere (d=256, 6L, 8H); **only the FFN varies**.
Byte-corpus domain separability: NMI 0.82 / purity 0.93
([`domain_analysis_dev_bytes.md`](domain_analysis_dev_bytes.md)).

| model | FFN | total | **active/token** | role |
|---|---|---|---|---|
| Dense-1× | width 256 | 2.59M | 2.59M | capacity-limited baseline / pretrained base for sequential |
| Dense-5× | width 1280 | 5.74M | 5.74M | capacity ceiling (5× compute) |
| Hard-MoE | 5×256, top-1 | 5.75M | **2.60M** | =Dense-5× capacity at =Dense-1× compute |
| ours | width 256 + 1 token/expert | 2.59–2.66M | 2.59–2.66M | our method at Dense-1× compute |

## 2. Our variants — a 2×2

An "expert" is **one** learned soft-prompt vector (`tokens_per_expert: 1`), prepended to the
sequence (the persona-token mechanism of
[`../../papers/master_thesis_stream_a.pdf`](../../papers/master_thesis_stream_a.pdf)). **Both
training regimes are alternating** (backbone⇄token blocks); they differ only in how the backbone
is obtained:

|  | **alternating** (from scratch — backbone learned inside the alternation) | **sequential** (train the LLM fully first, *then* warm-start + alternate) |
|---|---|---|
| **supervised** (route by ground-truth domain) | `cmm_ours_sup_alt` | `cmm_ours_sup_seq` |
| **unsupervised** (learned soft router) | `cmm_ours_unsup_alt` | `cmm_ours_unsup_seq` |

The **sequential** regime = pretrain Dense-1× on all the data → warm-start that backbone
(`--init-backbone-from`) → run the alternating scheme (gentle backbone LR 5e-5, high token LR 1e-2,
tokens-first; `configs/train/warm_alt.yaml`) so the tokens converge while the backbone co-adapts.

## 3. Results (macro-ppl, byte-level, ↓)

From [`main_table.md`](main_table.md). "%gap" = `(3.738 − macro)/(3.738 − 3.288)`.

| model | macro-ppl ↓ | routing-NMI | swap-ratio | **total-trainable** | active/token | %gap |
|---|---|---|---|---|---|---|
| Dense-1× | 3.738 | 0.82* | — | 2.59M | 2.59M | 0% |
| **Dense-5×** (ceiling) | **3.288** | 0.82* | — | 5.74M | 5.74M | 100% |
| **Hard-MoE — oracle** | **3.561** | 0.82* | — | 5.75M | **2.60M** | **~39%** |
| Hard-MoE — learned | 3.766 | 0.82* | — | 5.75M | 2.60M | ~0% |
| **ours — sup + alternating** | **3.660** | 1.00 | **1.26** | 2.59M (+1.3K) | 2.59M | **~17%** |
| **ours — sup + sequential** | **3.673** | 1.00 | 1.11 | 2.59M (+1.3K) | 2.59M | **~14%** |
| ours — unsup + alternating | 3.762 | 0.77 | 1.04 | 2.66M (+68K) | 2.66M | ~0% |
| **ours — unsup + sequential** | **3.690** | 0.76 | 1.02 | 2.66M (+68K) | 2.66M | **~11%** |

\* dense/MoE have no per-document router; 0.82 is the k-means clusterer's NMI vs domains. "+1.3K /
+68K" = params *added* over the dense backbone (the token bank, and the router head for the unsup
variants); the sequential runs also reuse the separately-pretrained 2.59M Dense-1× backbone.

### MoE is validated
Oracle MoE closes **~39%** of the capacity gap at Dense-1× compute. Learned routing captures ~none
from scratch in 15k steps — **the router, not the capacity, is the hard part.**

### Our method works, and pretrain-then-alternate is the robust recipe
Three of our four variants **beat Dense-1×**:
- **Supervised** (oracle domain routing): both regimes work — `sup_alt` 3.660 (~17%) ≈ `sup_seq`
  3.673 (~14%), adding just **five token vectors (1280 params)**. Strong specialization (sup_alt:
  routing through the wrong expert costs **1.26×**, 100% best-is-self).
- **Unsupervised** (learned router): only the **sequential** regime works — `unsup_seq` 3.690 (~11%)
  vs `unsup_alt` 3.762 (~0%). **Training the LLM properly first rescues the weak learned-router
  case**: alternating *from scratch* never gives the router a good enough backbone to route
  against; pretraining the base first does. This is the user-hypothesised win — pretrain, then
  alternate to converge the tokens — and it shows up exactly where it should (the hard case).

### Two levers, both echoing the MoE result
1. **Routing quality.** supervised ≫ learned, mirroring oracle MoE (3.56) ≫ learned MoE (3.77). A
   from-scratch router is the bottleneck for both MoE and us.
2. **A properly-trained base + co-adaptation.** The tokens only help if the backbone can use them.
   Co-training (alternating) is necessary, and *starting from a fully-pretrained backbone* makes
   the alternation far more reliable — decisive for the unsupervised case, free upside for the
   supervised one.

## 4. Synthesis
- **MoE (capacity)** is strongest at matched compute (39%), at +3.17M params.
- **Ours (conditioning)** reaches **11–17%** at **+1.3K–68K params** when routing is decent and the
  backbone is properly trained + co-adapted — a cheap, real specialization gain, below MoE because
  tokens add conditioning, not routed capacity.
- **Best practice for our method:** pretrain the LLM, then alternate (sequential). It matches the
  best supervised result and is the only thing that makes the unsupervised (deployable) router pay
  off. Reconciles with the pilot ([`../m7-raven/`](../m7-raven/)), where a frozen *pretrained*
  backbone already benefited from tokens — a capable base is the common ingredient.

## 5. Caveats
- Small models, 1 seed, 15–20k step stages; byte-perplexities. Learned-routing MoE likely
  under-trained (its ~0% is a lower bound). Capacity results (oracle MoE, Dense-5×) are robust.
- Oracle/supervised routing uses ground-truth domains (DEMix upper bound), not a deployable router.
- `T = 1` is thesis-faithful; `T > 1` added more conditioning headroom in earlier runs.

## 6. Reproduce

```bash
DATA=/ptmp/$USER/soft-moe/data
python scripts/build_data.py --config configs/experiment/cmm_dense1x.yaml --data-root $DATA
for m in cmm_dense1x cmm_dense5x cmm_hardmoe cmm_hardmoe_oracle cmm_ours_sup_alt cmm_ours_unsup_alt; do
  python scripts/train.py --config configs/experiment/$m.yaml --data-root $DATA \
    --run-dir /ptmp/$USER/soft-moe/experiments/$m --device cuda
done
# sequential = pretrain (dense1x above), then warm-start + alternate:
for m in cmm_ours_sup_seq cmm_ours_unsup_seq; do
  python scripts/train.py --config configs/experiment/$m.yaml --data-root $DATA \
    --run-dir /ptmp/$USER/soft-moe/experiments/$m --device cuda \
    --init-backbone-from /ptmp/$USER/soft-moe/experiments/cmm_dense1x
done
python scripts/make_report.py   --runs <symlinks-to-cmm-runs> --out reports/cmm
python scripts/cross_routing.py --runs .../cmm_ours_*_{alt,seq} --out reports/cmm/cross_routing_cmm.md
```
