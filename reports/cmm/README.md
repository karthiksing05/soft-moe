# Compute-matched MoE-validation test — does our method share MoE's strength?

**Run date:** 2026-06-30 · **Cluster:** Raven (A100, `mpib_gpu`) · **Corpus:** `dev_bytes`
(5 domains, byte-level) · **Training:** from scratch · **Seeds:** 1

This run answers two questions the earlier pretrained-backbone pilot ([`../m7-raven/`](../m7-raven/))
couldn't: **(1) does a faithful, reproducible recipe actually validate MoE's advantage, and
(2) does our EM-style expert-token method share that advantage?** Artifacts here:
[`main_table.md`](main_table.md), [`results.csv`](results.csv),
[`cross_routing_cmm.md`](cross_routing_cmm.md),
[`domain_analysis_dev_bytes.md`](domain_analysis_dev_bytes.md), [`figures/`](figures/).

## TL;DR

- **MoE's advantage is validated.** At *matched active compute*, an oracle-routed MoE reaches
  macro-ppl **3.55 vs the small dense model's 3.72**, closing **~46%** of the gap up to a
  5×-wider dense model (3.36) — i.e. it buys most of the big-model quality at the small-model
  compute. Learned top-1 routing recovers far less (~12%), showing the router, not the capacity,
  is the hard part.
- **Our expert tokens partially share it — but for a different reason.** At matched
  backbone-training budget, ours marginally beats its compute-equal dense baseline (3.69 vs 3.72)
  and its experts cleanly specialize (100% best-is-self), but it closes only **~8%** of the
  capacity gap. **Expert tokens add input-dependent *conditioning*, not *capacity*** — so in a
  capacity-starved regime they can't match a real MoE, which adds routed parameters.

## 1. The recipe (and why it's faithful)

The canonical, reproducible way to validate MoE is the **multi-domain / mixture-of-distributions
language-modeling test**, from the lineage:
**Adaptive Mixtures of Local Experts** (Jacobs, Jordan, Nowlan, Hinton, 1991) → **DEMix**
(Gururangan et al., 2021) → **Branch-Train-Merge / c-BTM** (Li et al., 2022; Gururangan et al.,
2023). Train *from scratch* on a mixture of distinct domains and compare **at matched active
compute**: a small dense model must share its capacity across all domains (interference), whereas
an MoE routes each input to a domain-specialized parameter subset.

We instantiate it nanoGPT-style: **byte-level** LM (vocab 257, no pretrained knowledge to
confound the capacity argument) on the 5 `dev` domains (wiki / news / reviews / arxiv / pubmed),
60,264 blocks of 512 bytes. Same attention everywhere (d=256, 6 layers, 8 heads); **only the FFN
varies**, so the comparison is purely about FFN capacity vs routing. Reproducible by anyone — the
data streams from public HF datasets and the models are tiny.

### Compute-matched model family (verified)

| model | FFN | total params | **active/token** | role |
|---|---|---|---|---|
| Dense-1× | width 256 | 2.57M | 2.57M | capacity-limited baseline |
| Dense-5× | width 1280 | 5.72M | 5.72M | capacity ceiling (5× compute) |
| Hard-MoE | 5 experts × 256, top-1 | 5.74M | **2.58M** | =Dense-5× capacity at =Dense-1× compute |
| ours (alt) | width 256 + expert tokens | 2.66M | 2.66M | our method at Dense-1× compute |

`active/token` is what one token actually flows through; for top-1 MoE the other 4 experts don't
run, so its active compute equals Dense-1× while its capacity equals Dense-5×. **That equality is
the whole point** — any quality the MoE gains over Dense-1× is free on compute.

## 2. Results (macro-perplexity, byte-level, lower is better)

From [`main_table.md`](main_table.md):

| model | macro-ppl ↓ | active/token | total params | % of capacity gap closed |
|---|---|---|---|---|
| Dense-1× | 3.716 | 2.57M | 2.57M | 0% (baseline) |
| **Dense-5×** (ceiling) | **3.358** | 5.72M | 5.72M | 100% |
| **Hard-MoE — oracle routing** | **3.553** | **2.58M** | 5.74M | **~46%** |
| Hard-MoE — learned routing | 3.673 | 2.58M | 5.74M | ~12% |
| ours (unsup, alt, 15k) | 3.898 | 2.66M | 2.66M | — (under-trained, see §3) |
| **ours (unsup, alt, 20k = matched)** | **3.686** | 2.66M | 2.66M | **~8%** |
| ours (sup, alt, 15k) | 3.849 | 2.59M | 2.59M | — (under-trained) |

"% of capacity gap closed" = `(3.716 − macro) / (3.716 − 3.358)` — how far a method moves from the
small-dense baseline toward the 5×-capacity ceiling, at (near) Dense-1× compute.

### MoE is validated
The **oracle MoE closes ~46% of the capacity gap at Dense-1× compute** — the textbook result:
most of a 5×-wider model's quality for the small model's per-token cost, because routing lets each
domain use its own FFN expert instead of competing for shared capacity. The **learned-vs-oracle
gap** (3.673 vs 3.553) is itself a faithful, well-known finding: top-1 routing is hard to learn
from scratch in a short run, so a learned router captures only ~12% — **the bottleneck is the
router, not the capacity**.

### Does ours share it? Partly, and for a different reason
At matched compute ours **marginally beats its dense baseline** (3.686 vs 3.716) and its experts
genuinely specialize — but it closes only **~8%** of the capacity gap, ~6× less than the oracle
MoE. The reason is mechanistic: **expert tokens add input-dependent conditioning, not FFN
capacity.** When the backbone itself is the bottleneck (small, from-scratch), conditioning a
fixed-capacity FFN can't substitute for adding routed parameters. This is the honest answer to
"do our alternating expert tokens have MoE's strength?": **they specialize, but they do not
deliver MoE's capacity-scaling benefit in a capacity-limited regime.**

## 3. Ruling out the under-training confound

The alternating M-step trains the backbone only ~75% of steps (the rest fine-tunes tokens), so at
a fixed 15k steps ours's backbone is under-trained vs a pure-dense run — and indeed the 15k ours
runs (3.85–3.90) were *worse* than Dense-1×. Re-running ours at **20k steps (≈15k backbone updates,
matched to dense)** recovers it to **3.686**, slightly *beating* Dense-1×. So the earlier deficit
was mostly under-training; the corrected, fair comparison is the 8%-gap number above.

## 4. The experts do specialize (cross-routing)

From [`cross_routing_cmm.md`](cross_routing_cmm.md): for every ours variant, **each domain is
modelled best by its own expert (100% best-is-self)**, and routing through the wrong expert raises
ppl (supervised 1.35×, unsupervised 1.09×). So the expert tokens carry real, domain-aligned
expertise — the specialization mechanism works. It simply yields a *small* macro-ppl gain here
because conditioning a capacity-limited FFN has limited headroom, not because the experts are
inert.

## 5. Reconciling with the pretrained pilot

In [`../m7-raven/`](../m7-raven/) (pretrained pythia-160m), expert tokens *did* help: near-dense
quality (28.5 ppl) and strong specialization (swap 1.18–1.22). The contrast is the point:

- **Pretrained / capacious backbone:** the bottleneck is *specialization/steering* of an
  already-capable model → expert-token conditioning helps, cheaply (15K–610K params).
- **From-scratch / capacity-limited backbone:** the bottleneck is *capacity* → only adding routed
  parameters (MoE) helps materially; conditioning gives a small edge.

**Our method is a cheap specialization layer, not a capacity-scaling mechanism.** It complements a
strong backbone rather than replacing MoE's conditional compute.

## 6. Caveats

- Small models, 1 seed, 15–20k steps. The learned-routing MoE is likely under-trained (its 12% is
  a lower bound; more steps / a router z-loss would raise it). The *capacity* result (oracle MoE,
  Dense-5×) is robust.
- The oracle MoE uses ground-truth domain routing (DEMix-style) — an upper bound that isolates
  capacity from router-learnability, not a deployable system.
- Byte-level + small scale: absolute ppls are byte-perplexities, not comparable to the pretrained
  pilot's token-perplexities.

## 7. Reproduce

```bash
DATA=/ptmp/$USER/soft-moe/data
python scripts/build_data.py --config configs/experiment/cmm_ours_unsup_alt.yaml --data-root $DATA
for m in cmm_dense1x cmm_dense5x cmm_hardmoe cmm_hardmoe_oracle \
         cmm_ours_sup_alt cmm_ours_unsup_alt cmm_ours_unsup_alt20k; do
  python scripts/train.py --config configs/experiment/$m.yaml --data-root $DATA \
    --run-dir /ptmp/$USER/soft-moe/experiments/$m --device cuda
done
python scripts/make_report.py    --runs <symlinks-to-cmm-runs> --out reports/cmm
python scripts/cross_routing.py  --runs .../cmm_ours_*_alt* --out reports/cmm/cross_routing_cmm.md
```
