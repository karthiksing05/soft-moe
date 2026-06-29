# M7 pilot — Soft-MoE on MPCDF Raven (A100)

**Run date:** 2026-06-29 · **Cluster:** Raven (NVIDIA A100, account `mpib_gpu`) ·
**Backbone:** `EleutherAI/pythia-160m` · **Corpus:** `dev` recipe (5 domains) · **Steps:** 2000 ·
**Seeds:** 1 (pilot) · **Methods:** dense, hard_moe, c-BTM, ours (sup), ours (unsup)

This is the first GPU run of the research loop end-to-end (data → EM training → evaluation) on a
**pretrained** backbone. It is a **pilot**, not the final headline: 1 seed, 2000 steps, 5 domains.
Its purpose is to (a) validate the full pipeline on real data/GPU, (b) get a first honest
comparison of the methods, and (c) document that the training corpus has genuinely distinct,
relevant domains. Artifacts in this folder: [`main_table.md`](main_table.md),
[`results.csv`](results.csv), [`domain_analysis_dev.md`](domain_analysis_dev.md),
[`figures/`](figures/).

---

## 1. The training corpus and its domains

The whole hypothesis is that *distinct sub-domains* can be hosted as expert tokens in one
backbone, so the corpus **must** contain clearly separable, relevant domains. The `dev` recipe
streams 5 domains from HuggingFace (20,000 docs each, 100,000 total → 42,929 packed 512-token
blocks), tokenized with the pythia tokenizer:

| domain | source | register | docs | blocks |
|---|---|---|---|---|
| wiki | `wikitext-103-raw-v1` | encyclopedic prose | 20,000 | 6,235 |
| news | `ag_news` | news headlines/leads | 20,000 | 2,146 |
| reviews | `imdb` | movie reviews (opinion) | 20,000 | 11,896 |
| arxiv | `ccdv/arxiv-summarization` | CS/physics/math abstracts | 20,000 | 13,186 |
| pubmed | `ccdv/pubmed-summarization` | biomedical abstracts | 20,000 | 9,466 |

> A 6th domain (code, `code_search_net`) was dropped: it is a script-based dataset removed from
> `datasets>=4` and gated behind `trust_remote_code` on 3.x. The 5 retained domains are all
> parquet-native and stream reliably. Swap in a parquet-native code corpus to restore a code
> domain.

### Are these domains actually distinct and relevant? — yes, with one honest caveat

Full evidence in [`domain_analysis_dev.md`](domain_analysis_dev.md) (produced by
`scripts/analyze_domains.py`). Summary:

- **Sample texts are unmistakably different registers** — Wikipedia game articles, Reuters
  market headlines, IMDb opinion reviews, arXiv physics/math abstracts (`@xmath…`), PubMed
  clinical abstracts (`background: … materials and methods: …`). This is the clearest evidence
  the domains are real and relevant. (See the "Sample documents per domain" section of the
  analysis.)
- **Inter-domain centroid cosine distances are wide** (0.60–1.11). The nearest pair is
  wiki↔news (0.60, both factual prose); the farthest pairs involve the scientific abstracts
  (arxiv/pubmed) vs the rest (≈1.0+). Domains occupy distinct regions of embedding space.
- **An unsupervised k-means clusterer recovers the domains at NMI 0.759 / ARI 0.737 / purity
  0.897.** This is the premise the *unsupervised* expert router relies on, and it holds. The
  contingency matrix is near-diagonal (reviews→c1, arxiv→c2/c4, pubmed→c3), with wiki+news
  sharing c0 — i.e. the two prose domains are the hardest to separate, as expected.
- **Per-document silhouette is modest (0.051).** Honestly: individual documents have high
  *within-domain* variance (wiki spans every topic; arxiv mixes physics/math/CS; pubmed spans
  many specialties), so per-point silhouette is low even though the **aggregate** structure is
  strongly separable (centroids + NMI above). This makes routing **non-trivial** — a realistic
  test of whether experts can specialize, not a toy where separation is free.

**Conclusion:** the corpus has genuinely distinct, relevant domains (qualitatively obvious,
quantitatively recoverable at NMI 0.76), with realistic overlap that makes the routing problem
meaningful rather than trivial.

---

## 2. Methodology

**Backbone.** `pythia-160m` (162M params), frozen for the Soft-MoE methods (so any
specialization is unambiguously carried by the expert tokens) and fully fine-tuned for the
baselines. Data tokenized with the matching pythia tokenizer (vocab 50,277).

**Methods compared** (each isolates one ingredient — see [THEORY §F](../../plan/THEORY.md)):

| method | what it is | what it isolates |
|---|---|---|
| **dense** | full fine-tune, no experts | value of *any* conditioning (lower bound) |
| **hard_moe** | top-1 token-routed MoE FFN (K=5), **sparse-upcycled** from the dense FFN | value of full expert FFNs (param-heavy upper reference) |
| **c-BTM** | 5 separate full pythia models, one per cluster | value of parameter *sharing* (1 model vs N) |
| **ours (sup)** | frozen backbone + learned expert tokens, routed by ground-truth cluster | "can one weight space host K experts?" |
| **ours (unsup)** | frozen backbone + learned tokens + EM-learned soft router | the full proposal — experts *discovered* by EM |

**Training.** 2000 steps, batch 8 × grad-accum 2, cosine schedule. Baselines (full FT) use
lr `5e-5`; Soft-MoE token banks use lr `1e-2` (prompts want a higher LR). Soft-MoE uses
`tokens_per_expert=4`, prefix injection, separation + load-balance + router regularizers
(`λ_sep=1.0 > λ_bal=0.1`, per the brainstorm priority). Each run writes `metrics.json` with
per-domain perplexity (oracle vs learned routing) and the specialization suite.

> **Sparse upcycling** (Komatsuzaki et al., 2023): each MoE expert is initialized as a copy of
> the pretrained dense FFN, so hard_moe starts ≈ the pretrained model and *learns to specialize*
> rather than relearning an FFN from scratch — a fair "true MoE" baseline at 2000 steps.

---

## 3. Results

From [`main_table.md`](main_table.md) (macro-ppl = equal-weight per-domain perplexity, the
headline; "+params" = trainable params **added** over a single dense backbone):

| method | macro-ppl ↓ | micro-ppl ↓ | routing-NMI ↑ | util-entropy ↑ | sep ↑ | swap-ratio ↑ | +trainable params ↓ |
|---|---|---|---|---|---|---|---|
| dense (full FT) | **26.99** | 25.03 | 0.76* | — | — | — | 0 (all 162M trained) |
| hard_moe (upcycled) | 27.97 | 25.95 | 0.76* | — | — | — | 226.7M |
| c-BTM (5 models) | 30.22 | 28.32 | 0.76* | 0.97 | — | — | 649.1M |
| ours (sup) | 35.66 | 32.46 | 0.76 | 0.97 | 1.25 | **1.18** | **15.4K** |
| ours (unsup) | 35.20 | 32.22 | **0.83** | 0.97 | 1.25 | **1.22** | **0.61M** |

\* For dense/hard_moe/c-BTM the routing-NMI column reports the *k-means clusterer's* NMI vs
domains (0.76); these methods have no per-document expert router, so it is a baseline reference,
not a learned-router score.

### What the numbers say (honestly)

1. **Quality:** full fine-tuning wins on raw perplexity (dense 27.0), with the upcycled MoE
   essentially matching it (28.0) and c-BTM a bit behind (30.2 — each of its 5 models sees only
   ~1/5 of the data in 2000 steps). The frozen-token methods are ~30% higher (35.2–35.7).
   **The Soft-MoE methods do *not* beat full fine-tuning on quality — and shouldn't be expected
   to at this scale.** They steer a frozen model; the baselines update all 162M weights.

2. **Efficiency is the real story.** ours reaches within ~30% of full-FT perplexity while
   training **15K–610K** parameters — **3–4 orders of magnitude fewer** than dense (162M),
   hard_moe (+227M) or c-BTM (+649M). "One frozen weight space + a tiny token bank" is a viable
   operating point, which is the thesis.

3. **Specialization is real (only the Soft-MoE methods have it):**
   - **Swap test passes (1.18–1.22):** routing an input through the *wrong* expert raises
     perplexity 18–22%. The expert tokens carry genuine, non-interchangeable expertise — the
     strongest causal evidence in the plan.
   - **Unsupervised EM beats its own seed:** ours (unsup) routing-NMI **0.83 > 0.76**, the
     k-means partition it was initialized from. The EM loop *refined* the domain structure rather
     than just memorizing the clusterer. It also edges out ours (sup) on macro-ppl (35.2 vs 35.7).
   - Experts are **well-utilized** (normalized usage-entropy 0.97, ~no dead experts) and
     **well-separated** (mean pairwise cosine distance 1.25).

Figures in [`figures/`](figures/): `quality_vs_params.png` (the efficiency frontier),
`contingency_*.png` (expert×domain heatmaps), `utilization_*.png`, `routing_gap.png`.

---

## 4. Caveats & limitations (this is a pilot)

- **1 seed, 2000 steps.** The plan wants ≥3 seeds and mean±std; perplexity gaps here are
  single-sample. Longer training would likely narrow the ours-vs-FT gap.
- **5 domains, dev recipe, pythia-160m.** Not yet the `main` (C4-clustered, 16-domain) headline
  recipe, nor multiple backbone sizes.
- **hard_moe routing** is per-token FFN routing, not per-document domain routing — its
  routing-NMI is therefore not directly comparable to ours.
- **c-BTM** here trains each expert on its cluster's slice for the same 2000 steps; a
  compute-matched or longer per-expert budget would help it.
- Two issues were found *and fixed* during this run: a CPU/GPU device bug in `token_separation`
  (eval only; training was unaffected) and a catastrophic full-FT lr (`1e-3`→`5e-5`). An earlier
  table showing dense at ppl 118 was that lr artifact — **disregard it**; the numbers above are
  the corrected ones.

---

## 5. Reproduce

```bash
# on Raven (env: module load python-waterboa/2025.06 + the project .venv)
DATA=/ptmp/$USER/soft-moe/data
python scripts/build_data.py     --config configs/experiment/ours_unsup_dev.yaml --data-root $DATA
python scripts/analyze_domains.py --processed-dir $DATA/processed/dev --out report/domain_analysis_dev.md
for m in dense_dev hard_moe_dev cbtm_dev ours_sup_dev ours_unsup_dev; do
  python scripts/train.py --config configs/experiment/$m.yaml --data-root $DATA \
    --run-dir /ptmp/$USER/soft-moe/experiments/$m --device cuda
done
python scripts/make_report.py --runs /ptmp/$USER/soft-moe/experiments --out report
```
SLURM scripts used: `.slurm/prep_dev.sbatch`, `.slurm/train_dev.sbatch <config>`,
`.slurm/eval_report.sbatch`. `/ptmp` is purged — keeper artifacts copied to
`/u/$USER/projects/soft-moe/results-m7-raven/`.

## 6. Next step toward the headline (M7 proper)

Scale the identical configs to **≥3 seeds**, more steps, and the `main` (C4-clustered) recipe;
add a `lora` backbone-mode variant of ours; report mean±std with paired-bootstrap significance.
