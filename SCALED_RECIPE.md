# A Training Recipe for Rigorously Testing the MoE-vs-Dense Advantage

## 1. Objective

"MoE beats dense" is not a single claim — it is at least three different claims that the
literature sometimes conflates:

1. At equal **active parameters / FLOPs per token**, an MoE reaches lower pretraining loss.
2. At equal **total parameters**, an MoE reaches lower pretraining loss.
3. At equal **accelerator memory**, an MoE reaches lower pretraining loss.

These three claims can point in different directions, and the size of the gap depends heavily
on architectural choices (expert granularity, shared experts, routing/balancing scheme) that
are easy to get wrong. A recipe that "best tests the advantage" therefore cannot be a single
MoE-vs-dense training run — it has to be a small **isoFLOP scaling-law study**, built with the
architecture choices the literature has shown to be closest to Pareto-optimal, evaluated on
tasks chosen specifically because they are known to make MoE and dense models diverge.

Sections 2 gives the literature review; Section 3 distills design principles from it; Sections
4–6 give the concrete recipe, compute grid, and evaluation protocol; Section 7 states the
hypotheses the recipe is designed to confirm or falsify.

---

## 2. Literature Review

### 2.1 Foundational architecture and routing

- **Shazeer et al., 2017 — "Outrageously Large Neural Networks"** (arXiv:1701.06538) introduced
  the sparsely-gated MoE layer for LSTMs: a learned router selects a top-k subset of expert
  feed-forward networks per token, with an auxiliary loss (coefficient of variation of expert
  load) added to prevent collapse onto a few experts.
- **GShard (Lepikhin et al., 2020, arXiv:2006.16668)** brought MoE to Transformers at scale,
  using top-2 routing, an expert **capacity factor** (a hard cap on tokens per expert per
  batch, beyond which tokens are dropped), and an auxiliary load-balancing loss.
- **Switch Transformer (Fedus, Zoph & Shazeer, JMLR 2022, arXiv:2101.03961)** simplified
  routing to top-1 and reformulated the load-balancing loss as a differentiable product of
  (fraction of tokens routed to expert *i*) and (average router probability for expert *i*),
  summed over experts and scaled by a coefficient (commonly α ≈ 0.01). It also formalized the
  capacity-factor/dropped-token trade-off and scaled MoE language models to 1.6T parameters.
- **Expert Choice routing (Zhou et al., NeurIPS 2022, arXiv:2202.09368)** flips the routing
  direction: experts pick their top-k tokens rather than tokens picking experts. This
  guarantees perfect load balance by construction and accelerated convergence >2× over top-1/
  top-2 token-choice routing in their 8B/64E setting — but it is **not causally safe** for
  autoregressive decoding (a token's routing can depend on future tokens in the batch), which
  is why essentially all autoregressive LLM-scale MoEs (Switch, GLaM, Mixtral, DeepSeek,
  OLMoE) use token-choice routing instead.

### 2.2 Training stability

- **ST-MoE (Zoph et al., 2022, arXiv:2202.08906)** is the standard reference on why MoE
  training is fragile and how to fix it. It surveys stability techniques (removing
  multiplicative interactions, injecting noise, constraining activations/gradients) and lands
  on the **router z-loss** — a penalty on the squared log-sum-exp of router logits — as the
  best trade-off: it removes divergence without hurting quality. Their large-scale stability
  study used a train capacity factor of 1.25 and an eval capacity factor of 2.0. ST-MoE also
  documents that fine-tuning quality/stability, not just pretraining loss, is a distinct
  failure mode worth tracking separately.

### 2.3 Scaling laws: the central methodological question

This is the crux of "what counts as a fair test."

- **Clark et al., 2022 — "Unified Scaling Laws for Routed Language Models"** (ICML,
  arXiv:2202.01169) fit a bilinear scaling law relating routed-model loss to the equivalent
  dense model size *N* and the number of experts *E*, letting dense and MoE performance be
  compared under one shared power law. Their headline finding, using **coarse-grained**
  experts (expert width = FFN width, as in GShard/Switch), was that the routing advantage
  **saturates** at large scale for a fixed base size.
- **Krajewski, Ludziejewski et al., 2024 — "Scaling Laws for Fine-Grained Mixture of
  Experts"** (ICML, arXiv:2402.07871) directly revisit and complicate that conclusion. They
  introduce **granularity** *G* as an explicit hyperparameter: splitting each expert's hidden
  dimension by *G* while routing to *G* fine-grained experts per original expert keeps active
  compute constant but changes the effective capacity/compute trade-off. Their central result:
  the common practice of setting expert width equal to the FFN width is **near-optimal at
  almost no compute budget**, and once granularity is tuned properly, **the efficiency gap
  between MoE and dense widens, rather than saturates, as model size and training budget
  grow.** This directly contradicts the coarse-grained reading in Clark et al. and is the
  single most important architectural lever for a recipe that wants to show the "best case"
  advantage.
- **Abnar et al., 2025 — "Parameters vs FLOPs: Scaling Laws for Optimal Sparsity"** (ICML,
  arXiv:2501.12370) define sparsity S = (E − K)/E and show that the **optimal sparsity level
  depends on which resource is constrained** (fixed parameter budget vs. fixed training
  compute), and — importantly — that the mapping from pretraining loss to downstream few-shot
  performance is **not architecture-invariant**: MoE and dense models can reach the same
  upstream loss but differ downstream. A follow-up, **"Optimal Sparsity of MoE Language
  Models for Reasoning Tasks"** (arXiv:2508.18672), sharpens this: on reasoning-heavy
  benchmarks (GSM8K, GSM-Plus) the optimal density shifts *back toward dense* once active
  parameter count is large enough, even though sparser models remain better at smaller active-
  parameter budgets. Sparsity is therefore not uniformly good — it is a task-dependent dial.
- **"Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient"** (arXiv:2502.05172)
  adds a third axis: instead of only comparing at equal FLOPs, it derives a joint scaling law
  in active parameters, dataset size, and (transformed) expert count, and shows that under a
  **fixed memory budget**, MoE can be memory-optimal too — a compute-matched MoE can beat an
  "overtrained," same-total-parameter dense model. This paper (280+ runs up to 2.7B active /
  5B total parameters) is a useful precedent for feasible academic-scale study design and for
  the idea that MoE also raises the optimal tokens-to-active-parameter ratio as expert count
  grows (i.e., MoE compute-optimal training runs want relatively more data per active
  parameter than dense compute-optimal runs do).

### 2.4 Architecture refinements that maximize the demonstrable advantage

- **DeepSeekMoE (Dai et al., 2024, arXiv:2401.06066)** operationalizes fine-grained routing
  with two combined strategies: (1) **fine-grained expert segmentation** — split each expert's
  FFN into *m* smaller pieces and route to more of them, at constant active compute, so
  different pieces of knowledge decompose into different experts instead of being forced into
  a shared expert; (2) **shared expert isolation** — reserve *K*ₛ experts that are *always*
  active for every token, absorbing common/high-frequency knowledge so the routed experts do
  not waste capacity re-learning it. DeepSeekMoE-16B matched DeepSeek-7B / LLaMA2-7B quality
  using roughly 40% of the compute, and the approach was validated again at 145B parameters.
- **DeepSeek-V3 (DeepSeek-AI, 2024, arXiv:2412.19437)** combines DeepSeekMoE with an
  **auxiliary-loss-free load-balancing** scheme: a per-expert bias term is added to routing
  scores for the top-k *selection* only (not the gate weights), and is nudged up or down each
  step based on whether that expert is over- or under-loaded. This avoids the interference
  between load-balancing and language-modeling gradients that a scalar auxiliary-loss
  coefficient always trades off, at the cost of extra bookkeeping (per-expert bias state).
- **Mixtral 8x7B (Jiang et al., Mistral AI, arXiv:2401.04088)** is the standard "plain"
  reference point: top-2 of 8 coarse-grained experts, no shared experts, and is a useful
  coarse-grained control condition to contrast against fine-grained designs.
- **OLMoE (Muennighoff et al., 2024, arXiv:2409.02060)** is the most useful reference for an
  *academic, fully reproducible* recipe: it is fully open (data, code, logs, intermediate
  checkpoints) and directly ablates the design choices above — granularity, shared experts,
  load-balancing + z-loss, routing algorithm, and upcycled vs. from-scratch initialization —
  at a real production scale (1B active / 7B total parameters, 5.1T tokens), training roughly
  2× faster than an equivalent dense model. It is the closest thing to a validated "reference
  configuration" for the recipe below.

### 2.5 Initialization: from-scratch vs. upcycling

- **Sparse Upcycling (Komatsuzaki et al., ICLR 2023, arXiv:2212.05055)** initializes an MoE
  from a dense checkpoint (each expert copied from the dense FFN) and continues training. This
  reuses roughly half of the dense model's sunk pretraining cost and outperformed
  from-scratch dense baselines on SuperGLUE/ImageNet in their setting.
- **Drop-Upcycling (arXiv:2502.19261)** identifies the failure mode this creates: naive
  upcycling starts every expert identical, so early in training experts have little incentive
  to differentiate, which slows specialization and can hurt long-horizon convergence relative
  to training from scratch. It proposes partial re-initialization to recover some of the
  benefit of random init while keeping most of the upcycling speed advantage.
- This is a genuine methodological fork: **from-scratch training isolates the architectural
  effect cleanly**; upcycling is more representative of how MoEs are actually built in
  practice (reusing an existing dense checkpoint) but confounds the comparison with
  initialization effects.

### 2.6 Where the pretraining advantage breaks down

- **Artetxe et al., 2022 — "Efficient Large Scale Language Modeling with Mixtures of
  Experts"** (Meta, EMNLP 2022, arXiv:2112.10684) is the single most important paper for
  recipe design, because it is the one paper that explicitly tests fine-tuning, not just
  pretraining loss and few-shot priming. Their finding: **with the exception of fine-tuning**,
  MoE is substantially more compute-efficient — matching dense quality with ~4× less compute
  at modest budgets, narrowing but persisting at their largest scale (a 1.1T-parameter MoE
  still beat a compute-equivalent 6.7B dense model). But **the compute-efficiency advantage
  does not transfer cleanly to full-shot fine-tuning**, and the size of the gap varies greatly
  by task and domain — evidence that MoE and dense models generalize differently in ways not
  fully explained by pretraining loss alone. Any recipe that only reports pretraining
  perplexity is not testing the full claim.

### 2.7 Systems and efficiency caveats

- Realized (wall-clock) speedups from sparsity are not the same as FLOP-count savings: they
  depend on the capacity factor / token-dropping trade-off (Switch, ST-MoE), and on all-to-all
  communication cost in distributed training, which itself depends on granularity — Krajewski
  et al. explicitly note that fine-grained MoE's wall-clock gains (needed to reach a given
  perplexity) can diverge from its FLOP-count gains depending on the training setup and
  hardware.
- Total parameters must still reside in aggregate accelerator memory even though only a
  fraction activate per token — a real cost that pure FLOP-matched comparisons hide, and the
  reason the memory-matched axis (§2.3, Joint MoE Scaling Laws) is a meaningful third
  comparison rather than a redundant one.

---

## 3. Design Principles Derived from the Review

1. **Compare curves, not points.** Every scaling-law paper above reaches its conclusion from
   an isoFLOP sweep across multiple compute budgets, not a single matched pair. A single
   pair-comparison recipe cannot distinguish "MoE wins at this scale" from "MoE wins at all
   scales," and the Clark-et-al.-vs-Krajewski-et-al. disagreement shows that the *trend* with
   scale is itself the finding worth reproducing.
2. **Use fine-grained experts + shared experts, not the textbook GShard/Switch config.**
   Coarse-grained MoE understates the architecture's potential and is the condition under
   which the advantage was found to saturate (§2.3). To test the *best case*, granularity and
   shared-expert count must be swept, not fixed at G=1.
3. **Stabilize both arms fairly.** Router z-loss and a load-balancing mechanism are not
   optional add-ons; without them, MoE training instability (§2.2) will contaminate the
   comparison with a confound unrelated to the architecture's ceiling.
4. **Report the gap on at least three matching axes** (active-FLOPs, total-parameters,
   memory), because they can disagree, and because "the MoE advantage" is frequently
   overclaimed by only reporting the one axis (active-FLOPs) that flatters it most.
5. **Evaluate downstream and after fine-tuning, not only upstream loss.** §2.3 and §2.6 both
   show pretraining loss and downstream/fine-tuned performance can diverge between
   architectures, and the direction of divergence is task-dependent (knowledge-heavy tasks vs.
   reasoning-heavy tasks vs. full fine-tuning).
6. **Separate the architecture effect from the initialization effect.** From-scratch training
   is the primary arm for a clean scientific comparison; upcycling is a secondary,
   practically-relevant arm.
7. **Track wall-clock and memory alongside FLOPs.** A recipe that only reports theoretical
   FLOPs cannot support real efficiency claims (§2.7).

---

## 4. The Training Recipe

### 4.1 Shared backbone (held identical across all arms)

Only the FFN sub-layer differs between the dense and MoE arms; everything else is identical so
that the FFN/routing design is the only variable under test.

| Component | Choice |
|---|---|
| Architecture family | Pre-norm decoder-only Transformer (Llama/OLMo-style) |
| Normalization | RMSNorm |
| Attention | Grouped-Query Attention, RoPE positional encoding |
| Non-MoE FFN activation | SwiGLU |
| Tokenizer | Fixed BPE tokenizer, identical vocabulary, identical across all runs |
| Sequence length | Fixed (e.g., 2048 or 4096) across all runs |
| Precision | bf16 mixed precision (fp8 optional at the largest budgets, matching both arms) |

### 4.2 Dense arm

Standard dense Transformer at a given active-parameter target N, varying depth/width to hit
that target with roughly balanced d_model/n_layers ratios typical of the model family chosen.

### 4.3 MoE arm

Following the fine-grained, shared-expert design that §2.3–2.4 show is closest to
Pareto-optimal:

| Parameter | Setting |
|---|---|
| Expert construction | Fine-grained: FFN split by granularity factor *G*; expert width = d_ff / G |
| Granularity sweep | G ∈ {1, 2, 4, 8} (G = 1 reproduces the coarse-grained Switch/GShard/Mixtral condition as a control) |
| Shared experts | 0 (control) vs. 1 vs. 2 always-on experts, per DeepSeekMoE |
| Routed experts (base, G=1) | e.g. 8 experts, top-2 active — scale expert count and top-k together with G to hold active FFN compute constant |
| Routing | Token-choice, softmax/sigmoid gate over routed experts (causal-safe; do **not** use Expert Choice for the primary autoregressive LM arm) |
| Active parameters | Matched to the dense arm's compute-optimal N at each compute budget (this is what makes the Axis-A comparison in §4.9 valid) |

### 4.4 Stability techniques (both MoE conditions)

| Technique | Setting |
|---|---|
| Load balancing (primary) | Auxiliary-loss-free bias-based balancing (DeepSeek-V3 style): per-expert bias added to top-k selection score only, updated each step by a small fixed step size toward balance |
| Load balancing (ablation arm) | Classic auxiliary load-balancing loss (Switch-style), coefficient α ≈ 0.01, to test whether the aux-loss-free scheme's reported advantage over the older auxiliary loss reproduces at this scale |
| Router z-loss | Always on for both balancing conditions; small coefficient (order 1e-3, per Zoph et al.), tuned by a short LR/coefficient sweep before the main runs |
| Capacity factor (if using a hard-capacity implementation rather than a fully dynamic one) | Train: 1.25, Eval: 2.0 (ST-MoE default), with dropped-token rate logged and kept low |

### 4.5 Data

- One fixed, deduplicated, well-documented open pretraining corpus (e.g. a FineWeb/DCLM-style
  web-text majority mixed with code and reference/encyclopedic text, following the OLMoE-style
  mixture) used identically — same shuffling, same document order — across **every** run in the
  grid. Data is the single biggest confound if it varies between arms; it must not vary.
- Held-out validation split from the same distribution, plus at least one clearly
  out-of-domain held-out split (e.g. held-out books/scientific text) to test whether any
  architecture generalizes better out-of-distribution, following Artetxe et al.'s in-/
  out-of-domain split methodology.

### 4.6 Optimization

- AdamW, warmup + cosine (or WSD) decay.
- **Do not assume dense-tuned learning rate transfers to MoE.** Run a small LR grid (e.g. 3–4
  values spanning roughly 2×) separately for the dense and MoE arms at one small compute
  budget, pick the best per architecture, and only then fix LR schedules for the full grid —
  MoE and dense models are not guaranteed to share an optimal LR.
- Same batch size in tokens, same weight decay, same gradient clipping, same random seed
  policy across arms at a given compute budget.
- **Multiple seeds per configuration point** (≥2–3) at the smaller compute budgets, since
  scaling-law curve fits are sensitive to per-run noise; this is standard practice in the
  papers reviewed above (e.g. the 280-run methodology in the Joint MoE Scaling Laws study).

### 4.7 Compute budget grid (isoFLOP sweep)

Approximate dense-equivalent operating points using the standard training-FLOPs
approximation C ≈ 6ND and a Chinchilla-style starting ratio D ≈ 20N, purely as an
**initial grid** — the actual compute-optimal (N, D) pair should be fit per architecture at
each budget with a small local sweep, since §2.3 shows MoE's optimal tokens-per-active-
parameter ratio tends to be *higher* than dense's as expert count grows, not identical to it.

| Compute budget C (FLOPs) | Illustrative dense-equivalent active params N | Illustrative tokens D |
|---|---|---|
| 1×10¹⁷ | ~30M | ~0.6B |
| 3×10¹⁷ | ~50M | ~1B |
| 1×10¹⁸ | ~90M | ~1.8B |
| 3×10¹⁸ | ~160M | ~3B |
| 1×10¹⁹ | ~290M | ~6B |
| 3×10¹⁹ | ~500M | ~10B |
| 1×10²⁰ (stretch budget) | ~900M | ~18B |

This spans roughly three orders of magnitude of compute and is comparable in scale to the
academic studies reviewed (Krajewski et al.'s fine-grained scaling-law study; the Joint MoE
Scaling Laws study's up-to-2.7B-active-parameter runs) — large enough to fit a real trend line,
small enough to be run on a university-scale GPU allocation (single-node to small multi-node,
depending on the top 1–2 budgets).

At each budget: train the dense arm at its own compute-optimal (N, D); train the MoE arm with
active parameters matched to that same N, but let D and the granularity/expert-count/top-k
combination vary to find the MoE arm's own compute-optimal point — this is what produces a
clean **Axis A (iso-active-FLOPs)** frontier comparison rather than a single-point comparison.

### 4.8 Initialization arms

- **Primary: from-scratch** for both dense and MoE, at every grid point — this is the arm used
  for the headline scaling-law comparison, since it has no initialization confound.
- **Secondary: sparse-upcycled MoE**, initialized from the matched from-scratch dense
  checkpoint at 2–3 of the grid points, continued for the same additional token budget as the
  from-scratch MoE arm, to test whether Artetxe/Komatsuzaki/Drop-Upcycling-style
  initialization effects change the conclusion at that scale.

### 4.9 The three matching axes

Report the MoE-vs-dense gap under all three, at every compute budget — this is the core
methodological contribution of the recipe (§3, principle 4):

| Axis | What is held equal | What it tests |
|---|---|---|
| **A — Active-FLOPs-matched** | Active parameters / FLOPs per token | "Does routing let you buy more effective capacity for the same per-token compute?" — the headline MoE claim |
| **B — Total-parameters-matched** | Total parameters (dense arm made as large as MoE's total param count, trained for the *same total training compute* as the MoE arm, and therefore comparatively data-starved) | "How much of the gain is genuine specialization vs. simply having more parameters?" — directly probes the finding in the Joint MoE Scaling Laws paper that a compute-matched MoE can beat an "overtrained," same-total-parameter dense model |
| **C — Memory-matched** | Peak accelerator memory footprint (dominated by parameter count, held roughly constant; account for activation/optimizer-state differences separately) | "Under a deployment/hardware memory constraint, which architecture wins?" — the axis on which conventional wisdom (MoE is memory-hungry) is most often wrong, per §2.3 |

---

## 5. Evaluation Protocol

Report all of the following per grid point, for both arms:

1. **Upstream**: held-out validation loss/perplexity, in-domain and out-of-domain.
2. **Knowledge-intensive few-shot** (where GLaM/Artetxe found MoE's largest wins): closed-book
   QA (e.g. TriviaQA, Natural Questions-style), broad knowledge benchmarks (e.g. MMLU-style).
3. **Commonsense / NLU few-shot**: HellaSwag, PIQA, ARC, WinoGrande-style tasks.
4. **Reasoning-heavy**: GSM8K-style math and code tasks — specifically because §2.3 shows this
   is where the sparsity advantage is most likely to *shrink or reverse* at larger active-
   parameter budgets, making it the sharpest test of whether "MoE is better" is scale- and
   task-conditional rather than universal.
5. **Full fine-tuning generalization**: fine-tune both matched arms (Axis A) on a small
   held-out downstream task suite and compare — this directly targets the Artetxe et al.
   finding that the pretraining-time compute advantage does not reliably carry over to
   fine-tuning, and is the check most training recipes skip.
6. **Systems metrics**: measured tokens/sec, step wall-clock time, peak memory, and (if
   multi-node) all-to-all communication time — reported alongside the FLOP-based numbers so
   that theoretical and realized efficiency can be told apart (§2.7).
7. **Expert utilization diagnostics** (MoE arm only): per-expert load distribution, dropped-
   token rate (if using hard capacity), and routing entropy — to confirm any observed gap is
   not an artifact of load-balancing failure rather than a true architectural effect.

---

## 6. Ablation Matrix

Run at 1–2 mid-sized compute budgets (not the full grid) to isolate each design choice's
contribution, holding everything else at the "primary" setting from §4:

| Design axis | Variants | Literature motivating the ablation |
|---|---|---|
| Granularity G | 1 (coarse control), 2, 4, 8 | Krajewski/Ludziejewski et al. 2024 |
| Shared experts | 0, 1, 2 | DeepSeekMoE (Dai et al. 2024) |
| Load balancing | Auxiliary loss (α≈0.01) vs. auxiliary-loss-free bias | Fedus et al. 2022; DeepSeek-V3 |
| Router z-loss | On vs. off | Zoph et al. 2022 (ST-MoE) |
| Top-k | 1, 2, 4 (interacting with G) | Shazeer et al. 2017; Krajewski et al. 2024 |
| Initialization | From-scratch vs. sparse-upcycled | Komatsuzaki et al. 2023; Drop-Upcycling 2025 |
| Routing algorithm (diagnostic only, non-causal) | Token-choice vs. Expert-choice | Zhou et al. 2022 |

---

## 7. Hypotheses the Recipe Is Designed to Test

Stated as falsifiable predictions grounded in §2, to be confirmed or overturned by the results
rather than assumed:

- **H1 (Axis A widens with scale, given fine granularity):** the active-FLOPs-matched gap
  between MoE and dense grows across the compute grid, replicating Krajewski/Ludziejewski
  rather than the saturation found by Clark et al. under coarse granularity — testable by
  comparing the G=1 control condition against G>1 conditions directly.
- **H2 (granularity is doing real work):** the G=1 control shows a visibly smaller or
  saturating gap relative to G>1, isolating how much of "the MoE advantage" in prior headline
  results (GLaM, Mixtral-style coarse MoE) was left on the table by not tuning granularity.
- **H3 (fine-tuning compresses the gap):** the Axis-A advantage measured on upstream loss and
  few-shot eval shrinks — and may reverse on some tasks — after full fine-tuning, replicating
  Artetxe et al. at smaller scale.
- **H4 (task-dependent optimal sparsity):** knowledge-intensive tasks favor the sparser/
  higher-total-parameter MoE configurations more than reasoning-heavy tasks do, and the
  reasoning-task gap narrows fastest as active parameters grow, replicating Abnar et al. and
  its reasoning-focused follow-up.
- **H5 (memory axis is not obviously anti-MoE):** on Axis C, the memory-matched MoE
  configuration is competitive with, and can beat, the memory-matched dense configuration,
  replicating the Joint MoE Scaling Laws finding rather than the conventional assumption that
  MoE is simply memory-expensive.
- **H6 (stabilization matters more than architecture at small scale):** at the smallest
  compute budgets, the choice of load-balancing scheme and z-loss on/off changes the outcome
  more than granularity does — i.e., an unstable MoE can lose to dense for reasons unrelated
  to its architectural ceiling, which is why §3's stabilization principle matters.

---

## 8. Practical Notes for Academic-Scale Compute

- The grid in §4.7 (30M–900M active-equivalent parameters, ~0.6B–18B tokens per run) is sized
  to be runnable on a single modern GPU node for the lower budgets and a small multi-node
  allocation for the top one or two budgets — comparable in scale to the Krajewski et al. and
  Joint MoE Scaling Laws studies, not to production-scale MoEs like DeepSeek-V3.
- If compute is tight, prioritize: (1) the full isoFLOP grid at G=1 vs. one strong fine-grained
  G for both arms (this alone tests H1/H2, the central claim), over (2) the full granularity ×
  shared-expert × balancing ablation matrix, over (3) the fine-tuning and memory-matched axes.
  All three are valuable, but (1) is the load-bearing result.
- Log everything needed to refit the scaling-law curves later (loss vs. compute per
  configuration, not just final numbers) — the point estimates from a small grid are noisy
  enough that the curve fit, not any single run, is the actual result.

---

## 9. References

- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer.* arXiv:1701.06538.
- Lepikhin, D. et al. (2020). *GShard: Scaling Giant Models with Conditional Computation and
  Automatic Sharding.* arXiv:2006.16668.
- Fedus, W., Zoph, B., & Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion
  Parameter Models with Simple and Efficient Sparsity.* JMLR 23. arXiv:2101.03961.
- Du, N. et al. (2022). *GLaM: Efficient Scaling of Language Models with Mixture-of-Experts.*
  ICML 2022. arXiv:2112.06905.
- Artetxe, M. et al. (2022). *Efficient Large Scale Language Modeling with Mixtures of
  Experts.* EMNLP 2022. arXiv:2112.10684.
- Zoph, B. et al. (2022). *ST-MoE: Designing Stable and Transferable Sparse Expert Models.*
  arXiv:2202.08906.
- Clark, A. et al. (2022). *Unified Scaling Laws for Routed Language Models.* ICML 2022.
  arXiv:2202.01169.
- Zhou, Y. et al. (2022). *Mixture-of-Experts with Expert Choice Routing.* NeurIPS 2022.
  arXiv:2202.09368.
- Komatsuzaki, A. et al. (2023). *Sparse Upcycling: Training Mixture-of-Experts from Dense
  Checkpoints.* ICLR 2023. arXiv:2212.05055.
- Dai, D. et al. (2024). *DeepSeekMoE: Towards Ultimate Expert Specialization in
  Mixture-of-Experts Language Models.* arXiv:2401.06066.
- Krajewski, J., Ludziejewski, J. et al. (2024). *Scaling Laws for Fine-Grained Mixture of
  Experts.* ICML 2024. arXiv:2402.07871.
- Jiang, A.Q. et al. (2024). *Mixtral of Experts.* Mistral AI. arXiv:2401.04088.
- Muennighoff, N. et al. (2024). *OLMoE: Open Mixture-of-Experts Language Models.*
  arXiv:2409.02060.
- DeepSeek-AI (2024). *DeepSeek-V3 Technical Report.* arXiv:2412.19437.
- Abnar, S. et al. (2025). *Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for
  Mixture-of-Experts Language Models.* ICML 2025. arXiv:2501.12370.
- *Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient.* (2025).
  arXiv:2502.05172.
- *Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization.* (2025).
  arXiv:2502.19261.
- *Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks.* (2025).
  arXiv:2508.18672.