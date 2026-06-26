# THEORY — Methodology & Design Space

> The conceptual companion to the implementation plans. Where [README](README.md),
> [01](01-data-collection.md), [02](02-training.md), [03](03-evaluation.md) say *what to build*,
> this doc says *why*, *what the choices are*, and *what each choice tests*. It formalizes the
> [`BRAINSTORM.md`](../BRAINSTORM.md) idea and enumerates the design space along four axes:
> **(A) the mixture / EM formulation**, **(B) where expertise is injected**,
> **(C) supervised ↔ unsupervised assignment**, **(D) regularization**, plus the
> **datasets** that make each of these testable.

---

## 0. One-sentence thesis

A single language model backbone `f_θ` can be turned into a *soft mixture of experts* not by
adding parameter-heavy expert FFNs, but by learning a small bank of **expert tokens**
`E = {e_1, …, e_K}` (soft-prompt / steering vectors) and an **assignment** of inputs to experts,
where both the tokens and the assignment are fit by an **Expectation–Maximization** procedure.
The experts share `f_θ`; expertise is carried by `e_k`, not by separate weights.

---

## LR. Literature review — where this sits

> Grounds the design space below in prior work. Citations are canonical references; an agent
> should **verify exact authors/years/venues** before they enter a paper (treat this as a
> starting bibliography, not ground truth). Our contribution lives in the *intersection* that no
> single line below occupies: **EM-discovered, parameter-shared, prompt-carried experts.**

### LR.1 Mixture-of-Experts: from gating to sparse routing
- **Adaptive Mixtures of Local Experts** (Jacobs, Jordan, Nowlan, Hinton, 1991) and
  **Hierarchical Mixtures of Experts** (Jordan & Jacobs, 1994) — the original MoE, trained with
  **EM**, where a gate routes inputs to expert networks. Our EM framing is a direct descendant,
  but our "experts" share one network and differ only by a conditioning token.
- **Sparsely-Gated MoE** (Shazeer et al., 2017), **GShard** (Lepikhin et al., 2020),
  **Switch Transformer** (Fedus, Zoph, Shazeer, 2021) — scale MoE via top-k token routing and a
  **load-balancing auxiliary loss** (our §D.2 borrows this directly). These add *parameters*
  (separate FFNs); we deliberately do not.
- **Routing as assignment**: **BASE Layers** (Lewis et al., 2021) and **Expert-Choice routing**
  (Zhou et al., 2022) cast routing as a balanced linear-assignment / optimal-transport problem —
  the theoretical basis for our **OT / Sinkhorn E-step** (§A.3) and balance-as-constraint idea.
- **Soft MoE** (Puigcerver, Riquelme, Mustafa, Houlsby, 2023) — replaces hard routing with a
  fully differentiable **soft** combination of experts. This is our namesake intuition (soft,
  not discrete partitioning) applied at the *token-bank* rather than FFN level.

### LR.2 Modular / mergeable expert LMs (the c-BTM lineage)
- **DEMix Layers** (Gururangan et al., 2021) — domain-expert FFN modules selected by a known
  domain label; mixture computed at inference. Closest "supervised expert" precursor.
- **Branch-Train-Merge** (Li et al., 2022) and **c-BTM: Cluster-Branch-Train-Merge / Scaling
  Expert LMs with Unsupervised Domain Discovery** (Gururangan et al., 2023) — k-means-cluster the
  corpus, train **one separate expert model per cluster**, ensemble by cluster posterior. This is
  our **primary baseline**; we keep their *unsupervised discovery* but collapse N models into one
  shared backbone + token bank. **Branch-Train-Mix / BTX** (Sukhbaatar et al., 2024) later mixes
  such experts back into a single MoE — a parameter-heavy analogue of what we do with prompts.

### LR.3 Parameter-efficient conditioning (what an "expert token" *is*)
- **Prefix-Tuning** (Li & Liang, 2021), **Prompt Tuning** (Lester et al., 2021), **P-Tuning**
  (Liu et al., 2021) — learn a small continuous prompt that steers a frozen LM. Our expert tokens
  (§B1–B2) are exactly this mechanism, repurposed as *the carrier of an expert*.
- **LoRA** (Hu et al., 2021) and mixtures of adapters — **Polytropon** (Ponti et al., 2023),
  **AdapterSoup** (Chronopoulou et al., 2023), **Mixture-of-LoRA-Experts (MoLE)** — route among
  low-rank weight updates. These motivate our heavier injection rungs (§B4–B5: FFN-subspace /
  per-expert LoRA) and the "training subspaces of the weight matrix" reading of the brainstorm.
- **Activation steering / representation engineering** (Subramani et al., 2022; Turner et al.,
  2023; Zou et al., 2023) — additive latent vectors that redirect behavior; basis for our
  steering-vector injection (§B3).

### LR.4 Mixture-of-Prompts (the MoP baseline)
- **SMoP: Sparse Mixture-of-Prompts** (Choi et al., 2023) and **Mixture of Soft Prompts**
  (Chen et al., 2023) — route among multiple soft prompts with a learned gate, trained
  **jointly** (no EM). This isolates exactly the ingredient we add — **EM-based discovery of the
  assignment** — making MoP the clean control for "does EM beat joint training?"

### LR.5 EM, mixture models & deep clustering (the assignment machinery)
- **EM algorithm** (Dempster, Laird, Rubin, 1977) — the formal backbone of §A.2.
- **Deep clustering via balanced assignment**: **DeepCluster** (Caron et al., 2018), **SwAV**
  (Caron et al., 2020), **Self-Labelling via Optimal Transport** (Asano, Rupprecht, Vedaldi,
  2020) — alternate representation learning with **Sinkhorn-balanced** cluster assignment. Direct
  template for our online-EM + OT-balanced E-step and for avoiding cluster collapse.
- **Cobweb** (Fisher, 1987) — incremental *hierarchical* concept formation; the clusterer behind
  our token-**path** / mixture-of-concepts idea (§B6, §C, §E.3).

### LR.6 Regularizers we draw on
- **Load balancing**: Switch-Transformer aux loss (Fedus et al., 2021) → §D.2.
- **Diversity**: **Determinantal Point Processes** (Kulesza & Taskar, 2012) → the log-det /
  volume separation term (§D.1).
- **Informativeness**: **Information Bottleneck** (Tishby, Pereira, Bialek, 1999; deep variational
  IB, Alemi et al., 2017) → the `I(x;z)` term that stops experts from ignoring the input (§D.4).
- **Discrete-routing differentiability**: **Gumbel-Softmax** (Jang et al., 2017; Maddison et al.,
  2017) → learned hard routing (§D.3).

### LR.7 Domains, data & adaptation
- **Don't Stop Pretraining** (Gururangan et al., 2020) — domain/task-adaptive pretraining;
  evidence that domain-specialized LMs beat generic ones, the premise our experts exploit.
- Corpora: **C4** (Raffel et al., 2020), **The Pile** (Gao et al., 2020), **S2ORC** (Lo et al.,
  2020), **M2D2** (Reid, Zhong, Schuster et al., 2022, hierarchical domains) — see §E.

### LR.8 The gap we fill
| Line of work | Experts are… | Assignment is… | Params | Missing piece we add |
|---|---|---|---|---|
| Sparse MoE / Soft MoE | FFN sub-modules | learned router (top-k/soft/OT) | **+N FFNs** | parameter sharing; prompt-carried experts |
| c-BTM / BTM | **separate models** | offline k-means | **×N models** | single shared backbone |
| Prefix/Prompt tuning | one prompt | none (single expert) | tiny | *multiple* experts + routing |
| SMoP / MoP | soft prompts | **jointly-trained** gate | tiny | **EM discovery** of the partition |
| Deep clustering (SwAV) | cluster centroids | EM / OT | — | a *language model* objective |
| **This work** | **expert tokens (prompts/subspaces)** | **EM (sup.↔unsup.)** | **tiny, shared** | — (the synthesis) |

The novelty claim is precisely the unoccupied cell: experts that are (i) **prompt/subspace-light
and parameter-shared** like prefix-tuning, (ii) **discovered by EM** like classic MoE and deep
clustering, and (iii) **multi-expert + domain-routed** like c-BTM — none of the prior lines is
all three at once.

---

## A. The mixture model & its EM

### A.1 Generative view

Treat the expert index `z ∈ {1..K}` as a latent variable. For a sequence `x` (or a document /
segment), the model is a mixture:

```
p(x)        = Σ_k  π_k · p(x | z=k)
p(x | z=k)  = Π_t  f_θ( x_t | x_<t , e_k )          # backbone conditioned on expert token e_k
```

- `π_k` = mixture prior (expert popularity). Can be uniform, learned, or data-derived.
- `f_θ(· | e_k)` = the *same* backbone, conditioned on expert token(s) `e_k` (mechanism in §B).
- This is exactly a **mixture of conditional language models that share parameters** — the
  difference from classic MoE (mixture over FFN sub-modules) and from c-BTM (mixture over
  *separate* models). Here the mixture components differ only in their conditioning vector.

### A.2 EM objective

We maximize the (regularized) marginal log-likelihood `Σ_i log p(x_i)`. Introduce
responsibilities `r_ik = p(z=k | x_i)`. The standard EM bound gives:

```
E-step:   r_ik  ∝  π_k · p(x_i | z=k)                 # posterior over experts
                =  π_k · exp( −NLL_θ(x_i | e_k) )

M-step:   maximize  Σ_i Σ_k  r_ik · [ log π_k − NLL_θ(x_i | e_k) ]  − Ω(E, θ, r)
            w.r.t.  {e_k}, θ (optionally), π
```

`Ω` is the regularizer (see §D). Three knobs immediately fall out — they are the main design
decisions:

1. **Hardness of the E-step** (§A.3): hard argmax vs soft posterior vs balanced (OT) assignment.
2. **What the M-step updates** (§B): only `e_k`? also `θ` (full / LoRA)? `π`?
3. **E-step cadence** (§A.4): per-batch online vs periodic corpus-level reassignment.

### A.3 E-step variants (the assignment operator)

| Variant | Responsibility `r_ik` | Behaves like | When to prefer |
|---|---|---|---|
| **Hard EM** | one-hot at `argmax_k [log π_k − NLL_ik]` | k-means / Viterbi-EM | clearest specialization, cheapest forward (1 expert/example) |
| **Soft EM** | softmax posterior over `k` | classic GMM-EM | smoother optimization, no premature collapse, but `top_k` forwards |
| **Top-k soft** | posterior restricted to top-k experts, renormalized | Sparse soft-MoE | compromise: `k=1–2` forwards, some smoothing |
| **Balanced / OT** | Sinkhorn-normalized so marginal over experts ≈ `π` (SwAV / Soft-MoE style) | optimal-transport clustering | **built-in load balancing**, prevents collapse without a balance penalty |
| **Router-amortized** | a learned head `g_φ(x) ≈ r_i·` trained to track EM responsibilities | inference-time routing | needed for deployment (no NLL-per-expert at test) |

> The **Balanced/OT E-step** is theoretically attractive: it *replaces* the load-balance
> regularizer (§D) with a hard constraint in the assignment step, which is often more stable
> than penalizing imbalance in the loss. Recommend implementing both and comparing.

### A.4 E-step cadence (the "EM lifestyle" from the brainstorm)

- **Online soft-EM** — recompute `r_ik` every batch from current params. Cheap, noisy, default.
- **Periodic hard reassignment** — every `e_step_every` steps, sweep the corpus, recompute
  assignments, freeze them for the next interval. This is the **c-BTM-flavored discovery loop**
  and gives the cleanest "experts acquired selectivity" story.
- **Annealed cadence** — start with frequent soft updates, lengthen the interval and harden the
  posterior (temperature → 0) over training, so experts crystallize. (See §D entropy annealing.)

---

## B. Where expertise lives — injection mechanisms

The brainstorm phrase *"training subspaces of the weight matrix based on the input"* is
mechanism-agnostic. Enumerated from lightest to heaviest; each is a config knob, and comparing
them is itself a research question (capacity vs parameter cost).

| # | Mechanism | What `e_k` is | Params/expert | "Subspace" interpretation |
|---|---|---|---|---|
| B1 | **Input soft-prompt** | `T` embedding vectors prepended to the token stream | `T·d` | conditions all layers indirectly via attention |
| B2 | **Prefix-tuning (KV)** | per-layer past key/values | `2·L·T·d` | injects a per-expert "context" at every layer — stronger |
| B3 | **Steering / bias vector** | additive vector into hidden state at layer ℓ | `d` (or `L·d`) | shifts the residual stream into an expert region |
| B4 | **FFN subspace gate** | low-rank modulation of an FFN weight: `W → W + U_k V_k` | `2·r·d` | **literally** trains an input-conditioned weight subspace (closest to brainstorm wording) |
| B5 | **LoRA-per-expert** | a LoRA adapter selected by `z` | `2·r·d·#mats` | per-expert parameter subspace; bridges toward true MoE |
| B6 | **Token *path* (hierarchical)** | a sequence of tokens from a Cobweb root→leaf path | varies | mixture-of-concepts: coarse→fine conditioning |

Design guidance:
- **B1/B2** are the purest test of the thesis ("can a *prompt* be an expert?") — start here with
  a **frozen** backbone so any specialization is unambiguously carried by `e_k`.
- **B4** most directly realizes "subspaces of the weight matrix"; it's the natural escalation if
  prompts lack capacity. Frame B1→B4 as a **capacity ladder** ablation.
- **B6** operationalizes the brainstorm's "hierarchical passing down" and "mixture of concepts":
  the expert is the *path* `e_root → … → e_leaf`, enabling interpolation between concepts.

---

## C. Supervised ↔ unsupervised — a continuum, not a binary

The brainstorm splits "supervised" and "unsupervised." In the EM frame these are the **same
algorithm with different E-steps** — the axis is *how much the assignment `r` is given vs
learned*. Enumerate the rungs:

| Rung | E-step source | `r_ik` is… | Tests |
|---|---|---|---|
| C1 **Fully supervised** | ground-truth domain label | fixed one-hot from data | "can one weight space hold K experts?" (capacity question, no discovery) |
| C2 **Clusterer-supervised** | k-means / Cobweb assignment, **frozen** before training | fixed one-hot from clusterer | c-BTM-style: discovery is offline, training just fits tokens |
| C3 **Warm-started EM** | init from C2, then let EM update `r` | learned, initialized | does refining the partition during training help? |
| C4 **Fully unsupervised EM** | random init, EM discovers everything | learned from scratch | the hard version — "experts acquire selectivity" |
| C5 **Semi-supervised** | labels for a subset, EM for the rest | mixed | realistic; labels anchor identifiability |
| C6 **Curriculum / annealed** | start supervised (C1/C2), decay label weight → unsupervised | scheduled | avoids the cold-start collapse of C4 |

Key theoretical point — **identifiability**: pure C4 has a label-permutation symmetry and is
prone to collapse / degenerate solutions. C2/C3/C5/C6 break the symmetry with structure or a few
labels. Recommend C2 and C4 as the two headline conditions, with C3/C6 as the bridge that shows
*how much* discovery costs vs supervision.

**Clusterer choice (for C2–C4 init)** maps onto the data structure:
- **k-means on sentence embeddings** — flat, fast, c-BTM-comparable. Default.
- **Cobweb** — produces a *hierarchy*; supports B6 token-paths and the "interpolate the best
  expert" idea. Use when testing hierarchical experts.
- **Spectral / agglomerative** — when domains are non-spherical in embedding space.
- **Likelihood-based (the model itself)** — in fully-unsupervised EM the "clusterer" *is*
  `r_ik ∝ exp(−NLL_ik)`; embeddings only seed it.

---

## D. Regularization options (the `Ω` term)

This is where the unsupervised version is won or lost. The brainstorm names two priorities:
**push tokens apart** (separation) > **load-balance** experts. Below is the full menu; each is a
weighted, independently-logged term (see [02 §4](02-training.md)). Group by what they prevent.

### D.1 Separation / diversity (prevent redundant experts) — *highest priority*
- **Pairwise cosine repulsion**: `Ω_sep = mean_{j≠k} cos(e_j, e_k)` (minimize). Simple, default.
- **Orthogonality penalty**: `‖EᵀE − I‖_F²` — drive the token bank toward an orthonormal basis
  (this is also the *fixed* baseline's init: orthogonal & constant).
- **Log-determinant / volume**: maximize `log det(EᵀE + εI)` — a determinantal (DPP-style)
  diversity prize; rewards spreading experts to span a large subspace, softer than hard ortho.
- **Hinge / margin repulsion**: only penalize pairs closer than margin `m` — lets experts cluster
  loosely while forbidding collapse.

### D.2 Load balancing (prevent dead / dominant experts) — *secondary priority*
- **Usage-entropy bonus**: maximize `H(mean_i r_i·)` — encourage uniform batch-level usage.
- **Switch-Transformer aux loss**: `K · Σ_k (frac_examples_k · mean_router_prob_k)` (minimize) —
  the standard MoE balance loss; pairs with a learned router.
- **OT / Sinkhorn assignment** (§A.3): balance as a *constraint*, not a penalty — often the most
  stable; consider it the "regularization-free" balancer.
- **KL-to-prior**: `KL( usage ‖ π )` — balance toward a chosen non-uniform prior (e.g. domain
  frequencies), more flexible than pure entropy.

### D.3 Confidence / sparsity of assignment (sharpen responsibilities)
- **Posterior-entropy penalty**: minimize `H(r_i·)` per example → push toward hard routing.
- **Temperature annealing**: shrink the E-step softmax temperature over training (soft→hard).
- **Gumbel-softmax / straight-through**: differentiable hard routing if the router is learned.

### D.4 Input–expert dependence (make experts *informative*, not arbitrary)
- **Mutual-information / Information-Bottleneck**: maximize `I(x; z)` (experts should depend on
  input) while bounding `I(z; nuisance)`. Prevents the trivial solution where `z` ignores `x`.
- **Router calibration loss**: cross-entropy of the amortized router `g_φ(x)` against EM `r` —
  keeps the deployable router faithful to the EM partition.

### D.5 Backbone / token magnitude (stability)
- **Token-norm penalty or fixed-norm projection**: keep `‖e_k‖` bounded so one expert can't win
  by sheer magnitude (a common collapse mode).
- **Backbone anchoring** (when `θ` is trainable): `‖θ − θ_0‖²` / EWC / KL-to-base, so shared
  weights don't drift away from a usable LM while specializing.

### D.6 Recommended default objective

```
L = Σ_i Σ_k r_ik · NLL_θ(x_i | e_k)
    + λ_sep   · Ω_sep            # D.1   (largest λ)
    + λ_bal   · Ω_balance        # D.2   (smaller; or replaced by OT E-step)
    + λ_ent   · Ω_conf           # D.3   (annealed)
    + λ_route · Ω_router         # D.4   (only if amortized router)
```
With the brainstorm's priority encoded as **`λ_sep > λ_bal`**, and `λ_ent` ramped up late. The
ablation in [03 §5](03-evaluation.md) sweeps these to validate the ordering empirically.

---

## E. Datasets — chosen to make each idea falsifiable

Data is not incidental: each core idea needs a corpus whose structure can *reveal* it. Map
datasets to claims.

### E.1 Diagnostic / synthetic (identifiability first)
- **Controlled domain mixtures** — concatenate `K` maximally-distinct generators (e.g. natural
  language + code + arithmetic + a templated formal language). Ground-truth `z` is known and
  domains are near-orthogonal, so *if* the method can't separate these, it can't separate
  anything. Validates the **supervised capacity claim (C1)** and the **swap test**.
- **Synthetic "rotated subspace" data** — sequences whose statistics live in known, controllable
  subspaces; lets you measure whether expert tokens recover the *true* number and geometry of
  components (identifiability of K, separation metrics ground-truthed).

### E.2 Naturally domain-separable (the main corpora)
- **c-BTM setup: C4 and/or S2ORC**, k-means-clustered — the **direct baseline corpus**. Use it
  so Ours vs c-BTM is apples-to-apples (same partition, single model vs N models).
- **The Pile by constituent** (PubMed, GitHub, ArXiv, FreeLaw, StackExchange, …) — strong,
  human-labeled domain boundaries → clean supervised labels for C1/C5 and routing-accuracy.
- **M2D2** (Wikipedia + S2ORC, two-level domain hierarchy) — purpose-built for *hierarchical*
  domain adaptation → tests the **Cobweb / token-path (B6)** and coarse↔fine expert idea.

### E.3 Hierarchical / concept-structured (for the Cobweb story)
- Any corpus with a **taxonomy** (M2D2's hierarchy, arXiv category trees, Amazon/product
  category trees, Wikipedia category graph). The hierarchy gives ground-truth for evaluating
  whether the **token path** recovers coarse→fine concept structure (the "mixture of concepts"
  claim and "interpolate the best expert").

### E.4 Skill / task mixtures (does an "expert" = a task?)
- **Instruction / task mixtures** (FLAN-style task clusters, or a multi-task suite) — tests
  whether experts specialize by *skill* rather than surface domain; relevant to the Persona-LLM
  motivation in the brainstorm.

### E.5 Selection criteria (what makes a dataset "good" here)
1. **Separability** — domains distinguishable in embedding space (measure silhouette / NMI of
   the clusterer vs labels *before* committing — emitted by [01](01-data-collection.md)).
2. **Per-domain held-out depth** — enough test tokens per domain for stable per-domain ppl.
3. **Label availability** — needed for C1/C5 and for *scoring* unsupervised routing.
4. **Scale ladder** — same recipe at toy/dev/main sizes so claims survive scaling.
5. **Comparability** — overlaps the published baselines' corpora (c-BTM → C4/S2ORC) for fair head-to-head.

---

## F. What each baseline isolates (theory of the comparison)

| Baseline | Controls for | Conclusion if Ours ≥ it |
|---|---|---|
| **Dense** | the value of *any* conditioning | conditioning helps at all |
| **Ours (fixed orthogonal)** | the value of *learning* the tokens | learning the subspace matters (key ablation) |
| **MoP** | the value of *EM* (vs jointly-trained prompts) | EM discovery beats joint training |
| **c-BTM** | the value of *parameter sharing* (1 model vs N) | one weight space rivals N models |
| **Hard MoE** | the value of *full expert FFNs* | soft tokens approach true MoE at a fraction of params |

The scientific payload is the **chain of ablations**, not any single number: each baseline
removes exactly one ingredient (conditioning → learning → EM → sharing → parameter cost).

---

## G. Open theoretical questions / risks

1. **Capacity of a prompt as an expert.** Do `T·d` numbers carry enough to specialize a frozen
   backbone, or is B4/B5 (weight-subspace) required? → the capacity-ladder ablation answers this.
2. **Identifiability & collapse in C4.** Without labels or OT balancing, EM may collapse experts
   or find permutation-degenerate optima → mitigations: OT E-step, centroid warm-start (C2/C3),
   separation + magnitude penalties, annealed temperature.
3. **Posterior–router mismatch.** The EM posterior uses per-expert NLL (unavailable at test); the
   amortized router must approximate it → measure the oracle-vs-routed gap ([03 §2](03-evaluation.md)).
4. **Granularity of the latent.** Should `z` be per-document, per-segment, or per-token? Per-token
   routing is closer to classic MoE but breaks the "one expert per input/domain" framing → start
   per-document/segment; note per-token as future work.
5. **Does separation trade off quality?** Maximal orthogonality (the fixed baseline) may over-
   constrain; the `λ_sep` sweep tests whether learned, *partially* overlapping experts are better.
6. **Number of experts K vs true domains.** Under/over-clustering behavior — does EM merge/split
   gracefully? (k-sweep ablation; Cobweb's hierarchy offers a principled K.)

---

## H. Minimal theory-to-experiment crosswalk

| Claim | Condition | Primary metric | Doc |
|---|---|---|---|
| One weight space holds K experts | C1 supervised, frozen backbone | per-domain ppl ≈ hard-MoE | [03 §2](03-evaluation.md) |
| Learning tokens > fixing them | Ours-learned vs Ours-fixed | macro-ppl + separation | [03 §3,§5](03-evaluation.md) |
| Experts acquire selectivity unsup. | C4 / C3 | routing NMI ≫ chance, swap-test gap | [03 §3](03-evaluation.md) |
| Sharing rivals N models | Ours vs c-BTM | ppl at ≪ params | [03 §2](03-evaluation.md) |
| Hierarchy = mixture of concepts | B6 + Cobweb on M2D2 | path-vs-taxonomy alignment | [03 §5](03-evaluation.md) |
| Separation should outrank balance | λ_sep/λ_bal sweep | specialization vs quality frontier | [03 §5](03-evaluation.md) |
