# 02 — Training

> Owns: `src/softmoe/models/`, `src/softmoe/training/`, `configs/model/`, `configs/train/`,
> `configs/experiment/`, `scripts/train.py`.
> Goal: implement the **expert-token model** and the **EM training loop**, plus every baseline,
> behind a single config-driven `train.py`. This is the core contribution — get it right.

## 1. The core module: `ExpertTokenBank` (`models/expert_tokens.py`)

An expert is **not** a separate FFN or model. It is a small set of learnable embedding vectors
("expert tokens") injected into the backbone's input/prefix so that the *same* weights compute
differently per expert. This is the soft-prompt / prefix-tuning mechanism, used as the carrier
of expertise.

```python
class ExpertTokenBank(nn.Module):
    """
    n_experts: number of experts (= n_clusters from data, or a chosen K)
    tokens_per_expert: T soft tokens per expert (T=1 default; >1 = a token "path")
    d_model: backbone hidden size
    init: 'random' | 'orthogonal' | 'from_cluster_centroids'
    trainable: if False -> the FIXED/orthogonal-constant baseline (ablation)
    """
    # embeddings: Parameter[n_experts, tokens_per_expert, d_model]
    def forward(self, expert_ids) -> Tensor:        # -> [B, T, d_model] prefix to inject
    def separation_loss(self) -> Tensor:            # pushes experts apart (see §4)
    def orthogonal_init(self): ...                  # Gram-Schmidt / random orthonormal
```

Injection strategies (config `model.injection`), implement at least the first two:
- **`prefix`** — prepend the T expert tokens to the input embedding sequence (cheapest, like
  prompt-tuning). Attention mask + position ids handled so labels still align.
- **`prefix_kv`** — prefix-tuning proper: expert tokens become past key/values at each layer
  (stronger conditioning). Use `peft`-style hooks or manual `past_key_values`.
- (optional) **`addition`** — add a per-expert bias vector into hidden states at a chosen layer.

## 2. The wrapper: `SoftMoE` (`models/soft_moe.py`)

```python
class SoftMoE(nn.Module):
    backbone: AutoModelForCausalLM     # gpt2 / pythia-160m, frozen or LoRA or full per config
    tokens:   ExpertTokenBank
    router:   Router                   # produces expert assignment (see §3)
    def forward(self, batch) -> {loss, logits, aux}:
        expert_ids, route_info = self.router(batch, self.tokens)   # E-step output (or detached)
        prefix = self.tokens(expert_ids)
        out = self.backbone(inputs_embeds=inject(prefix, batch.input_ids), labels=batch.labels)
        return {loss: out.loss, logits: out.logits,
                aux: {route_info, separation, load_balance}}
```
- **Backbone tuning mode** (config `model.backbone_mode`): `frozen` (only tokens learn — purest
  test of the hypothesis), `lora`, or `full`. Default `frozen` for the headline claim, with
  `lora` as a stronger variant.
- Keep `forward` identical in signature for all baselines so the trainer is method-agnostic.

## 3. The router / E-step (`models/router.py`)

The router answers: *which expert does this input use?* Two modes.

- **Supervised** (`SupervisedRouter`): expert id = `batch.domain_id` or `batch.cluster_id`
  (config picks which). No learned routing; this is the clean "can one weight space hold many
  experts" experiment.
- **Unsupervised / learned** (`SoftRouter`): a tiny head maps a pooled representation of the
  input (e.g. mean of first-k backbone hidden states, or the sentence embedding from data) to a
  distribution over experts. Two assignment styles:
  - **EM-hard:** E-step assigns argmax expert (responsibilities r_ik ∈ {0,1}); M-step trains.
  - **EM-soft:** responsibilities are softmax/posterior; loss is the responsibility-weighted
    mixture (run the forward once per top-k expert and combine — keep top-k small, default 1–2).

### The EM view (make this explicit in code + docstring)
- **E-step:** given current tokens+backbone, compute responsibility `r_ik` = posterior that
  input *i* belongs to expert *k* (∝ exp(−NLL_ik) × prior_k, or router softmax). For supervised,
  `r_ik` is the one-hot label.
- **M-step:** fix responsibilities, take gradient step minimizing `Σ_i Σ_k r_ik · NLL_ik` plus
  regularizers. Update tokens (and backbone if not frozen) and the prior/router.
- Alternate E and M on a configurable cadence (`train.em.e_step_every` steps; full re-assign vs
  online/streaming responsibilities). Start with **online soft-EM** (cheap) and offer a periodic
  **hard re-assignment** sweep over the corpus (more faithful to c-BTM-style discovery).

## 4. Losses & regularizers (`training/losses.py`)

Total objective:
```
L = L_lm  +  λ_sep · L_separation  −  λ_bal · H_load   (+ λ_route · L_route)
```
- **`L_lm`** — standard causal LM cross-entropy (responsibility-weighted in soft-EM).
- **`L_separation`** — push expert tokens apart. Implement at least:
  - mean pairwise cosine similarity (minimize), and
  - optionally a repulsion / log-det (volume) term on the token matrix.
  Per the brainstorm, **separation is the higher-priority regularizer** → default `λ_sep > λ_bal`.
- **`H_load` (load balance)** — entropy of the batch-wise expert-usage distribution (maximize)
  *or* the Switch-Transformer auxiliary balance loss (minimize importance×load product). Keeps
  experts from collapsing to one. Lower priority than separation.
- **`L_route`** (optional) — cross-entropy training the router head toward current
  responsibilities (so the learned router tracks the EM assignment).
- Every term is individually weighted in `configs/train/*.yaml`, logged separately to W&B, and
  unit-tested for correctness on toy tensors (`tests/test_losses.py`).

## 5. The EM trainer (`training/em_trainer.py`)

```python
class EMTrainer:
    def fit(self, model, data, cfg):
        for step in range(cfg.max_steps):
            batch = next(loader)
            if cfg.em.mode == 'hard' and step % cfg.em.e_step_every == 0:
                self.reassign(model, data)         # corpus-level E-step (optional)
            out = model(batch)                     # online E-step inside router + M-step forward
            loss = combine(out, cfg.lambdas)
            loss.backward(); clip; opt.step(); sched.step(); opt.zero_grad()
            log({lm, sep, balance, utilization, per-expert counts})
            if step % cfg.eval_every == 0: validate(); maybe_checkpoint()
```
Requirements:
- `accelerate` or plain DDP for multi-GPU; gradient accumulation; mixed precision; grad clip.
- **Checkpoint** tokens + router + (optional) backbone/LoRA + optimizer + RNG + resolved config.
- Log **expert utilization histogram** every N steps — this is the primary "is it specializing?"
  signal during training.
- Deterministic: seed everything; dump `resolved_config.yaml` and git SHA into the run dir.

## 6. Baselines (`models/baselines/`) — same `forward` contract

| Baseline | File | Sketch |
|----------|------|--------|
| **Dense** | `dense.py` | backbone only, no tokens/router. Lower bound. |
| **Hard MoE** | `hard_moe.py` | replace backbone FFN(s) with top-k token-routed experts (small N). Reference upper bound. Can use a minimal from-scratch MoE FFN, not a full library. |
| **c-BTM** | `cbtm.py` | train **N separate** small experts (one per cluster) via the *same* trainer (loop over clusters); at eval, ensemble by cluster posterior. This reuses the data clusterer from [01](01-data-collection.md). |
| **MoP** | `mop.py` | per-domain soft prompts, selected/averaged by a router but **trained without EM** (joint, fixed assignment). Isolates the EM contribution. |
| **Ours (fixed)** | uses `ExpertTokenBank(trainable=False, init='orthogonal')` | isolates *learning* the embeddings. |

All baselines are selected purely by `configs/experiment/<name>.yaml` → no separate scripts.

## 7. Config example (`configs/experiment/ours_unsup.yaml`)

```yaml
defaults: [/data/dev, /model/softmoe_pythia160m, /train/em_soft]
model:
  backbone_mode: frozen
  injection: prefix_kv
  expert_tokens: { n_experts: 6, tokens_per_expert: 4, init: from_cluster_centroids, trainable: true }
  router: { kind: soft, top_k: 1, pool: meanhidden }
train:
  max_steps: 20000
  em: { mode: soft, e_step_every: 500 }
  lambdas: { sep: 1.0, balance: 0.1, route: 0.5 }   # sep > balance, per brainstorm
  optimizer: { name: adamw, lr: 1.0e-3, wd: 0.0 }    # tokens want a higher LR than full FT
  scheduler: cosine
seed: 0
```

## 8. Deliverables / acceptance

- **M2:** `SoftMoE.forward` runs on toy batch; `tests/test_model.py` confirms grads flow to
  `ExpertTokenBank` (and *not* to a frozen backbone), loss finite, shapes correct under each
  injection mode.
- **M3:** supervised variant on toy corpus beats the dense baseline on per-domain val ppl
  (overfit-to-learn sanity).
- **M4:** unsupervised EM produces non-uniform, stable expert utilization and rising token
  separation; experts correlate with ground-truth domains above chance (logged each eval).
- **M5:** all six methods train through `scripts/train.py --config configs/experiment/<m>.yaml`
  and write a comparable `experiments/<run>/metrics.json`.

## 9. Risks the agent should watch

- **Expert collapse** (all inputs → one expert): the load-balance term and a warmup that seeds
  tokens from cluster centroids mitigate it; log utilization early.
- **Frozen backbone too weak:** if `frozen` shows no specialization signal, escalate to `lora`
  before concluding negative — make this a planned ablation, not a surprise.
- **Soft-EM cost:** top-k>1 multiplies forward passes; keep `top_k` small, prefer hard
  re-assignment sweeps for the discovery signal.
- **Tokens-per-expert vs path:** start `tokens_per_expert=1`; only add the hierarchical "token
  path" (Cobweb root→leaf) once the flat version works.
