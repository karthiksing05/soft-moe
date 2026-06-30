# Soft-MoE: Implementation Plan

> Coding-agent brief. These documents specify *what to build and in what order* so an
> autonomous coding agent can scaffold this repository and run the full research loop:
> **data collection → training → evaluation**. Read this file first, then the three plan
> docs in order. The source idea lives in [`../BRAINSTORM.md`](../BRAINSTORM.md); the
> methodology, design-space enumeration, and literature review live in [`THEORY.md`](THEORY.md).

## 1. The research question, in one paragraph

Mixture-of-Experts (MoE) gets its power by routing different inputs to different *parameter
subsets*. We ask: **can a single LLM's weight space be softly partitioned into experts by
conditioning on learned "expert tokens" (soft-prompt embeddings), trained via an
Expectation–Maximization loop?** Instead of N separate expert models (as in classic MoE or
c-BTM), we keep **one** frozen-or-lightly-tuned backbone and learn a small bank of expert
token embeddings. The E-step assigns each input to an expert (cluster); the M-step trains the
backbone and the selected expert tokens. We study two regimes and four baselines.

### Two regimes
- **Supervised** — domain boundaries are known (or set by a clusterer). One expert token (or
  short token path) per sub-domain. Goal: show one weight space can host multiple experts.
- **Unsupervised** — expert tokens *and* their assignment over the corpus are learned jointly
  through EM, with a regularizer that (a) pushes expert tokens apart and (b) load-balances
  expert usage across each batch. Separation is prioritized over balance.

### Baselines (all must be implemented and comparable head-to-head)
1. **Dense** — single backbone, no expert tokens (lower bound).
2. **Hard MoE** — classic top-k token-routed FFN MoE (upper-bound reference).
3. **c-BTM** — cluster → branch → train N expert *models* → merge at inference.
4. **MoP** — Mixture-of-Prompts: per-domain soft prompts but routed/averaged, no EM.
5. **Ours (fixed)** — our method with a *fixed, orthogonal, constant* token bank (ablation
   isolating "does *learning* the embeddings help?").
6. **Ours (learned)** — the full proposal, supervised and unsupervised variants.

## 2. Proposed repository structure

The coding agent should create exactly this layout. Each plan doc owns the dirs it creates.

```
soft-moe/
├── plan/                      # these docs (already exists)
├── papers/                    # reference PDFs (already exists)
├── BRAINSTORM.md
├── pyproject.toml             # package + deps (see §4)
├── README.md                  # short user-facing readme (agent writes last)
├── .gitignore                 # ignores data/, outputs/, wandb/, *.ckpt
│
├── configs/                   # all run config (yaml). NEVER hardcode hyperparams.
│   ├── data/                  # one yaml per dataset recipe
│   ├── model/                 # backbone + expert-token specs
│   ├── train/                 # optimizer, EM schedule, regularizer weights
│   ├── eval/                  # metric suites
│   └── experiment/            # top-level configs composing the above
│
├── src/softmoe/
│   ├── __init__.py
│   ├── data/                  # → 01-data-collection.md
│   │   ├── download.py
│   │   ├── clustering.py      # k-means + Cobweb clusterers behind one interface
│   │   ├── dataset.py         # torch Dataset/collator, returns (tokens, cluster_id)
│   │   └── build.py           # end-to-end: raw → sharded tokenized corpus + assignments
│   ├── models/                # → 02-training.md
│   │   ├── expert_tokens.py   # the ExpertTokenBank module (the core contribution)
│   │   ├── soft_moe.py        # wraps a HF causal LM + injects expert tokens
│   │   ├── router.py          # E-step assignment (supervised id / learned soft)
│   │   └── baselines/         # dense.py, hard_moe.py, cbtm.py, mop.py
│   ├── training/              # → 02-training.md
│   │   ├── em_trainer.py      # the EM loop (E-step assign, M-step optimize)
│   │   ├── losses.py          # LM loss + separation + load-balance regularizers
│   │   └── callbacks.py       # checkpointing, logging, early stop
│   ├── eval/                  # → 03-evaluation.md
│   │   ├── perplexity.py
│   │   ├── specialization.py  # routing acc, separation, balance, utilization
│   │   ├── harness.py         # runs a model over the metric suite
│   │   └── report.py          # aggregates runs → tables/plots
│   └── utils/                 # seeding, config loading, logging, distributed
│
├── scripts/                   # thin CLI entrypoints (argparse → calls src/)
│   ├── build_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── make_report.py
│
├── experiments/               # run outputs (gitignored): ckpts, logs, metrics.json
├── data/                      # raw + processed corpora (gitignored)
└── tests/                     # pytest: shapes, EM-step correctness, metric math
```

## 3. Build order & milestones

The agent should implement in this dependency order, with a working smoke test at each gate.

| # | Milestone | Doc | Done when |
|---|-----------|-----|-----------|
| M0 | Scaffold repo, `pyproject.toml`, config loader, seeding, CI-able `pytest` stub | this file | `pip install -e .` works; `pytest` green on empty suite |
| M1 | Data pipeline on a *tiny* multi-domain corpus | [01](01-data-collection.md) | `scripts/build_data.py --config configs/data/toy.yaml` produces sharded tokens + cluster assignments |
| M2 | `ExpertTokenBank` + `SoftMoE` wrapper; forward pass runs | [02](02-training.md) | unit test: loss is finite, grads reach expert tokens only |
| M3 | EM trainer; supervised variant overfits toy corpus | [02](02-training.md) | per-domain val ppl drops below dense baseline on toy |
| M4 | Regularizers + unsupervised EM assignment | [02](02-training.md) | experts specialize (utilization entropy < uniform, separation ↑) |
| M5 | All baselines (dense, hard-MoE, c-BTM, MoP, fixed-token) | [02](02-training.md) | each trains via the *same* `train.py` interface |
| M6 | Eval harness + specialization metrics + report | [03](03-evaluation.md) | `make_report.py` emits the comparison table across all methods |
| M7 | Scale-up run on real multi-domain corpus | all | full results table reproduced from `configs/experiment/main.yaml` |

## 4. Conventions the agent must follow

- **Stack:** Python ≥3.10, PyTorch, HuggingFace `transformers`/`datasets`/`tokenizers`,
  `scikit-learn` (k-means), `concept_formation` or a local Cobweb impl, `wandb` (optional,
  gated behind a flag), `hydra-core` *or* a thin yaml loader for config composition.
- **Config-first:** no hyperparameters in code. Everything flows from `configs/`. A run is
  fully described by one `configs/experiment/*.yaml`.
- **Reproducibility:** every run seeds torch/numpy/python, logs the resolved config and git
  SHA into its `experiments/<run-id>/` dir, and writes a single `metrics.json`.
- **One model interface:** every method (baselines included) exposes the same
  `forward(batch) -> {loss, logits, aux}` and is constructed from config, so `train.py` and
  `evaluate.py` are method-agnostic.
- **Start tiny, then scale:** every component must work on a CPU-sized toy config before any
  GPU/real-data run. Toy configs live next to real ones in each `configs/` subdir.
- **Backbone default:** `gpt2` (124M) or `EleutherAI/pythia-160m` for dev; make it a config
  knob so a larger model can be swapped in for the headline run.

## 5. Definition of done for the whole project

A single command sequence reproduces the paper's main table:
```bash
python scripts/build_data.py    --config configs/data/main.yaml
python scripts/train.py         --config configs/experiment/<method>.yaml   # for each method
python scripts/make_report.py   --runs experiments/ --out reports/main
```
The table reports per-domain and average perplexity plus specialization metrics for all six
methods, in both supervised and unsupervised regimes where applicable.
