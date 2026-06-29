# Soft-MoE — EM-discovered expert tokens in a single LLM

How can a single language-model backbone be **softly partitioned into experts** by conditioning
on a small bank of learned *expert tokens* (soft prompts), with the assignment of inputs to
experts fit by **Expectation–Maximization** — instead of training N separate expert models (MoE,
c-BTM) or adding parameter-heavy expert FFNs?

The experts share one backbone; expertise is carried by the token `e_k`, not by separate weights.
This repo implements the full research loop — **data → EM training → evaluation** — for that idea,
its two regimes (supervised / unsupervised), and five baselines, all behind one config-driven CLI.

See [`BRAINSTORM.md`](BRAINSTORM.md) for the seed idea and [`plan/`](plan/) for the full design:
[`plan/README.md`](plan/README.md) (build order), [`plan/THEORY.md`](plan/THEORY.md) (methodology,
design space, literature review), and the three stage docs
([01 data](plan/01-data-collection.md), [02 training](plan/02-training.md),
[03 evaluation](plan/03-evaluation.md)).

## Install

```bash
python -m venv .venv --system-site-packages   # reuse a system torch if present
.venv/bin/python -m pip install -e ".[dev]"
# optional extras: ".[embed]" (sentence-transformers), ".[cobweb]", ".[wandb]"
```

## Quickstart (offline, ~10s on CPU)

The `synth` recipe builds 4 near-orthogonal synthetic domains and a tiny local GPT-2 — no network,
no model download. End-to-end for the headline method:

```bash
python scripts/build_data.py  --config configs/experiment/ours_unsup.yaml
python scripts/train.py       --config configs/experiment/ours_unsup.yaml --device cpu
python scripts/evaluate.py    --run experiments/ours_unsup_seed0
python scripts/make_report.py --runs experiments --out reports/latest
```

Run every method (`dense`, `hard_moe`, `cbtm`, `mop`, `ours_fixed`, `ours_sup`, `ours_unsup`)
the same way — only the `--config` changes — then `make_report.py` emits the comparison table
(`reports/latest/main_table.md`, `results.csv`) and figures. Curated, committed run reports live
in `reports/<run>/` (e.g. [`reports/m7-raven/`](reports/m7-raven/)); `reports/latest/` is the
gitignored scratch dump for ad-hoc runs.

## Layout

```
configs/        # all hyperparameters. data/ model/ train/ eval/ experiment/ (hydra-style `defaults:`)
src/softmoe/
  data/         # download/synthetic → embed → cluster (kmeans/cobweb) → tokenize+shard → splits
  models/       # ExpertTokenBank (the contribution) + SoftMoE wrapper + Router; baselines/
  training/     # EM trainer, losses (LM + separation + load-balance + router), callbacks
  eval/         # perplexity (per-domain, oracle vs routed), specialization metrics, harness, report
  cli/          # build_data / train / evaluate / make_report  (mirrored by scripts/)
scripts/        # thin CLI entrypoints
tests/          # pytest: data, losses, model (grad flow), metric math, hermetic end-to-end smoke
experiments/    # run outputs (gitignored): checkpoints, metrics.json, resolved_config.yaml
reports/        # curated run reports (tracked, e.g. reports/m7-raven/); reports/latest/ is scratch
data/           # gitignored corpora
```

## Key ideas in code

- **`models/expert_tokens.py`** — `ExpertTokenBank`: learnable soft prompts; `trainable=False,
  init='orthogonal'` *is* the fixed-orthogonal ablation baseline.
- **`models/router.py`** — the E-step: `SupervisedRouter` (label-routed) or `SoftRouter` (amortized,
  EM-hard / EM-soft top-k).
- **`training/em_trainer.py`** — online soft-EM by default; optional periodic likelihood-based hard
  reassignment (the c-BTM-flavored discovery loop) amortized into the router.
- **`eval/specialization.py`** — routing NMI/accuracy, utilization entropy, token separation,
  expert×domain contingency, and the **swap test** (route through the wrong expert).

Every method exposes the same `forward(batch) -> {loss, logits, per_example_nll, aux}`, so
`train.py` / `evaluate.py` are method-agnostic and a run is fully described by one
`configs/experiment/*.yaml`.

## Scaling up (the headline run, M7)

`configs/data/{dev,main}.yaml` and `configs/model/backbone_{gpt2,pythia160m}.yaml` swap in real
HF corpora and pretrained backbones. The headline run trains on real multi-domain data with a
pretrained backbone over ≥3 seeds — that is GPU/cluster work. Drive it on the **MPCDF clusters**
via the Claude Code HPC skills wired into this repo (see [`CLAUDE.md`](CLAUDE.md) and
`mpcdf-hpc-skills/`): e.g. `/toolbox:setup-repo`, `/data:get-dataset`, `/training:submit`,
`/slurm:monitor`. `/ptmp` is purged — persist keeper checkpoints to archive/object storage.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q     # 27 hermetic tests, ~5s, no network
```
