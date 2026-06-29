# MPCDF HPC Skills for Claude Code

A unified Claude Code skill library for research workflows on the MPCDF clusters —
**Raven** (NVIDIA A100), **Dais** (NVIDIA H200), **Viper-GPU** (AMD MI300A / ROCm),
and **Viper-CPU**. Claude handles SSH, environment setup, data movement, job
submission, and monitoring; you confirm the steps that matter.

It covers what these clusters are used for here:

- **Train LLMs** — `/training:submit` (single-GPU sweeps on Raven, 8× H200 distributed on Dais).
- **Gather & consolidate data** — `/data:transfer`, `/data:get-dataset` (rsync/Globus/DataShare/Nexus-S3 + HuggingFace datasets onto `/ptmp`).
- **Run open-model inference** — `/toolbox:run-vllm` (self-hosted vLLM via Slurm + Apptainer) or `/inference:serve` (managed MPCDF LLM Inference Service).
- **Cluster plumbing** — `/cluster:login`, `/cluster:setup-env`, `/cluster:open-vs-code`, `/git:clone`, `/slurm:monitor`.

Everything is grounded in the MPCDF docs (https://docs.mpcdf.mpg.de/doc/index.html).
Per-cluster facts live in `reference/MPCDF-CLUSTER-FACTS.md`; skills read from there
rather than hard-coding values — and re-verify live (`sinfo -s`, `module avail`)
before submitting.

## Prerequisites

- [Claude Code](https://claude.ai/code) installed.
- An MPCDF account with access to the target cluster(s); 2FA configured for the gateway.
- VS Code with the Remote-SSH extension (optional, for `/cluster:open-vs-code`).

## Quick start

1. Install the library — see [INSTALL.md](INSTALL.md).
2. Open Claude Code in this directory.
3. Connect once, then run a workflow:
   ```
   /cluster:login            # resolves the cluster and opens an SSH ControlMaster session
   /toolbox:setup-project    # clone → model → vLLM → optional run
   ```
   or run a single capability directly, e.g. `/training:submit`, `/data:get-dataset`,
   `/toolbox:run-vllm`, `/slurm:monitor`.

## Layout

```
CLAUDE.md                     orchestration guide (roles, rules, catalogues)
.claude/
  settings.json               Bash allowlist
  commands/                   slash commands (cluster / git / data / inference / training / slurm / toolbox)
  security_protocols/         controlled · navigating · automated
reference/
  MPCDF-CLUSTER-FACTS.md      single source of truth for cluster values
  SLURM-REFERENCE.md          job skeletons + per-cluster resource blocks
skills/
  cluster/  git/  data/  inference/  training/  slurm/
```

See [CLAUDE.md](CLAUDE.md) for the full skill and command catalogues.
