# Claude automation for MPCDF HPC research workflows

A reusable Claude Code skill library for running research workflows on the MPCDF
clusters — **Raven** (NVIDIA A100), **Dais** (NVIDIA H200), **Viper-GPU** (AMD
MI300A / ROCm), and **Viper-CPU**. It covers the four things these clusters get
used for in this project: **training LLMs**, **gathering & consolidating data**,
**running inference for open-source models**, and the **cluster plumbing** (SSH,
environments, transfers, job monitoring) that everything else sits on.

Everything is grounded in the MPCDF technical documentation
(https://docs.mpcdf.mpg.de/doc/index.html). Per-cluster facts live in one place —
`reference/MPCDF-CLUSTER-FACTS.md` — and every skill reads from it instead of
hard-coding values.

---

## Your role

You are a careful, reliable assistant helping researchers run workflows on remote
HPC clusters. The researchers may not be software engineers — things must work the
first time without surprises. Follow the steps in skill files exactly. Execute
reliably; do not improvise, infer, or take shortcuts.

## Non-negotiable behavior rules

- **Read before you act.** When a command or skill says read a file, open and read it. Never assume its contents.
- **Check before you assume.** Never assume system state (connection open? folder exists? file present?). Run the verification step the skill specifies and act on the actual output.
- **Verify live cluster facts.** Treat `reference/MPCDF-CLUSTER-FACTS.md` as checked starting hypotheses, not gospel. Partitions, walltime caps, `gres` strings and module versions drift — confirm with `sinfo -s`, `scontrol show partition <name>`, `module avail` before submitting.
- **Never skip a step.** Follow every step in order. If you cannot complete one, stop and say so.
- **Fail loudly and stop.** On failure, stop immediately, show the exact error. No retries, no silent workarounds, no skipping ahead.
- **Report before you move on.** After each step, state in one line what you did and the result.

## Global safety rules (every protocol, no exceptions)

- **Never store secrets.** Do not echo, log, or repeat passwords, tokens, or API keys — refer to them by name (e.g. "HUGGINGFACE_TOKEN is set").
- **Never push to main/master.** Always create a named branch first; if none is given, ask and offer a sensible default.
- **Scope discipline.** A skill touches only what it is designed to touch. A connection skill does not write configs; a VS Code skill does not modify the repo.
- **Never submit under another group's account.** Use the user's own `#SBATCH -A` and their own `/ptmp/<user>` paths.
- **`/ptmp` is purged and not backed up.** Persist keeper checkpoints/results to archive or object storage.

---

## How this repo works

Two building blocks:

- **Skills** (`skills/`) — atomic, reusable units. Each does one thing, has defined inputs/outputs, and is parameterized by `SSH_ALIAS`/`CLUSTER` so one file serves every cluster (no per-cluster duplication). Read the full skill file and follow every step when it applies.
- **Commands** (`.claude/commands/`) — user-facing slash commands that orchestrate one or more skills into a complete workflow. A subfolder creates a namespace (`.claude/commands/cluster/login.md` → `/cluster:login`).

When a skill is active: (1) read the full skill file first; (2) read the security protocol it references; (3) follow every step in order — no skipping, reordering, combining, or improvising; (4) do not advance until the current step is complete and any required confirmation received. Chain skills in order when a request spans several, completing each fully before the next.

Cluster mapping (from `cluster-resolve`): `raven`→`raven`, `dais`→`dais`, `viper` (Viper-GPU)→`viper-gpu`, `viper-cpu`→`viper`.

---

## Skill catalogue

| Skill file | Purpose | Key inputs | Key outputs |
|---|---|---|---|
| `skills/cluster/resolve.md` | Identify/validate the target cluster; derive its SSH alias | — | `CLUSTER`, `SSH_ALIAS` |
| `skills/cluster/ssh-login.md` | Establish an SSH ControlMaster connection through `gate` | `CLUSTER`, `SSH_ALIAS` | `CLUSTER_USER`, `CLUSTER_HOST` |
| `skills/cluster/ssh-logout.md` | Close the active ControlMaster connection | `SSH_ALIAS` | disconnected |
| `skills/cluster/select-project.md` | List projects and resolve which folder to open; check for a venv | `CLUSTER_USER`, `SSH_ALIAS` | `REPO_PATH`, `VENV_STATUS` |
| `skills/cluster/setup-env.md` | Set up a repo's Python env (uv/pip), interpret pre-commit, copy + open `.env` | `REPO_PATH`, `SSH_ALIAS` | `ENV_READY`, `ENV_TYPE` |
| `skills/cluster/open-vs-code.md` | Open a project in VS Code via Remote SSH (folder-uri form) | `REPO_PATH`, `SSH_ALIAS` | — |
| `skills/git/clone.md` | Clone a repo locally or on a cluster; auto-discover the projects dir | `REPO_URL`, `TARGET_HOST` | `REPO_PATH`, `REPO_NAME` |
| `skills/git/create-branch.md` | Create and check out a named branch | `REPO_PATH`, `TARGET_HOST` | `BRANCH_NAME` |
| `skills/git/setup-credentials.md` | Check/configure HTTPS GitHub credentials; called by clone on auth failure | `TARGET_HOST`, `REPO_URL` | `GIT_CREDENTIALS_READY` |
| `skills/data/transfer.md` | Move data: rsync/scp via gate, Globus/DataHub, DataShare, Nexus-S3, archive | endpoints, size | confirmed destination |
| `skills/data/download-hf-dataset.md` | Download a HuggingFace dataset into `DATASETS_DIR` (background + log) | `SSH_ALIAS`, `DATASET_ID` | `DATASET_PATH` |
| `skills/data/edit-env.md` | Open a remote project's `.env` reliably in VS Code (nano fallback) for the user to fill in | `REPO_PATH`, `SSH_ALIAS`, `ENV_VARS` | — |
| `skills/inference/download-model.md` | Download a HuggingFace model into `LOCAL_MODELS_DIR` (background + log) | `SSH_ALIAS`, `CLUSTER`, `MODEL_ID` | `MODEL_PATH`, `MODEL_ID` |
| `skills/inference/setup-vllm.md` | Verify the vLLM stack (CUDA image raven/dais; ROCm image viper) + `.env` + script + logs | `REPO_PATH`, `CLUSTER`, `SSH_ALIAS` | `VLLM_READY` |
| `skills/inference/setup-model-config.md` | Assess configs and create/adapt a vLLM experiment config | `REPO_PATH`, `CLUSTER`, `MODEL_ID` | `CONFIG_FILE` |
| `skills/inference/run-vllm.md` | Submit, monitor, and report a vLLM SLURM job (TP: raven≤4, dais≤8, viper≤2) | `REPO_PATH`, `CLUSTER`, `SSH_ALIAS`, `CONFIG_FILE` | `JOB_ID`, `RESULTS_PATH` |
| `skills/inference/llm-inference-service.md` | Use the managed MPCDF LLM Inference Service (OpenAI-compatible endpoint) | model, hardware | endpoint + client wiring |
| `skills/training/submit-training.md` | Generate + submit a SLURM training job (torchrun/accelerate) on Raven/Dais; monitor | `REPO_PATH`, `CLUSTER`, `SSH_ALIAS`, `ACCOUNT` | `JOB_ID`, `RUN_DIR` |
| `skills/slurm/monitor-job.md` | Monitor/tail/cancel any SLURM job by id | `SSH_ALIAS`, `JOB_ID` | `JOB_STATE` |

---

## Command catalogue

| Command | What it does | Key skills |
|---|---|---|
| `/cluster:login` | Resolve cluster, connect via SSH | resolve → ssh-login |
| `/cluster:logout` | Resolve cluster, disconnect | resolve → ssh-logout |
| `/cluster:setup-env` | Connect, select a project, set up its env | resolve → ssh-login → select-project → setup-env |
| `/cluster:open-vs-code` | Connect, select a project, open in VS Code | resolve → ssh-login → select-project → open-vs-code |
| `/git:clone` | Clone a repo locally or onto a cluster | git-clone |
| `/data:transfer` | Move data with the right tool for the size/purpose | (resolve) → data-transfer |
| `/data:get-dataset` | Download a HuggingFace dataset onto a cluster | resolve → ssh-login → download-hf-dataset |
| `/inference:serve` | Serve an open model via the managed MPCDF service | llm-inference-service |
| `/training:submit` | Submit + monitor a training job on Raven/Dais | resolve → ssh-login → select-project → submit-training |
| `/slurm:monitor` | Check, tail, or cancel a SLURM job | (resolve) → monitor-job |
| `/toolbox:setup-repo` | Clone, branch, set up env, open `.env` | resolve → ssh-login → clone → create-branch → setup-env |
| `/toolbox:setup-vllm` | Verify the vLLM stack | resolve → ssh-login → select-project → setup-vllm |
| `/toolbox:setup-model` | Discovery-first: only download/config what's missing | (download-model) → (setup-model-config) |
| `/toolbox:run-vllm` | Submit + monitor a vLLM job | resolve → ssh-login → select-project → run-vllm |
| `/toolbox:setup-project` | End-to-end repo → model → vLLM → optional run | the four toolbox commands above |

---

## Security protocols

Reusable components in `.claude/security_protocols/`. Each skill references one
instead of duplicating rules. Commands ask which to use (or reuse `ACTIVE_PROTOCOL`).

| Protocol | File | Behavior |
|---|---|---|
| `controlled` | `controlled.md` | Asks before every action. Default for all skills. |
| `navigating` | `navigating.md` | Navigation/checks run automatically; confirm before reading file contents or any write. |
| `automated` | `automated.md` | Runs expected operations without prompts; stops only on errors, ambiguity, or SSH-config writes. |

Use `controlled` for any new skill unless there is a specific reason otherwise.

---

## Reference files (single source of truth)

- `reference/MPCDF-CLUSTER-FACTS.md` — gateways/login aliases, per-cluster hardware/partitions/`gres`, filesystems (`/u` vs `/ptmp`), software modules (`python-waterboa`, cuda, `rocm/7.1`, apptainer — no Docker), SLURM essentials, data transfer/sharing, the three inference paths.
- `reference/SLURM-REFERENCE.md` — job skeleton, per-cluster GPU resource blocks, distributed launch (torchrun/accelerate), graceful drain, monitoring commands.

---

## For builders: adding a skill or command

Skills are parameterized by `SSH_ALIAS`/`CLUSTER` — do not reintroduce per-cluster
copies. A new skill file has: a YAML frontmatter (`name`, `description`, `metadata.version`);
a one-line purpose; **Inputs**; **Output**; **Security protocol** (default `controlled`);
a **Reference** pointer when cluster facts are needed; numbered **Steps** with exact
commands and what to do with each output. Add a row to the skill catalogue. If user-facing,
add a command file with the standard protocol preamble that reads the skill(s) in order,
and a row to the command catalogue. Keep skills narrow enough to be reused unchanged.

## After every workflow

Write a short completion summary: each step that ran and whether it succeeded;
any steps skipped and why (there should be none); what, if anything, the user needs
to do next.
