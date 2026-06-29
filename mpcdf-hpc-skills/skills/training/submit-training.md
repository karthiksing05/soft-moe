---
name: training-submit
description: >
  Generate and submit a SLURM training job (single-GPU or single-node multi-GPU
  via torchrun/accelerate) on Raven or Dais, then track it to completion. Use
  when the user says "train [model] on raven/dais", "submit a training run",
  "launch the fine-tune", "start the EM training", "kick off a sweep", or
  "run train.py on the cluster". Builds the sbatch script from the repo's train
  entrypoint + the cluster resource block, submits, polls, and reports.
metadata:
  version: "1.0.0"
---

# Skill: training-submit

Build a SLURM batch script for a training run, submit it, and monitor it.
Targets **Raven** (A100, small-scale / single-GPU sweeps) and **Dais**
(8× H200, full-scale distributed). **Not** Viper-GPU — that ROCm cluster is for
inference only in this project. Operates over SSH from the laptop.

## Inputs
- `REPO_PATH`, `SSH_ALIAS`, `CLUSTER` (`raven` or `dais`), `CLUSTER_USER`.
- `TRAIN_ENTRYPOINT`: training script path relative to `REPO_PATH` (resolved in Step 1).
- `ACCOUNT`: value for `#SBATCH -A` (asked once if not set).

## Output
On success: `JOB_ID`, `JOB_STATUS=COMPLETED`, `RUN_NAME`, `LOG_PATH`, `RUN_DIR`.
On failure: `JOB_STATUS=FAILED|SUBMISSION_ERROR` + last 30 log lines.

## Security protocol
Caller's protocol; default `.claude/security_protocols/controlled.md`. Also:
- NEVER push to git, modify `.env`/configs, or cancel a running job without explicit confirmation.
- Writing the generated sbatch script counts as a file write — confirm under `controlled`/`navigating`.

## Reference
`reference/SLURM-REFERENCE.md` (skeleton, per-cluster resource blocks, distributed launch, graceful drain) and `reference/MPCDF-CLUSTER-FACTS.md` (gres, partitions, modules).

> Cluster guard: if `CLUSTER=viper`, stop — "Viper-GPU is ROCm inference only in this project; train on Raven (A100) or Dais (H200)." Always confirm live partition/gres with `sinfo -s` and `module avail` before relying on the values below.

---

## Step 1: Resolve entrypoint, run name, account, resources

Verify the connection:
```bash
ssh -O check <SSH_ALIAS> 2>&1
```
No `Master running` → stop, tell the user to log in first.

Find the training entrypoint if not provided:
```bash
ssh <SSH_ALIAS> "find <REPO_PATH> -maxdepth 3 -name '*.py' \( -name 'train*.py' -o -path '*train*' \) ! -path '*/.git/*' 2>/dev/null | head -10"
```
One clear match → use it. Multiple → numbered menu. None → ask the user for the path. Store as `TRAIN_ENTRYPOINT`.

Ask once for any of these not already known: `RUN_NAME` (default: `<reponame>-<date>`), `ACCOUNT` (`#SBATCH -A`), number of GPUs, and any script args (e.g. `--config ...`). Under `automated`, use sensible defaults and only stop if `ACCOUNT` is genuinely unknown — never submit under a guessed account.

Pick the resource block from `reference/SLURM-REFERENCE.md`:
- `raven`: `--gres=gpu:a100:<N>` (N=1..4), `--cpus-per-task=18`.
- `dais`: `--partition=gpu --gres=gpu:h200:<N>` (N=1..8), `--cpus-per-task=12`, `--mem=0` for a full node.

Confirm live availability (informational, do not block on a transient busy state):
```bash
ssh <SSH_ALIAS> "sinfo -s 2>/dev/null | head -20"
```

## Step 2: Generate the sbatch script

Compose the script (substitute real values; one task per GPU for distributed):
```bash
#!/bin/bash -l
#SBATCH -J <RUN_NAME>
#SBATCH -o <REPO_PATH>/.log/%x_%j.out
#SBATCH -e <REPO_PATH>/.log/%x_%j.err
#SBATCH -D <REPO_PATH>
#SBATCH -A <ACCOUNT>
#SBATCH --time=24:00:00
#SBATCH --nodes=1
<cluster resource block: gres / partition / ntasks-per-node=<N> / cpus-per-task / mem>
#SBATCH --signal=B:USR1@300        # warn 5 min before kill for graceful drain

module purge
module load <python-waterboa/<ver>> <cuda/<ver>>      # CUDA stack; see cluster facts
export PYTHONPATH=.

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29500
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$SLURM_GPUS_ON_NODE \
  --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  <TRAIN_ENTRYPOINT> <script args>
```
For a single-GPU run, drop torchrun and call `srun python <TRAIN_ENTRYPOINT> <args>`.
For `accelerate`, replace the launch line with `srun accelerate launch --num_processes $SLURM_GPUS_ON_NODE <TRAIN_ENTRYPOINT> <args>`.

Show the full script. Under `controlled`/`navigating`, ask `Write and submit this script? [y/n]`. Under `automated`, proceed.

Ensure the log dir exists, then write the script to the repo:
```bash
ssh <SSH_ALIAS> "mkdir -p <REPO_PATH>/.log"
cat << 'SBATCH_EOF' | ssh <SSH_ALIAS> "cat > <REPO_PATH>/.slurm/train_<RUN_NAME>.sbatch"
<script content>
SBATCH_EOF
```
(Create `<REPO_PATH>/.slurm` first with `mkdir -p` if missing.)

## Step 3: Submit
```bash
ssh <SSH_ALIAS> "cd <REPO_PATH> && sbatch .slurm/train_<RUN_NAME>.sbatch 2>&1"
```
Extract `Submitted batch job <ID>` → `JOB_ID`. No id → show output, `JOB_STATUS=SUBMISSION_ERROR`, stop.

## Step 4: Track (PENDING is normal)
Timestamped status log. First check no delay, then prepend `sleep 30 && `:
```bash
ssh <SSH_ALIAS> "squeue -j <JOB_ID> -h -o '%T %R' 2>/dev/null || echo NOT_IN_QUEUE"
```
`RUNNING` → tail the log each cycle:
```bash
ssh <SSH_ALIAS> "tail -n 60 <REPO_PATH>/.log/<RUN_NAME>_<JOB_ID>.out 2>/dev/null"
```
Watch for `CUDA out of memory` / `Traceback` → surface and ask whether to `scancel`. Every 20 PENDING polls (~10 min), ask whether to keep waiting. `NOT_IN_QUEUE` after RUNNING → finished → Step 5.

## Step 5: Outcome report
```bash
ssh <SSH_ALIAS> "sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS,ExitCode 2>/dev/null | head -5"
```
```
──────────────────────────────────────────
Job ID:   <JOB_ID>   Status: COMPLETED | FAILED
Run:      <RUN_NAME>   Cluster: <CLUSTER>
Log:      <REPO_PATH>/.log/<RUN_NAME>_<JOB_ID>.out
Outputs:  <RUN_DIR — e.g. /ptmp/<user>/runs/<RUN_NAME>>
──────────────────────────────────────────
```
**FAILED** → include the last 30 lines of the `.out`/`.err` logs; `JOB_STATUS=FAILED`.
Remind the user: `/ptmp` is purged and not backed up — copy keeper checkpoints to archive or object storage (see `skills/data/transfer.md`).

## Orchestration contract
```
JOB_ID=<value>
JOB_STATUS=COMPLETED|FAILED
RUN_NAME=<value>
LOG_PATH=<REPO_PATH>/.log/<RUN_NAME>_<JOB_ID>.out
RUN_DIR=<value>
```
