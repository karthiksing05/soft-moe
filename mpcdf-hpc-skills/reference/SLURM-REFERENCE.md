# SLURM Reference (shared)

Job-template building blocks used by the `training` and `inference` skills.
Cluster-specific values (gres, partitions, modules) come from
`reference/MPCDF-CLUSTER-FACTS.md`. Always confirm live values with `sinfo -s`
and `module avail` before submitting.

## Skeleton

```bash
#!/bin/bash -l
#SBATCH -J <job-name>
#SBATCH -o <repo>/.log/%x_%j.out      # %x=name, %j=jobid
#SBATCH -e <repo>/.log/%x_%j.err
#SBATCH -D ./                          # working dir
#SBATCH -A <your-account>
#SBATCH --time=24:00:00                # 24h cap (verify)
#SBATCH --mail-type=none
# --- resource block: see per-cluster examples below ---

module purge
module load <python/cuda/rocm/apptainer ...>
export PYTHONPATH=.
srun <command>
```

## Per-cluster GPU resource blocks

**Raven (A100 40 GB, 4/node):**
```bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4         # 1–4
#SBATCH --ntasks-per-node=4       # one task per GPU for distributed
#SBATCH --cpus-per-task=18        # 72 cores / 4
```
Interactive debug (15 min): `srun --time=00:15:00 --partition=gpudev --gres=gpu:a100:1 --pty bash`

**Dais (H200 ~140 GB, 8/node):**
```bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8         # 1–8; use 8 + --mem=0 for a full node
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12        # ~96 cores / 8
#SBATCH --mem=0
```

**Viper-GPU (MI300A, 2/node, ROCm):**
```bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2              # 1–2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=24        # 48 cores / 2; keeps task near its APU socket
module load rocm/7.1
```

## Distributed training launch (single node, multi-GPU)

Prefer `torchrun` or `accelerate launch`. One process per GPU:

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29500
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$SLURM_GPUS_ON_NODE \
  --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  scripts/train.py <args>
```

Multi-node: set `--nnodes=$SLURM_NNODES` and let `srun` start one launcher per
node; `torchrun` handles per-node local ranks. For `accelerate`, generate/commit
a config and call `accelerate launch --num_processes ...`.

## Graceful drain (long workers)

Design workers to stop cleanly before walltime instead of dying mid-item:
- Pass a budget such as `--min-remaining-hours 0.5`; the worker loop checks
  elapsed vs `$SLURM_JOB_END_TIME` and exits between items when the margin is
  hit, flushing partial results first.
- Trap `SIGTERM`/`SIGUSR1` (Slurm can be told to send a warning signal before
  the kill with `#SBATCH --signal=B:USR1@300`) and checkpoint on it.

## Monitoring

- Queue state: `squeue -j <id> -h -o '%T %R'` → `PENDING <reason>` (normal on a
  busy cluster), `RUNNING`, or absent (finished/failed).
- Live log: `tail -n 80 <repo>/.log/<job>_<id>.out`.
- Finished accounting: `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode`.
- Estimated start: `squeue -j <id> --start`.
