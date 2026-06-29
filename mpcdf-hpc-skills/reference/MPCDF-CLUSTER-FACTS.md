# MPCDF Cluster Facts (source of truth)

This file is the single authoritative reference for cluster-specific values used
by every skill in this library. Skills read values from here instead of
hard-coding them. Grounded in the MPCDF technical documentation
(https://docs.mpcdf.mpg.de/doc/index.html), verified June 2026.

> **Golden rule (from the thesis operating manual): verify live facts before
> baking them into a job.** Partition names, walltime caps, `gres` strings and
> module versions change. Treat every value below as a *checked starting
> hypothesis*, and confirm against the running scheduler before submitting:
> `sinfo -s`, `scontrol show partition <name>`, `module avail`. The most stable
> anchors are the GPU types (A100 on Raven, H200 on Dais, MI300A on Viper-GPU).

---

## Access — gateways and login nodes

All clusters sit behind login-only SSH gateways. There is **no module
environment on the gateways** — you cannot compile or run there.

| Item | Value |
|---|---|
| Gateway alias | `gate.mpcdf.mpg.de` (round-robins to `gate1`/`gate2`) |
| Gateway 2FA | **Enforced.** Password + OTP required. SHA256 key-exchange only. |
| Gateway $HOME | Tiny, local to each gate, not the cluster home. |
| Gate reboots | gate1 Tue 03:45, gate2 Sat 03:45 (German time) → sessions ≤ 7 days |

Login node aliases (each round-robins across that cluster's login nodes — prefer
the alias over a pinned node like `raven01i`):

| Cluster | SSH login alias | Reach via |
|---|---|---|
| Raven | `raven.mpcdf.mpg.de` | `ProxyJump gate` |
| Viper-CPU | `viper.mpcdf.mpg.de` | `ProxyJump gate` |
| Viper-GPU | `viper-gpu.mpcdf.mpg.de` (confirm against the Viper-GPU guide; some setups reach GPU login nodes via `viper`) | `ProxyJump gate` |
| Dais | `dais.mpcdf.mpg.de` | `ProxyJump gate` |

Recommended `~/.ssh/config` block (ControlMaster keeps the submit/rsync loop
fast — note that a stale ControlMaster socket can hang later ssh; delete the
socket file to recover):

```
Host gate
    HostName gate.mpcdf.mpg.de
    User <MPCDF-USERNAME>
    ServerAliveInterval 120
    ControlMaster auto
    ControlPath ~/.ssh/cm_sockets/%r@%h:%p
    ControlPersist 4h

Host raven viper viper-gpu dais
    HostName %h.mpcdf.mpg.de
    User <MPCDF-USERNAME>
    ProxyJump gate
    ControlMaster auto
    ControlPath ~/.ssh/cm_sockets/%r@%h:%p
    ControlPersist 4h
```

(`HostName %h.mpcdf.mpg.de` expands `viper-gpu` → `viper-gpu.mpcdf.mpg.de`. If a
cluster's real login hostname differs, give it its own `Host` block.)

---

## Per-cluster hardware, partitions, gres

### Raven (NVIDIA, CUDA) — `CLUSTER=raven`, `SSH_ALIAS=raven`
- GPU nodes: 192 × (2× Intel Xeon IceLake-SP 8360Y, 72 cores, 512 GB RAM, **4× NVIDIA A100-SXM4 40 GB**, NVLink).
- gres string: `--gres=gpu:a100:<1-4>`
- Interactive/dev GPU partition: `--partition=gpudev --gres=gpu:a100:1` (15-min cap). For batch GPU jobs, request `--gres=gpu:a100:N`; verify the batch partition name with `sinfo`.
- High-bandwidth IB nodes: `--constraint="gpu-bw"`.
- Debug QoS: add `--qos=debug` for higher priority, 15-min limit.
- Best for: CUDA training (single-GPU and up to 4-GPU/node), CUDA inference (vLLM TP≤4), ASR/diarisation/voice-embedding (pyannote/WhisperX) — these need **cuDNN 8** (bundle it and set `LD_LIBRARY_PATH`).

### Dais (NVIDIA H200, CUDA) — `CLUSTER=dais`, `SSH_ALIAS=dais`
- GPU nodes: **8× NVIDIA H200 (~140 GB)** per node, ~96 CPU cores (12/GPU), ~2 TB RAM.
- gres string: `--gres=gpu:h200:<1-8>`. `--mem=250000` ≈ 1/8 node (per GPU); `--mem=0` = whole node.
- Partition: `gpu`. Exclusive full node = request all 8 GPUs + `--mem=0`.
- Default Python module: `python-waterboa`.
- Container module: `apptainer/1.4.1`.
- Best for: large/multi-GPU CUDA training (the **8× H200 full-scale target**), big-model inference (vLLM TP up to 8), LLM-judge workloads.

### Viper-GPU (AMD MI300A, ROCm) — `CLUSTER=viper`, `SSH_ALIAS=viper-gpu`
- GPU nodes: **2× AMD MI300A APUs** per node. Each APU = 24 CPU cores + 1 GPU on one socket (48 cores, 2 GPUs/node).
- gres string: `--gres=gpu:<1-2>`. Place tasks topologically close to their GPU: e.g. `--ntasks-per-node=2 --cpus-per-task=1 --gres=gpu:2` puts task 0 on socket 0/GPU 0 and task 1 on socket 1/GPU 1.
- ROCm modules: `rocm/7.0`, `rocm/7.1`.
- Optional fast node-local scratch via NVMe-over-Fabrics (attach with Slurm flags; see the Viper-GPU guide).
- Best for: **ROCm inference only** for this project's stack (vLLM **TP≤2** with the ROCm image). The CUDA-only ASR/diarisation/torch-CUDA stack does **not** run here.

### Viper-CPU (AMD CPU) — `CLUSTER=viper-cpu`, `SSH_ALIAS=viper`
- CPU-only AMD nodes. Best for: CPU-bound data work (blocking, metadata mining, prep) when you don't want to burn a GPU allocation.

> **CUDA vs ROCm split (critical):** the ASR / diarisation / voice-embedding /
> torch-CUDA stack is NVIDIA-only → Raven or Dais. Viper-GPU (ROCm) is for LLM
> **inference** only in this project.

---

## File systems (GPFS)

| Path | Role | Properties |
|---|---|---|
| `/u/<user>` | Home | GPFS, fast, **backed up**, limited quota. Code, configs, small artefacts. |
| `/ptmp/<user>` | Scratch | GPFS, fast, large. **NOT backed up. Purged.** Model weights, datasets, checkpoints, job logs. |
| `$JOB_TMPDIR`, `$JOB_SHMTMPDIR` | Node-local | Per-Slurm-job, for apps that need true node-local storage. |
| Never use | `/tmp`, `$TMPDIR`, `/dev/shm` directly | Use `/ptmp` for shared scratch instead. |

Cluster-specific scratch mounts also exist (e.g. `/raven/ptmp/<user>`,
`/viper/ptmp/<user>`). `/ptmp/<user>` is the portable choice; confirm the exact
mount on each cluster. **Persist anything you care about off `/ptmp`** (your own
object store / archive) — it is purged.

Recommended layout used by these skills:
- `LOCAL_MODELS_DIR = /ptmp/<user>/models` — HuggingFace model weights.
- `DATASETS_DIR = /ptmp/<user>/datasets` — downloaded datasets.
- Repos / code → `/u/<user>/projects/<repo>` (small, backed up).
- Run outputs / checkpoints → `/ptmp/<user>/runs/<experiment>` (then copy keepers to archive/object store).

---

## Software stack

| Need | Module / tool |
|---|---|
| Python | `module load python-waterboa/<ver>` (the current default; **Anaconda is deprecated** for licensing reasons). |
| CUDA (Raven/Dais) | `module load cuda/<ver>` |
| ROCm (Viper-GPU) | `module load rocm/7.1` (or `7.0`) |
| Containers | `module load apptainer` (Apptainer/Singularity; **Docker is not available** on the clusters) |
| Per-project env | `uv` (preferred when `uv.lock` present) or `pip` venv at `<repo>/.venv` |
| Audio decode | `module load ffmpeg/7.1` (torchaudio cannot decode MP3 otherwise) |

Set `export PYTHONPATH=.` from the repo root inside job scripts (interactive
shells do not set it).

---

## SLURM essentials

- **Walltime:** general cap is **24 h**. Longer runs need the documented long-job handling; design every long worker to **drain gracefully** before walltime rather than die mid-item.
- **Login nodes enforce resource limits** — no heavy work on login nodes. Push everything through `sbatch`. `srun`/`salloc` interactive sessions are for debugging only.
- **Account:** set `#SBATCH -A <your-account>`. Never submit under another group's account.
- **Order in a script:** shebang `#!/bin/bash -l`, then `#SBATCH` directives, then `module purge` + `module load …`, then `srun …`.
- Useful: `squeue --me`, `squeue -j <id> -h -o '%T %R'`, `scontrol show job <id>`, `sacct -j <id>`, `sinfo -s`, `scancel <id>`.

---

## Containers (Apptainer)

- Supported runtime: **Apptainer/Singularity** (Charliecloud also available). **No Docker.** Convert Docker images: `apptainer pull out.sif docker://<image>`.
- Build/pull on a login node with `module load apptainer` (pulls are network + I/O, generally fine on login nodes; heavy conversions may belong in a job).
- Run on GPUs: CUDA → `apptainer exec --nv <image.sif> …`; ROCm (Viper-GPU) → `apptainer exec --rocm <image.sif> …`.
- vLLM images: a CUDA image (Raven/Dais) and a ROCm image (Viper-GPU). **Reuse prebuilt images; do not rebuild vLLM on a login node.**

---

## Data transfer & sharing

| Method | When |
|---|---|
| `rsync`/`scp`/`sftp` via `ProxyJump gate` | Small–medium transfers, repo deploys. `rsync -av -e 'ssh -J <user>@gate.mpcdf.mpg.de' src/ <user>@raven.mpcdf.mpg.de:/ptmp/<user>/dst/` |
| **Globus Online** / MPCDF DataHub | Large-scale, resumable, scheduled transfers and staging to HPC. |
| **Nexus-S3** object storage | S3-compatible store for derived artefacts / publishing (opt-in via SelfService). |
| **DataShare** (Nextcloud) | Web sync-and-share for smaller datasets / files / external collaborators. |
| `archive.mpcdf.mpg.de` | Tape archive for long-term retention of keepers. |

For HuggingFace models/datasets, download **directly on the cluster** into
`/ptmp` (the compute side has internet on login nodes); see the data skills.

---

## Running open-model inference — three paths

1. **MPCDF LLM Inference Service** — `https://llm.mpcdf.mpg.de`. A managed web app
   that submits the Slurm job and routes an **OpenAI-compatible REST endpoint**
   for you, using vLLM or Ollama. Connected to **Dais and Viper-GPU**. Log in
   with Kerberos credentials; you must have logged directly into the target HPC
   system at least once first. Best for **interactive** evaluation / user
   studies. For non-interactive benchmarks/offline eval, MPCDF recommends Slurm
   batch jobs (example scripts in the *LLMs-meet-MPCDF* GitLab repo).
2. **Self-hosted vLLM via Slurm + Apptainer** — full control, best for batch /
   offline evaluation pipelines. This is what the `inference` skills automate.
3. **GWDG Chat AI** — external alternative for "standard" open models, with a
   chat UI and an inference API.

---

## Accounts / identifiers

- Use **your own** `--account` / `#SBATCH -A` string and **your own** scratch
  paths. Identifiers seen in handoffs from other groups (e.g. `mpib_gpu`,
  `mpib_inst`, `/raven/ptmp/<someone>/…`) are examples — never submit under
  another group's account.
- No Raven/Dais/Viper access yet? Apply via the MPCDF helpdesk / SelfService.
