---
name: inference-run-vllm
description: >
  Submit, monitor, and report a vLLM inference SLURM job on Raven, Dais, or
  Viper-GPU; track it to completion and hand off a structured outcome. Use when
  the user says "run [model] on raven/dais/viper", "submit the [config] job",
  "launch inference for [experiment]", "monitor the vllm job", or "check if the
  job is done". Resolves config + cluster + tensor-parallel size from context.
metadata:
  version: "3.0.0"
---

# Skill: inference-run-vllm

Submit a vLLM job to SLURM, track PENDING→RUNNING→done, report a structured
outcome. Operates over SSH from the laptop.

## Inputs
- `REPO_PATH`, `SSH_ALIAS`, `CLUSTER`, `CLUSTER_USER`.
- `CONFIG_FILE`: YAML under `configs/`, relative to `REPO_PATH`; resolved in Step 1.

## Output
On success: `JOB_ID`, `JOB_STATUS=COMPLETED`, `EXPERIMENT_NAME`, `LOG_PATH`, `RESULTS_PATH`, `CONFIG_FILE`.
On failure: `JOB_STATUS=FAILED|SUBMISSION_ERROR` + `ERROR_SUMMARY` (last 20 inference + 10 vLLM log lines).

## Security protocol
Caller's protocol; default `.claude/security_protocols/controlled.md`. Also:
- NEVER modify `.env`/configs; NEVER cancel a running job without explicit confirmation.

## Reference
`reference/MPCDF-CLUSTER-FACTS.md` (Per-cluster GPU → tensor-parallel limits) and `reference/SLURM-REFERENCE.md`.

> Tensor-parallel ceiling per single node: Raven TP≤4 (A100×4), Dais TP≤8 (H200×8), Viper-GPU TP≤2 (MI300A×2). Size `--tp` to fit the model on one node.

---

## Step 1: Resolve CONFIG_FILE + validate prerequisites
If both provided (pipeline mode), skip resolution. Else, if the user named a model/experiment, find a matching vllm config:
```bash
ssh <SSH_ALIAS> "find <REPO_PATH>/configs -name '*.yaml' | xargs grep -l 'model_provider: vllm' 2>/dev/null"
```
Partial-match the name; multiple → numbered menu; none → list all vllm configs and ask.

Validate prerequisites:
```bash
# Container (CUDA on raven/dais; ROCm on viper)
ssh <SSH_ALIAS> "test -f <REPO_PATH>/container_images/vllm-openai.sif -o -f <REPO_PATH>/container_images/vllm-rocm.sif && echo SIF_FOUND || echo SIF_MISSING"
ssh <SSH_ALIAS> "cd <REPO_PATH> && grep -s 'LOCAL_MODELS_DIR' .env | grep -v '^#' | grep -v '=$' && echo ENV_OK || echo ENV_MISSING"
ssh <SSH_ALIAS> "test -f <REPO_PATH>/scripts/inference/<CLUSTER>/inference.sh && echo SCRIPT_FOUND || echo SCRIPT_MISSING"
ssh <SSH_ALIAS> "test -d <REPO_PATH>/.log && echo LOG_DIR_FOUND || echo LOG_DIR_MISSING"
```
Any `MISSING` → stop with the precise reason (point to `/toolbox:setup-vllm`). `LOG_DIR_MISSING` → offer `mkdir -p <REPO_PATH>/.log` with confirmation.

Read experiment name:
```bash
ssh <SSH_ALIAS> "grep '^experiment_name:' <REPO_PATH>/<CONFIG_FILE> | awk '{print \$2}'"
```
Store as `EXPERIMENT_NAME` (or `unknown`).

## Step 2: Submit
```bash
ssh <SSH_ALIAS> "cd <REPO_PATH> && uv run python run_experiment.py --mode generate --config_file <CONFIG_FILE> --use_local_model --cluster <CLUSTER> 2>&1"
```
Extract `Submitted batch job <ID>` → `JOB_ID`. No id → show output, `JOB_STATUS=SUBMISSION_ERROR`, stop. Confirm: Job ID, Experiment, Config, Cluster, Log `<REPO_PATH>/.log/inference_<JOB_ID>.out`.

## Step 3: Track queue (PENDING is normal)
Keep a timestamped status log. First check (no delay), then prepend `sleep 30 && `:
```bash
ssh <SSH_ALIAS> "squeue -j <JOB_ID> -h -o '%T %R' 2>/dev/null || echo NOT_IN_QUEUE"
```
`PENDING <reason>` → log + continue. `RUNNING` → Step 4. `NOT_IN_QUEUE` before RUNNING → rejected/cancelled → Step 5. Every 20 PENDING polls (~10 min) ask whether to keep waiting.

## Step 4: vLLM readiness + progress
Each cycle (prepend `sleep 30 && `):
```bash
ssh <SSH_ALIAS> "tail -n 80 <REPO_PATH>/.log/inference_<JOB_ID>.out 2>/dev/null"
ssh <SSH_ALIAS> "grep -E 'ERROR|Traceback|CUDA out of memory|out of memory|HIP out of memory' <REPO_PATH>/.log/vllm_<JOB_ID>.out 2>/dev/null | tail -5"
```
Milestone 1 = `vLLM server is ready!`. On OOM/Traceback → log it, ask `Cancel the job? scancel <JOB_ID> [y/n]`. Every 3 cycles confirm the job is still in `squeue`; `NOT_IN_QUEUE` before `Generation finished` → Step 5. Milestone 2 = `Preparing questions...` → `Generating responses...` → `Generation finished`. ~20 cycles without readiness → ask whether to keep waiting.

## Step 5: Outcome report
Resolve the results timestamp dir:
```bash
ssh <SSH_ALIAS> "ls -1t <REPO_PATH>/experiments/<EXPERIMENT_NAME>/ 2>/dev/null | head -1"
```
Store as `TIMESTAMP` (`unknown` if empty).
```
──────────────────────────────────────────
Job ID:     <JOB_ID>     Status: COMPLETED | FAILED
Experiment: <EXPERIMENT_NAME>     Config: <CONFIG_FILE>     Cluster: <CLUSTER>
Log:        <REPO_PATH>/.log/inference_<JOB_ID>.out
Results:    <REPO_PATH>/experiments/<EXPERIMENT_NAME>/<TIMESTAMP>/
──────────────────────────────────────────
```
**FAILED** → include last 20 lines of `inference_<JOB_ID>.out` + last 10 of `vllm_<JOB_ID>.out`; `JOB_STATUS=FAILED`.

## Orchestration contract
```
JOB_ID=<value>
JOB_STATUS=COMPLETED|FAILED
EXPERIMENT_NAME=<value>
LOG_PATH=<REPO_PATH>/.log/inference_<JOB_ID>.out
RESULTS_PATH=<REPO_PATH>/experiments/<EXPERIMENT_NAME>/<TIMESTAMP>/
CONFIG_FILE=<value>
```
