---
name: inference-setup-vllm
description: >
  Set up and verify the vLLM inference stack on an MPCDF cluster: checks the vLLM
  Apptainer image (offers to pull a CUDA image on Raven/Dais; reports the ROCm
  image on Viper-GPU), that LOCAL_MODELS_DIR is set and exists, and that the
  inference script and log dir are in place. Use when the user says "check vllm
  setup", "is vllm ready", "set up vllm on raven/dais/viper", or before submitting
  a job.
metadata:
  version: "3.0.0"
---

# Skill: inference-setup-vllm

## Inputs
- `REPO_PATH`, `SSH_ALIAS`, `CLUSTER` (`raven`|`dais`|`viper`).

## Output
On success: `VLLM_READY=true`. On blocked: `VLLM_READY=false` + reason + next step.

## Security protocol
Caller's protocol; default `.claude/security_protocols/controlled.md`. Also:
- NEVER write to `.env`/config; NEVER pull/copy images without confirmation; create dirs only after confirmation.

## Reference
`reference/MPCDF-CLUSTER-FACTS.md` (Containers; Per-cluster GPU). Confirm GPU type matches the image (CUDA vs ROCm).

---

## Step 1: Container image (differs by cluster)

### Raven / Dais (CUDA → `apptainer exec --nv`)
```bash
ssh <SSH_ALIAS> "test -f <REPO_PATH>/container_images/vllm-openai.sif && echo SIF_FOUND || echo SIF_MISSING"
```
`SIF_FOUND` → continue. `SIF_MISSING` → offer to build:
```
vLLM CUDA container is missing. I can build it:
  module load apptainer
  apptainer pull <REPO_PATH>/container_images/vllm-openai.sif docker://vllm/vllm-openai:<ver>
Build now? [y/n]
```
On `y`:
```bash
ssh <SSH_ALIAS> "module load apptainer && apptainer pull <REPO_PATH>/container_images/vllm-openai.sif docker://vllm/vllm-openai:<ver>"
ssh <SSH_ALIAS> "test -f <REPO_PATH>/container_images/vllm-openai.sif && echo SIF_FOUND || echo SIF_STILL_MISSING"
```
`SIF_STILL_MISSING` → show output, stop, `VLLM_READY=false`. On `n` → stop with manual build instructions, `VLLM_READY=false`.

### Viper-GPU (ROCm → `apptainer exec --rocm`)
Check the ROCm image (confirm the path with the user / Viper-GPU guide; a system image may exist, otherwise pull a repo-local one):
```bash
ssh <SSH_ALIAS> "test -f <REPO_PATH>/container_images/vllm-rocm.sif && echo SIF_FOUND || echo SIF_MISSING"
```
`SIF_MISSING` → offer to pull a ROCm vLLM image:
```
ROCm vLLM container is missing. I can pull one:
  module load apptainer
  apptainer pull <REPO_PATH>/container_images/vllm-rocm.sif docker://rocm/vllm:<ver>
(If your site provides a system ROCm vLLM image, give me its path instead.)
Pull now? [y/n]
```
Proceed analogously; verify; on failure stop with `VLLM_READY=false`.

## Step 2: LOCAL_MODELS_DIR in .env
**2a.** `ssh <SSH_ALIAS> "test -f <REPO_PATH>/.env && echo ENV_FOUND || echo ENV_MISSING"` — `ENV_MISSING` → stop with copy-from-example instructions, `VLLM_READY=false`.
**2b.** Read it:
```bash
ssh <SSH_ALIAS> "grep '^LOCAL_MODELS_DIR=' <REPO_PATH>/.env | head -1 | cut -d'=' -f2-"
```
Report the raw value. Empty/whitespace → stop with instructions, `VLLM_READY=false`. Non-empty → store as `LOCAL_MODELS_DIR_VALUE` and verify:
```bash
ssh <SSH_ALIAS> "source <REPO_PATH>/.env 2>/dev/null && test -d \"\$LOCAL_MODELS_DIR\" && echo DIR_FOUND || echo DIR_MISSING"
```
`DIR_MISSING`:
- automated → `Creating <LOCAL_MODELS_DIR_VALUE>...` then `mkdir -p`, verify; still missing → stop `VLLM_READY=false`.
- controlled/navigating → ask `Create it now? [y/n] (n if there's a typo in .env)`; on `y` create + verify; on `n` stop `VLLM_READY=false`.
`DIR_FOUND` → continue.

## Step 3: Inference script
```bash
ssh <SSH_ALIAS> "test -f <REPO_PATH>/scripts/inference/<CLUSTER>/inference.sh && echo SCRIPT_FOUND || echo SCRIPT_MISSING"
```
`SCRIPT_MISSING` → stop (script should be in the repo; re-check the clone or `/toolbox:setup-repo`), `VLLM_READY=false`.

## Step 4: Log directory
```bash
ssh <SSH_ALIAS> "test -d <REPO_PATH>/.log && echo LOG_DIR_FOUND || echo LOG_DIR_MISSING"
```
`LOG_DIR_MISSING` → create with confirmation (`mkdir -p <REPO_PATH>/.log`), verify; still missing → stop `VLLM_READY=false`.

## Step 5: Report
```
vLLM setup verified on <CLUSTER>.
  Container:        <container path>   ✓   (CUDA: --nv | ROCm: --rocm)
  LOCAL_MODELS_DIR: <value>            ✓
  Inference script: scripts/inference/<CLUSTER>/inference.sh  ✓
  Log directory:    .log/              ✓
VLLM_READY=true
```
