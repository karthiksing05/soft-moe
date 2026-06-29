---
name: inference-download-model
description: >
  Download a HuggingFace model into LOCAL_MODELS_DIR on any MPCDF cluster (or
  locally), using the project's .env and uv environment, as a background job with
  a log. Use when the user says "download [model] from huggingface", "pull
  [model] weights", "get the qwen/llama/deepseek weights", or "I need [model] for
  vllm". Reads LOCAL_MODELS_DIR from .env; saves to $LOCAL_MODELS_DIR/<org>/<name>
  which is exactly the layout vLLM expects. No code changes needed.
metadata:
  version: "2.0.0"
---

# Skill: inference-download-model

## Inputs
- `REPO_PATH`, `SSH_ALIAS`, `CLUSTER` (or `local`); resolved in Step 1 if absent.
- `MODEL_ID`: e.g. `Qwen/Qwen2.5-32B-Instruct`; asked if absent.

## Output
On success: `MODEL_PATH` (`$LOCAL_MODELS_DIR/<MODEL_ID>`), `MODEL_ID`.
On failure: exact error + failing step. Stop. Do not retry around missing infra.

## Security protocol
Caller's protocol; default `.claude/security_protocols/controlled.md`. Also:
- NEVER echo/log `HUGGINGFACE_TOKEN` or any secret; reference by name.
- NEVER write to `.env` or any config; NEVER overwrite an existing model dir without confirmation.

> Command convention: `ssh <SSH_ALIAS> "<cmd>"` is the remote form; omit the wrapper when local.

---

## Step 1: Resolve context
If `CLUSTER`+`REPO_PATH` set, use them. Else ask `local` or a cluster (invoke `cluster-resolve` for a cluster). If remote, require `ssh -O check <SSH_ALIAS>` → `Master running`, else stop and tell the user to log in first. If `REPO_PATH` unknown, discover toolbox projects:
```bash
ssh <SSH_ALIAS> "ls -d ~/*/configs/ ~/projects/*/configs/ 2>/dev/null | sed 's|/configs/||'"
```
One → use it; many → ask; none → suggest `/toolbox:setup-repo` and stop.

## Step 2: Pre-flight .env (LOCAL_MODELS_DIR + token)
```bash
ssh <SSH_ALIAS> "grep -s '^LOCAL_MODELS_DIR=' <REPO_PATH>/.env 2>/dev/null | head -1"
ssh <SSH_ALIAS> "grep -iE '^[A-Z_]*(HF|HUGGING)[A-Z_]*TOKEN[A-Z_]*=' <REPO_PATH>/.env 2>/dev/null | grep -v '=YOUR_' | grep -v '=$' | head -1"
```
Both set → `✓ .env: LOCAL_MODELS_DIR and HF token set.` Else open `.env` for editing (read `skills/data/edit-env.md`), wait, re-check. If `LOCAL_MODELS_DIR` still unset → stop. If token unset → one-line warning (`⚠ HUGGINGFACE_TOKEN not set — gated models will fail`) and continue.
Directory check:
```bash
ssh <SSH_ALIAS> "test -d \"<LOCAL_MODELS_DIR>\" && echo DIR_FOUND || echo DIR_MISSING"
```
`DIR_MISSING` → stop (tell the user to create it / re-run setup-vllm which can create it).

## Step 3: Resolve MODEL_ID
If absent, ask for the full id; validate exactly one `/`.

## Step 4: Already present?
```bash
ssh <SSH_ALIAS> "test -d \"<LOCAL_MODELS_DIR>/<MODEL_ID>\" && echo MODEL_EXISTS || echo MODEL_MISSING"
```
`MODEL_EXISTS` → list contents, ask re-download/overwrite. No → return existing path, stop. Yes/`MODEL_MISSING` → continue.

## Step 5: Confirm + start (background, logged)
Note size/time (30B+ ≈ 60–150 GB, 30–90 min). SLUG = MODEL_ID with `/`→`_`. automated → proceed; else `Proceed? [y/n]`.
```bash
ssh <SSH_ALIAS> "mkdir -p <REPO_PATH>/.log"
ssh <SSH_ALIAS> "cd <REPO_PATH> && set -a && source .env && set +a && nohup uv run python -c \"from huggingface_hub import snapshot_download; import os; tok=next((v for k,v in os.environ.items() if 'token' in k.lower() and ('hf' in k.lower() or 'hugging' in k.lower()) and v and 'YOUR_' not in v), None); snapshot_download('<MODEL_ID>', local_dir=os.path.join(os.environ['LOCAL_MODELS_DIR'], '<MODEL_ID>'), token=tok)\" > <REPO_PATH>/.log/download_<SLUG>.log 2>&1 & echo \$!"
```
Capture PID as `DOWNLOAD_PID`. Confirm started.

## Step 6: Monitor (every 60s)
```bash
ssh <SSH_ALIAS> "ps -p <DOWNLOAD_PID> >/dev/null 2>&1 && echo RUNNING || echo NOT_RUNNING"
ssh <SSH_ALIAS> "tail -n 5 <REPO_PATH>/.log/download_<SLUG>.log 2>/dev/null"
```
Subsequent checks prepend `sleep 60 && `. `NOT_RUNNING` → Step 7. After 60 RUNNING polls (~60 min), ask whether to keep waiting.

## Step 7: Verify + report
```bash
ssh <SSH_ALIAS> "grep -iE 'error|traceback|exception|401|403|404' <REPO_PATH>/.log/download_<SLUG>.log | tail -10"
ssh <SSH_ALIAS> "test -d \"<LOCAL_MODELS_DIR>/<MODEL_ID>\" && echo DIR_OK || echo DIR_MISSING; du -sh \"<LOCAL_MODELS_DIR>/<MODEL_ID>\" 2>/dev/null; ls \"<LOCAL_MODELS_DIR>/<MODEL_ID>/\" | head"
```
`DIR_OK` + no errors → report path/size, then:
```
MODEL_PATH=<LOCAL_MODELS_DIR>/<MODEL_ID>
MODEL_ID=<MODEL_ID>
```
Else show last 20 log lines and stop.
