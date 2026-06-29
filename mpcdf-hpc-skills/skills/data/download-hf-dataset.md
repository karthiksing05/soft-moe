---
name: data-download-hf-dataset
description: >
  Download a HuggingFace dataset (or a filtered slice) into DATASETS_DIR on an
  MPCDF cluster, using the project's uv/pip env, as a background job with a log.
  Use when the user says "download the [dataset]", "pull [dataset] from
  huggingface", "get the training data", or a pipeline needs a dataset staged on
  /ptmp before training/eval. Mirrors download-model but for datasets.
metadata:
  version: "1.0.0"
---

# Skill: data-download-hf-dataset

Stage a HuggingFace dataset into `DATASETS_DIR` (default `/ptmp/<user>/datasets`)
on the cluster. Works locally or remote via `SSH_ALIAS`.

## Inputs
- `REPO_PATH`, `SSH_ALIAS`, `CLUSTER` (from prior skills or asked).
- `DATASET_ID`: e.g. `allenai/c4`, `HuggingFaceFW/fineweb`; asked if absent.
- Optional `CONFIG`/`SPLIT`/`REVISION` and a filter predicate.

## Output
On success: `DATASET_PATH` (`$DATASETS_DIR/<DATASET_ID>`), `DATASET_ID`.
On failure: exact error + failing step. Stop.

## Security protocol
Caller's protocol; default `.claude/security_protocols/controlled.md`. Also:
- NEVER echo/log `HUGGINGFACE_TOKEN`; reference by name.
- NEVER overwrite an existing dataset dir without confirmation.

> Command convention: commands shown as `ssh <SSH_ALIAS> "<cmd>"` (remote). Run the inner command directly when local.

---

## Step 1: Resolve context + DATASETS_DIR
Resolve `CLUSTER`/`REPO_PATH` like the other toolbox skills (verify `ssh -O check <SSH_ALIAS>` shows `Master running` if remote). Read the target dir from `.env`, falling back to `/ptmp/<user>/datasets`:
```bash
ssh <SSH_ALIAS> "grep -s '^DATASETS_DIR=' <REPO_PATH>/.env 2>/dev/null | head -1 | cut -d'=' -f2-"
```
Empty → DATASETS_DIR=`/ptmp/<CLUSTER_USER>/datasets`. Ensure it exists (create with confirmation per protocol):
```bash
ssh <SSH_ALIAS> "mkdir -p <DATASETS_DIR>"
```

## Step 2: Resolve DATASET_ID
If absent, ask for the full HF id (`org/name`). Validate one `/`.

## Step 3: Already present?
```bash
ssh <SSH_ALIAS> "test -d \"<DATASETS_DIR>/<DATASET_ID>\" && echo EXISTS || echo MISSING"
```
`EXISTS` → list contents, ask whether to re-download/overwrite. On no → return existing path. On yes / `MISSING` → continue.

## Step 4: Confirm + start (background, logged)
Print a summary (id, destination, cluster, log path `<REPO_PATH>/.log/dataset_<SLUG>.log` where SLUG = id with `/`→`_`). automated → proceed; otherwise `Proceed? [y/n]`.
```bash
ssh <SSH_ALIAS> "mkdir -p <REPO_PATH>/.log"
```
Two acquisition modes:
- **Whole dataset / files** (fast, resumable): `huggingface_hub.snapshot_download(repo_id, repo_type='dataset', local_dir=...)`.
- **Filtered slice** (recommended for huge corpora — stage only what you need): `datasets.load_dataset(id, config, split=..., streaming=True)`, filter, then `to_parquet`/`save_to_disk` under `DATASETS_DIR`.

Background launch (snapshot mode example):
```bash
ssh <SSH_ALIAS> "cd <REPO_PATH> && set -a && source .env && set +a && nohup uv run python -c \"from huggingface_hub import snapshot_download; import os; tok=next((v for k,v in os.environ.items() if 'token' in k.lower() and ('hf' in k.lower() or 'hugging' in k.lower()) and v and 'YOUR_' not in v), None); snapshot_download('<DATASET_ID>', repo_type='dataset', local_dir=os.path.join('<DATASETS_DIR>','<DATASET_ID>'), token=tok)\" > <REPO_PATH>/.log/dataset_<SLUG>.log 2>&1 & echo \$!"
```
Capture the PID.

## Step 5: Monitor
Poll every 60s: `ssh <SSH_ALIAS> "ps -p <PID> >/dev/null 2>&1 && echo RUNNING || echo NOT_RUNNING"` and `tail -n 5` the log. Append timestamped status lines. After ~60 RUNNING polls, ask whether to keep waiting.

## Step 6: Verify + report
```bash
ssh <SSH_ALIAS> "grep -iE 'error|traceback|exception|401|403|404' <REPO_PATH>/.log/dataset_<SLUG>.log | tail -10"
ssh <SSH_ALIAS> "du -sh \"<DATASETS_DIR>/<DATASET_ID>\" 2>/dev/null; ls \"<DATASETS_DIR>/<DATASET_ID>/\" | head"
```
Clean + present → report path and size. Errors/missing → show last 20 log lines and stop.
```
DATASET_PATH=<DATASETS_DIR>/<DATASET_ID>
DATASET_ID=<DATASET_ID>
```
Reminder: `/ptmp` is purged — push the curated slice to Nexus-S3/archive (see `skills/data/transfer.md`) if it must persist.
