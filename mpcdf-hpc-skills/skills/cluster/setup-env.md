---
name: cluster-setup-env
description: >
  Set up the Python environment for a repository on any MPCDF cluster. Reads repo
  docs to understand requirements, detects the package manager (uv or pip),
  confirms choices, sets up the env only if absent, handles pre-commit sanely, and
  exports a gitignored package snapshot. Works for any cluster via SSH_ALIAS.
metadata:
  version: "2.0.0"
---

# Skill: cluster-setup-env

Single parameterized setup-env for all clusters (replaces the per-cluster copies).

## Inputs
- `REPO_PATH` (from `select-project`/`git-clone`), `SSH_ALIAS`, `CLUSTER`.

## Output
On success: `ENV_READY=true`, `ENV_TYPE` (`uv` | `pip`). On failure: state the problem and stop.

## Security protocol
Apply the protocol set by the calling command/session; default
`.claude/security_protocols/controlled.md`. In addition this skill:
- NEVER writes secret values to `.env` — copies the template and delegates opening.
- NEVER overwrites an existing `.env`.
- NEVER modifies an existing virtual environment without explicit confirmation.

## Reference
`reference/MPCDF-CLUSTER-FACTS.md` (Software stack: `python-waterboa`, `uv`).

---

## Step 0: Verify the connection
```bash
ssh -O check <SSH_ALIAS> 2>&1
```
No `Master running` → stop: "No active connection. Run the login command first."

## Step 1: Confirm REPO_PATH
Use the session value if present; otherwise ask for the full path on the cluster.

## Step 2: Read repo docs and config
```bash
ssh <SSH_ALIAS> "cat <REPO_PATH>/README.md 2>/dev/null || echo FILE_MISSING"
ssh <SSH_ALIAS> "ls <REPO_PATH>/docs/ 2>/dev/null && cat <REPO_PATH>/docs/*.md 2>/dev/null || echo NO_DOCS"
ssh <SSH_ALIAS> "cat <REPO_PATH>/pyproject.toml 2>/dev/null || echo NO_PYPROJECT"
ssh <SSH_ALIAS> "test -f <REPO_PATH>/uv.lock && echo UV_LOCK_FOUND || echo NO_UV_LOCK"
ssh <SSH_ALIAS> "test -f <REPO_PATH>/requirements.txt && cat <REPO_PATH>/requirements.txt || echo NO_REQUIREMENTS"
ssh <SSH_ALIAS> "cat <REPO_PATH>/.env.example 2>/dev/null || echo NO_ENV_EXAMPLE"
```
Determine: uv (pyproject.toml + uv.lock) vs pip; optional extras; whether pre-commit is mentioned; required env vars (from `.env.example`).

## Step 3: Select package manager
- **automated:** uv if unambiguous (pyproject + uv.lock) → print `✓ Package manager: uv (detected)`; pip if requirements.txt and no uv.lock; if neither matches cleanly, STOP and ask `[uv / pip]`.
- **controlled:** recommend and ask `Use uv? [y/n]` / `Use pip? [y/n]`; if unclear, explain and ask to choose.
Store as `ENV_TYPE`.

## Step 4: Skip if env already exists
```bash
ssh <SSH_ALIAS> "test -d <REPO_PATH>/.venv && echo VENV_EXISTS || echo VENV_MISSING"
```
`VENV_EXISTS`: automated → `✓ .venv exists — skipping`, go to Step 8. controlled → ask whether to skip to the `.env` check; only recreate on explicit confirmation.

## Step 5: Install the package manager if needed (uv)
```bash
ssh <SSH_ALIAS> "which uv 2>/dev/null && echo UV_FOUND || echo UV_MISSING"
```
`UV_MISSING`: automated → install immediately; controlled → ask. Then:
```bash
ssh <SSH_ALIAS> "curl -LsSf https://astral.sh/uv/install.sh | sh"
ssh <SSH_ALIAS> "which uv && echo UV_OK || echo UV_FAILED"
```
`UV_FAILED` → stop with the error.

## Step 6: Optional extras (pip only)
Skip for uv. For pip, if extras were found, present Default / Dev / other extras and ask which. Store as `PIP_EXTRAS` (default `.`).

## Step 7: Install dependencies
**uv:**
```bash
ssh <SSH_ALIAS> "cd <REPO_PATH> && uv sync"
```
**pip:**
```bash
ssh <SSH_ALIAS> "cd <REPO_PATH> && python3 -m venv .venv"
ssh <SSH_ALIAS> "cd <REPO_PATH> && source .venv/bin/activate && pip install --upgrade pip && pip install wheel && pip install -e '.[<PIP_EXTRAS>]'"
```
(controlled → show the commands and ask first.) On failure: show output and stop. Then verify:
```bash
ssh <SSH_ALIAS> "test -d <REPO_PATH>/.venv && echo VENV_OK || echo VENV_FAILED"
```
`VENV_FAILED` → stop.

## Step 7b: Pre-commit (only if docs mention it)
Run (uv): `ssh <SSH_ALIAS> "cd <REPO_PATH> && uv run pre-commit run --all-files 2>&1"`
Run (pip): `ssh <SSH_ALIAS> "cd <REPO_PATH> && source .venv/bin/activate && pre-commit run --all-files 2>&1"`

**Interpret the result — do not treat all non-zero exits as fatal:**
- **Auto-fixers only** (`end-of-file-fixer` / `trailing-whitespace` are the only failing hooks): they edited files in place and exited 1; expected on a fresh clone. Print `✓ Pre-commit: auto-formatted files. No action needed.` and continue.
- **Style/lint hooks** (flake8, ruff, mypy, isort) failed: pre-existing repo issues, not introduced by setup. Print `⚠ Pre-commit: pre-existing style issues (see output). They do not block setup.` and continue.
- **Security/structural hooks** (`Detect Private Key`, import errors, broken installs) failed: stop, show the full hook output, tell the user.

## Step 8: Check `.env` and delegate opening
```bash
ssh <SSH_ALIAS> "test -f <REPO_PATH>/.env && echo ENV_EXISTS || echo ENV_MISSING"
```
`ENV_EXISTS` → `✓ .env already exists.` Go to Step 9.
`ENV_MISSING` and `.env.example` found → copy the template, then delegate opening:
```bash
ssh <SSH_ALIAS> "cp <REPO_PATH>/.env.example <REPO_PATH>/.env"
```
Read `skills/data/edit-env.md` and follow every step, passing `CLUSTER`, `SSH_ALIAS`, `REPO_PATH`, and `ENV_VARS` from Step 2.
`ENV_MISSING` and no `.env.example` → `⚠ No .env.example — skipping .env setup.` Continue.

## Step 9: Snapshot installed packages
**uv:** `ssh <SSH_ALIAS> "cd <REPO_PATH> && uv pip freeze > installed-packages.txt"`
**pip:** `ssh <SSH_ALIAS> "cd <REPO_PATH> && source .venv/bin/activate && pip freeze > installed-packages.txt"`
```bash
ssh <SSH_ALIAS> "grep -qxF 'installed-packages.txt' <REPO_PATH>/.gitignore 2>/dev/null || echo 'installed-packages.txt' >> <REPO_PATH>/.gitignore"
```

## Step 10: Report
```
✓ Environment ready
  Repo: <REPO_PATH>   Env: <ENV_TYPE>   Venv: <REPO_PATH>/.venv
  Snapshot: installed-packages.txt (gitignored)
  Activate: source .venv/bin/activate   |   uv run python ...
```
Return `ENV_READY=true`, `ENV_TYPE`.
