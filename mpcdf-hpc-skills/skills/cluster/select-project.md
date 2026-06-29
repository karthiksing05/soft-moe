---
name: cluster-select-project
description: >
  List projects on an MPCDF cluster and resolve which folder to open. Handles
  user selection or auto-matches a named request, and checks for a virtual
  environment. Works for any cluster via SSH_ALIAS. Outputs REPO_PATH and
  VENV_STATUS for calling skills.
metadata:
  version: "2.0.0"
---

# Skill: cluster-select-project

This replaces the previously duplicated per-cluster select-project skills; it is
parameterized by `SSH_ALIAS` so one file serves every cluster.

## Inputs
- `CLUSTER_USER`: must be set (from `cluster-ssh-login`). If not, stop and tell the calling workflow to run the login skill first.
- `SSH_ALIAS`: from `cluster-resolve`.

## Output
On success: `REPO_PATH`, `VENV_STATUS` (`VENV_FOUND` | `NO_VENV`).
On failure: state the problem and stop.

## Security protocol
Follow the protocol set by the calling command. If called directly, read and apply
`.claude/security_protocols/controlled.md`.

---

## Step 1: List projects
```bash
ssh <SSH_ALIAS> "ls -1 /u/<CLUSTER_USER>/projects/ 2>/dev/null || echo NO_PROJECTS_DIR"
```
`NO_PROJECTS_DIR` → tell the user, ask for a full path, store as `REPO_PATH`, skip to Step 3.
Otherwise filter to directories only (never loose files like `.ipynb`).

## Step 2: Resolve which project
**User named a project** → match by partial name:
- Exactly one match → `REPO_PATH = /u/<CLUSTER_USER>/projects/<match>`. Proceed.
- Multiple → list them, ask the user to pick.
- None → tell the user, ask for a full path.

**User named none** → numbered menu:
```
Projects in /u/<CLUSTER_USER>/projects/:
  1. <name>
  2. ...
Which would you like? (number or full path)
```
Number → `REPO_PATH = /u/<CLUSTER_USER>/projects/<selected>`. Full path → use directly.

## Step 3: Verify the folder
```bash
ssh <SSH_ALIAS> "test -d <REPO_PATH> && echo FOLDER_OK || echo FOLDER_MISSING"
```
`FOLDER_MISSING` → tell the user and stop.

## Step 4: Check for a virtual environment
```bash
ssh <SSH_ALIAS> "test -d <REPO_PATH>/.venv && echo VENV_FOUND || echo NO_VENV"
```
Store as `VENV_STATUS` (informational — `uv run` manages its own env; the caller decides whether to act on it). Return `REPO_PATH`, `VENV_STATUS`.
