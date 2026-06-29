---
name: cluster-open-vs-code
description: >
  Open an MPCDF cluster project folder in VS Code via Remote SSH, with the venv
  auto-detected. Use when the user wants to open VS Code for a cluster project,
  view/edit in an IDE, inspect logs or results after a run, or says "open in VS
  Code", "open the project", "browse the results", "open the log". Works for any
  cluster via SSH_ALIAS.
metadata:
  version: "2.0.0"
---

# Skill: cluster-open-vs-code

Opens VS Code connected to a cluster project over Remote SSH using the reliable
**folder-uri** form (the `--remote` form sometimes opens a locally cached copy
instead of connecting).

## Inputs
- `REPO_PATH`, `SSH_ALIAS`. Optional from session: `LOG_PATH`, `CONFIG_FILE` (relative to REPO_PATH), `RESULTS_PATH` (a directory, shown not opened).

## Output
VS Code opens the folder (and any requested files). On failure: state the problem and stop.

## Security protocol
Follow the calling command's protocol; default `.claude/security_protocols/controlled.md`.

---

## Step 1: Resolve REPO_PATH
Use the session value, else ask for the full path (e.g. `/u/<user>/projects/<repo>`).

## Step 2: Check the VS Code CLI
```bash
which code 2>/dev/null && echo CODE_FOUND || echo CODE_MISSING
```
`CODE_MISSING` → tell the user to install it (Command Palette → "Shell Command: Install 'code' command in PATH") and stop.

## Step 3: Check the Remote SSH extension
```bash
code --list-extensions 2>/dev/null | grep -i "ms-vscode-remote.remote-ssh" && echo EXT_FOUND || echo EXT_MISSING
```
`EXT_MISSING` → tell the user to install extension `ms-vscode-remote.remote-ssh` and stop.

## Step 4: Identify extra files from workflow context
If any of `LOG_PATH`, `CONFIG_FILE`, `RESULTS_PATH` are present, list them and ask once whether to open the log + config alongside the project. On `y`, set `EXTRA_FILES` to the full remote paths of `LOG_PATH` and `<REPO_PATH>/<CONFIG_FILE>` (skip absent ones); note `RESULTS_PATH` for the report only. On `n`/none, `EXTRA_FILES` empty.

## Step 5: Show and confirm
```
I will open VS Code with:
  code --folder-uri "vscode-remote://ssh-remote+<SSH_ALIAS><REPO_PATH>"
  [one --file-uri line per EXTRA_FILES, if any]
VS Code manages its own SSH connection (independent of the ControlMaster socket);
opening it does not affect running jobs.
Proceed? [y/n]
```
Wait for explicit `y`; else stop.

## Step 6: Open
```bash
code --folder-uri "vscode-remote://ssh-remote+<SSH_ALIAS><REPO_PATH>"
```
If `EXTRA_FILES` is non-empty, wait 2s and open each:
```bash
sleep 2 && code --file-uri "vscode-remote://ssh-remote+<SSH_ALIAS><file_path>"
```
Any failure → show the error and stop.

## Step 7: Report
```
✓ VS Code opened
  Project: <REPO_PATH>   Host: <SSH_ALIAS>
  Venv: .venv (auto-detected by the Python extension, if present)
  Opened: <EXTRA_FILES or "project folder only">
  Results: <RESULTS_PATH if present>
```
