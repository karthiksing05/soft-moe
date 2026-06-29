---
name: git-clone
description: >
  Clone a git repository into a target directory — locally or on a remote host
  via SSH (any MPCDF cluster alias). Use when the user asks to clone a repo,
  download a repository, pull a project to the cluster, or when a workflow needs
  a local copy of a remote repo. Auto-discovers a sensible target folder.
metadata:
  version: "2.0.0"
---

# Skill: git-clone

Clone a repo into a target dir, only if not already present. Local or remote.
If the caller provides TARGET_HOST and TARGET_DIR, use them without asking.

## Output
On success: `REPO_PATH`, `REPO_NAME`, `TARGET_HOST`. On failure: state and stop.

## Security protocol
Caller's protocol; default `.claude/security_protocols/controlled.md`.
This skill also: NEVER clones into an existing directory; NEVER overwrites files.

---

## Step 1: Repository URL
If provided, use it. Else ask. Derive `REPO_NAME` from the final path segment minus `.git`.

## Step 2: Execution context + target dir
If TARGET_HOST and TARGET_DIR provided, skip. Else ask `local` or an SSH alias (e.g. `raven`, `dais`, `viper-gpu`). Store as TARGET_HOST.

Discover the home dir:
- local: `echo $HOME`
- remote: `ssh <TARGET_HOST> "echo \$HOME"`
Store as HOME_DIR.

Find standard project folders:
- local: `find <HOME_DIR> -maxdepth 1 -type d \( -name "projects" -o -name "repos" -o -name "code" -o -name "src" -o -name "work" \) 2>/dev/null`
- remote: `ssh <TARGET_HOST> "find <HOME_DIR> -maxdepth 1 -type d \( -name 'projects' -o -name 'repos' -o -name 'code' -o -name 'src' -o -name 'work' \) 2>/dev/null"`

Interpret:
- exactly one → `TARGET_DIR` = it; print `✓ Target directory: <TARGET_DIR>`.
- multiple → numbered menu, ask.
- none → ask for a folder name (create under HOME_DIR) or a full absolute path.

Natural-language match ("the projects folder", "projects") against discovered names before concluding nothing was found.

Set `REPO_PATH = <TARGET_DIR>/<REPO_NAME>`.

## Step 3: Already present?
- local: `test -d <REPO_PATH> && echo EXISTS || echo MISSING`
- remote: `ssh <TARGET_HOST> "test -d <REPO_PATH> && echo EXISTS || echo MISSING"`
`EXISTS` → "Repository already at <REPO_PATH>. Skipping clone." Return REPO_PATH. Stop.

## Step 4: Clone
automated → run; controlled → show command + `Proceed? [y/n]`.
- local: `git clone <REPO_URL> <REPO_PATH>`
- remote: `ssh <TARGET_HOST> "git clone <REPO_URL> <REPO_PATH>"`

On failure: if the error includes `Authentication failed`, `could not read Username`, `Invalid username or password`, `403`, or `Repository not found` → read `skills/git/setup-credentials.md`, follow it (pass TARGET_HOST, REPO_URL), then re-run the clone once. Any other error or a second failure → show full output and stop.

## Step 5: Verify
`test -d <REPO_PATH> && echo CLONE_OK || echo CLONE_FAILED` (remote: wrap in ssh). `CLONE_FAILED` → stop.

## Step 6: Report
```
✓ Repository cloned
  Path: <REPO_PATH>   Host: <TARGET_HOST>
```
Return REPO_PATH, REPO_NAME, TARGET_HOST.
