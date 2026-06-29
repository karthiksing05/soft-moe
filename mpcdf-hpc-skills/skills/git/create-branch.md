---
name: git-create-branch
description: >
  Create and check out a named git branch in an existing repo, locally or on a
  remote host via SSH. Use when the user wants to start work on a branch, or when
  a workflow needs a working branch before committing (never work on main/master).
metadata:
  version: "2.0.0"
---

# Skill: git-create-branch

Create + check out a branch. Uses TARGET_HOST and REPO_PATH if the caller provides them.

## Output
`BRANCH_NAME`. On failure: state and stop.

## Security protocol
Caller's protocol; default `.claude/security_protocols/controlled.md`.

---

## Step 1: Branch name
Ask `What should the new branch be named? (Enter for default: setup/env)`. Enter → `setup/env`.

## Step 2: Ensure it does not exist
- local: `git -C <REPO_PATH> branch --list <BRANCH_NAME>`
- remote: `ssh <TARGET_HOST> "git -C <REPO_PATH> branch --list <BRANCH_NAME>"`
Non-empty → stop, tell the user it already exists. Empty → continue.

## Step 3: Create + checkout
`git -C <REPO_PATH> checkout -b <BRANCH_NAME>` (remote: wrap in ssh). Failure → show and stop.

## Step 4: Verify
`git -C <REPO_PATH> branch --show-current` (remote: wrap). Matches → continue, else stop.

## Step 5: Report
```
✓ Branch created: <BRANCH_NAME>   Repo: <REPO_PATH>   Host: <TARGET_HOST>
```
Return BRANCH_NAME.
