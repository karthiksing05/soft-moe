---
name: git-setup-credentials
description: >
  Check and configure HTTPS git credentials for GitHub on a host (called by
  git-clone on auth failure). Use before cloning a GitHub HTTPS repo on a fresh
  cluster, or when a clone fails with 401/403/"could not read Username".
metadata:
  version: "2.0.0"
---

# Skill: git-setup-credentials

Verify and configure HTTPS credentials for github.com on TARGET_HOST. HTTPS only.

## Output
`GIT_CREDENTIALS_READY = true`. On failure: state and stop.

## Security protocol
Caller's protocol. Also non-negotiable: NEVER echo/log/print tokens; refer to the token by name only.

---

## Step 1: Skip if not HTTPS
REPO_URL starts with `git@` or `ssh://` → `ℹ SSH URL — no HTTPS creds needed.` Set ready, return.
`https://` → continue.

## Step 2: Existing creds?
- local: `printf 'protocol=https\nhost=github.com\n' | git credential fill 2>/dev/null | grep -q 'password=' && echo CREDS_FOUND || echo CREDS_MISSING`
- remote: wrap the same pipeline in `ssh <TARGET_HOST> "..."`
`CREDS_FOUND` → set ready, return. `CREDS_MISSING` → continue.

## Step 3: Ask
GitHub username → `GIT_USERNAME`. GitHub PAT (repo scope; https://github.com/settings/tokens/new) → `GIT_PAT` (do not echo/log).

## Step 4: Helper
`git config --global credential.helper store` (remote: wrap in ssh). Failure → show, stop.

## Step 5: Store (token via stdin, never in the command string)
- local: `printf 'protocol=https\nhost=github.com\nusername=<GIT_USERNAME>\npassword=<GIT_PAT>\n' | git credential approve`
- remote: `printf 'protocol=https\nhost=github.com\nusername=<GIT_USERNAME>\npassword=<GIT_PAT>\n' | ssh <TARGET_HOST> "git credential approve"`
Do not show output. Failure → show error without the token, stop.

## Step 6: Verify
Re-run the Step 2 check. `CREDS_MISSING` → stop with an error.

## Step 7: Report
```
✓ GitHub credentials configured on <TARGET_HOST>
  Username: <GIT_USERNAME>   Token: [stored, not shown]
```
Set `GIT_CREDENTIALS_READY = true`, return.
