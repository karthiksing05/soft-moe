---
name: cluster-ssh-login
description: >
  Establish an SSH ControlMaster connection to an MPCDF cluster (Raven, Dais,
  Viper-GPU, Viper-CPU) through the gate gateway, so later commands run without
  re-authentication. Use when the user wants to connect / log into a cluster,
  access MPCDF, or when any workflow needs an active cluster connection. Works
  for any cluster via the SSH_ALIAS resolved by cluster-resolve.
metadata:
  version: "2.0.0"
---

# Skill: cluster-ssh-login

## Inputs
- `CLUSTER`, `SSH_ALIAS` from `cluster-resolve`. If not set, read `skills/cluster/resolve.md` and run it first.

## Output
On success: `CLUSTER_USER` (MPCDF username), `CLUSTER_HOST`, `<SSH_ALIAS>_CONNECTED=true`.
On any failure: state the problem precisely and stop. No fallback, no retry.

## Security protocol
Follow the protocol set by the calling command. If called directly, read and apply
`.claude/security_protocols/controlled.md`.
Additional, non-negotiable:
- NEVER write to `~/.ssh/config` without explicit confirmation.
- NEVER overwrite or delete existing `~/.ssh/config` entries.
- NEVER ask for a password or OTP — you neither receive nor need them.

## Reference
Read the "Access" section of `reference/MPCDF-CLUSTER-FACTS.md` for hostnames and
the recommended config block.

---

## Step 1: Ask for username
Ask: "What is your MPCDF username?" Never infer it. Store as `CLUSTER_USER`.

## Step 2: Check for an existing connection
```bash
ssh -O check <SSH_ALIAS> 2>&1
```
Output contains `Master running` → skip to Step 6. Otherwise continue.

## Step 3: Check SSH config
```bash
grep -nE "^Host .*\b<SSH_ALIAS>\b" ~/.ssh/config 2>/dev/null
```
- **Entry present** with `ProxyJump gate` + `ControlMaster` + `ControlPath` → continue to Step 4.
- **Entry present but missing settings** → name exactly which lines are missing, show what you will add, ask `May I append these? [y/n]`. On `y` append only the missing lines; else stop.
- **No entry** → show exactly what you will write (substitute the real username for `<CLUSTER_USER>` and the real hostname for `<SSH_ALIAS>` per the facts file), e.g.:
  ```
  Host gate
      HostName gate.mpcdf.mpg.de
      User <CLUSTER_USER>
      ServerAliveInterval 120
      ControlMaster auto
      ControlPath ~/.ssh/cm_sockets/%r@%h:%p
      ControlPersist 4h

  Host <SSH_ALIAS>
      HostName <login-host per facts file>
      User <CLUSTER_USER>
      ProxyJump gate
      ControlMaster auto
      ControlPath ~/.ssh/cm_sockets/%r@%h:%p
      ControlPersist 4h

  May I append this? [y/n]
  ```
  Wait for explicit `y`, then append with a heredoc. Else stop.

## Step 4: Ensure the socket directory exists
```bash
test -d ~/.ssh/cm_sockets && echo EXISTS || echo MISSING
```
`MISSING` → (automated: create immediately; controlled/navigating: ask `Proceed? [y/n]`) then:
```bash
mkdir -p ~/.ssh/cm_sockets && chmod 700 ~/.ssh/cm_sockets
```

## Step 5: Ask the user to log in manually
The gateway enforces 2FA, which only the user can satisfy. Tell them:
```
No active connection found. Open a NEW terminal and run:
    ssh <SSH_ALIAS>
You'll be prompted for:
  1. Password for gate.mpcdf.mpg.de
  2. One-Time Password (OTP)
  3. Password for the cluster login node
Leave that terminal open. Press Enter here when you're logged in.
```
Wait for Enter. Do not proceed until confirmed.

## Step 6: Verify the connection
```bash
ssh -O check <SSH_ALIAS> 2>&1
```
No `Master running` → show output and stop (login not detected).
```bash
ssh <SSH_ALIAS> "hostname && echo CLUSTER_CONNECTED"
```
Store the hostname as `CLUSTER_HOST`. Missing hostname or marker → show output and stop.

## Step 7: Report
```
✓ Connected to <CLUSTER>
  User:   <CLUSTER_USER>
  Host:   <CLUSTER_HOST>
  Alias:  <SSH_ALIAS>
  Valid:  ~4 hours (re-run on expiry)
```
Return `CLUSTER_USER`, `CLUSTER_HOST`, `<SSH_ALIAS>_CONNECTED=true`.
