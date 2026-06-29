---
name: cluster-ssh-logout
description: >
  Close the active SSH ControlMaster connection to an MPCDF cluster (Raven, Dais,
  Viper-GPU, Viper-CPU). Use when the user wants to disconnect, log out of the
  cluster, or end the session. Works for any cluster via SSH_ALIAS.
metadata:
  version: "2.0.0"
---

# Skill: cluster-ssh-logout

## Inputs
- `SSH_ALIAS` from `cluster-resolve`. If not set, ask which cluster to disconnect from and map it.

## Output
On success: `<SSH_ALIAS>_DISCONNECTED=true`. On failure: state the problem and stop.

## Security protocol
Follow the protocol set by the calling command. If called directly, read and apply
`.claude/security_protocols/controlled.md`.

---

## Step 1: Check the socket
```bash
ssh -O check <SSH_ALIAS> 2>&1
```
No `Master running` → tell the user there is nothing to close. Stop.
`Master running` → continue.

## Step 2: Close (show command, confirm first)
```bash
ssh -O exit <SSH_ALIAS> 2>&1
```
Wait for explicit `y`. Else stop.

## Step 3: Verify
```bash
ssh -O check <SSH_ALIAS> 2>&1
```
Still `Master running` → show output and stop with an error. Otherwise the connection is closed.

## Step 4: Report
```
✓ Disconnected from <SSH_ALIAS>. Re-run the login skill to reconnect.
```
Return `<SSH_ALIAS>_DISCONNECTED=true`.
