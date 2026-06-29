---
name: cluster-resolve
description: >
  Identify and validate which MPCDF cluster a workflow runs on (raven, dais,
  viper, viper-cpu). Reads the cluster from the request or asks, maps it to its
  SSH alias, and confirms the alias is reachable in config. Use as a prerequisite
  in any cluster workflow that needs to know its target. Trigger on "on raven",
  "on dais", "on viper", or when another skill needs CLUSTER resolved.
metadata:
  version: "2.0.0"
---

# Skill: cluster-resolve

Resolve and validate the target cluster, and derive its SSH alias.

## Output
On success: `CLUSTER` (one of `raven`, `dais`, `viper`, `viper-cpu`) and
`SSH_ALIAS` (the `~/.ssh/config` Host to use).
On any failure: state the problem precisely and stop.

## Security protocol
Follow the protocol set by the calling command. If called directly, read and apply
`.claude/security_protocols/controlled.md`.

## Reference
Read `reference/MPCDF-CLUSTER-FACTS.md` (the "Access" and "Per-cluster" sections).

---

## Step 1: Identify the cluster

If the request already names a cluster ("on raven", "for dais", "viper"), store it
as `CLUSTER` and continue. Otherwise ask:

```
Which cluster should this run on? (raven / dais / viper / viper-cpu)

  raven      NVIDIA A100 40GB — CUDA training & inference, ASR/diarisation
  dais       NVIDIA H200 — large/multi-GPU CUDA training, big-model inference
  viper      AMD MI300A (ROCm) — vLLM inference only (TP<=2)
  viper-cpu  AMD CPU — CPU-bound data prep / mining
```

Validate the answer is one of the four. If not, ask again.

## Step 2: Derive SSH_ALIAS

| CLUSTER | SSH_ALIAS |
|---|---|
| `raven` | `raven` |
| `dais` | `dais` |
| `viper` | `viper-gpu` |
| `viper-cpu` | `viper` |

## Step 3: Check the alias is configured

```bash
grep -qE "^Host .*\b<SSH_ALIAS>\b" ~/.ssh/config 2>/dev/null && echo ALIAS_FOUND || echo ALIAS_MISSING
```

`ALIAS_MISSING` → tell the user the alias is not in `~/.ssh/config` and that the
login skill will offer to add it (showing exactly what it will append). Continue
— the login skill handles writing config with confirmation.

## Step 4: Report
```
✓ Cluster resolved: <CLUSTER>  (ssh alias: <SSH_ALIAS>)
```
Return `CLUSTER` and `SSH_ALIAS`.
