---
name: slurm-monitor-job
description: >
  Monitor any SLURM job on a cluster by job ID — show queue state, stream the
  log tail, report final accounting, and (with confirmation) cancel it. Use when
  the user says "check job <id>", "is my job running", "what's the status of
  <id>", "tail the log for <id>", "why is my job pending", or "cancel job <id>".
  Cluster-agnostic; works for training and inference jobs alike.
metadata:
  version: "1.0.0"
---

# Skill: slurm-monitor-job

Inspect and (optionally) cancel a SLURM job over SSH. Read-only by default;
cancellation always requires explicit confirmation.

## Inputs
- `SSH_ALIAS` (or `CLUSTER` → derive: `viper`→`viper-gpu`, `viper-cpu`→`viper`, else identity).
- `JOB_ID`: the SLURM job id (asked if not provided).
- `LOG_PATH`: optional; the `.out` file to tail.

## Output
`JOB_STATE` (PENDING/RUNNING/COMPLETED/FAILED/CANCELLED/UNKNOWN) and a short report.

## Security protocol
Caller's protocol; default `.claude/security_protocols/controlled.md`. Also:
- NEVER cancel a job without explicit `y`. All squeue/sacct/tail are read-only.

## Reference
`reference/SLURM-REFERENCE.md` (Monitoring).

---

## Step 1: Resolve target
Verify the connection:
```bash
ssh -O check <SSH_ALIAS> 2>&1
```
No `Master running` → stop, tell the user to log in first. Ask for `JOB_ID` if not set.

## Step 2: Queue state
```bash
ssh <SSH_ALIAS> "squeue -j <JOB_ID> -h -o '%T %M %L %R' 2>/dev/null || echo NOT_IN_QUEUE"
```
- `PENDING <reason>` → normal on a busy cluster; report the reason. Optionally show the estimate:
  ```bash
  ssh <SSH_ALIAS> "squeue -j <JOB_ID> --start 2>/dev/null"
  ```
- `RUNNING <elapsed> <left> <nodelist>` → report node(s) and time left.
- `NOT_IN_QUEUE` → the job has finished or never started → Step 4 (accounting).

## Step 3: Live log (if RUNNING and LOG_PATH known)
If `LOG_PATH` is not set, try to discover it:
```bash
ssh <SSH_ALIAS> "ls -1t ~/**/.log/*_<JOB_ID>.out 2>/dev/null | head -1"
```
Then:
```bash
ssh <SSH_ALIAS> "tail -n 80 <LOG_PATH> 2>/dev/null"
```
Surface any `ERROR` / `Traceback` / `out of memory` lines. To keep watching, prepend `sleep 30 && ` and repeat.

## Step 4: Final accounting (finished jobs)
```bash
ssh <SSH_ALIAS> "sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode 2>/dev/null | head -10"
```
Map the `State` to `JOB_STATE`. If `FAILED`/`CANCELLED` and a log is known, show the last 20 lines.

## Step 5: Optional cancel
Only if the user asked to cancel. Show the command and require confirmation:
```
Cancel job <JOB_ID> on <SSH_ALIAS>?
  scancel <JOB_ID>
Proceed? [y/n]
```
On explicit `y`:
```bash
ssh <SSH_ALIAS> "scancel <JOB_ID>"
```
Verify it left the queue:
```bash
ssh <SSH_ALIAS> "squeue -j <JOB_ID> -h 2>/dev/null || echo GONE"
```
Report. On anything other than `y`: do nothing.

## Step 6: Report
```
Job <JOB_ID> on <SSH_ALIAS>
  State:    <JOB_STATE> <reason if pending>
  Elapsed:  <from sacct, if available>
  Log:      <LOG_PATH or "unknown">
```
