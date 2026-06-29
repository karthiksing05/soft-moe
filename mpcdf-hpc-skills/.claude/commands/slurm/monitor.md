# /slurm:monitor — Check, tail, or cancel a SLURM job

If `ACTIVE_PROTOCOL` is already set in this session, use it and tell the user:
  "Using active security protocol: <ACTIVE_PROTOCOL>"
Otherwise ask:
```
Which security mode should this workflow use?
  - automated:  runs without y/n prompts; stops only on errors or ambiguous inputs
  - navigating: navigation and checks run automatically; confirmation required before any write
  - controlled: asks for confirmation before every action

Enter a mode:
```
If the input is not one of the three options, ask again. Store as ACTIVE_PROTOCOL.
Read `.claude/security_protocols/<ACTIVE_PROTOCOL>.md`.

1. Resolve the target: use `SSH_ALIAS` from session context, or read
   `skills/cluster/resolve.md` to derive it from `CLUSTER`.
2. Read `skills/slurm/monitor-job.md` and follow every step exactly.
   Pass `SSH_ALIAS` and `JOB_ID` (ask the user if not provided).
   Cancellation always requires explicit confirmation.
