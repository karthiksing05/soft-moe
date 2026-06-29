# /cluster:logout — Disconnect from a cluster

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

1. Read `skills/cluster/resolve.md` and follow every step exactly. → `CLUSTER`, `SSH_ALIAS`.
   If `CLUSTER` is already confirmed in this session, the skill uses it directly.
2. Read `skills/cluster/ssh-logout.md` and follow every step exactly. Pass `SSH_ALIAS`.
