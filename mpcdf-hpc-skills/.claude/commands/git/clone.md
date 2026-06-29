# /git:clone — Clone a git repository

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

1. Read `skills/git/clone.md` and follow every step exactly, in order.
   If cloning onto a cluster, it will resolve the target host via an SSH alias
   and auto-discover the projects directory.
2. Do not proceed to the next step until the current step is complete and any
   required confirmation has been received.
