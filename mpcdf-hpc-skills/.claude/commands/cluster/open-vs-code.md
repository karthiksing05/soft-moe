# /cluster:open-vs-code — Open a cluster project in VS Code

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

1. Read `skills/cluster/resolve.md`. → `CLUSTER`, `SSH_ALIAS`.
2. Read `skills/cluster/ssh-login.md`. Pass `CLUSTER`, `SSH_ALIAS`. → `CLUSTER_USER`.
   Skip if the connection is already confirmed active.
3. Read `skills/cluster/select-project.md`. Pass `CLUSTER_USER`, `SSH_ALIAS`. → `REPO_PATH`.
   Skip if `REPO_PATH` is already set.
4. Read `skills/cluster/open-vs-code.md`. Pass `REPO_PATH`, `SSH_ALIAS` (folder-uri form).
