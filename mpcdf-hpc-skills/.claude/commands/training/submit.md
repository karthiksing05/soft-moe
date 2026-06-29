# /training:submit — Submit and monitor a SLURM training job

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

Targets Raven (A100) or Dais (8× H200). Not Viper-GPU (ROCm inference only).

1. Read `skills/cluster/resolve.md`. → `CLUSTER`, `SSH_ALIAS`.
   If `CLUSTER=viper`, stop: training belongs on Raven or Dais.
2. Ensure a connection: if not connected, read `skills/cluster/ssh-login.md`. → `CLUSTER_USER`.
3. Read `skills/cluster/select-project.md`. Pass `CLUSTER_USER`, `SSH_ALIAS`. → `REPO_PATH`.
   Skip if `REPO_PATH` is already set.
4. Read `skills/training/submit-training.md` and follow every step exactly.
   Pass `REPO_PATH`, `CLUSTER`, `SSH_ALIAS`, `CLUSTER_USER`. → `JOB_ID`, `RUN_DIR`.
