# /toolbox:setup-vllm — Verify the vLLM inference stack

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

Execute in order, completing each fully before the next:

1. Read `skills/cluster/resolve.md`. → `CLUSTER`, `SSH_ALIAS`.
2. Check for an active connection:
   ```bash
   ssh -O check <SSH_ALIAS> 2>&1
   ```
   If the output lacks `Master running`, read `skills/cluster/ssh-login.md`
   (pass `CLUSTER`, `SSH_ALIAS`).
3. Read `skills/cluster/select-project.md`. Pass `CLUSTER_USER`, `SSH_ALIAS`. → `REPO_PATH`.
4. Read `skills/inference/setup-vllm.md`. Pass `REPO_PATH`, `CLUSTER`, `SSH_ALIAS`. → `VLLM_READY`.
   (CUDA image on raven/dais; ROCm image on viper.)
