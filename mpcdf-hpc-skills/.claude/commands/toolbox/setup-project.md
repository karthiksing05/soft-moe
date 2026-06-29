# /toolbox:setup-project — End-to-end: repo → model → vLLM → (optional) run

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

Runs the full toolbox setup, chaining the toolbox commands in sequence.
`CLUSTER`/`SSH_ALIAS`/`REPO_PATH` resolve in Step 1 and carry forward.

1. Read `.claude/commands/toolbox/setup-repo.md` and follow every step.
   → `CLUSTER`, `SSH_ALIAS`, `REPO_PATH`, `BRANCH_NAME`.
2. Read `.claude/commands/toolbox/setup-vllm.md` and follow every step.
   Pass the carried context. → `VLLM_READY`.
3. Read `.claude/commands/toolbox/setup-model.md` and follow every step.
   → `CONFIG_FILE`.
4. Ask: `Setup complete. Submit a vLLM inference job now? (yes / no)`
   - yes → read `.claude/commands/toolbox/run-vllm.md`, pass `CONFIG_FILE`.
   - no  → stop. The user can run `/toolbox:run-vllm` later.
