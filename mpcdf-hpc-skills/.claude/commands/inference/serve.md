# /inference:serve — Serve an open model via the MPCDF LLM Inference Service

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

This command uses the managed service at https://llm.mpcdf.mpg.de (Dais /
Viper-GPU), which submits the Slurm job and routes an OpenAI-compatible endpoint
for you. For non-interactive batch evaluation, prefer `/inference:run-vllm`.

1. Read `skills/inference/llm-inference-service.md` and follow every step exactly.
   It guides Kerberos login, model/hardware/framework selection, and wiring an
   OpenAI-compatible client to the routed endpoint.
