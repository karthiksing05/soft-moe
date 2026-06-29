---
name: inference-llm-service
description: >
  Use the MPCDF LLM Inference Service (https://llm.mpcdf.mpg.de) — a managed web
  app that submits the Slurm job and routes an OpenAI-compatible REST endpoint for
  an open model on Dais or Viper-GPU, using vLLM or Ollama. Use when the user
  wants an interactive endpoint for an open model without writing job scripts,
  asks "how do I serve a model on MPCDF", "spin up an endpoint", or wants to
  evaluate a model interactively. For non-interactive batch eval, prefer the
  self-hosted vLLM skills (run-vllm) instead.
metadata:
  version: "1.0.0"
---

# Skill: inference-llm-service

The managed alternative to self-hosting vLLM. Best for **interactive** model
evaluation and user studies. For extensive benchmarks / offline evaluation,
MPCDF recommends Slurm batch jobs instead → use `skills/inference/run-vllm.md`.

Reference: the "Running open-model inference" section of
`reference/MPCDF-CLUSTER-FACTS.md`, and
https://docs.mpcdf.mpg.de/doc/computing/software/data_analytics-machine_learning.html

## What it does
- Web UI at `https://llm.mpcdf.mpg.de`; log in with **Kerberos** credentials.
- You pick hardware + framework (vLLM or Ollama) + model; it submits the Slurm
  job and routes an **OpenAI-compatible REST endpoint** you call from your
  laptop / existing tools.
- Connected to **Dais** and **Viper-GPU**.

## Security protocol
Caller's protocol; default `.claude/security_protocols/controlled.md`. Claude does
not drive the web UI or handle Kerberos credentials — it guides the user and then
helps wire client code to the endpoint.

---

## Step 1: Confirm prerequisites
Tell the user:
```
To use the LLM Inference Service:
  1. Active MPCDF account with HPC access enabled.
  2. You must have logged in directly at least once to the target system
     (Dais or Viper-GPU) before creating your first endpoint there.
  3. Go to https://llm.mpcdf.mpg.de and sign in with your Kerberos credentials.
```
If the user has never logged into the target system, point them to the login
command (`/cluster:login`) first.

## Step 2: Guide endpoint creation
Walk the user through the UI: choose the system (Dais for big models / more
GPUs; Viper-GPU for ROCm), the framework (vLLM for OpenAI-compatible serving;
Ollama for quick local-style use), the model, and hardware. The service submits
the job and exposes the endpoint URL + any token.

## Step 3: Wire up a client (OpenAI-compatible)
Once the user has the endpoint URL (and key, if shown), help them call it. The
endpoint speaks the OpenAI schema, so the `openai` SDK works by overriding
`base_url`:
```python
from openai import OpenAI
client = OpenAI(base_url="<ENDPOINT_URL>/v1", api_key="<TOKEN_OR_PLACEHOLDER>")
resp = client.chat.completions.create(
    model="<MODEL_NAME_AS_SHOWN_BY_SERVICE>",
    messages=[{"role": "user", "content": "..."}],
)
print(resp.choices[0].message.content)
```
Never hard-code the token in shared code; read it from env. Confirm reachability:
```bash
curl -s <ENDPOINT_URL>/v1/models -H "Authorization: Bearer $LLM_TOKEN" | head
```

## Step 4: When to switch to self-hosting
If the user needs extensive batch generation, reproducible offline evaluation,
or a custom serving config, recommend the Slurm-based path:
```
For non-interactive / large-scale runs, use the self-hosted vLLM job instead:
  /toolbox:run-vllm   (skills/inference/run-vllm.md)
Example batch scripts also live in the LLMs-meet-MPCDF GitLab repo.
```

## Step 5: Report
State the endpoint URL, the system it runs on, the framework, and that the
endpoint persists only as long as its Slurm job; the user re-creates it via the
UI when needed.
