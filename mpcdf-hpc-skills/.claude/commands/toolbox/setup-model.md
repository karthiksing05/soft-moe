# /toolbox:setup-model — Ensure a model has weights and a vLLM config (discovery-first)

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

This command ensures a model has weights downloaded and a vLLM config ready.
It discovers what already exists before asking anything, then only does what is missing.

`CLUSTER`, `SSH_ALIAS`, and `REPO_PATH` must already be set from a prior skill in
this session (e.g. from `/toolbox:setup-repo` or `/toolbox:setup-vllm`). If they
are not set, tell the user to run one of those first to establish cluster context.

---

**Step 1: Ask for the model**

```
Which HuggingFace model do you want to use for inference?
Enter the full model identifier, for example:
  Qwen/Qwen2.5-32B-Instruct
  meta-llama/Meta-Llama-3-70B-Instruct
  deepseek-ai/deepseek-llm-67b-chat

Model ID:
```
Store as `MODEL_ID`. Validate it contains exactly one `/`. If not, ask again.

**Step 2: Discover current state** (run all checks without stopping between them)

Read `LOCAL_MODELS_DIR` from `.env`:
```bash
ssh <SSH_ALIAS> "grep -s '^LOCAL_MODELS_DIR=' <REPO_PATH>/.env 2>/dev/null | head -1 | cut -d'=' -f2-"
```
Empty → note it; handle via the special case below.

Check weights:
```bash
ssh <SSH_ALIAS> "test -d \"<LOCAL_MODELS_DIR>/<MODEL_ID>\" && echo MODEL_EXISTS || echo MODEL_MISSING"
```
Check for an existing config (search by the model-name slug after `/`, lowercased):
```bash
ssh <SSH_ALIAS> "find <REPO_PATH>/configs -name '*.yaml' 2>/dev/null | xargs grep -li '<model_name_slug>' 2>/dev/null | head -3"
```
One or more paths → `CONFIG_STATUS=FOUND`, store first match as `EXISTING_CONFIG`. Else `CONFIG_STATUS=MISSING`.

**Step 3: Report state and act**
```
Model:   <MODEL_ID>
Weights: <✓ present at LOCAL_MODELS_DIR/MODEL_ID | ✗ not downloaded>
Config:  <✓ found at EXISTING_CONFIG | ✗ none found>
```
Branch on what is missing:
- **Both present** → nothing to do. Set `CONFIG_FILE=<EXISTING_CONFIG>` and stop.
- **Weights present, config missing** → print `✓ Weights present. Setting up config...`, then read
  `skills/inference/setup-model-config.md` (pass `CLUSTER`, `SSH_ALIAS`, `REPO_PATH`, `MODEL_ID`). → `CONFIG_FILE`.
- **Weights missing, config present** → print `✓ Config exists. Downloading weights...`, set
  `CONFIG_FILE=<EXISTING_CONFIG>`, then read `skills/inference/download-model.md`
  (pass `CLUSTER`, `SSH_ALIAS`, `REPO_PATH`, `MODEL_ID`). → `MODEL_PATH`.
- **Both missing** → read `skills/inference/download-model.md` (→ `MODEL_PATH`), then
  `skills/inference/setup-model-config.md` (→ `CONFIG_FILE`). Pass `CLUSTER`, `SSH_ALIAS`, `REPO_PATH`, `MODEL_ID` to both.

**Special case — `LOCAL_MODELS_DIR` not set:** the `.env` is not filled in. Tell the user:
```
LOCAL_MODELS_DIR is not set in <REPO_PATH>/.env.
Run /toolbox:setup-vllm first — it checks all prerequisites including .env.
```
Stop here.
