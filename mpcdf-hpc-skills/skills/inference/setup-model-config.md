---
name: inference-setup-model-config
description: >
  Assess and set up a vLLM experiment config for the toolbox project on any MPCDF
  cluster. Reads every existing config first, then branches: reuse an existing
  config for the model, adapt a blueprint to a new model, or guide first-config
  creation for a new project. Use when the user says "set up a config", "create a
  config for [model]", "what configs do I have", or after downloading a model.
metadata:
  version: "2.0.0"
---

# Skill: inference-setup-model-config

Understand the config landscape, then guide to the right config. **Read every
config before presenting anything** — never show filenames alone.

Transformation rules when adapting a blueprint:
- `experiment_name` → unchanged
- `models:` → replaced: key = MODEL_ID, `model_provider: vllm`, `api_base: http://localhost:8000/v1`, `generation_arguments` carried over, provider-specific fields removed
- `evals:` and `response_strategies:` → kept exactly

## Inputs
- `REPO_PATH`, `SSH_ALIAS`, `CLUSTER`, `MODEL_ID`.

## Output
On success: `CONFIG_FILE` (relative to REPO_PATH). On failure: exact error + step. Stop.

## Security protocol
Caller's protocol; default `.claude/security_protocols/controlled.md`. Also: never overwrite a config without confirmation; never modify an existing config (only create new); never write `.env`.

> `ssh <SSH_ALIAS> "..."` remote form; omit when local.

---

## Step 1: Resolve context
Resolve CLUSTER/REPO_PATH/MODEL_ID (verify `Master running` if remote). Validate MODEL_ID has one `/`.

## Step 2: Read all configs
```bash
ssh <SSH_ALIAS> "find <REPO_PATH>/configs -name '*.yaml' | sort"
```
None → Branch C. Else read all in one pass:
```bash
ssh <SSH_ALIAS> "for f in \$(find <REPO_PATH>/configs -name '*.yaml' | sort); do echo \"=== \$f ===\"; cat \"\$f\"; echo; done"
```
For each, note internally: `experiment_name`; `models:` entries (key, provider, api_base, generation_arguments); `evals:` task names; whether any model key contains the model-name part of MODEL_ID (after `/`, case-insensitive). Build understanding silently.

## Step 3: Branch
- A model key matches MODEL_ID (exactly or contains the name part) → **Branch A**.
- Else → **Branch B**.

## Branch A: config already exists
```
I found an existing config for <MODEL_ID>:
  File: configs/<path>   Experiment: <name>   Evals: <list>   Provider: vLLM
Use this config as-is? [Y/n]  (n → see all configs and pick a different blueprint)
```
Y/Enter → `CONFIG_FILE` = this path → Step 8. n → show the Branch B menu and let them pick.

## Branch B: adapt a blueprint (one line per config)
```
No config found yet for <MODEL_ID>. Pick a blueprint:

  1. <experiment_name>  [<eval_1> · <eval_2> · ...]  — configs/<path>
  2. <experiment_name>  [<eval_1> · <eval_2> · ...]  — configs/<path>
  ...
<one contextual note:>
  - all same experiment+evals → "Adapting any runs <MODEL_ID> on the same evaluation."
  - differing → "These run different experiments; pick the matching evaluation."
  - only API configs → "Adapting converts the model block to a local vLLM endpoint; prompts/eval logic unchanged."
Which should I use as a blueprint? Enter a number:
```
Store `BLUEPRINT_PATH`; extract `BLUEPRINT_GENERATION_ARGS` and `BLUEPRINT_EXPERIMENT_NAME`. → Step 4.

## Branch C: new project
Discover prompt/question/data material:
```bash
ssh <SSH_ALIAS> "find <REPO_PATH> -maxdepth 4 \( -name '*.yaml' -o -name '*.json' -o -name '*.txt' -o -name '*.csv' \) ! -path '*/.git/*' ! -path '*/__pycache__/*' ! -path '*/configs/*' | sort | head -30"
ssh <SSH_ALIAS> "ls <REPO_PATH>/"
```
Offer two paths: (1) **Open VS Code** — read `skills/cluster/open-vs-code.md`, open the project, tell the user to start a Claude Code session there to design the config against their files; stop (hands off). (2) **Answer questions here** — ask experiment name, eval tasks (+ prompt paths), response format; build a skeleton:
```yaml
experiment_name: <answer1>
models:
  <MODEL_ID>:
    model_provider: vllm
    api_base: http://localhost:8000/v1
    generation_arguments:
      temperature: 0.7
      max_tokens: 512
evals:
  <task>:
    prompts: <path or TODO>
response_strategies:
  default:
    type: <answer3>
```
Show it; `Write to configs/<experiment_slug>_<model_slug>.yaml? [y/n]` (or "vscode"). y → `OUTPUT_FILENAME` set, skip to Step 6 with the skeleton. vscode → Path 1. n → stop.

## Step 4: Output filename (Branch B)
Derive from the blueprint name, swapping the model suffix for a slug of MODEL_ID (e.g. `Qwen/Qwen2.5-32B-Instruct` → `qwen25_32b`), same subdir. automated → use it; else ask `Use this name? [y]` or a different one. Store `OUTPUT_FILENAME`.

## Step 5: Existing file check
```bash
ssh <SSH_ALIAS> "test -f <REPO_PATH>/configs/<OUTPUT_FILENAME> && echo EXISTS || echo MISSING"
```
`EXISTS` → `Overwrite? [y/n]`; n → keep + report `CONFIG_FILE=configs/<OUTPUT_FILENAME>` + stop.

## Step 6: Construct + review
Apply the transformation rules (or use the skeleton). Show the full result. automated → write; else `Write this file? [y/n]`.

## Step 7: Write + verify
```bash
cat << 'HEREDOC' | ssh <SSH_ALIAS> "cat > <REPO_PATH>/configs/<OUTPUT_FILENAME>"
<config content>
HEREDOC
ssh <SSH_ALIAS> "test -f <REPO_PATH>/configs/<OUTPUT_FILENAME> && echo WRITTEN || echo FAILED"
```
(Local: redirect directly.) `FAILED` → report error, stop.

## Step 8: Report
```
Config ready.
  File: configs/<OUTPUT_FILENAME>   Model: <MODEL_ID>   Experiment: <name>   Evals: <list>
Run it with /toolbox:run-vllm
CONFIG_FILE=configs/<OUTPUT_FILENAME>
```
