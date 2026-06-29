---
name: data-edit-env
description: >
  Open the .env of a remote MPCDF project reliably in VS Code for the user to
  fill in secret values; falls back to SSH/nano instructions if VS Code is
  unavailable. Use after copying .env.example -> .env, or whenever the user
  needs to set LOCAL_MODELS_DIR, HUGGINGFACE_TOKEN, MinIO/Mongo creds, etc.
metadata:
  version: "1.0.0"
---

# Skill: data-edit-env

Claude never writes secret values itself — it opens the file for the user.

## Inputs
- `REPO_PATH`, `SSH_ALIAS`, `CLUSTER`; optional `ENV_VARS` (list of required keys from `.env.example`).

## Security protocol
Caller's protocol; default `.claude/security_protocols/controlled.md`.
Non-negotiable: NEVER write, echo, or log secret values. Refer to secrets by name only.

---

## Step 1: VS Code available?
```bash
which code 2>/dev/null && echo CODE_FOUND || echo CODE_MISSING
```

## Step 2: Open (folder-uri context, then file-uri — reliable Remote SSH)
`CODE_FOUND`:
```bash
code --folder-uri "vscode-remote://ssh-remote+<SSH_ALIAS><REPO_PATH>"
sleep 2 && code --file-uri "vscode-remote://ssh-remote+<SSH_ALIAS><REPO_PATH>/.env"
```
Then tell the user which keys to fill (from `ENV_VARS`), e.g.:
```
.env opened in VS Code on <CLUSTER>. Fill in (at least):
  LOCAL_MODELS_DIR=/ptmp/<user>/models      # where model weights live (scratch)
  HUGGINGFACE_TOKEN=hf_...                   # only for gated models
  <other ENV_VARS>
Save the file, then continue.
```

`CODE_MISSING`:
```
VS Code CLI not found. Edit the file manually on <CLUSTER>:
  ssh <SSH_ALIAS>
  nano <REPO_PATH>/.env
Fill in: <ENV_VARS>. Save, then continue.
```

Wait for the user to confirm they have saved before returning.
