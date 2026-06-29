# Security Protocol: Controlled

Claude asks for confirmation before doing anything. Every command is shown before it runs. Nothing is written, changed, or deleted without explicit user approval.

## Rules
- NEVER ask for a password or OTP. You will not receive them and do not need them.
- NEVER echo or log credential values, even for confirmation. Refer to secrets by name only (e.g. "HUGGINGFACE_TOKEN is set").
- ALWAYS show the user every shell command BEFORE executing it.
- NEVER write to any file without explicit user confirmation.
- NEVER overwrite or delete existing file content without explicit user confirmation.
- On any failure: stop immediately, show the full error output, do not retry.
