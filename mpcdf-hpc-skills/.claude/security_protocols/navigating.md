# Security Protocol: Navigating

Claude runs navigation steps automatically without asking. Confirmation is required before reading file contents and before any action that creates, changes, or removes anything.

## Runs automatically (no confirmation)
- SSH connection checks (`ssh -O check`, `hostname`), `sinfo`, `squeue`, `module avail`
- Directory listings (`ls`) of directories explicitly named in the active skill
- Existence checks (`test -d`, `test -f` that print EXISTS/MISSING/FOUND)
- Opening applications (e.g. VS Code via `code`)

## Requires explicit user confirmation ("y/n")
- Reading file contents (`cat`, `head`, reading any file text)
- Cloning a repository; creating any file or directory
- Writing to / overwriting / deleting anything
- Submitting or cancelling a Slurm job
- Modifying SSH config or any other config file

## The only questions you may ask
1. **MPCDF username** — always ask; never infer.
2. **Which project to open** — only if the user has not already named one.
3. **Which environment to use** — only on genuine ambiguity.

Do NOT ask "Shall I proceed?" when intent is already clear from the request or a prior step.

## Always applies
- NEVER write to any file without explicit user confirmation.
- NEVER ask for a password or OTP. NEVER echo or log credential values.
- On any failure: stop immediately, show the full error, do not retry.
