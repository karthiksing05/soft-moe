# Security Protocol: Automated

Claude proceeds without confirmation for all expected, safe operations. The user sees progress output and errors — not a sequence of y/n prompts.

## Runs automatically (no confirmation)
- All read operations: directory listings, reading files, connection/scheduler checks
- Any command the active skill explicitly defines as part of its workflow
- Installing packages, creating venvs, `uv sync`, syncing dependencies
- Copying template files (e.g. `.env.example` → `.env`) when no file exists
- Creating directories explicitly named in the skill; appending to `.gitignore`
- Pulling a prebuilt container image named in the skill
- Submitting and monitoring a Slurm job the workflow was asked to run
- Running pre-commit if the repo docs mention it

## ALWAYS requires explicit user input
- **MPCDF username** — ask once at SSH login. Never infer.
- **Repository URL**, **branch name**, **`--account` string** — ask if not provided.
- **SSH config writes** — show exactly what will be appended to `~/.ssh/config` and ask "May I append this? [y/n]".
- **Choosing between pip extras** when the docs make no choice obvious.
- **Cancelling a running job** — always confirm.

## Immediate stop (no retry, no fallback)
- Any command exits non-zero or with unexpected output
- A file/dir the workflow expected to create fresh already exists (e.g. `.env`) — report and ask
- Ambiguous inputs not resolvable from context

## Output style
After each step print one line: `✓ <what was done>` or `✗ <what failed — full error follows>`. State results, not intentions.

## Always applies
- NEVER store or echo credential values. Refer by name only.
- NEVER push to `main`/`master`. NEVER overwrite an existing `.env`.
- NEVER modify an existing virtual environment without asking.
- On any failure: stop immediately and show the full error.
