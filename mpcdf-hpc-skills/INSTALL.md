# Installing into Claude Code

I can't install this onto your machine from this chat — what you have is a
downloadable folder. Below are the two ways to wire it into Claude Code. The
**project method** is the one this library is designed for, because the slash
commands and the orchestration guide only work when the whole repo is present.

## Method 1 — Project repo (recommended; gives you the slash commands)

Claude Code automatically picks up `CLAUDE.md`, `.claude/commands/`, and
`.claude/settings.json` from the working directory.

1. Unzip the download somewhere stable, e.g. `~/mpcdf-hpc-skills`.
2. Open that folder in Claude Code (or `cd` into it and start `claude`):
   ```
   cd ~/mpcdf-hpc-skills
   claude
   ```
3. The slash commands are now available — type `/` to see them (`/cluster:login`,
   `/toolbox:run-vllm`, `/training:submit`, …). `CLAUDE.md` is loaded as project
   context, and `.claude/settings.json` pre-allows the read-only/SSH commands the
   skills use.

To use it as the home for your own work, keep your project repos elsewhere on the
cluster (the skills clone into `/u/<user>/projects/...`) and just run Claude Code
from this folder. To share with teammates, commit this folder to a git repo; when
they clone it, they get the same commands and skills.

## Method 2 — Skills-only ZIP upload (claude.ai)

If you only want the individual **skills** (the `skills/**/*.md` files, each of
which has the required `name` + `description` YAML frontmatter) available in the
claude.ai UI, you can upload them under **Settings → Customize → Skills**. Note:

- ZIP each skill so the **folder is at the root of the archive**, not just its contents.
- This route does **not** bring the `/…` slash commands or `CLAUDE.md` orchestration —
  those are a Claude Code project feature (Method 1). Uploaded skills trigger by
  description match instead of being invoked by a command.

## After installing — sanity checks

- Frontmatter present on every skill (`name`, `description` within limits): already done.
- No hardcoded secrets anywhere: confirmed — secrets are only ever read from `.env`
  by name and never written or echoed.
- `.claude/settings.json` allowlist matches the commands the skills run (ssh, grep,
  ls, test, find, mkdir, which, code, printf, sleep, cat, …). Widen it in that file
  if your protocol prompts more than you'd like in `automated` mode.

## First run

```
/cluster:login         # pick the cluster; you'll be asked for your MPCDF username and
                       # to complete password + OTP in a separate terminal (Claude never
                       # sees your password or OTP)
```
Then chain into a workflow, e.g. `/toolbox:setup-project` or `/training:submit`.
