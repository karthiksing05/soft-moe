# soft-moe — research repo + MPCDF HPC automation

This repository explores **soft-MoE-style partitioning within a single LLM**: how to
condition subspaces of the weight matrix on the input so one model learns multiple
"experts" (supervised and unsupervised variants). See `BRAINSTORM.md`, `README.md`,
and the `plan/` folder (`01-data-collection`, `02-training`, `03-evaluation`,
`THEORY.md`) for the research design.

Executing that plan — collecting data, training, and running inference — happens on
the **MPCDF clusters** (Raven, Dais, Viper-GPU, Viper-CPU). The reusable Claude Code
skill library for those workflows lives in `mpcdf-hpc-skills/` and is wired into this
working directory:

- Slash commands: `.claude/commands/` → `mpcdf-hpc-skills/.claude/commands/`
- Security protocols: `.claude/security_protocols/` → `mpcdf-hpc-skills/.claude/security_protocols/`
- Permission allowlist: `.claude/settings.json` → `mpcdf-hpc-skills/.claude/settings.json`
- Skill files (read by literal relative path): `skills/` → `mpcdf-hpc-skills/skills/`
- Cluster facts (single source of truth): `reference/` → `mpcdf-hpc-skills/reference/`

These are symlinks, so `mpcdf-hpc-skills/` stays the single source of truth — edit the
files there, not the links. The full operating guide for the HPC workflows (your role,
non-negotiable behavior rules, global safety rules, and the skill/command catalogues)
is imported below.

@mpcdf-hpc-skills/CLAUDE.md
