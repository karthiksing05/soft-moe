# soft-moe — research repo + MPCDF HPC automation

This repository contrasts the **EM expert-token finetuning** technique from
`papers/master_thesis_stream_a.pdf` — conditioning a *single* shared backbone on a bank of
EM-trained *expert tokens* (a two-phase Phase-A/Phase-B protocol) — against the **standard
MoE** (real per-expert capacity) and **general finetuning** (a dense model), in a
capacity-constrained multi-domain setting where the MoE's capacity is necessary. See
`README.md` for the experiment, findings, and repository map. The headline comparison lives
in `configs/experiment/*_d256.yaml` and `reports/comparison/`.

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
