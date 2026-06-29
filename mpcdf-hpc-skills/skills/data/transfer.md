---
name: data-transfer
description: >
  Move data to/from / between MPCDF systems: rsync/scp/sftp through the gate
  gateway for small-medium transfers, Globus Online / DataHub for large-scale
  staging, DataShare for sharing, Nexus-S3 object storage for derived artefacts,
  and the archive system for long-term retention. Use when the user wants to
  upload/download data, copy a dataset to /ptmp, deploy a repo to the cluster,
  stage files, or sync results back. Picks the right tool for the size/purpose.
metadata:
  version: "1.0.0"
---

# Skill: data-transfer

Choose and run the right transfer method. Read the "Data transfer & sharing" and
"File systems" sections of `reference/MPCDF-CLUSTER-FACTS.md` first.

## Inputs (resolve interactively if absent)
- Direction & endpoints: source, destination (local path, `<SSH_ALIAS>:/ptmp/...`, S3 bucket, etc.).
- Approximate size (determines the method).

## Output
On success: confirmed destination path/location + a verification listing.
On failure: the exact error and the step that failed. Stop.

## Security protocol
Caller's protocol; default `.claude/security_protocols/controlled.md`. In addition:
- NEVER put secrets in command lines or URLs; read credentials from `.env`/env, by name only.
- Confirm before any transfer that overwrites existing files.
- Destinations on the cluster default to **`/ptmp/<user>/...`** (scratch) for large data, never `/u` (home, small quota) and never `/tmp`.

---

## Step 0: Pick the method

| Size / purpose | Method | Skill section |
|---|---|---|
| Small–medium (≤ tens of GB), repo deploy, results pull | `rsync` over `ProxyJump gate` | Step 1 |
| Large / many files / resumable / scheduled | Globus Online (DataHub) | Step 2 |
| Share with collaborators / external download | DataShare (Nextcloud) | Step 3 |
| Derived artefacts, S3-compatible, publishing | Nexus-S3 | Step 4 |
| Long-term keepers | Archive (`archive.mpcdf.mpg.de`) | Step 5 |

Confirm the chosen method with the user if ambiguous.

## Step 1: rsync / scp / sftp (via gate)
The cluster aliases already `ProxyJump gate`, so `rsync … <SSH_ALIAS>:…` works directly when a ControlMaster session is live. If not, use an explicit jump.

Upload to scratch:
```bash
rsync -avh --progress <local_src>/ <SSH_ALIAS>:/ptmp/<user>/<dst>/
```
Pull results back:
```bash
rsync -avh --progress <SSH_ALIAS>:/ptmp/<user>/runs/<exp>/ <local_dst>/
```
Explicit jump (no live ControlMaster, or cross-system):
```bash
rsync -avh -e 'ssh -J <user>@gate.mpcdf.mpg.de' <src>/ <user>@raven.mpcdf.mpg.de:/ptmp/<user>/<dst>/
```
Cluster ⇄ cluster: prefer routing through a host that has both mounts, or stage via Globus. Show the command; confirm (controlled) or run (automated). On non-zero exit, show output and stop.

Verify:
```bash
ssh <SSH_ALIAS> "ls -lh /ptmp/<user>/<dst>/ | head"
```

## Step 2: Globus Online / DataHub (large-scale)
For large or scheduled transfers, point the user to the MPCDF DataHub Globus
endpoint and the staging docs
(https://docs.mpcdf.mpg.de/doc/data/globusonline/index.html). Claude does not
drive the Globus web flow; summarise the steps: log in to Globus with MPCDF SSO,
select the MPCDF DataHub collection and the other endpoint, transfer the path on
`/ptmp` or home. Use Globus Flows for repeatable staging.

## Step 3: DataShare (sharing)
For sharing smaller datasets/files or exposing for external download, use
DataShare (Nextcloud). Point to the `ds`/`pocli` CLI and the web client; do not
handle credentials in plaintext.

## Step 4: Nexus-S3 (object storage for artefacts)
For derived artefacts (embeddings, registries, checkpoints) and publishing, use
Nexus-S3 (opt-in via SelfService). Configure an S3 client (e.g. `aws`/`mc`) with
credentials kept in env/`.env` (never echoed). Typical: `mc cp` / `aws s3 cp` to
the project bucket. This is the recommended place to **persist anything on
`/ptmp` that must survive purging.**

## Step 5: Archive (long-term)
For long-term retention, use the MPCDF archive (`archive.mpcdf.mpg.de`) per the
backup-and-archive docs. Reach it through the gate jump.

## Step 6: Report
State the method used, the exact destination, the verification listing, and — for
anything written to `/ptmp` — a reminder that scratch is purged and not backed up,
so keepers should be pushed to Nexus-S3 or the archive (Steps 4–5).
