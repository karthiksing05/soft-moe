# /toolbox:setup-repo — Clone, branch, and set up a project environment

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

Execute in order, completing each fully before the next:

1. Read `skills/cluster/resolve.md`. → `CLUSTER`, `SSH_ALIAS`.
2. Read `skills/cluster/ssh-login.md`. Pass `CLUSTER`, `SSH_ALIAS`. → `CLUSTER_USER`.
3. Read `skills/git/clone.md`. Pass `TARGET_HOST=<SSH_ALIAS>`. → `REPO_PATH`.
4. Read `skills/git/create-branch.md`. Pass `TARGET_HOST=<SSH_ALIAS>`, `REPO_PATH`. → `BRANCH_NAME`.
   (Never work on main/master; create a named branch first.)
5. Read `skills/cluster/setup-env.md`. Pass `REPO_PATH`, `SSH_ALIAS`.
   Step 8 of that skill delegates `.env` opening to `skills/data/edit-env.md`,
   so the user is prompted to fill in `LOCAL_MODELS_DIR`, `HUGGINGFACE_TOKEN`, etc.
