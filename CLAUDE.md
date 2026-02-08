# rlee-tools

## Working Repository

The default repository is `/workspace/vllm`. However, the user may specify a different worktree (e.g. `/workspace/vllm-rendered`, `/workspace/vllm-embed-shape`). When a plan or task references a specific worktree path, **always use that worktree** instead of the default. You can list all worktrees with `git -C /workspace/vllm worktree list`.

## Environment Setup

**IMPORTANT:** Before starting work, verify that a `.venv` directory exists in the vllm repo (or the respective worktree) you've been assigned to:

```bash
ls -la <worktree>/.venv
```

If the `.venv` directory does not exist, **STOP immediately** and notify the user for further instructions. Do not proceed without a working virtual environment.

Once confirmed, activate the environment using:
```bash
source <worktree>/.venv/bin/activate
```

If you need to create a new environment, use:
```bash
uv venv
```
