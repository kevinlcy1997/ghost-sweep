# Receiving Codex Handoff

This bundle helps continue the `ghost-sweep` project on another machine.

## What This Bundle Contains

- `.codex/skills`
- `.codex/plugins`
- `.codex/memories`
- `.codex/memory`
- `.codex/config.toml`
- `.codex/AGENTS.md`
- `.codex/RTK.md`
- `rtk.exe`
- this handoff file
- a repo WIP patch, if generated beside this file

Sensitive/session-heavy Codex files are intentionally excluded from the portable bundle, including `auth.json`, `secrets`, `.sandbox-secrets`, `sessions`, large logs, and SQLite runtime state. Re-auth GitHub, Notion, or other connectors on the receiving machine if needed.

## Install On The Receiving Machine

1. Extract the zip somewhere temporary.

2. Copy the `.codex` folder into the receiving user's home directory:

```powershell
Copy-Item -Recurse -Force ".\.codex" "$env:USERPROFILE\.codex"
```

3. Copy RTK into the same expected path:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.local\bin" | Out-Null
Copy-Item -Force ".\rtk.exe" "$env:USERPROFILE\.local\bin\rtk.exe"
```

4. Add RTK to `PATH` if it is not already there:

```powershell
[Environment]::SetEnvironmentVariable(
  "Path",
  [Environment]::GetEnvironmentVariable("Path", "User") + ";$env:USERPROFILE\.local\bin",
  "User"
)
```

Restart the terminal after changing `PATH`.

5. Review machine-specific paths in:

```text
%USERPROFILE%\.codex\config.toml
%USERPROFILE%\.codex\AGENTS.md
```

Replace any old `C:\Users\Kevin\...` paths if the receiving Windows username is different.

6. Re-auth connectors if needed:

- GitHub connector / `gh auth login`
- Notion connector
- OpenAI/Codex auth
- Headroom, if used

## Ghost Sweep Repo Handoff

Original repo path:

```text
C:\Users\Kevin\Documents\Project\ghost-sweep
```

Notion project memory hub:

```text
https://app.notion.com/p/3906e655eec381e0b77cc1daf194a227
```

Ask the receiving Codex session to read/search this Notion hub first:

```text
Read Ghost Sweep Project Memory in Notion, then continue the spatial context feature pack work.
```

## Current Coding Checkpoint

Current task: add a Stage 2 spatial context feature pack based on feature correlation analysis.

Important files:

```text
docs/superpowers/specs/2026-07-01-spatial-context-feature-pack-design.md
docs/superpowers/plans/2026-07-01-spatial-context-feature-pack.md
ghost_ranking_features.py
analysis/run_zone_ranking_experiment.py
tests/test_engineered_ranking_features.py
tests/test_two_stage_experiment.py
```

Current verified status before this handoff:

```powershell
C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_engineered_ranking_features.py tests/test_two_stage_experiment.py -q -p no:cacheprovider --basetemp .pytest_tmp_spatial_features
# 9 passed

C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_spatial_sampling.py tests/test_model_iteration.py tests/test_multi_horizon_iteration.py tests/test_two_stage_experiment.py tests/test_engineered_ranking_features.py -q -p no:cacheprovider --basetemp .pytest_tmp_spatial_features
# 17 passed
```

Full two-stage training timed out after around 10 minutes:

```powershell
C:\Users\Kevin\.local\bin\rtk.exe C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe analysis/run_two_stage_experiment.py
# timed out after ~604 seconds
```

Likely cause: hotspot distance computation is correct but too slow for the full training table. Next step is to optimize/batch/cache hotspot distance computation, then rerun full two-stage training and write a Notion checkpoint.

## Suggested Resume Prompt

```text
Continue Ghost Sweep spatial context feature pack. Read Notion Ghost Sweep Project Memory and the plan at docs/superpowers/plans/2026-07-01-spatial-context-feature-pack.md. Current status: feature tests and focused modeling tests pass; full analysis/run_two_stage_experiment.py timed out after 10 minutes, likely due to hotspot distance computation. Optimize or batch that computation, rerun training, then write a Notion checkpoint.
```

