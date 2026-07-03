# Ghost Sweep — Headroom Proxy for GitHub Copilot CLI

**Date:** 2026-07-03
**Status:** Draft
**Scope:** Phase 1 — repo-local setup and launch flow for Headroom-wrapped GitHub Copilot CLI on Windows

---

## 1. Problem Statement

The repository does not currently provide a repeatable way to run GitHub Copilot CLI through Headroom. The user wants “the proxy” for their GitHub Copilot usage, but the documented upstream support is for **GitHub Copilot CLI**, not the VS Code Copilot chat extension.

The goal of this phase is to add a boring, local, reversible workflow that lets a developer in this repo launch Copilot CLI through a Headroom proxy, verify it works, and stop without permanently mutating the global Copilot setup.

---

## 2. Approved Scope

The following decisions are locked for v1:

- **Target:** GitHub Copilot CLI only
- **Platform:** Windows-first, using PowerShell
- **Install style:** repo-local and reversible
- **Proxy mode:** Headroom upstream, not a custom proxy written in this repo
- **Auth path:** prefer existing Copilot login reuse, support explicit token fallback
- **Persistence:** no durable global install in v1

Important consequence: this design does **not** attempt to route the VS Code Copilot chat extension itself through Headroom. Upstream documents `headroom wrap copilot`; it does not clearly document a supported path for the VS Code extension transport.

---

## 3. Environment Facts

Current local environment facts observed during design:

- `copilot` CLI is installed (`GitHub Copilot CLI 1.0.63`)
- Node and npm are installed
- repo Python venv exists at `.venv`
- `.venv\Scripts\python.exe` is Python `3.12.10`
- Docker is not installed
- MSVC `link.exe` is not on `PATH`
- Rust is not installed

These matter because upstream Headroom documentation says Windows may require a native build path with MSVC and Rust when a usable wheel is not available.

---

## 4. Objectives and Non-Goals

### 4.1 Objectives

1. Give this repo a one-command path to launch Copilot CLI through Headroom.
2. Keep the setup local to the repo instead of changing the user’s global Copilot configuration.
3. Fail fast on missing Windows prerequisites with concrete messages.
4. Support both standard GitHub.com accounts and enterprise-hosted Copilot deployments.
5. Leave behind one small smoke test that proves the proxy path works.

### 4.2 Non-Goals

- No custom compression proxy implementation in this repo.
- No VS Code Copilot extension interception.
- No Docker-based flow in v1.
- No background Windows service or persistent proxy daemon in v1.
- No automatic secrets discovery or token storage logic beyond what Headroom already ships.

---

## 5. Approaches Considered

### 5.1 Recommended: Repo-local wrapper around Headroom

Use Headroom as designed and add a repo-local PowerShell entrypoint that:

- checks prerequisites
- installs Headroom into a dedicated local venv
- launches `headroom wrap copilot --subscription`
- runs a smoke test

Why this wins: smallest diff, uses upstream behavior, easy to delete later.

### 5.2 Alternative: Docker or WSL wrapper

Use a containerized Headroom flow and keep native Windows toolchain concerns out of the repo.

Why not for v1: Docker is absent on this machine, so this adds setup work before value.

### 5.3 Rejected: Write a custom proxy in Ghost Sweep

Build a local HTTP proxy or wrapper here that mimics the part of Headroom the user needs.

Why rejected: duplicate functionality, higher maintenance, more fragile than using upstream.

---

## 6. Chosen Design

V1 adds a single Windows-first repo entrypoint, `deploy\headroom.ps1`, plus a small dedicated local virtual environment at `.headroom-venv`.

The script acts as a thin orchestrator over upstream Headroom. It does not reimplement proxy logic. It exposes a few explicit actions:

- `preflight` — verify Copilot CLI, Python, and Windows build prerequisites
- `install` — create `.headroom-venv` and install `headroom-ai`
- `launch` — start `headroom wrap copilot --subscription` with passthrough Copilot args
- `verify` — run a smoke prompt and confirm `HEADROOM_OK`

This keeps the surface area small and matches the existing repo style of practical deployment scripts.

---

## 7. File and Responsibility Plan

| File | Responsibility |
|------|----------------|
| `deploy\headroom.ps1` | Single entrypoint for preflight, install, launch, and verify |
| `docs\superpowers\specs\2026-07-03-headroom-copilot-proxy-design.md` | This design record |

No extra helper modules are needed in v1. If the PowerShell script becomes hard to read, helper extraction can happen later.

---

## 8. Command Surface

Suggested command shape:

```powershell
.\deploy\headroom.ps1 preflight
.\deploy\headroom.ps1 install
.\deploy\headroom.ps1 launch -- --model claude-sonnet-4-20250514
.\deploy\headroom.ps1 verify
```

Behavior rules:

- use `.headroom-venv`, not the existing ML `.venv`
- pass everything after `--` straight through to Copilot CLI
- prefer `headroom wrap copilot --subscription`
- do not modify persistent global Copilot config in v1

---

## 9. Install and Launch Flow

### 9.1 Preflight

`preflight` checks:

- `copilot` command exists
- Python 3.10+ is available
- if Headroom is not already installed, whether Windows build prerequisites are present:
  - `link.exe`
  - `rustc`

If prerequisites are missing, the script stops with a concrete message instead of attempting install.

### 9.2 Install

`install`:

1. creates `.headroom-venv`
2. upgrades pip
3. installs `headroom-ai[proxy]`
4. verifies `headroom --version`

The install stays isolated from repo ML dependencies and avoids unneeded extras in v1.

### 9.3 Launch

`launch`:

1. activates or directly calls the `.headroom-venv` interpreter
2. runs `headroom wrap copilot --subscription`
3. forwards any model or prompt args after `--`

Default launch should not guess a model. If the user passes no model, Copilot CLI/Headroom behavior stands as-is.

### 9.4 Verify

`verify` runs a small end-to-end prompt:

```text
Reply with exactly: HEADROOM_OK
```

Success criteria: the command completes and prints `HEADROOM_OK`.

---

## 10. Auth and Enterprise Handling

The design must support two auth paths:

### 10.1 Preferred path

Reuse the user’s existing Copilot CLI login and let Headroom discover the reusable bearer token.

### 10.2 Fallback path

If discovery fails, the script should tell the user exactly which environment variables Headroom supports:

- `GITHUB_COPILOT_TOKEN`
- `GITHUB_COPILOT_GITHUB_TOKEN`

For enterprise or custom-domain deployments, the script should support passthrough environment configuration rather than inventing new config files. Supported upstream variables include:

- `GITHUB_COPILOT_API_URL`
- `GITHUB_COPILOT_ENTERPRISE_DOMAIN`

The script should not print secret values.

---

## 11. Error Handling

The flow should stop early and clearly in the following cases:

- Copilot CLI missing
- Python missing or too old
- native Windows build prerequisites missing
- Headroom install failure
- Copilot auth discovery failure
- verify prompt not returning the exact smoke string

Each failure should end with one next action, not a wall of logs. Example: install MSVC build tools, install Rust, run `copilot` login once, or export a fallback token variable.

---

## 12. Testing and Verification

V1 leaves behind one runnable check instead of a large test suite.

The required check is a script-level smoke test driven by the `verify` action:

- install succeeds
- wrapped Copilot launch succeeds
- the response is exactly `HEADROOM_OK`

This is enough for the setup script because the logic is orchestration, not business logic.

---

## 13. Rollout Plan

Implementation order:

1. add `deploy\headroom.ps1`
2. implement `preflight`
3. implement `install`
4. implement `launch`
5. implement `verify`
6. run the smoke check locally as far as local prerequisites allow

If local prerequisites block a full install, the script is still acceptable for v1 if:

- preflight detects the blocker correctly
- install fails with the expected guidance
- no unrelated repo behavior changes

---

## 14. Out of Scope for Later

Possible later additions, not part of this design:

- VS Code Copilot extension routing if upstream documents a supported path
- Docker-based fallback flow
- persistent background proxy mode
- shared config file for model defaults
- telemetry/dashboard wrappers around Headroom stats

---

## 15. Acceptance Criteria

This design is successful when:

1. the repo contains a documented, local Headroom setup path for Copilot CLI
2. the script does not alter the user’s normal global Copilot flow unless explicitly run
3. missing Windows prerequisites are reported clearly
4. a user with the needed prerequisites can run the verify flow and get `HEADROOM_OK`
