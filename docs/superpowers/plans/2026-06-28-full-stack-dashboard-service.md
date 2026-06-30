# Full-Stack Dashboard Service Plan

I'm using the writing-plans skill with cost-aware multi-agent planning.

## Goal

Replace the fragile embedded-data dashboard with a local full-stack service that renders a dashboard shell and serves robust JSON APIs for fixed-grid coverage, feature marts, model summaries, predictions, and artifact traceability.

## Agent and Model Routing

| Workstream | Agent Role | Suggested Model Tier | Cost Control | Review Gate |
| --- | --- | --- | --- | --- |
| API contract | Builder | mid-tier | TDD tests for paths and response shapes only | Tests fail before implementation and pass after |
| Service implementation | Builder | mid-tier | Python stdlib server, no new dependency install | API tests pass and endpoints are paginated/filtered |
| Dashboard shell | Builder | mid-tier | Static HTML/JS fetched from service, no embedded data blob | Dashboard test verifies fetch-based UI contract |
| Verification | Reviewer | low-cost | Run focused tests, smoke API, then full suite | Command evidence and local URL are reported |

## Tasks

### 1. API Contract Tests

Files: `tests/test_dashboard_service.py`

**Agent Role:** Builder  
**Suggested Model Tier:** mid-tier  
**Why This Tier:** The task defines a compact HTTP/data contract over existing CSV/JSON artifacts.  
**Inputs Needed:** Existing artifacts under `analysis/geo`, `analysis/data_discovery`, and `analysis/dashboard_manifest_latest.json`.  
**Expected Output:** Failing tests for summary, coverage, timeseries, predictions, artifacts, and dashboard HTML.  
**Review Gate:** Tests fail for missing service module before implementation.

Steps:
- Add tests using direct handler/app helper functions rather than network sockets where possible.
- Assert JSON response shapes, pagination, filters, and dashboard HTML script references.

### 2. Service Implementation

Files: `analysis/dashboard_service.py`

**Agent Role:** Builder  
**Suggested Model Tier:** mid-tier  
**Why This Tier:** One focused module using Python standard library patterns.  
**Inputs Needed:** Task 1 tests and artifact file paths.  
**Expected Output:** Passing service tests and a runnable `python analysis/dashboard_service.py --port 8765`.  
**Review Gate:** Endpoints return bounded JSON, not giant embedded payloads.

Steps:
- Implement CSV/JSON loading helpers with caching based on file mtime.
- Implement `/api/summary`, `/api/coverage`, `/api/timeseries`, `/api/predictions`, `/api/artifacts`.
- Implement `/` dashboard page and static JS/CSS inline shell.

### 3. Verification

Files: tests and generated service only.

**Agent Role:** Reviewer  
**Suggested Model Tier:** low-cost  
**Why This Tier:** Deterministic command execution and smoke checks.  
**Inputs Needed:** Completed implementation.  
**Expected Output:** Passing focused tests, full suite, and a local URL.  
**Review Gate:** Report exact commands and whether the server was started.

Plan complete and saved to `docs/superpowers/plans/2026-06-28-full-stack-dashboard-service.md`.

Recommended execution: Inline Execution. The code surface is small and tightly coupled enough that TDD in the main session is cheaper than dispatching agents.
