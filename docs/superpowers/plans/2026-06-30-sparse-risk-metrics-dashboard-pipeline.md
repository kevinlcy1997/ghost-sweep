# Sparse Risk Metrics Dashboard Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. This plan includes cost-aware model routing; use the cheapest suggested tier that can satisfy each task's review gate, and escalate only when the listed escalation trigger occurs. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one coherent next step across model science, product dashboard, and pipeline hygiene: sparse ranking metrics plus probability-style outputs, dashboard API/UI exposure, and a repeatable retrain/startup path.

**Architecture:** Keep the model artifact producer as the source of truth, writing richer metrics and prediction columns from `analysis/run_model_iteration.py`. Let `analysis/dashboard_service.py` render those artifacts through bounded API endpoints and an interactive Leaflet map. Keep pipeline orchestration lightweight by extending `start-dev.ps1` rather than adding a new service.

**Tech Stack:** Python, pandas, scikit-learn, LightGBM, H3, stdlib HTTP server, Leaflet, PowerShell, pytest.

---

## File Structure

- `ghost_ranking_metrics.py`: pure metric helpers for sparse ranking, calibration, and group hit-rate evaluation.
- `analysis/run_model_iteration.py`: model selection, holdout scoring, prediction artifact columns, metadata, and HTML report output.
- `analysis/run_multi_horizon_experiment.py`: cross-horizon summary fields and report display.
- `analysis/dashboard_service.py`: API endpoints and HTML/JS dashboard that expose model metrics, probability bands, and horizon-specific map overlays.
- `analysis/build_dashboard_manifest.py`: traceability list for model metadata, summaries, and prediction artifacts.
- `start-dev.ps1`: optional `-RetrainModels` path to refresh model artifacts before serving dashboard and MLflow.
- `tests/test_zone_ranking_metrics.py`: unit tests for new sparse ranking/calibration metrics.
- `tests/test_model_iteration.py`: tests for selection logic and prediction artifact fields.
- `tests/test_dashboard_service.py`: API and HTML assertions for model metrics, horizon map overlay, and probability display.
- `tests/test_ci.py`: dev script assertions for retraining support.

## Agent and Model Routing

| Workstream | Agent Role | Suggested Model Tier | Cost Control | Review Gate |
| --- | --- | --- | --- | --- |
| Metric helpers | Builder | mid-tier | Edit one pure Python module plus focused tests | Unit tests cover sparse positives, calibration, and group hit-rate edge cases |
| Model artifact producer | Builder | high-capability | Touch only iteration scripts and tests | Metrics are consistent across folds, holdout, metadata, prediction CSVs, and selection sort |
| Dashboard API/UI | Builder | mid-tier | Touch service and dashboard tests only | API returns model metrics and map overlay properties without embedding large payloads |
| Pipeline hygiene | Builder | low-cost | Mechanical PowerShell and manifest updates | Tests confirm retrain flag, MLflow path, and manifest traceability |
| Final validation | Reviewer | low-cost | Run listed pytest slices and artifact smoke commands only | Failures are summarized with exact command and likely owner |

### Task 1: Add Sparse Ranking And Calibration Metrics

**Files:**
- Modify: `ghost_ranking_metrics.py`
- Modify: `tests/test_zone_ranking_metrics.py`

**Agent Role:** Builder
**Suggested Model Tier:** mid-tier
**Why This Tier:** The task is a focused pure-function implementation with deterministic tests.
**Inputs Needed:** Existing `precision_at_k`, `recall_at_k`, `top_decile_lift`, and current ranking metric tests.
**Expected Output:** New helpers for `brier_score`, `expected_calibration_error`, `group_hit_rate_at_k`, and `risk_band`, with passing metric tests.
**Review Gate:** Main agent verifies no model-training logic leaks into metric helpers and all empty/single-class edge cases are deterministic.

- [ ] **Step 1: Write failing tests**

Add tests asserting: Brier score equals mean squared probability error; expected calibration error returns zero for empty inputs and a bounded value for two bins; group hit-rate counts how many positive districts appear in top-k predictions; risk bands map probabilities to low, elevated, high, and critical.

- [ ] **Step 2: Run failing metric tests**

Run: `python -m pytest tests/test_zone_ranking_metrics.py -q`
Expected: FAIL because the new helper functions are not imported or defined.

- [ ] **Step 3: Implement minimal metric helpers**

Implement the functions in `ghost_ranking_metrics.py` using NumPy arrays, stable descending score order, and explicit input-length validation.

- [ ] **Step 4: Run metric tests**

Run: `python -m pytest tests/test_zone_ranking_metrics.py -q`
Expected: PASS.

### Task 2: Enrich Model Selection And Prediction Artifacts

**Files:**
- Modify: `analysis/run_model_iteration.py`
- Modify: `analysis/run_multi_horizon_experiment.py`
- Modify: `tests/test_model_iteration.py`

**Agent Role:** Builder
**Suggested Model Tier:** high-capability
**Why This Tier:** This changes model selection semantics and artifact contracts across horizons.
**Inputs Needed:** Task 1 metrics, current model iteration script, existing prediction CSV header.
**Expected Output:** Fold, summary, holdout, metadata, and predictions include practical sparse metrics and probability/risk fields.
**Review Gate:** Main agent verifies selection no longer leans on brittle `precision_at_20` alone and the output remains backward-compatible with `score`.

- [ ] **Step 1: Write failing tests**

Add tests for `select_satisfactory_model` preferring practical lift/AP/precision@100 over a spiky `precision@20`, and for a small prediction frame formatter producing `rank`, `probability`, `risk_band`, and preserving `score`.

- [ ] **Step 2: Run failing tests**

Run: `python -m pytest tests/test_model_iteration.py -q`
Expected: FAIL because the formatter and new selection fields do not exist.

- [ ] **Step 3: Implement model artifact changes**

Update `_score_predictions` to include `rows`, `positives`, `base_rate`, `precision_at_100`, `recall_at_50`, `recall_at_100`, `brier_score`, `expected_calibration_error`, and district/region hit-rate metrics where group columns are available. Add a small `format_prediction_artifact()` helper to sort by probability and assign rank/risk bands. Update summary aggregation, selection sorting, metadata, and report copy.

- [ ] **Step 4: Run model tests**

Run: `python -m pytest tests/test_model_iteration.py tests/test_multi_horizon_iteration.py -q`
Expected: PASS.

### Task 3: Surface Metrics And Probability Overlays In Dashboard

**Files:**
- Modify: `analysis/dashboard_service.py`
- Modify: `tests/test_dashboard_service.py`

**Agent Role:** Builder
**Suggested Model Tier:** mid-tier
**Why This Tier:** The service already has endpoint patterns; this adds bounded endpoints and UI elements using the same conventions.
**Inputs Needed:** Prediction CSV columns from Task 2, metadata JSON paths, existing dashboard API dispatch tests.
**Expected Output:** Dashboard exposes `/api/model-metrics`, `/api/grid.geojson?horizon=...`, probability-colored map overlay, model KPI panel, and prediction table columns.
**Review Gate:** Main agent verifies the API still streams bounded data and the HTML does not embed large datasets.

- [ ] **Step 1: Write failing tests**

Add tests that `/api/model-metrics` returns horizon rows, `/api/grid.geojson?horizon=30m&min_events=0` includes probability/risk properties when predictions exist, and `/` contains model metric fetch/render hooks.

- [ ] **Step 2: Run failing dashboard tests**

Run: `python -m pytest tests/test_dashboard_service.py -q`
Expected: FAIL because the endpoint and UI hooks do not exist.

- [ ] **Step 3: Implement endpoint and UI changes**

Read `multi_horizon_summary_latest.csv`, `best_iterated_model_metadata_{30m,1h,2h}.json`, and selected prediction CSVs. Merge predictions onto grid GeoJSON by `h3_zone`/`zone_id`. In the browser, color the map by selected horizon probability when available and keep event-count coloring as fallback.

- [ ] **Step 4: Run dashboard tests**

Run: `python -m pytest tests/test_dashboard_service.py -q`
Expected: PASS.

### Task 4: Add Retrain And Traceability Hygiene

**Files:**
- Modify: `analysis/build_dashboard_manifest.py`
- Modify: `start-dev.ps1`
- Modify: `tests/test_ci.py`

**Agent Role:** Builder
**Suggested Model Tier:** low-cost
**Why This Tier:** This is a mechanical orchestration and manifest update with clear assertions.
**Inputs Needed:** Existing manifest groups, current PowerShell script, model artifact paths from Task 2.
**Expected Output:** Manifest tracks model summaries, metadata, and prediction artifacts; `start-dev.ps1 -RetrainModels` reruns multi-horizon training and rebuilds the manifest before startup.
**Review Gate:** Main agent verifies the flag is opt-in and does not make normal startup slow.

- [ ] **Step 1: Write failing tests**

Extend CI tests to assert `RetrainModels`, `run_multi_horizon_experiment.py`, and `build_dashboard_manifest.py` are present in `start-dev.ps1`, and model metadata paths are present in the manifest builder.

- [ ] **Step 2: Run failing CI tests**

Run: `python -m pytest tests/test_ci.py -q`
Expected: FAIL until the script and manifest are updated.

- [ ] **Step 3: Implement script and manifest changes**

Add `[switch]$RetrainModels`, run `analysis/run_multi_horizon_experiment.py` when set, and always rebuild the manifest after refresh or retrain. Add model metadata and reports to `ARTIFACT_GROUPS`.

- [ ] **Step 4: Run CI tests**

Run: `python -m pytest tests/test_ci.py -q`
Expected: PASS.

### Task 5: Validate The Combined Path

**Files:**
- Read: all files modified above

**Agent Role:** Reviewer
**Suggested Model Tier:** low-cost
**Why This Tier:** The task runs deterministic tests and smoke checks only.
**Inputs Needed:** Completed Tasks 1-4.
**Expected Output:** Command evidence, known residual risk, and next recommended run command.
**Review Gate:** Main agent checks failures for real regressions before claiming completion.

- [ ] **Step 1: Run focused test suite**

Run: `python -m pytest tests/test_zone_ranking_metrics.py tests/test_model_iteration.py tests/test_multi_horizon_iteration.py tests/test_dashboard_service.py tests/test_ci.py -q`
Expected: PASS.

- [ ] **Step 2: Smoke dashboard APIs**

Run: `python - <<'PY'\nfrom analysis import dashboard_service as s\nfor path in ['/api/summary','/api/model-metrics','/api/predictions?horizon=30m&limit=3','/api/grid.geojson?horizon=30m&min_events=0']:\n    status, _, body = s.dispatch('GET', path)\n    print(path, status, len(body))\nPY`
Expected: each status is 200 and body length is greater than 2.

- [ ] **Step 3: Optional artifact refresh**

Run: `python analysis/run_multi_horizon_experiment.py`
Expected: writes `analysis/multi_horizon_summary_latest.csv`, per-horizon metadata, prediction CSVs, and `analysis/multi_horizon_report.html`.

## Self-Review

- Spec coverage: A is covered by Tasks 1-2, B by Task 3, and C by Task 4, with Task 5 validating the integrated path.
- Placeholder scan: No TBD/TODO placeholders remain; each task names exact files and commands.
- Type consistency: New metrics use numeric floats; prediction artifacts preserve `score` and add `probability`, `rank`, and `risk_band`.
- Dependency ordering: Dashboard work depends on model artifact columns, and pipeline work depends on final artifact names.
- Agent independence: Each task lists bounded files, inputs, expected output, and review gate.
- Cost fit: Pure metrics and dashboard edits use mid-tier; model selection uses high-capability; PowerShell/manifest and verification use low-cost.
- Parallel safety: Task 1 and Task 4 can run independently; Task 2 depends on Task 1; Task 3 depends on Task 2 output contract.
- Review gates: Every task has a pass/fail test checkpoint before the next task.

Plan complete and saved to `docs/superpowers/plans/2026-06-30-sparse-risk-metrics-dashboard-pipeline.md`.

Recommended execution: Subagent-Driven. I dispatch focused agents by task, use cheaper tiers for bounded read/review/mechanical work, reserve high-capability reasoning for architecture and integration review, and checkpoint after each diff.

Alternative execution: Inline Execution. I execute the plan in this session with the same checkpoints, but with less parallelism.
