# Hierarchical Spatial Cascade Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare a coarse-to-fine H3 hierarchy against the current accepted baseline, first by sweeping parent size (`res8` vs `res7`) with a soft cascade, then by comparing hierarchy styles on the winning parent size.

**Architecture:** Keep the existing activity stage and the accepted fine-cell baseline intact. Add hierarchy helpers inside `analysis/run_two_stage_experiment.py`, derive parent H3 cells from existing `zone_id` values, and run the hierarchy variants as experiment-only artifact paths so the baseline contract stays recoverable until a winner is proven.

**Tech Stack:** Python, pandas, NumPy, h3, LightGBM, scikit-learn pipelines, pytest, existing two-stage experiment and error-analysis scripts.

---

## File Map

- **Modify:** `analysis/run_two_stage_experiment.py`
  - Add hierarchy config dataclass and helper functions
  - Derive res8/res7 parents from existing res9 `zone_id`
  - Run Phase A parent-size sweep and Phase B hierarchy bake-off
  - Emit hierarchy-specific comparison artifacts without replacing baseline artifacts early

- **Modify:** `tests/test_two_stage_experiment.py`
  - Add unit tests for parent mapping, parent-label construction, soft routing, full-tree mass conservation, candidate gating, and whole-stack selection rules

- **Modify:** `WORKLOG.md`
  - Log experiment setup, commands, test results, keep/reject decision, and resume notes

- **Generated artifacts (no source edits expected):**
  - `analysis/hierarchy_parent_sweep_latest.csv`
  - `analysis/hierarchy_variant_comparison_latest.csv`
  - `analysis/hierarchy_predictions_<variant>_<horizon>_latest.csv`
  - `analysis/hierarchy_metadata_<variant>_<horizon>.json`
  - regenerated `analysis/two_stage_summary_latest.csv`
  - regenerated `analysis/spatial_model_error_summary_latest.csv`
  - regenerated `analysis/dashboard_manifest_latest.json`

## Ground Rules

- Use an **isolated worktree** when executing this plan.
- Do **not** change the activity-stage model family in this plan.
- Do **not** replace accepted baseline artifact names until the hierarchy winner is proven useful.
- Do **not** change `probability` semantics in the baseline path.
- Follow TDD for each helper or selection rule.
- Commit after each task.

---

### Task 1: Add hierarchy scaffolding and selection helpers

**Files:**
- Modify: `analysis/run_two_stage_experiment.py`
- Modify: `tests/test_two_stage_experiment.py`

- [ ] **Step 1: Write the failing tests for parent mapping and whole-stack gate**

Add these tests near the existing helper tests in `tests/test_two_stage_experiment.py`:

```python
def test_parent_zone_id_moves_one_or_two_h3_levels_up():
    zone_id = compute_h3_zone(22.3154, 114.1698, resolution=9)

    parent_res8 = _parent_zone_id(zone_id, 8)
    parent_res7 = _parent_zone_id(zone_id, 7)

    assert h3.get_resolution(parent_res8) == 8
    assert h3.get_resolution(parent_res7) == 7
    assert _parent_zone_id(zone_id, 8) == h3.cell_to_parent(zone_id, 8)


def test_prepare_parent_targets_marks_parent_positive_if_any_child_is_positive():
    frame = pd.DataFrame(
        {
            "target_time": ["t1", "t1", "t1"],
            "zone_id": [
                compute_h3_zone(22.3154, 114.1698, resolution=9),
                compute_h3_zone(22.3160, 114.1700, resolution=9),
                compute_h3_zone(22.3300, 114.1800, resolution=9),
            ],
            "actual": [0, 1, 0],
        }
    )

    prepared = _prepare_hierarchy_child_frame(frame, target_col="actual", parent_resolution=8)

    positive_parent = prepared.loc[prepared["actual"] == 1, "parent_target"].iloc[0]
    assert positive_parent == 1
    assert set(prepared["parent_target"].unique()) <= {0, 1}


def test_whole_stack_gate_rejects_two_hour_only_win():
    comparison = pd.DataFrame(
        [
            {"variant": "baseline", "horizon": "30m", "dispatch_precision_at_50": 0.10},
            {"variant": "baseline", "horizon": "1h", "dispatch_precision_at_50": 0.12},
            {"variant": "baseline", "horizon": "2h", "dispatch_precision_at_50": 0.04},
            {"variant": "candidate", "horizon": "30m", "dispatch_precision_at_50": 0.08},
            {"variant": "candidate", "horizon": "1h", "dispatch_precision_at_50": 0.10},
            {"variant": "candidate", "horizon": "2h", "dispatch_precision_at_50": 0.09},
        ]
    )

    assert _whole_stack_wins(comparison, "candidate", "baseline") is False
```

- [ ] **Step 2: Run the targeted tests and verify RED**

Run:

```powershell
C:\Users\kevlam03\OneDrive - Robert Half\Documents\Ghost_Sweep\.venv\Scripts\python.exe -m pytest tests\test_two_stage_experiment.py -k "parent_zone_id or prepare_parent_targets or whole_stack_gate" -q -p no:cacheprovider --basetemp .pytest_tmp_hierarchy_task1_red
```

Expected: FAIL with missing `_parent_zone_id`, `_prepare_hierarchy_child_frame`, or `_whole_stack_wins`.

- [ ] **Step 3: Write the minimal hierarchy helper implementation**

Add this code to `analysis/run_two_stage_experiment.py` near the other helper functions:

```python
@dataclass(frozen=True)
class HierarchySpec:
    name: str
    kind: str
    parent_resolution: int


def _parent_zone_id(zone_id: str, parent_resolution: int) -> str:
    zone_id = str(zone_id)
    if h3.get_resolution(zone_id) <= parent_resolution:
        raise ValueError(f"parent resolution {parent_resolution} must be coarser than child zone")
    return str(h3.cell_to_parent(zone_id, parent_resolution))


def _prepare_hierarchy_child_frame(
    frame: pd.DataFrame,
    target_col: str,
    parent_resolution: int,
) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["parent_zone_id"] = prepared["zone_id"].map(
        lambda zone_id: _parent_zone_id(str(zone_id), parent_resolution)
    )
    prepared["parent_target"] = (
        prepared.groupby(["target_time", "parent_zone_id"])[target_col]
        .transform("max")
        .astype(int)
    )
    return prepared


def _whole_stack_wins(comparison: pd.DataFrame, variant: str, baseline: str = "baseline") -> bool:
    pivot = comparison.pivot(index="variant", columns="horizon", values="dispatch_precision_at_50")
    base = pivot.loc[baseline]
    challenger = pivot.loc[variant]
    return bool(
        challenger["30m"] >= base["30m"]
        and challenger["1h"] >= base["1h"]
        and challenger.mean() > base.mean()
    )
```

- [ ] **Step 4: Re-run the targeted tests and verify GREEN**

Run:

```powershell
C:\Users\kevlam03\OneDrive - Robert Half\Documents\Ghost_Sweep\.venv\Scripts\python.exe -m pytest tests\test_two_stage_experiment.py -k "parent_zone_id or prepare_parent_targets or whole_stack_gate" -q -p no:cacheprovider --basetemp .pytest_tmp_hierarchy_task1_green
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```powershell
git add analysis\run_two_stage_experiment.py tests\test_two_stage_experiment.py
git commit -m "feat: add hierarchy experiment scaffolding"
```

---

### Task 2: Implement the soft cascade routing primitive and Phase A parent-size sweep

**Files:**
- Modify: `analysis/run_two_stage_experiment.py`
- Modify: `tests/test_two_stage_experiment.py`

- [ ] **Step 1: Write failing tests for soft routing and parent-size sweep output**

Append these tests to `tests/test_two_stage_experiment.py`:

```python
def test_route_children_soft_weights_child_scores_by_parent_probability():
    children = pd.DataFrame(
        {
            "target_time": ["t1", "t1", "t1"],
            "zone_id": ["a", "b", "c"],
            "parent_zone_id": ["p1", "p1", "p2"],
            "activity_probability": [0.5, 0.5, 0.5],
            "spatial_probability": [0.9, 0.4, 0.8],
            "actual": [1, 0, 0],
        }
    )
    parents = pd.DataFrame(
        {
            "target_time": ["t1", "t1"],
            "parent_zone_id": ["p1", "p2"],
            "parent_probability": [0.7, 0.3],
        }
    )

    routed = _route_children_soft(children, parents)

    assert list(routed.sort_values("hierarchy_score", ascending=False)["zone_id"]) == ["a", "b", "c"]


def test_choose_best_parent_resolution_prefers_res8_when_it_wins_whole_stack():
    comparison = pd.DataFrame(
        [
            {"variant": "soft_res8", "horizon": "30m", "dispatch_precision_at_50": 0.11},
            {"variant": "soft_res8", "horizon": "1h", "dispatch_precision_at_50": 0.13},
            {"variant": "soft_res8", "horizon": "2h", "dispatch_precision_at_50": 0.05},
            {"variant": "soft_res7", "horizon": "30m", "dispatch_precision_at_50": 0.09},
            {"variant": "soft_res7", "horizon": "1h", "dispatch_precision_at_50": 0.11},
            {"variant": "soft_res7", "horizon": "2h", "dispatch_precision_at_50": 0.06},
            {"variant": "baseline", "horizon": "30m", "dispatch_precision_at_50": 0.10},
            {"variant": "baseline", "horizon": "1h", "dispatch_precision_at_50": 0.12},
            {"variant": "baseline", "horizon": "2h", "dispatch_precision_at_50": 0.04},
        ]
    )

    assert _choose_best_parent_resolution(comparison, baseline="baseline") == 8
```

- [ ] **Step 2: Run the targeted tests and verify RED**

Run:

```powershell
C:\Users\kevlam03\OneDrive - Robert Half\Documents\Ghost_Sweep\.venv\Scripts\python.exe -m pytest tests\test_two_stage_experiment.py -k "route_children_soft or choose_best_parent_resolution" -q -p no:cacheprovider --basetemp .pytest_tmp_hierarchy_task2_red
```

Expected: FAIL with missing `_route_children_soft` or `_choose_best_parent_resolution`.

- [ ] **Step 3: Implement the soft routing helper and Phase A selector**

Add this code to `analysis/run_two_stage_experiment.py`:

```python
PARENT_STAGE_NUMERIC_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "zone_event_count_1h",
    "zone_event_count_3h",
    "zone_event_count_24h",
    "zone_event_count_7d",
    "neighbor_event_count_24h",
    "ring2_event_count_24h",
]


def _build_parent_stage_frame(
    children: pd.DataFrame,
    target_col: str,
    parent_resolution: int,
) -> pd.DataFrame:
    prepared = _prepare_hierarchy_child_frame(children, target_col, parent_resolution)
    grouped = prepared.groupby(["target_time", "parent_zone_id"], as_index=False).agg(
        {
            "hour": "first",
            "day_of_week": "first",
            "is_weekend": "first",
            "zone_event_count_1h": "sum",
            "zone_event_count_3h": "sum",
            "zone_event_count_24h": "sum",
            "zone_event_count_7d": "sum",
            "neighbor_event_count_24h": "sum",
            "ring2_event_count_24h": "sum",
            target_col: "max",
        }
    )
    grouped["parent_target"] = grouped.pop(target_col).astype(int)
    return grouped


def _fit_parent_stage(parent_frame: pd.DataFrame) -> pd.DataFrame:
    feature_cols = PARENT_STAGE_NUMERIC_FEATURES
    model = LGBMClassifier(
        n_estimators=160,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    model.fit(parent_frame[feature_cols], parent_frame["parent_target"].astype(int))
    scored = parent_frame[["target_time", "parent_zone_id"]].copy()
    scored["parent_probability"] = model.predict_proba(parent_frame[feature_cols])[:, 1]
    return scored


def _route_children_soft(
    children: pd.DataFrame,
    parents: pd.DataFrame,
    per_target_time_quota: int = 10,
) -> pd.DataFrame:
    routed = children.merge(
        parents[["target_time", "parent_zone_id", "parent_probability"]],
        on=["target_time", "parent_zone_id"],
        how="left",
    ).copy()
    routed["parent_probability"] = routed["parent_probability"].fillna(0.0)
    routed["hierarchy_score"] = (
        routed["activity_probability"].astype(float)
        * routed["parent_probability"].astype(float)
        * routed["spatial_probability"].astype(float)
    )
    routed["probability"] = routed["hierarchy_score"]
    return assign_dispatch_rank(routed, per_target_time_quota=per_target_time_quota)


def _choose_best_parent_resolution(comparison: pd.DataFrame, baseline: str = "baseline") -> int:
    winners = []
    for variant in sorted(v for v in comparison["variant"].unique() if v != baseline):
        if _whole_stack_wins(comparison, variant, baseline):
            winners.append(variant)
    if "soft_res8" in winners:
        return 8
    if "soft_res7" in winners:
        return 7
    raise RuntimeError("Neither parent resolution beat baseline on the whole-stack gate.")
```

Then add the single-horizon evaluator and the Phase A runner:

```python
def run_hierarchy_horizon(
    events: list[dict],
    horizon_minutes: int,
    spec: HierarchySpec,
    resolution: int = DEFAULT_H3_RESOLUTION,
) -> dict[str, Any]:
    target_col = target_for_horizon(horizon_minutes)
    lookback_hours = _effective_lookback_hours(events)
    lookback_days = max(1, int(lookback_hours // 24))
    activity_df = build_activity_training_data(
        events,
        horizon_minutes=horizon_minutes,
        lookback_hours=lookback_hours,
        resolution=resolution,
    )
    spatial_df = build_zone_ranking_training_data(
        events,
        forecast_hours=max(1, horizon_minutes // 60),
        horizon_minutes=horizon_minutes,
        target_col=target_col,
        lookback_days=lookback_days,
        resolution=resolution,
    )
    split = make_positive_count_holdout(spatial_df, target_col)
    child_holdout = spatial_df.loc[split.holdout_mask].copy()
    activity_holdout = _fit_activity_holdout(
        activity_df,
        activity_target_for_horizon(horizon_minutes),
        _select_model(_evaluate_activity_candidates(activity_df, activity_target_for_horizon(horizon_minutes), horizon_minutes)[1], "activity"),
        horizon_minutes,
        _artifact_paths(horizon_slug(horizon_minutes)),
    )
    child_holdout["activity_probability"] = _probabilities_for_time(
        child_holdout["target_time"],
        activity_holdout["holdout_predictions"],
        activity_holdout["holdout_metrics"]["base_rate"],
    )
    parent_frame = _build_parent_stage_frame(child_holdout, target_col, spec.parent_resolution)
    parent_scores = _fit_parent_stage(parent_frame)
    routed = _route_children_soft(
        _prepare_hierarchy_child_frame(child_holdout, target_col, spec.parent_resolution),
        parent_scores,
        per_target_time_quota=_dispatch_quota_for_target(target_col),
    )
    holdout_metrics = _score_spatial_predictions(
        child_holdout[target_col],
        routed["probability"].to_numpy(),
        groups={"district": child_holdout["district"], "region": child_holdout["region"]},
    )
    holdout_metrics.update(_dispatch_topk_metrics(routed, k=50))
    return {
        "variant": spec.name,
        "horizon": horizon_slug(horizon_minutes),
        "dispatch_precision_at_50": holdout_metrics["dispatch_precision_at_50"],
        "precision_at_50": holdout_metrics["precision_at_50"],
    }


def run_hierarchy_parent_sweep(events: list[dict], horizons: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "variant": "baseline",
            "parent_resolution": DEFAULT_H3_RESOLUTION,
            "horizon": item["horizon"],
            "dispatch_precision_at_50": item["spatial_model"]["holdout_metrics"]["dispatch_precision_at_50"],
            "precision_at_50": item["spatial_model"]["holdout_metrics"]["precision_at_50"],
        }
        for item in [run_two_stage_horizon(events, horizon_minutes=horizon) for horizon in horizons]
    ]
    for parent_resolution in (8, 7):
        spec = HierarchySpec(name=f"soft_res{parent_resolution}", kind="soft", parent_resolution=parent_resolution)
        for horizon in horizons:
            metrics = run_hierarchy_horizon(events, horizon, spec)
            rows.append({"variant": spec.name, "parent_resolution": parent_resolution, **metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "hierarchy_parent_sweep_latest.csv", index=False)
    return frame
```

- [ ] **Step 4: Re-run the targeted tests and verify GREEN**

Run:

```powershell
C:\Users\kevlam03\OneDrive - Robert Half\Documents\Ghost_Sweep\.venv\Scripts\python.exe -m pytest tests\test_two_stage_experiment.py -k "route_children_soft or choose_best_parent_resolution" -q -p no:cacheprovider --basetemp .pytest_tmp_hierarchy_task2_green
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```powershell
git add analysis\run_two_stage_experiment.py tests\test_two_stage_experiment.py
git commit -m "feat: add soft hierarchy parent sweep"
```

---

### Task 3: Implement the full-tree and candidate-generation hierarchy variants

**Files:**
- Modify: `analysis/run_two_stage_experiment.py`
- Modify: `tests/test_two_stage_experiment.py`

- [ ] **Step 1: Write failing tests for full-tree mass conservation and candidate gating**

Append these tests to `tests/test_two_stage_experiment.py`:

```python
def test_route_children_tree_normalizes_child_mass_inside_each_parent():
    children = pd.DataFrame(
        {
            "target_time": ["t1", "t1", "t1"],
            "zone_id": ["a", "b", "c"],
            "parent_zone_id": ["p1", "p1", "p2"],
            "activity_probability": [1.0, 1.0, 1.0],
            "spatial_probability": [0.6, 0.4, 1.0],
            "actual": [1, 0, 0],
        }
    )
    parents = pd.DataFrame(
        {
            "target_time": ["t1", "t1"],
            "parent_zone_id": ["p1", "p2"],
            "parent_probability": [0.7, 0.3],
        }
    )

    routed = _route_children_tree(children, parents)
    p1_sum = routed.loc[routed["parent_zone_id"] == "p1", "hierarchy_probability"].sum()
    p2_sum = routed.loc[routed["parent_zone_id"] == "p2", "hierarchy_probability"].sum()

    assert round(p1_sum, 6) == 0.7
    assert round(p2_sum, 6) == 0.3


def test_route_children_candidates_drops_children_outside_selected_parents():
    children = pd.DataFrame(
        {
            "target_time": ["t1", "t1", "t1"],
            "zone_id": ["a", "b", "c"],
            "parent_zone_id": ["p1", "p2", "p3"],
            "activity_probability": [0.5, 0.5, 0.5],
            "spatial_probability": [0.7, 0.6, 0.9],
            "actual": [1, 0, 0],
        }
    )
    parents = pd.DataFrame(
        {
            "target_time": ["t1", "t1", "t1"],
            "parent_zone_id": ["p1", "p2", "p3"],
            "parent_probability": [0.5, 0.3, 0.2],
        }
    )

    routed = _route_children_candidates(children, parents, max_parents=2)

    assert set(routed["parent_zone_id"]) == {"p1", "p2"}
```

- [ ] **Step 2: Run the targeted tests and verify RED**

Run:

```powershell
C:\Users\kevlam03\OneDrive - Robert Half\Documents\Ghost_Sweep\.venv\Scripts\python.exe -m pytest tests\test_two_stage_experiment.py -k "route_children_tree or route_children_candidates" -q -p no:cacheprovider --basetemp .pytest_tmp_hierarchy_task3_red
```

Expected: FAIL with missing `_route_children_tree` or `_route_children_candidates`.

- [ ] **Step 3: Implement the two remaining hierarchy arms**

Add this code to `analysis/run_two_stage_experiment.py`:

```python
def _route_children_tree(children: pd.DataFrame, parents: pd.DataFrame) -> pd.DataFrame:
    routed = children.merge(
        parents[["target_time", "parent_zone_id", "parent_probability"]],
        on=["target_time", "parent_zone_id"],
        how="left",
    ).copy()
    routed["parent_probability"] = routed["parent_probability"].fillna(0.0)
    child_mass = routed.groupby(["target_time", "parent_zone_id"])["spatial_probability"].transform("sum")
    routed["conditional_child_probability"] = np.where(
        child_mass > 0,
        routed["spatial_probability"] / child_mass,
        0.0,
    )
    routed["hierarchy_probability"] = (
        routed["activity_probability"].astype(float)
        * routed["parent_probability"].astype(float)
        * routed["conditional_child_probability"].astype(float)
    )
    routed["probability"] = routed["hierarchy_probability"]
    return assign_dispatch_rank(routed, per_target_time_quota=10)


def _route_children_candidates(
    children: pd.DataFrame,
    parents: pd.DataFrame,
    max_parents: int = 5,
) -> pd.DataFrame:
    keep = (
        parents.sort_values(["target_time", "parent_probability"], ascending=[True, False])
        .groupby("target_time", group_keys=False)
        .head(max_parents)
    )
    routed = children.merge(
        keep[["target_time", "parent_zone_id"]],
        on=["target_time", "parent_zone_id"],
        how="inner",
    ).copy()
    routed["hierarchy_score"] = (
        routed["activity_probability"].astype(float)
        * routed["spatial_probability"].astype(float)
    )
    routed["probability"] = routed["hierarchy_score"]
    return assign_dispatch_rank(routed, per_target_time_quota=10)
```

Then add a dispatcher:

```python
def _route_hierarchy_children(
    children: pd.DataFrame,
    parents: pd.DataFrame,
    spec: HierarchySpec,
) -> pd.DataFrame:
    if spec.kind == "soft":
        return _route_children_soft(children, parents)
    if spec.kind == "tree":
        return _route_children_tree(children, parents)
    if spec.kind == "candidate":
        return _route_children_candidates(children, parents)
    raise ValueError(f"Unknown hierarchy kind: {spec.kind}")
```

- [ ] **Step 4: Re-run the targeted tests and verify GREEN**

Run:

```powershell
C:\Users\kevlam03\OneDrive - Robert Half\Documents\Ghost_Sweep\.venv\Scripts\python.exe -m pytest tests\test_two_stage_experiment.py -k "route_children_tree or route_children_candidates" -q -p no:cacheprovider --basetemp .pytest_tmp_hierarchy_task3_green
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```powershell
git add analysis\run_two_stage_experiment.py tests\test_two_stage_experiment.py
git commit -m "feat: add hierarchy routing variants"
```

---

### Task 4: Wire the experiment runner, comparison artifacts, and keep/reject logic

**Files:**
- Modify: `analysis/run_two_stage_experiment.py`
- Modify: `tests/test_two_stage_experiment.py`

- [ ] **Step 1: Write failing tests for Phase B selection and comparison artifact rows**

Append this test to `tests/test_two_stage_experiment.py`:

```python
def test_choose_best_hierarchy_variant_requires_whole_stack_win():
    comparison = pd.DataFrame(
        [
            {"variant": "baseline", "horizon": "30m", "dispatch_precision_at_50": 0.10},
            {"variant": "baseline", "horizon": "1h", "dispatch_precision_at_50": 0.12},
            {"variant": "baseline", "horizon": "2h", "dispatch_precision_at_50": 0.04},
            {"variant": "soft", "horizon": "30m", "dispatch_precision_at_50": 0.11},
            {"variant": "soft", "horizon": "1h", "dispatch_precision_at_50": 0.13},
            {"variant": "soft", "horizon": "2h", "dispatch_precision_at_50": 0.05},
            {"variant": "tree", "horizon": "30m", "dispatch_precision_at_50": 0.09},
            {"variant": "tree", "horizon": "1h", "dispatch_precision_at_50": 0.11},
            {"variant": "tree", "horizon": "2h", "dispatch_precision_at_50": 0.08},
        ]
    )

    assert _choose_best_hierarchy_variant(comparison, baseline="baseline") == "soft"
```

- [ ] **Step 2: Run the targeted tests and verify RED**

Run:

```powershell
C:\Users\kevlam03\OneDrive - Robert Half\Documents\Ghost_Sweep\.venv\Scripts\python.exe -m pytest tests\test_two_stage_experiment.py -k "choose_best_hierarchy_variant" -q -p no:cacheprovider --basetemp .pytest_tmp_hierarchy_task4_red
```

Expected: FAIL with missing `_choose_best_hierarchy_variant`.

- [ ] **Step 3: Implement the comparison writers and variant chooser**

Add this code to `analysis/run_two_stage_experiment.py`:

```python
def _choose_best_hierarchy_variant(comparison: pd.DataFrame, baseline: str = "baseline") -> str:
    variants = [variant for variant in comparison["variant"].unique() if variant != baseline]
    winners = [variant for variant in variants if _whole_stack_wins(comparison, variant, baseline)]
    if not winners:
        raise RuntimeError("No hierarchy variant beat baseline on the whole-stack gate.")
    ranked = sorted(
        winners,
        key=lambda variant: comparison.loc[comparison["variant"] == variant, "dispatch_precision_at_50"].mean(),
        reverse=True,
    )
    return ranked[0]


def _write_hierarchy_comparison(frame: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["variant", "horizon"]).to_csv(path, index=False)
    return path
```

Then wire a hierarchy-specific runner:

```python
def run_hierarchy_variant_bakeoff(
    events: list[dict],
    horizons: list[int],
    specs: list[HierarchySpec],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "variant": "baseline",
            "horizon": item["horizon"],
            "dispatch_precision_at_50": item["spatial_model"]["holdout_metrics"]["dispatch_precision_at_50"],
        }
        for item in [run_two_stage_horizon(events, horizon_minutes=horizon) for horizon in horizons]
    ]
    for spec in specs:
        for horizon in horizons:
            rows.append(run_hierarchy_horizon(events, horizon, spec))
    return pd.DataFrame(rows)


def run_hierarchy_experiment(
    horizons: list[int] | None = None,
    resolution: int = DEFAULT_H3_RESOLUTION,
) -> dict[str, Any]:
    events = _load_events().to_dict("records")
    horizon_list = horizons or HORIZONS

    parent_sweep = run_hierarchy_parent_sweep(events, horizon_list)
    winning_parent_resolution = _choose_best_parent_resolution(parent_sweep, baseline="baseline")

    bakeoff_specs = [
        HierarchySpec(name="soft", kind="soft", parent_resolution=winning_parent_resolution),
        HierarchySpec(name="tree", kind="tree", parent_resolution=winning_parent_resolution),
        HierarchySpec(name="candidate", kind="candidate", parent_resolution=winning_parent_resolution),
    ]
    comparison = run_hierarchy_variant_bakeoff(events, horizon_list, bakeoff_specs)
    winner = _choose_best_hierarchy_variant(comparison, baseline="baseline")

    _write_hierarchy_comparison(parent_sweep, "hierarchy_parent_sweep_latest.csv")
    _write_hierarchy_comparison(comparison, "hierarchy_variant_comparison_latest.csv")
    return {
        "winning_parent_resolution": winning_parent_resolution,
        "winning_variant": winner,
    }
```

- [ ] **Step 4: Re-run the targeted tests and verify GREEN**

Run:

```powershell
C:\Users\kevlam03\OneDrive - Robert Half\Documents\Ghost_Sweep\.venv\Scripts\python.exe -m pytest tests\test_two_stage_experiment.py -k "choose_best_hierarchy_variant" -q -p no:cacheprovider --basetemp .pytest_tmp_hierarchy_task4_green
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```powershell
git add analysis\run_two_stage_experiment.py tests\test_two_stage_experiment.py
git commit -m "feat: wire hierarchy experiment comparison flow"
```

---

### Task 5: Execute the experiment, regenerate artifacts, log the decision, and keep or reject

**Files:**
- Modify: `WORKLOG.md`
- Generated: `analysis/hierarchy_parent_sweep_latest.csv`
- Generated: `analysis/hierarchy_variant_comparison_latest.csv`
- Generated: `analysis/two_stage_summary_latest.csv`
- Generated: `analysis/spatial_model_error_summary_latest.csv`
- Generated: `analysis/dashboard_manifest_latest.json`

- [ ] **Step 1: Run the focused validation suite before the experiment**

Run:

```powershell
C:\Users\kevlam03\OneDrive - Robert Half\Documents\Ghost_Sweep\.venv\Scripts\python.exe -m pytest tests\test_ghost_ranking_features.py tests\test_two_stage_experiment.py tests\test_ranking_metrics.py tests\test_spatial_sampling.py tests\test_spatial_model_error_analysis.py tests\test_spatial_ranking_diagnostics.py -q -p no:cacheprovider --basetemp .pytest_tmp_hierarchy_full
```

Expected: PASS.

- [ ] **Step 2: Run the hierarchy experiment**

Run:

```powershell
C:\Users\kevlam03\OneDrive - Robert Half\Documents\Ghost_Sweep\.venv\Scripts\python.exe -c "from analysis.run_two_stage_experiment import run_hierarchy_experiment; print(run_hierarchy_experiment())"
```

Expected: prints the winning parent resolution and winning hierarchy variant, and writes the comparison CSVs.

- [ ] **Step 3: Re-run error analysis and rebuild the manifest**

Run:

```powershell
C:\Users\kevlam03\OneDrive - Robert Half\Documents\Ghost_Sweep\.venv\Scripts\python.exe analysis\analyze_spatial_model_errors.py --k 50
C:\Users\kevlam03\OneDrive - Robert Half\Documents\Ghost_Sweep\.venv\Scripts\python.exe analysis\build_dashboard_manifest.py
```

Expected: both commands exit `0`.

- [ ] **Step 4: Inspect the comparison artifacts against baseline**

Run:

```powershell
Import-Csv analysis\hierarchy_parent_sweep_latest.csv | Format-Table variant,horizon,dispatch_precision_at_50 -AutoSize
Import-Csv analysis\hierarchy_variant_comparison_latest.csv | Format-Table variant,horizon,dispatch_precision_at_50 -AutoSize
Import-Csv analysis\two_stage_summary_latest.csv | Select-Object horizon,spatial_model,spatial_dispatch_precision_at_50,spatial_precision_at_50 | Format-Table -AutoSize
```

Expected:

- one of `soft_res8` or `soft_res7` wins the whole-stack parent sweep
- one hierarchy variant either beats baseline cleanly or all hierarchy variants fail the gate

- [ ] **Step 5: Update `WORKLOG.md` with the result**

Append a block that records:

```markdown
## 2026-07-08 Hierarchical spatial cascade experiment

Current objective:
- Compare res8/res7 hierarchy routing and hierarchy variants against the accepted baseline.

Files inspected:
- `analysis/hierarchy_parent_sweep_latest.csv`
- `analysis/hierarchy_variant_comparison_latest.csv`
- `analysis/two_stage_summary_latest.csv`
- `analysis/spatial_model_error_summary_latest.csv`

Files changed:
- `WORKLOG.md`

Commands run:
- `C:\Users\kevlam03\OneDrive - Robert Half\Documents\Ghost_Sweep\.venv\Scripts\python.exe -m pytest tests\test_ghost_ranking_features.py tests\test_two_stage_experiment.py tests\test_ranking_metrics.py tests\test_spatial_sampling.py tests\test_spatial_model_error_analysis.py tests\test_spatial_ranking_diagnostics.py -q -p no:cacheprovider --basetemp .pytest_tmp_hierarchy_full`
- `C:\Users\kevlam03\OneDrive - Robert Half\Documents\Ghost_Sweep\.venv\Scripts\python.exe -c "from analysis.run_two_stage_experiment import run_hierarchy_experiment; print(run_hierarchy_experiment())"`
- `C:\Users\kevlam03\OneDrive - Robert Half\Documents\Ghost_Sweep\.venv\Scripts\python.exe analysis\analyze_spatial_model_errors.py --k 50`
- `C:\Users\kevlam03\OneDrive - Robert Half\Documents\Ghost_Sweep\.venv\Scripts\python.exe analysis\build_dashboard_manifest.py`

Test results:
- Focused hierarchy validation passed; record the exact pytest summary line from the command above (for example, `29 passed, 5 warnings in 4.12s`).
- Parent sweep result; record one line for `soft_res8` and one line for `soft_res7`, each with `30m` / `1h` / `2h` dispatch precision@50 values, then state which parent resolution won the whole-stack gate.
- Variant comparison result; record the winning variant name and whether the final decision was keep or reject, or explicitly note that all hierarchy variants failed the whole-stack gate.

Blockers:
- `ghost_alerts.db` in linked worktrees may resolve to an empty ignored SQLite file; if that happens, copy the populated repo-root database into the worktree before rerunning the experiment.

Next steps:
- If a hierarchy arm wins the whole-stack gate, keep the winning code path and artifact set.
- If all arms fail, leave the worktree branch as a rejected experiment and do not merge it to `main`.
```

- [ ] **Step 6: Apply the keep/reject rule**

If **no** hierarchy arm beats baseline on the whole-stack gate:

```powershell
git status --short
git branch --show-current
```

Expected:

- the worktree branch remains a rejected-experiment record
- no hierarchy change is merged back to `main`

If a hierarchy arm **does** beat baseline:

```powershell
git add analysis\run_two_stage_experiment.py tests\test_two_stage_experiment.py WORKLOG.md analysis\hierarchy_parent_sweep_latest.csv analysis\hierarchy_variant_comparison_latest.csv analysis\two_stage_summary_latest.csv analysis\spatial_model_error_summary_latest.csv analysis\dashboard_manifest_latest.json
git commit -m "feat: add hierarchical spatial cascade experiment"
```

Expected:

- reject path restores the accepted baseline
- keep path commits the winning hierarchy experiment with supporting artifacts

- [ ] **Step 7: If rejected, re-run the focused suite after the revert**

Run:

```powershell
C:\Users\kevlam03\OneDrive - Robert Half\Documents\Ghost_Sweep\.venv\Scripts\python.exe -m pytest tests\test_ghost_ranking_features.py tests\test_two_stage_experiment.py tests\test_ranking_metrics.py tests\test_spatial_sampling.py tests\test_spatial_model_error_analysis.py tests\test_spatial_ranking_diagnostics.py -q -p no:cacheprovider --basetemp .pytest_tmp_hierarchy_revert
```

Expected: PASS on the restored accepted baseline.
