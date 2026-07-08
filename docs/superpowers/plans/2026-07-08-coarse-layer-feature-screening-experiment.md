# Coarse-Layer Feature Screening Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an analysis-first coarse-context screen that measures `district`, `police_zone`, and `res8` signal on the current `res9` rows, then runs only the smallest real ablations that add new information over the accepted baseline.

**Architecture:** Keep the accepted two-stage pipeline intact. Extend `ghost_ranking_features.py` with incremental `police_zone` and `res8` context columns, add a Phase A signal screen in `analysis/run_two_stage_experiment.py`, and only if short-horizon signal clears the gate, reuse the existing horizon runner with variant-specific spatial feature sets and variant-specific artifact slugs. The accepted baseline already contains `district` and `region`, so treat `district` as an analysis control and use the current `region` field as the `police_zone` proxy instead of inventing a new GIS dependency.

**Tech Stack:** Python, pandas, NumPy, h3, scikit-learn, LightGBM, pytest, existing two-stage experiment scripts.

---

## File Map

- **Modify:** `ghost_ranking_features.py`
  - Add `res8_zone` and `police_zone` context columns to the res9 training rows
  - Add bounded recent-activity and relative-position features for `police_zone` and `res8`
  - Leave the current district feature pack in place

- **Modify:** `analysis/run_two_stage_experiment.py`
  - Add Phase A coarse-layer signal diagnostics and the short-horizon gate
  - Add a local spatial preprocessor that can accept ablation-specific feature sets
  - Add Phase B ablation wiring with variant-specific artifact names
  - Keep the accepted `run_two_stage_experiment()` baseline entrypoint unchanged

- **Modify:** `tests/test_ghost_ranking_features.py`
  - Add focused coverage for `police_zone` / `res8` enrichment and relative features

- **Modify:** `tests/test_two_stage_experiment.py`
  - Add focused coverage for signal scoring, gate behavior, feature-pack selection, whole-stack variant selection, and Phase B reject behavior

- **Modify:** `WORKLOG.md`
  - Record the implementation-plan handoff now
  - Record the final experiment outcome when this plan is executed

- **Modify:** `WORKLOG_INDEX.md`
  - Add the implementation-plan milestone now
  - Add the experiment milestone later when execution adds a new top-level `##` entry

- **Generated artifacts (no source edits expected):**
  - `analysis\coarse_layer_signal_30m_latest.csv`
  - `analysis\coarse_layer_signal_1h_latest.csv`
  - `analysis\coarse_layer_signal_2h_latest.csv`
  - `analysis\coarse_layer_ablation_comparison_latest.csv`
  - variant-specific spatial artifacts such as `analysis\spatial_zone_predictions_baseline_plus_res8_30m_latest.csv`
  - regenerated `analysis\two_stage_summary_latest.csv` only if a winning variant clears the whole-stack gate

## Ground Rules

- Do **not** add another hierarchy, parent gate, or parent-to-child normalization path.
- Do **not** promise a `baseline + district` rerun; the accepted baseline already uses district-aware counts, ratios, percentiles, and the `district` categorical column.
- Do **not** add a new GIS dataset for police boundaries in this pass; use the existing broad `region` field as the `police_zone` proxy.
- Do **not** overwrite the accepted baseline artifacts until a Phase B winner actually clears the whole-stack gate.
- Follow TDD for every helper and commit after every task.

---

### Task 1: Add `police_zone` and `res8` context features to the ranking rows

**Files:**
- Modify: `ghost_ranking_features.py`
- Modify: `tests/test_ghost_ranking_features.py`

- [ ] **Step 1: Write the failing feature-builder tests**

Add these tests to `tests/test_ghost_ranking_features.py`:

```python
import h3
import pandas as pd

from ghost_ranking_features import (
    add_engineered_ranking_features,
    build_zone_ranking_training_data,
)
from ghost_zones import compute_h3_zone


def test_build_zone_ranking_training_data_adds_police_zone_and_res8_context():
    events = [
        _event(22.3154, 114.1698, f"2026-06-{day:02d} 10:00:00")
        for day in range(1, 18)
    ]

    rows = build_zone_ranking_training_data(events, resolution=9)

    assert "police_zone" in rows.columns
    assert "res8_zone" in rows.columns
    assert set(rows["police_zone"]) == set(rows["region"])
    assert set(rows["res8_zone"].map(h3.get_resolution)) == {8}


def test_add_engineered_ranking_features_adds_police_zone_and_res8_relative_features():
    zone_a = compute_h3_zone(22.3154, 114.1698, resolution=9)
    zone_b = compute_h3_zone(22.3160, 114.1703, resolution=9)
    target_time = pd.Timestamp("2026-06-02 09:00:00")
    frame = pd.DataFrame(
        {
            "target_time": [target_time, target_time],
            "zone_id": [zone_a, zone_b],
            "zone_lat": [22.3154, 22.3160],
            "zone_lng": [114.1698, 114.1703],
            "district": ["Kowloon City", "Kowloon City"],
            "region": ["Kowloon West", "Kowloon West"],
            "police_zone": ["Kowloon West", "Kowloon West"],
            "res8_zone": [str(h3.cell_to_parent(zone_a, 8)), str(h3.cell_to_parent(zone_b, 8))],
            "hour": [9, 9],
            "day_of_week": [0, 0],
            "zone_event_count_3h": [2, 1],
            "zone_event_count_24h": [3, 1],
            "zone_event_count_7d": [5, 2],
            "district_event_count_3h": [3, 3],
            "district_event_count_24h": [4, 4],
            "district_active_zones_24h": [2, 2],
            "zone_same_hour_rate": [0.8, 0.2],
            "district_same_hour_rate": [0.5, 0.5],
        }
    )

    enhanced = add_engineered_ranking_features(frame)
    row_a = enhanced.loc[enhanced["zone_id"] == zone_a].iloc[0]

    assert row_a["police_zone_event_count_24h"] == 4
    assert row_a["res8_event_count_24h"] == 4
    assert row_a["zone_24h_share_of_police_zone"] == 0.75
    assert row_a["zone_7d_rank_in_res8"] == 1.0
    assert row_a["zone_same_hour_percentile_in_police_zone"] == 1.0
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_ghost_ranking_features.py -k "police_zone or res8" -q -p no:cacheprovider --basetemp .pytest_tmp_coarse_task1_red
```

Expected: FAIL with missing `police_zone`, missing `res8_zone`, or missing engineered coarse-layer columns.

- [ ] **Step 3: Add the minimal coarse-context implementation**

Add these helpers near `_add_district_relative_features()` in `ghost_ranking_features.py`:

```python
def _coarse_parent_zone_id(zone_id: str, parent_resolution: int) -> str:
    zone_id = str(zone_id)
    current_resolution = h3.get_resolution(zone_id)
    if current_resolution <= parent_resolution:
        return zone_id
    return str(h3.cell_to_parent(zone_id, parent_resolution))


def _add_group_context_features(
    enhanced: pd.DataFrame,
    group_col: str,
    prefix: str,
) -> pd.DataFrame:
    for column, default in {
        f"{prefix}_event_count_3h": 0.0,
        f"{prefix}_event_count_24h": 0.0,
        f"{prefix}_event_count_7d": 0.0,
        f"{prefix}_same_hour_rate": 0.0,
        f"{prefix}_active_zones_24h": 0,
        f"zone_24h_share_of_{prefix}": 0.0,
        f"zone_7d_rank_in_{prefix}": 0.0,
        f"zone_same_hour_percentile_in_{prefix}": 0.0,
    }.items():
        if column not in enhanced:
            enhanced[column] = default

    if not {"target_time", group_col}.issubset(enhanced.columns):
        return enhanced

    group_cols = ["target_time", group_col]
    enhanced[f"{prefix}_event_count_3h"] = enhanced.groupby(group_cols)["zone_event_count_3h"].transform("sum")
    enhanced[f"{prefix}_event_count_24h"] = enhanced.groupby(group_cols)["zone_event_count_24h"].transform("sum")
    enhanced[f"{prefix}_event_count_7d"] = enhanced.groupby(group_cols)["zone_event_count_7d"].transform("sum")
    enhanced[f"{prefix}_same_hour_rate"] = enhanced.groupby(group_cols)["zone_same_hour_rate"].transform("mean")
    enhanced[f"{prefix}_active_zones_24h"] = enhanced.groupby(group_cols)["zone_event_count_24h"].transform(
        lambda values: int((values > 0).sum())
    )
    enhanced[f"zone_24h_share_of_{prefix}"] = [
        _safe_divide(zone_count, parent_total)
        for zone_count, parent_total in zip(
            enhanced["zone_event_count_24h"],
            enhanced[f"{prefix}_event_count_24h"],
        )
    ]
    enhanced[f"zone_7d_rank_in_{prefix}"] = enhanced.groupby(group_cols)["zone_event_count_7d"].rank(
        method="min",
        ascending=False,
    )
    enhanced[f"zone_same_hour_percentile_in_{prefix}"] = enhanced.groupby(group_cols)["zone_same_hour_rate"].rank(
        method="max",
        pct=True,
    )
    return enhanced
```

Inside `add_engineered_ranking_features()`, insert these lines immediately after `enhanced = df.copy()`:

```python
    if "police_zone" not in enhanced and "region" in enhanced:
        enhanced["police_zone"] = enhanced["region"]
    if "res8_zone" not in enhanced and "zone_id" in enhanced:
        enhanced["res8_zone"] = enhanced["zone_id"].map(
            lambda zone_id: _coarse_parent_zone_id(str(zone_id), 8)
        )
```

Still inside `add_engineered_ranking_features()`, call the new helpers before the existing district-relative helper:

```python
    enhanced = _add_group_context_features(enhanced, "police_zone", "police_zone")
    enhanced = _add_group_context_features(enhanced, "res8_zone", "res8")
    enhanced = _add_district_relative_features(enhanced)
```

Inside the `row` dictionary in `build_zone_ranking_training_data()`, add these exact keys:

```python
                "police_zone": region,
                "res8_zone": _coarse_parent_zone_id(zone_id, 8),
```

- [ ] **Step 4: Re-run the focused tests and verify GREEN**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_ghost_ranking_features.py -k "police_zone or res8" -q -p no:cacheprovider --basetemp .pytest_tmp_coarse_task1_green
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```powershell
git add ghost_ranking_features.py tests\test_ghost_ranking_features.py
git commit -m "feat: add coarse context feature pack"
```

---

### Task 2: Add Phase A signal diagnostics and the short-horizon gate

**Files:**
- Modify: `analysis/run_two_stage_experiment.py`
- Modify: `tests/test_two_stage_experiment.py`

- [ ] **Step 1: Write the failing signal-screen tests**

Add these tests to `tests/test_two_stage_experiment.py`:

```python
def test_screen_coarse_layer_prefers_feature_with_within_time_signal():
    frame = pd.DataFrame(
        {
            "target_time": ["t1", "t1", "t2", "t2"],
            "alert_next_30m": [1, 0, 1, 0],
            "res8_event_count_24h": [9, 1, 8, 1],
            "zone_24h_share_of_res8": [0.9, 0.1, 0.8, 0.2],
            "res8_zone": ["a", "b", "a", "b"],
            "noise_feature": [1.0, 1.0, 1.0, 1.0],
        }
    )

    summary = _screen_coarse_layer(
        frame,
        target_col="alert_next_30m",
        horizon_minutes=30,
        layer_name="res8",
        feature_cols=["noise_feature", "res8_event_count_24h", "zone_24h_share_of_res8", "res8_zone"],
    )

    assert summary["best_feature"] == "res8_event_count_24h"
    assert summary["screen_pass"] is True
    assert summary["within_time_positive_rank"] > 0.8


def test_phase_a_gate_requires_a_short_horizon_pass():
    analysis = pd.DataFrame(
        [
            {"horizon": "30m", "layer": "district", "screen_pass": False},
            {"horizon": "1h", "layer": "police_zone", "screen_pass": False},
            {"horizon": "2h", "layer": "res8", "screen_pass": True},
        ]
    )

    assert _phase_a_gate(analysis) is False

    analysis.loc[analysis["horizon"] == "1h", "screen_pass"] = True

    assert _phase_a_gate(analysis) is True
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_two_stage_experiment.py -k "screen_coarse_layer or phase_a_gate" -q -p no:cacheprovider --basetemp .pytest_tmp_coarse_task2_red
```

Expected: FAIL with missing `_screen_coarse_layer` or missing `_phase_a_gate`.

- [ ] **Step 3: Add the Phase A helpers**

Add this code to `analysis/run_two_stage_experiment.py` near the other helper functions:

```python
from sklearn.feature_selection import mutual_info_classif


COARSE_LAYER_ANALYSIS_FEATURES = {
    "district": [
        "district_event_count_24h",
        "district_same_hour_rate",
        "zone_24h_share_of_district",
        "zone_same_hour_percentile_in_district",
        "district",
    ],
    "police_zone": [
        "police_zone_event_count_24h",
        "police_zone_same_hour_rate",
        "zone_24h_share_of_police_zone",
        "zone_same_hour_percentile_in_police_zone",
        "police_zone",
    ],
    "res8": [
        "res8_event_count_24h",
        "res8_same_hour_rate",
        "zone_24h_share_of_res8",
        "zone_same_hour_percentile_in_res8",
        "res8_zone",
    ],
}
PHASE_A_MUTUAL_INFORMATION_MIN = 0.001
PHASE_A_TARGET_RATE_LIFT_MIN = 1.10
PHASE_A_POSITIVE_RANK_MIN = 0.55


def _feature_signal_summary(
    frame: pd.DataFrame,
    target_col: str,
    feature_col: str,
) -> dict[str, Any]:
    scored = frame[["target_time", target_col, feature_col]].copy().dropna(subset=[feature_col])
    if scored.empty:
        return {
            "feature": feature_col,
            "mutual_information": 0.0,
            "target_rate_lift": 0.0,
            "within_time_positive_rank": 0.0,
        }

    target = scored[target_col].astype(int)
    raw = scored[feature_col]
    discrete = raw.dtype == "object" or str(raw.dtype).startswith("category")
    if discrete:
        encoded = pd.Series(pd.factorize(raw.fillna("Unknown"))[0], index=scored.index, dtype=float)
        local_strength = scored.groupby(["target_time", feature_col])[target_col].transform("mean")
        mi_input = encoded.to_frame()
    else:
        encoded = pd.to_numeric(raw, errors="coerce").fillna(0.0)
        local_strength = encoded
        mi_input = encoded.to_frame()

    ranked = local_strength.groupby(scored["target_time"]).rank(method="average", pct=True)
    top_bucket = ranked >= 0.8
    base_rate = float(target.mean()) if len(target) else 0.0
    top_rate = float(target[top_bucket].mean()) if top_bucket.any() else 0.0
    mutual_information = 0.0
    if len(scored) >= 4 and target.nunique() > 1:
        mutual_information = float(
            mutual_info_classif(
                mi_input,
                target,
                discrete_features=discrete,
                random_state=42,
            )[0]
        )

    return {
        "feature": feature_col,
        "mutual_information": mutual_information,
        "target_rate_lift": float(top_rate / base_rate) if base_rate > 0 else 0.0,
        "within_time_positive_rank": float(ranked[target == 1].mean()) if int(target.sum()) else 0.0,
    }


def _screen_coarse_layer(
    frame: pd.DataFrame,
    target_col: str,
    horizon_minutes: int,
    layer_name: str,
    feature_cols: list[str],
) -> dict[str, Any]:
    rows = [
        _feature_signal_summary(frame, target_col, feature_col)
        for feature_col in feature_cols
        if feature_col in frame.columns
    ]
    ranked = pd.DataFrame(rows).sort_values(
        ["within_time_positive_rank", "mutual_information", "target_rate_lift"],
        ascending=[False, False, False],
    )
    best = ranked.iloc[0].to_dict()
    screen_pass = bool(
        best["mutual_information"] >= PHASE_A_MUTUAL_INFORMATION_MIN
        and best["target_rate_lift"] >= PHASE_A_TARGET_RATE_LIFT_MIN
        and best["within_time_positive_rank"] >= PHASE_A_POSITIVE_RANK_MIN
    )
    return {
        "horizon_minutes": horizon_minutes,
        "horizon": horizon_slug(horizon_minutes),
        "layer": layer_name,
        "best_feature": best["feature"],
        "mutual_information": best["mutual_information"],
        "target_rate_lift": best["target_rate_lift"],
        "within_time_positive_rank": best["within_time_positive_rank"],
        "screen_pass": screen_pass,
    }


def _phase_a_gate(analysis: pd.DataFrame) -> bool:
    if analysis.empty:
        return False
    short_horizon = analysis[analysis["horizon"].isin({"30m", "1h"})]
    return bool(short_horizon["screen_pass"].any())
```

- [ ] **Step 4: Re-run the focused tests and verify GREEN**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_two_stage_experiment.py -k "screen_coarse_layer or phase_a_gate" -q -p no:cacheprovider --basetemp .pytest_tmp_coarse_task2_green
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```powershell
git add analysis\run_two_stage_experiment.py tests\test_two_stage_experiment.py
git commit -m "feat: add coarse layer signal screening"
```

---

### Task 3: Parameterize the spatial pipeline for ablation-specific feature sets

**Files:**
- Modify: `analysis/run_two_stage_experiment.py`
- Modify: `tests/test_two_stage_experiment.py`

- [ ] **Step 1: Write the failing feature-pack tests**

Add these tests to `tests/test_two_stage_experiment.py`:

```python
def test_spatial_feature_sets_add_only_requested_incremental_packs():
    numeric, categorical = _spatial_feature_sets(("res8",))

    assert "res8_event_count_24h" in numeric
    assert "police_zone_event_count_24h" not in numeric
    assert "district" in categorical
    assert "res8_zone" in categorical


def test_spatial_feature_sets_keep_district_in_baseline_only():
    numeric, categorical = _spatial_feature_sets(())

    assert "district_event_count_24h" in numeric
    assert "district" in categorical
    assert "police_zone_event_count_24h" not in numeric
    assert "res8_zone" not in categorical
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_two_stage_experiment.py -k "spatial_feature_sets" -q -p no:cacheprovider --basetemp .pytest_tmp_coarse_task3_red
```

Expected: FAIL with missing `_spatial_feature_sets`.

- [ ] **Step 3: Add the local spatial preprocessor and feature-pack selector**

Replace the imported shared pipeline usage in `analysis/run_two_stage_experiment.py` with local helpers:

```python
COARSE_LAYER_INCREMENTAL_PACKS = {
    "police_zone": {
        "numeric": [
            "police_zone_event_count_3h",
            "police_zone_event_count_24h",
            "police_zone_event_count_7d",
            "police_zone_same_hour_rate",
            "police_zone_active_zones_24h",
            "zone_24h_share_of_police_zone",
            "zone_7d_rank_in_police_zone",
            "zone_same_hour_percentile_in_police_zone",
        ],
        "categorical": [],
    },
    "res8": {
        "numeric": [
            "res8_event_count_3h",
            "res8_event_count_24h",
            "res8_event_count_7d",
            "res8_same_hour_rate",
            "res8_active_zones_24h",
            "zone_24h_share_of_res8",
            "zone_7d_rank_in_res8",
            "zone_same_hour_percentile_in_res8",
        ],
        "categorical": ["res8_zone"],
    },
}


def _spatial_feature_sets(extra_layers: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    numeric = list(NUMERIC_FEATURES)
    categorical = list(CATEGORICAL_FEATURES)
    for layer in extra_layers:
        numeric.extend(COARSE_LAYER_INCREMENTAL_PACKS[layer]["numeric"])
        categorical.extend(COARSE_LAYER_INCREMENTAL_PACKS[layer]["categorical"])
    return list(dict.fromkeys(numeric)), list(dict.fromkeys(categorical))


def _spatial_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def _make_spatial_pipeline(
    candidate: Candidate,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", _spatial_preprocessor(numeric_features, categorical_features)),
            ("model", candidate.factory()),
        ]
    )


def _fit_spatial_candidate(
    candidate: Candidate,
    train_sample: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str,
) -> Pipeline | None:
    feature_cols = numeric_features + categorical_features
    pipeline = _make_spatial_pipeline(candidate, numeric_features, categorical_features)
    if candidate.kind == "ranker":
        ranker_sample, group_sizes = _prepare_ranker_training_frame(train_sample, target_col)
        if ranker_sample.empty or not group_sizes:
            return None
        pipeline.fit(
            ranker_sample[feature_cols],
            ranker_sample[target_col],
            model__group=group_sizes,
        )
        return pipeline

    pipeline.fit(train_sample[feature_cols], train_sample[target_col])
    return pipeline


def _evaluate_spatial_candidates(
    df: pd.DataFrame,
    target_col: str,
    horizon_minutes: int,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    feature_cols = numeric_features + categorical_features
    splits = make_purged_rolling_splits(df, horizon_minutes=horizon_minutes)
    rows: list[dict[str, Any]] = []
    for candidate in _candidate_models():
        for split in splits:
            train_sample = sample_spatial_training_rows(
                df.loc[split.train_mask],
                target_col,
                negative_ratio=5,
                inactive_negative_fraction=0.02,
            )
            if train_sample.empty or train_sample[target_col].nunique() < 2:
                continue
            pipeline = _fit_spatial_candidate(
                candidate,
                train_sample,
                numeric_features,
                categorical_features,
                target_col,
            )
            if pipeline is None:
                continue
            scores = _predict_spatial_candidate_scores(
                pipeline,
                candidate,
                df.loc[split.validation_mask],
                feature_cols,
            )
            scores = _blend_recent_spatial_prior_scores(
                df.loc[split.validation_mask],
                scores,
                alpha=_spatial_blend_alpha(target_col),
            )
            metrics = _score_spatial_predictions(
                df.loc[split.validation_mask, target_col],
                scores,
                groups={
                    "district": df.loc[split.validation_mask, "district"],
                    "region": df.loc[split.validation_mask, "region"],
                },
            )
            metrics.update(
                _neighbor_hit_metrics(
                    df.loc[split.validation_mask],
                    target_col,
                    scores,
                )
            )
            metrics.update(
                _operational_spatial_metrics(
                    df.loc[split.validation_mask],
                    target_col,
                    scores,
                )
            )
            rows.append(
                {
                    "model": candidate.name,
                    "fold": split.metadata["fold"],
                    "train_rows": int(len(train_sample)),
                    "unsampled_train_rows": split.metadata["train_rows"],
                    "validation_rows": split.metadata["validation_rows"],
                    **metrics,
                    **{f"split_{key}": value for key, value in split.metadata.items()},
                }
            )
    fold_results = pd.DataFrame(rows)
    if fold_results.empty:
        raise ValueError("No valid spatial folds had both classes after sampling.")
    summary = (
        fold_results.groupby("model", sort=False)
        .agg(
            folds=("fold", "count"),
            median_precision_at_20=("precision_at_20", "median"),
            median_precision_at_50=("precision_at_50", "median"),
            median_precision_at_100=("precision_at_100", "median"),
            median_recall_at_50=("recall_at_50", "median"),
            median_recall_at_100=("recall_at_100", "median"),
            median_average_precision=("average_precision", "median"),
            median_top_decile_lift=("top_decile_lift", "median"),
            median_neighbor_hit_rate_at_20=("neighbor_hit_rate_at_20", "median"),
            median_neighbor_hit_rate_at_50=("neighbor_hit_rate_at_50", "median"),
            median_neighbor_hit_rate_at_100=("neighbor_hit_rate_at_100", "median"),
            median_group_precision_at_50=("group_precision_at_50", "median"),
            median_group_recall_at_50=("group_recall_at_50", "median"),
            median_district_hit_rate_at_50=("district_hit_rate_at_50", "median"),
            median_region_hit_rate_at_50=("region_hit_rate_at_50", "median"),
            median_brier_score=("brier_score", "median"),
            median_expected_calibration_error=("expected_calibration_error", "median"),
            median_roc_auc=("roc_auc", "median"),
        )
        .reset_index()
    )
    return fold_results, summary


def _fit_spatial_holdout(
    df: pd.DataFrame,
    target_col: str,
    chosen: dict[str, Any],
    activity_holdout_predictions: pd.DataFrame,
    paths: dict[str, Path],
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> dict[str, Any]:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    feature_cols = numeric_features + categorical_features
    split = make_positive_count_holdout(df, target_col, min_positives=50)
    train_sample = sample_spatial_training_rows(
        df.loc[split.train_mask],
        target_col,
        negative_ratio=5,
        inactive_negative_fraction=0.02,
    )
    selected = next(item for item in _candidate_models() if item.name == chosen["model"])
    pipeline = _fit_spatial_candidate(
        selected,
        train_sample,
        numeric_features,
        categorical_features,
        target_col,
    )
    if pipeline is None:
        raise ValueError(f"Could not fit spatial candidate: {selected.name}")
    spatial_scores = _predict_spatial_candidate_scores(
        pipeline,
        selected,
        df.loc[split.holdout_mask],
        feature_cols,
    )
    spatial_scores = _blend_recent_spatial_prior_scores(
        df.loc[split.holdout_mask],
        spatial_scores,
        alpha=_spatial_blend_alpha(target_col),
    )
    default_activity = (
        float(activity_holdout_predictions["activity_probability"].mean())
        if not activity_holdout_predictions.empty
        else 1.0
    )
    activity_scores = _probabilities_for_time(
        df.loc[split.holdout_mask, "target_time"],
        activity_holdout_predictions,
        default_activity,
    )
    base = df.loc[
        split.holdout_mask,
        ["target_time", "zone_id", "district", "region", "zone_lat", "zone_lng"],
    ].copy()
    base["spatial_probability"] = np.clip(spatial_scores, 0.0, 1.0)
    base["actual"] = df.loc[split.holdout_mask, target_col].astype(int).to_numpy()
    predictions = combine_activity_and_spatial_scores(base, activity_scores)
    predictions = assign_dispatch_rank(
        predictions,
        per_target_time_quota=_dispatch_quota_for_target(target_col),
    )
    predictions.to_csv(paths["spatial_predictions"], index=False)
    joblib.dump(pipeline, paths["spatial_model"])
    holdout_metrics = _score_spatial_predictions(
        df.loc[split.holdout_mask, target_col],
        predictions["spatial_probability"].to_numpy(),
        groups={
            "district": df.loc[split.holdout_mask, "district"],
            "region": df.loc[split.holdout_mask, "region"],
        },
    )
    holdout_metrics.update(
        _neighbor_hit_metrics(
            df.loc[split.holdout_mask],
            target_col,
            predictions["spatial_probability"].to_numpy(),
        )
    )
    holdout_metrics.update(
        _operational_spatial_metrics(
            df.loc[split.holdout_mask],
            target_col,
            predictions["spatial_probability"].to_numpy(),
        )
    )
    holdout_metrics.update(_dispatch_topk_metrics(predictions, k=50))

    return {
        "model_path": _relative(paths["spatial_model"]),
        "predictions_path": _relative(paths["spatial_predictions"]),
        "holdout_metrics": holdout_metrics,
        "holdout_split": split.metadata,
        "training_rows_sampled": int(len(train_sample)),
    }


def run_two_stage_horizon(
    events: list[dict],
    horizon_minutes: int,
    resolution: int = DEFAULT_H3_RESOLUTION,
    spatial_layers: tuple[str, ...] = (),
    artifact_slug: str | None = None,
) -> dict[str, Any]:
    slug = artifact_slug or horizon_slug(horizon_minutes)
    paths = _artifact_paths(slug)
    target_col = target_for_horizon(horizon_minutes)
    activity_target = activity_target_for_horizon(horizon_minutes)
    lookback_hours = _effective_lookback_hours(events)
    lookback_days = max(1, int(lookback_hours // 24))
    numeric_features, categorical_features = _spatial_feature_sets(spatial_layers)
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
    if activity_df.empty or spatial_df.empty:
        raise RuntimeError("Two-stage feature tables are empty.")

    activity_folds, activity_summary = _evaluate_activity_candidates(
        activity_df,
        activity_target,
        horizon_minutes,
    )
    spatial_folds, spatial_summary = _evaluate_spatial_candidates(
        spatial_df,
        target_col,
        horizon_minutes,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    activity_chosen = _select_model(activity_summary, "activity")
    spatial_chosen = _select_model(spatial_summary, "spatial")

    activity_holdout = _fit_activity_holdout(
        activity_df,
        activity_target,
        activity_chosen,
        horizon_minutes,
        paths,
    )
    spatial_holdout = _fit_spatial_holdout(
        spatial_df,
        target_col,
        spatial_chosen,
        activity_holdout["holdout_predictions"],
        paths,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    activity_folds.to_csv(paths["activity_folds"], index=False)
    spatial_folds.to_csv(paths["spatial_folds"], index=False)
    activity_metadata = {
        "stage": "activity",
        "resolution": resolution,
        "horizon_minutes": horizon_minutes,
        "horizon_slug": slug,
        "target_col": activity_target,
        "chosen_model": activity_chosen,
        "fold_summary": activity_summary.to_dict("records"),
        "folds_path": _relative(paths["activity_folds"]),
        "features": ACTIVITY_NUMERIC_FEATURES + ACTIVITY_CATEGORICAL_FEATURES,
        "training_rows": int(len(activity_df)),
        **{key: value for key, value in activity_holdout.items() if key != "holdout_predictions"},
    }
    spatial_metadata = {
        "stage": "spatial",
        "resolution": resolution,
        "horizon_minutes": horizon_minutes,
        "horizon_slug": slug,
        "target_col": target_col,
        "chosen_model": spatial_chosen,
        "fold_summary": spatial_summary.to_dict("records"),
        "folds_path": _relative(paths["spatial_folds"]),
        "features": numeric_features + categorical_features,
        "spatial_layers": list(spatial_layers),
        "training_rows": int(len(spatial_df)),
        **spatial_holdout,
    }
    paths["activity_metadata"].write_text(json.dumps(activity_metadata, indent=2), encoding="utf-8")
    paths["spatial_metadata"].write_text(json.dumps(spatial_metadata, indent=2), encoding="utf-8")

    return {
        "horizon_minutes": horizon_minutes,
        "horizon": slug,
        "activity_model": activity_metadata,
        "spatial_model": spatial_metadata,
        "activity_metadata_path": _relative(paths["activity_metadata"]),
        "spatial_metadata_path": _relative(paths["spatial_metadata"]),
        "activity_predictions_path": _relative(paths["activity_predictions"]),
        "predictions_path": _relative(paths["spatial_predictions"]),
    }
```

- [ ] **Step 4: Re-run the focused tests and verify GREEN**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_two_stage_experiment.py -k "spatial_feature_sets" -q -p no:cacheprovider --basetemp .pytest_tmp_coarse_task3_green
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```powershell
git add analysis\run_two_stage_experiment.py tests\test_two_stage_experiment.py
git commit -m "feat: parameterize coarse ablation feature sets"
```

---

### Task 4: Wire the top-level screening runner and Phase B comparison flow

**Files:**
- Modify: `analysis/run_two_stage_experiment.py`
- Modify: `tests/test_two_stage_experiment.py`

- [ ] **Step 1: Write the failing runner tests**

Add these tests to `tests/test_two_stage_experiment.py`:

```python
import analysis.run_two_stage_experiment as two_stage


def test_choose_best_coarse_variant_requires_whole_stack_win():
    comparison = pd.DataFrame(
        [
            {"variant": "baseline", "horizon": "30m", "dispatch_precision_at_50": 0.10},
            {"variant": "baseline", "horizon": "1h", "dispatch_precision_at_50": 0.12},
            {"variant": "baseline", "horizon": "2h", "dispatch_precision_at_50": 0.04},
            {"variant": "baseline_plus_res8", "horizon": "30m", "dispatch_precision_at_50": 0.11},
            {"variant": "baseline_plus_res8", "horizon": "1h", "dispatch_precision_at_50": 0.13},
            {"variant": "baseline_plus_res8", "horizon": "2h", "dispatch_precision_at_50": 0.05},
            {"variant": "baseline_plus_police_zone", "horizon": "30m", "dispatch_precision_at_50": 0.09},
            {"variant": "baseline_plus_police_zone", "horizon": "1h", "dispatch_precision_at_50": 0.11},
            {"variant": "baseline_plus_police_zone", "horizon": "2h", "dispatch_precision_at_50": 0.08},
        ]
    )

    assert _choose_best_coarse_variant(comparison, baseline="baseline") == "baseline_plus_res8"


def test_run_coarse_layer_feature_screening_skips_phase_b_when_gate_fails(monkeypatch, tmp_path):
    analysis = pd.DataFrame(
        [
            {"horizon": "30m", "layer": "district", "screen_pass": False},
            {"horizon": "1h", "layer": "police_zone", "screen_pass": False},
            {"horizon": "2h", "layer": "res8", "screen_pass": True},
        ]
    )
    called = {"phase_b": False}

    monkeypatch.setattr(two_stage, "_load_events", lambda: pd.DataFrame([{"create_dt": "2026-06-01 00:00:00"}]))
    monkeypatch.setattr(two_stage, "_run_phase_a_signal_screen", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(
        two_stage,
        "_run_phase_b_ablations",
        lambda *args, **kwargs: called.__setitem__("phase_b", True),
    )
    monkeypatch.setattr(two_stage, "COARSE_LAYER_ABLATION_PATH", tmp_path / "coarse_layer_ablation.csv")

    result = two_stage.run_coarse_layer_feature_screening(horizons=[30])

    assert result["decision"] == "reject"
    assert result["winning_variant"] is None
    assert called["phase_b"] is False
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_two_stage_experiment.py -k "choose_best_coarse_variant or run_coarse_layer_feature_screening" -q -p no:cacheprovider --basetemp .pytest_tmp_coarse_task4_red
```

Expected: FAIL with missing `_choose_best_coarse_variant`, missing `run_coarse_layer_feature_screening`, or missing `_run_phase_b_ablations`.

- [ ] **Step 3: Add the Phase B runner and comparison artifact writer**

Add this code to `analysis/run_two_stage_experiment.py`:

```python
COARSE_LAYER_SIGNAL_PATH_TEMPLATE = "coarse_layer_signal_{slug}_latest.csv"
COARSE_LAYER_ABLATION_PATH = OUTPUT_DIR / "coarse_layer_ablation_comparison_latest.csv"
COARSE_LAYER_ABLATION_VARIANTS = {
    "baseline": (),
    "baseline_plus_res8": ("res8",),
    "baseline_plus_police_zone": ("police_zone",),
    "baseline_plus_res8_police_zone": ("res8", "police_zone"),
}


def _whole_stack_wins(comparison: pd.DataFrame, variant: str, baseline: str = "baseline") -> bool:
    pivot = comparison.pivot(index="variant", columns="horizon", values="dispatch_precision_at_50")
    base = pivot.loc[baseline]
    challenger = pivot.loc[variant]
    return bool(
        challenger["30m"] >= base["30m"]
        and challenger["1h"] >= base["1h"]
        and challenger.mean() > base.mean()
    )


def _choose_best_coarse_variant(
    comparison: pd.DataFrame,
    baseline: str = "baseline",
) -> str | None:
    candidates = [variant for variant in comparison["variant"].unique() if variant != baseline]
    winners = [
        (
            variant,
            float(
                comparison.loc[comparison["variant"] == variant, "dispatch_precision_at_50"].mean()
            ),
        )
        for variant in candidates
        if _whole_stack_wins(comparison, variant, baseline)
    ]
    if not winners:
        return None
    winners.sort(key=lambda item: item[1], reverse=True)
    return winners[0][0]


def _write_coarse_layer_signal_table(
    rows: list[dict[str, Any]],
    path: Path,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values(["horizon_minutes", "layer"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _write_coarse_layer_ablation_comparison(
    results_by_variant: dict[str, list[dict[str, Any]]],
    path: Path = COARSE_LAYER_ABLATION_PATH,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, results in results_by_variant.items():
        for item in results:
            spatial = item["spatial_model"]["holdout_metrics"]
            rows.append(
                {
                    "variant": variant,
                    "horizon_minutes": item["horizon_minutes"],
                    "horizon": item["horizon"],
                    "dispatch_precision_at_50": spatial.get("dispatch_precision_at_50", 0.0),
                    "spatial_precision_at_50": spatial.get("precision_at_50", 0.0),
                    "group_recall_at_50": spatial.get("group_recall_at_50", 0.0),
                    "neighbor_hit_rate_at_50": spatial.get("neighbor_hit_rate_at_50", 0.0),
                    "spatial_model": item["spatial_model"]["chosen_model"]["model"],
                    "predictions_path": item["predictions_path"],
                }
            )
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "variant",
                "horizon_minutes",
                "horizon",
                "dispatch_precision_at_50",
                "spatial_precision_at_50",
                "group_recall_at_50",
                "neighbor_hit_rate_at_50",
                "spatial_model",
                "predictions_path",
            ]
        )
    else:
        frame = frame.sort_values(["variant", "horizon_minutes"])
    frame.to_csv(path, index=False)
    return frame


def _run_phase_a_signal_screen(
    events: list[dict[str, Any]],
    horizons: list[int],
    resolution: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon_minutes in horizons:
        target_col = target_for_horizon(horizon_minutes)
        lookback_hours = _effective_lookback_hours(events)
        spatial_df = build_zone_ranking_training_data(
            events,
            forecast_hours=max(1, horizon_minutes // 60),
            horizon_minutes=horizon_minutes,
            target_col=target_col,
            lookback_days=max(1, int(lookback_hours // 24)),
            resolution=resolution,
        )
        for layer_name, feature_cols in COARSE_LAYER_ANALYSIS_FEATURES.items():
            rows.append(
                _screen_coarse_layer(
                    spatial_df,
                    target_col=target_col,
                    horizon_minutes=horizon_minutes,
                    layer_name=layer_name,
                    feature_cols=feature_cols,
                )
            )
        slug = horizon_slug(horizon_minutes)
        _write_coarse_layer_signal_table(
            [row for row in rows if row["horizon"] == slug],
            OUTPUT_DIR / COARSE_LAYER_SIGNAL_PATH_TEMPLATE.format(slug=slug),
        )
    return pd.DataFrame(rows)


def _run_phase_b_ablations(
    events: list[dict[str, Any]],
    horizons: list[int],
    resolution: int,
) -> dict[str, list[dict[str, Any]]]:
    return {
        variant: [
            run_two_stage_horizon(
                events,
                horizon_minutes=horizon_minutes,
                resolution=resolution,
                spatial_layers=layers,
                artifact_slug=f"{variant}_{horizon_slug(horizon_minutes)}",
            )
            for horizon_minutes in horizons
        ]
        for variant, layers in COARSE_LAYER_ABLATION_VARIANTS.items()
    }


def run_coarse_layer_feature_screening(
    horizons: list[int] | None = None,
    resolution: int = DEFAULT_H3_RESOLUTION,
) -> dict[str, Any]:
    events = _load_events().to_dict("records")
    chosen_horizons = horizons or HORIZONS
    analysis = _run_phase_a_signal_screen(events, chosen_horizons, resolution)
    if not _phase_a_gate(analysis):
        _write_coarse_layer_ablation_comparison({})
        return {
            "decision": "reject",
            "reason": "No coarse layer cleared the 30m/1h analysis gate.",
            "winning_variant": None,
            "analysis_rows": int(len(analysis)),
            "ablation_path": str(_relative(COARSE_LAYER_ABLATION_PATH)),
        }

    results_by_variant = _run_phase_b_ablations(events, chosen_horizons, resolution)
    comparison = _write_coarse_layer_ablation_comparison(results_by_variant)
    winning_variant = _choose_best_coarse_variant(comparison, baseline="baseline")
    if winning_variant:
        write_two_stage_summary(results_by_variant[winning_variant], SUMMARY_PATH)
    return {
        "decision": "keep" if winning_variant else "reject",
        "reason": "" if winning_variant else "No ablation variant beat the accepted baseline on the whole-stack gate.",
        "winning_variant": winning_variant,
        "analysis_rows": int(len(analysis)),
        "ablation_rows": int(len(comparison)),
        "ablation_path": str(_relative(COARSE_LAYER_ABLATION_PATH)),
    }
```

- [ ] **Step 4: Re-run the focused tests and verify GREEN**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_two_stage_experiment.py -k "choose_best_coarse_variant or run_coarse_layer_feature_screening" -q -p no:cacheprovider --basetemp .pytest_tmp_coarse_task4_green
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```powershell
git add analysis\run_two_stage_experiment.py tests\test_two_stage_experiment.py
git commit -m "feat: wire coarse layer ablation runner"
```

---

### Task 5: Run the experiment, record the outcome, and leave the baseline intact unless a winner proves itself

**Files:**
- Modify: `WORKLOG.md`
- Modify: `WORKLOG_INDEX.md`

- [ ] **Step 1: Run the smallest full validation slice**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_ghost_ranking_features.py tests\test_two_stage_experiment.py -q -p no:cacheprovider --basetemp .pytest_tmp_coarse_full
```

Expected: PASS.

- [ ] **Step 2: Run the coarse-layer screen**

Run:

```powershell
.venv\Scripts\python.exe -c "from analysis.run_two_stage_experiment import run_coarse_layer_feature_screening; import json; print(json.dumps(run_coarse_layer_feature_screening(), indent=2))"
```

Expected: JSON result with `decision`, `winning_variant`, and the analysis / ablation artifact paths.

- [ ] **Step 3: Refresh downstream artifacts only if a winner exists**

If the JSON output shows a non-null `winning_variant`, run:

```powershell
.venv\Scripts\python.exe analysis\analyze_spatial_model_errors.py --k 50
.venv\Scripts\python.exe analysis\build_dashboard_manifest.py
```

Expected: updated error-summary and dashboard-manifest artifacts for the winning variant.

- [ ] **Step 4: Record the experiment in the worklog**

Append a top-level `## 2026-07-08 Coarse-layer feature screening experiment` entry to `WORKLOG.md` using `WORKLOG_TEMPLATE.md`, and update `WORKLOG_INDEX.md` in the same change. Record:

1. The analysis gate result for `district`, `police_zone`, and `res8` at `30m`, `1h`, and `2h`
2. Whether Phase B ran
3. The exact comparison rows from `analysis\coarse_layer_ablation_comparison_latest.csv`
4. Whether the accepted baseline stayed in place or a new winner cleared the whole-stack gate

- [ ] **Step 5: Commit the recorded outcome**

```powershell
git add WORKLOG.md WORKLOG_INDEX.md analysis\run_two_stage_experiment.py ghost_ranking_features.py tests\test_ghost_ranking_features.py tests\test_two_stage_experiment.py
git commit -m "exp: record coarse layer feature screening"
```
