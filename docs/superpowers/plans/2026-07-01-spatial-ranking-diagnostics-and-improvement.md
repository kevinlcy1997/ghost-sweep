# Spatial Ranking Diagnostics and Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) and superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax tracking. This plan includes cost-aware model routing; use the cheapest suggested tier that can satisfy each task's review gate, and escalate only when the listed escalation trigger occurs.

**Goal:** Diagnose why Stage 2 spatial precision@20 and precision@50 are zero, then improve or reframe the spatial ranking objective so top-ranked zones become operationally useful.

**Architecture:** Add a lightweight diagnostics layer around existing two-stage prediction artifacts before changing model training. Use the diagnostics to decide whether misses are near positives, in the correct district/region, poorly calibrated, or genuinely wrong, then add one narrow model improvement path focused on top-k ranking quality. Keep the current two-stage runner and feature builders intact except for scoped additions that are covered by tests.

**Tech Stack:** Python, pandas, NumPy, h3, scikit-learn/LightGBM, pytest, existing `analysis/run_two_stage_experiment.py`, `ghost_ranking_metrics.py`, and generated CSV/JSON artifacts.

---

## Current Baseline

The latest completed run produced these Stage 2 holdout results in `analysis/two_stage_summary_latest.csv`:

| Horizon | Spatial precision@20 | Spatial precision@50 | Spatial AP | Top-decile lift | District hit-rate@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30m | 0.0 | 0.0 | 0.001463 | 2.599759 | 0.583333 |
| 1h | 0.0 | 0.0 | 0.000573 | 0.175426 | 0.571429 |
| 2h | 0.0 | 0.0 | 0.000580 | 0.363619 | 0.181818 |

Assumption for this plan: the product workflow cares about a short top-k list, but a near miss in an adjacent H3 cell or same street/district may still be useful enough to track separately from exact-zone precision.

## Agent and Model Routing

| Workstream | Agent Role | Suggested Model Tier | Cost Control | Review Gate |
| --- | --- | --- | --- | --- |
| Artifact and metric map | Investigator | low-cost | Read-only; inspect only listed artifact/test/metric files | Main agent confirms artifact columns, current metric definitions, and no code edits |
| Near-miss diagnostics | Builder | mid-tier | One new analysis script plus focused tests | Script emits exact top-k miss table and tests pass |
| Spatial metric design | Architect | high-capability | One judgment pass after diagnostics exist | Metric definitions distinguish exact, neighbor, district, and region hits without leakage |
| Metric implementation | Builder | mid-tier | Edit metric module and two-stage scoring only | Tests prove new metrics and existing metrics still pass |
| Ranking improvement experiment | Builder | mid-tier | Narrow changes to spatial candidate selection/model config; no broad refactor | Full runner completes and top-k/near-hit metrics are compared against baseline |
| Final validation and checkpoint | Reviewer | low-cost | Run listed commands only and summarize outputs | Summary CSV, diagnostics CSV, tests, and Notion checkpoint are present |

## Task 1: Map Current Spatial Artifacts and Label Sparsity

**Files:**
- Read: `analysis/two_stage_summary_latest.csv`
- Read: `analysis/spatial_zone_predictions_30m_latest.csv`
- Read: `analysis/spatial_zone_predictions_1h_latest.csv`
- Read: `analysis/spatial_zone_predictions_2h_latest.csv`
- Read: `analysis/spatial_model_metadata_30m.json`
- Read: `analysis/spatial_model_metadata_1h.json`
- Read: `analysis/spatial_model_metadata_2h.json`
- Create: `analysis/spatial_ranking_diagnostic_snapshot_latest.csv`

**Agent Role:** Investigator

**Suggested Model Tier:** low-cost

**Why This Tier:** This is deterministic artifact inspection and summarization with no source edits.

**Inputs Needed:** The files listed above and this task section.

**Expected Output:** A CSV snapshot with row counts, positive counts, base rates, score ranges, exact top-k hit counts, and distinct target-time counts for each horizon.

**Review Gate:** Main agent verifies the snapshot values match the source artifacts and confirms which horizon should be optimized first.

- [x] **Step 1: Create a read-only artifact profiler**

Run this command from repo root:

```bash
.venv-ghost/bin/python - <<'PY'
import json
from pathlib import Path
import pandas as pd

rows = []
for slug in ["30m", "1h", "2h"]:
    pred_path = Path(f"analysis/spatial_zone_predictions_{slug}_latest.csv")
    meta_path = Path(f"analysis/spatial_model_metadata_{slug}.json")
    pred = pd.read_csv(pred_path)
    meta = json.loads(meta_path.read_text())
    target_col = "actual" if "actual" in pred.columns else "target"
    score_col = "spatial_probability" if "spatial_probability" in pred.columns else "probability"
    ordered = pred.sort_values(["target_time", score_col], ascending=[True, False])
    top20 = ordered.groupby("target_time").head(20)
    top50 = ordered.groupby("target_time").head(50)
    rows.append({
        "horizon": slug,
        "prediction_rows": len(pred),
        "target_times": pred["target_time"].nunique(),
        "positives": int(pred[target_col].sum()),
        "base_rate": float(pred[target_col].mean()),
        "score_min": float(pred[score_col].min()),
        "score_p50": float(pred[score_col].median()),
        "score_p95": float(pred[score_col].quantile(0.95)),
        "score_max": float(pred[score_col].max()),
        "exact_hits_at_20": int(top20[target_col].sum()),
        "exact_hits_at_50": int(top50[target_col].sum()),
        "metadata_precision_at_50": meta["holdout_metrics"]["precision_at_50"],
        "metadata_top_decile_lift": meta["holdout_metrics"]["top_decile_lift"],
    })

out = pd.DataFrame(rows)
out.to_csv("analysis/spatial_ranking_diagnostic_snapshot_latest.csv", index=False)
print(out.to_string(index=False))
PY
```

Expected: command exits `0` and writes `analysis/spatial_ranking_diagnostic_snapshot_latest.csv` with three rows.

- [x] **Step 2: Review score compression**

Run:

```bash
.venv-ghost/bin/python - <<'PY'
import pandas as pd
for slug in ["30m", "1h", "2h"]:
    pred = pd.read_csv(f"analysis/spatial_zone_predictions_{slug}_latest.csv")
    score_col = "spatial_probability" if "spatial_probability" in pred.columns else "probability"
    print(slug, pred[score_col].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())
PY
```

Expected: output shows whether probabilities are saturated near `0` or `1`, or too flat to rank top-k.

- [x] **Step 3: Commit only if a snapshot artifact should be versioned**

If `analysis/spatial_ranking_diagnostic_snapshot_latest.csv` is intended to be retained, commit it with:

```bash
git add analysis/spatial_ranking_diagnostic_snapshot_latest.csv
git commit -m "analysis: capture spatial ranking diagnostic snapshot"
```

Expected: commit succeeds. If generated analysis snapshots are not committed in this repo, leave it untracked and note the path in the final report.

## Task 2: Add Exact and Near-Miss Top-K Diagnostic Script

**Files:**
- Create: `analysis/diagnose_spatial_ranking.py`
- Create: `tests/test_spatial_ranking_diagnostics.py`
- Read: `ghost_zones.py`

**Agent Role:** Builder

**Suggested Model Tier:** mid-tier

**Why This Tier:** This adds a focused analysis script and tests using existing prediction artifacts and H3 helpers; implementation is straightforward but needs careful metric semantics.

**Inputs Needed:** This task section, existing prediction CSV schema, and `ghost_zones.py` H3 helper patterns.

**Expected Output:** A tested script that reports exact, ring-1, ring-2, district, and region hit rates for top-k predictions by target time.

**Review Gate:** Tests pass and the script output explains whether top-ranked misses are near actual positives.

Escalate to high-capability if prediction artifacts do not contain district, region, `zone_id`, `target_time`, score, and label columns consistently.

- [x] **Step 1: Write failing tests for near-miss diagnostics**

Create `tests/test_spatial_ranking_diagnostics.py`:

```python
import pandas as pd

from analysis.diagnose_spatial_ranking import summarize_topk_near_misses


def test_summarize_topk_near_misses_counts_neighbor_hit():
    rows = pd.DataFrame(
        [
            {
                "target_time": "2026-06-01 10:00:00",
                "zone_id": "a",
                "spatial_probability": 0.9,
                "actual": 0,
                "district": "D1",
                "region": "R1",
            },
            {
                "target_time": "2026-06-01 10:00:00",
                "zone_id": "b",
                "spatial_probability": 0.1,
                "actual": 1,
                "district": "D1",
                "region": "R1",
            },
        ]
    )

    summary = summarize_topk_near_misses(
        rows,
        k=1,
        neighbor_lookup={"a": {"b"}},
        ring2_lookup={"a": set()},
    )

    assert summary.iloc[0]["exact_hits"] == 0
    assert summary.iloc[0]["ring1_hits"] == 1
    assert summary.iloc[0]["district_hits"] == 1
    assert summary.iloc[0]["region_hits"] == 1
```

- [x] **Step 2: Run test verify RED**

Run:

```bash
.venv-ghost/bin/python -m pytest tests/test_spatial_ranking_diagnostics.py -q -p no:cacheprovider --basetemp .pytest_tmp_spatial_rank_diag
```

Expected: FAIL with `ModuleNotFoundError` or missing function.

- [x] **Step 3: Implement diagnostic script**

Create `analysis/diagnose_spatial_ranking.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import h3
import pandas as pd


def _score_column(frame: pd.DataFrame) -> str:
    if "spatial_probability" in frame:
        return "spatial_probability"
    if "probability" in frame:
        return "probability"
    if "score" in frame:
        return "score"
    raise ValueError("Prediction frame must include spatial_probability, probability, or score.")


def _label_column(frame: pd.DataFrame) -> str:
    if "actual" in frame:
        return "actual"
    if "target" in frame:
        return "target"
    raise ValueError("Prediction frame must include actual or target label column.")


def _h3_neighbors(zone_id: str, radius: int) -> set[str]:
    return set(h3.grid_disk(zone_id, radius)) - {zone_id}


def summarize_topk_near_misses(
    predictions: pd.DataFrame,
    k: int = 50,
    neighbor_lookup: dict[str, set[str]] | None = None,
    ring2_lookup: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    score_col = _score_column(predictions)
    label_col = _label_column(predictions)
    neighbor_lookup = neighbor_lookup or {}
    ring2_lookup = ring2_lookup or {}
    rows: list[dict[str, object]] = []

    for target_time, group in predictions.groupby("target_time", sort=True):
        positives = group[group[label_col].astype(int) == 1]
        positive_zones = set(positives["zone_id"].astype(str))
        positive_districts = set(positives.get("district", pd.Series(dtype=str)).astype(str))
        positive_regions = set(positives.get("region", pd.Series(dtype=str)).astype(str))
        top = group.sort_values(score_col, ascending=False).head(k)

        exact_hits = 0
        ring1_hits = 0
        ring2_hits = 0
        district_hits = 0
        region_hits = 0
        for row in top.itertuples(index=False):
            zone_id = str(getattr(row, "zone_id"))
            exact = zone_id in positive_zones
            ring1 = bool(neighbor_lookup.get(zone_id, _h3_neighbors(zone_id, 1)) & positive_zones)
            ring2 = bool(ring2_lookup.get(zone_id, _h3_neighbors(zone_id, 2)) & positive_zones)
            district = str(getattr(row, "district", "")) in positive_districts
            region = str(getattr(row, "region", "")) in positive_regions
            exact_hits += int(exact)
            ring1_hits += int(exact or ring1)
            ring2_hits += int(exact or ring1 or ring2)
            district_hits += int(district)
            region_hits += int(region)

        rows.append(
            {
                "target_time": target_time,
                "k": k,
                "positives": int(len(positives)),
                "exact_hits": exact_hits,
                "ring1_hits": ring1_hits,
                "ring2_hits": ring2_hits,
                "district_hits": district_hits,
                "region_hits": region_hits,
                "exact_precision": exact_hits / k if k else 0.0,
                "ring1_precision": ring1_hits / k if k else 0.0,
                "ring2_precision": ring2_hits / k if k else 0.0,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose exact and near-miss spatial top-k ranking.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--horizon", required=True)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    summary = summarize_topk_near_misses(predictions, k=args.k)
    output = Path(args.output or f"analysis/spatial_topk_near_miss_{args.horizon}_latest.csv")
    summary.to_csv(output, index=False)
    print(summary.drop(columns=["target_time"]).mean(numeric_only=True).to_string())
    print(output)


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run test verify GREEN**

Run:

```bash
.venv-ghost/bin/python -m pytest tests/test_spatial_ranking_diagnostics.py -q -p no:cacheprovider --basetemp .pytest_tmp_spatial_rank_diag
```

Expected: PASS.

- [x] **Step 5: Generate diagnostics for all horizons**

Run:

```bash
.venv-ghost/bin/python analysis/diagnose_spatial_ranking.py --predictions analysis/spatial_zone_predictions_30m_latest.csv --horizon 30m --k 50
.venv-ghost/bin/python analysis/diagnose_spatial_ranking.py --predictions analysis/spatial_zone_predictions_1h_latest.csv --horizon 1h --k 50
.venv-ghost/bin/python analysis/diagnose_spatial_ranking.py --predictions analysis/spatial_zone_predictions_2h_latest.csv --horizon 2h --k 50
```

Expected: each command exits `0` and writes `analysis/spatial_topk_near_miss_30m_latest.csv`, `analysis/spatial_topk_near_miss_1h_latest.csv`, and `analysis/spatial_topk_near_miss_2h_latest.csv`.

- [x] **Step 6: Commit**

```bash
git add analysis/diagnose_spatial_ranking.py tests/test_spatial_ranking_diagnostics.py
git commit -m "analysis: add spatial ranking near-miss diagnostics"
```

Expected: commit succeeds.

## Task 3: Add Near-Miss Metrics to Spatial Evaluation

**Files:**
- Modify: `ghost_ranking_metrics.py`
- Modify: `analysis/run_two_stage_experiment.py`
- Modify: `tests/test_two_stage_experiment.py`
- Create or Modify: `tests/test_ranking_metrics.py`

**Agent Role:** Architect

**Suggested Model Tier:** high-capability

**Why This Tier:** Metric design affects model interpretation and future optimization. The worker must avoid leakage and distinguish product-useful near misses from exact label hits.

**Inputs Needed:** Task 2 diagnostic outputs, existing `ghost_ranking_metrics.py`, and current two-stage metadata schema.

**Expected Output:** Spatial metadata includes exact and near-miss top-k metrics while preserving existing exact precision and AP metrics.

**Review Gate:** New metrics are mathematically clear, tested on small examples, and present in `analysis/spatial_model_metadata_30m.json`, `analysis/spatial_model_metadata_1h.json`, and `analysis/spatial_model_metadata_2h.json` after a run.

Escalate remains high-capability if diagnostics show near misses are frequent but exact hits are absent, because deciding the primary metric becomes a product/modeling judgment.

- [x] **Step 1: Write metric tests**

Add this to `tests/test_ranking_metrics.py`:

```python
import pandas as pd

from ghost_ranking_metrics import near_miss_hit_rate_at_k


def test_near_miss_hit_rate_at_k_counts_neighbor_matches_by_group():
    predictions = pd.DataFrame(
        [
            {"target_time": "t1", "zone_id": "a", "score": 0.9, "actual": 0},
            {"target_time": "t1", "zone_id": "b", "score": 0.1, "actual": 1},
            {"target_time": "t2", "zone_id": "c", "score": 0.8, "actual": 0},
            {"target_time": "t2", "zone_id": "d", "score": 0.7, "actual": 0},
        ]
    )

    result = near_miss_hit_rate_at_k(
        predictions,
        k=1,
        neighbor_lookup={"a": {"b"}, "c": set()},
        score_col="score",
        label_col="actual",
    )

    assert result == 0.5
```

- [x] **Step 2: Run test verify RED**

Run:

```bash
.venv-ghost/bin/python -m pytest tests/test_ranking_metrics.py::test_near_miss_hit_rate_at_k_counts_neighbor_matches_by_group -q -p no:cacheprovider --basetemp .pytest_tmp_spatial_metrics
```

Expected: FAIL because `near_miss_hit_rate_at_k` does not exist.

- [x] **Step 3: Implement metric helper**

Add to `ghost_ranking_metrics.py`:

```python
def near_miss_hit_rate_at_k(
    predictions,
    k: int,
    neighbor_lookup: dict[str, set[str]],
    score_col: str = "score",
    label_col: str = "actual",
    group_col: str = "target_time",
    zone_col: str = "zone_id",
) -> float:
    if predictions.empty:
        return 0.0
    hits = []
    for _, group in predictions.groupby(group_col, sort=False):
        positives = set(group.loc[group[label_col].astype(int) == 1, zone_col].astype(str))
        if not positives:
            continue
        top = group.sort_values(score_col, ascending=False).head(k)
        group_hit = any(
            str(row[zone_col]) in positives
            or bool(neighbor_lookup.get(str(row[zone_col]), set()) & positives)
            for _, row in top.iterrows()
        )
        hits.append(int(group_hit))
    return float(sum(hits) / len(hits)) if hits else 0.0
```

- [x] **Step 4: Wire metric into two-stage scoring**

In `analysis/run_two_stage_experiment.py`, extend `_score_spatial_predictions` call sites or post-process spatial predictions so metadata contains:

```python
"neighbor_hit_rate_at_20": ...,
"neighbor_hit_rate_at_50": ...,
"neighbor_hit_rate_at_100": ...,
```

Use the same holdout predictions already passed to exact precision metrics. Do not compute neighbors using future labels beyond the same validation/holdout frame.

- [x] **Step 5: Run metric and two-stage tests**

Run:

```bash
.venv-ghost/bin/python -m pytest tests/test_ranking_metrics.py tests/test_two_stage_experiment.py -q -p no:cacheprovider --basetemp .pytest_tmp_spatial_metrics
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add ghost_ranking_metrics.py analysis/run_two_stage_experiment.py tests/test_ranking_metrics.py tests/test_two_stage_experiment.py
git commit -m "feat: add spatial near-miss ranking metrics"
```

Expected: commit succeeds.

## Task 4: Run One Ranking Improvement Experiment

**Files:**
- Modify: `analysis/run_two_stage_experiment.py`
- Modify: `tests/test_two_stage_experiment.py`
- Read: `analysis/spatial_model_folds_30m_latest.csv`
- Read: `analysis/spatial_topk_near_miss_30m_latest.csv`

**Agent Role:** Builder

**Suggested Model Tier:** mid-tier

**Why This Tier:** This is a bounded modeling iteration using existing candidate models and sampling hooks. It should not alter the architecture unless diagnostics contradict the assumption.

**Inputs Needed:** Task 2 and Task 3 outputs, current spatial fold metrics, and `sample_spatial_training_rows` behavior.

**Expected Output:** One controlled change to spatial training or model selection, with before/after metrics in summary artifacts.

**Review Gate:** The full runner completes, exact metrics do not regress silently, and at least one of neighbor-hit@50, district-hit@50, or top-decile lift improves for the selected primary horizon.

Escalate to high-capability if the model improvement requires redefining the target, changing H3 resolution, or adding non-local data sources.

- [x] **Step 1: Choose primary horizon from diagnostics**

Run:

```bash
.venv-ghost/bin/python - <<'PY'
import pandas as pd
summary = pd.read_csv("analysis/two_stage_summary_latest.csv")
print(summary[["horizon", "spatial_average_precision", "spatial_top_decile_lift", "spatial_precision_at_50"]].to_string(index=False))
PY
```

Expected: `30m` is the default primary horizon because it has the best top-decile lift and spatial AP in the current baseline.

- [x] **Step 2: Write failing test for model selection tie-breaker**

In `tests/test_two_stage_experiment.py`, add:

```python
from analysis.run_two_stage_experiment import _select_model


def test_select_spatial_model_prefers_top_k_before_decile_lift():
    summary = pd.DataFrame(
        [
            {"model": "lift_only", "median_precision_at_50": 0.0, "median_top_decile_lift": 5.0, "median_average_precision": 0.01},
            {"model": "top_k", "median_precision_at_50": 0.02, "median_top_decile_lift": 2.0, "median_average_precision": 0.01},
        ]
    )

    assert _select_model(summary, "spatial")["model"] == "top_k"
```

- [x] **Step 3: Run test verify RED**

Run:

```bash
.venv-ghost/bin/python -m pytest tests/test_two_stage_experiment.py::test_select_spatial_model_prefers_top_k_before_decile_lift -q -p no:cacheprovider --basetemp .pytest_tmp_spatial_selection
```

Expected: FAIL if current spatial selection prioritizes top-decile lift before precision@50.

- [x] **Step 4: Update spatial model selection order**

In `analysis/run_two_stage_experiment.py`, update `_select_model(summary, stage="spatial")` sorting to prefer top-k usefulness:

```python
ranked = summary.sort_values(
    by=["median_precision_at_50", "median_precision_at_100", "median_average_precision", "median_top_decile_lift"],
    ascending=[False, False, False, False],
)
```

If Task 3 added `median_neighbor_hit_rate_at_50`, include it before exact precision only if the product accepts adjacent-zone hits:

```python
by=["median_neighbor_hit_rate_at_50", "median_precision_at_50", "median_average_precision", "median_top_decile_lift"]
```

- [x] **Step 5: Run focused tests**

Run:

```bash
.venv-ghost/bin/python -m pytest tests/test_two_stage_experiment.py tests/test_engineered_ranking_features.py -q -p no:cacheprovider --basetemp .pytest_tmp_spatial_selection
```

Expected: PASS.

- [x] **Step 6: Run full experiment**

Run:

```bash
.venv-ghost/bin/python analysis/run_two_stage_experiment.py
```

Expected: command completes and refreshes `analysis/two_stage_summary_latest.csv`, `analysis/spatial_model_metadata_30m.json`, `analysis/spatial_model_metadata_1h.json`, and `analysis/spatial_model_metadata_2h.json`.

- [x] **Step 7: Compare before/after metrics**

Run:

```bash
.venv-ghost/bin/python - <<'PY'
import pandas as pd
df = pd.read_csv("analysis/two_stage_summary_latest.csv")
print(df[["horizon", "spatial_model", "spatial_precision_at_20", "spatial_precision_at_50", "spatial_average_precision", "spatial_top_decile_lift"]].to_string(index=False))
PY
```

Expected: output is pasted into the task notes. If exact precision remains zero, keep the change only if Task 3 near-miss metrics or AP/lift improve and the product accepts that tradeoff.

Result after full runner: selected spatial model remains `lightgbm_conservative`; exact holdout precision@20 and precision@50 remain `0.0` for all horizons. New near-miss instrumentation shows operationally useful adjacent-zone signal instead of exact-zone hits: 30m neighbor-hit@50 `0.043860`, 1h neighbor-hit@50 `0.026549`, 2h neighbor-hit@50 `0.017857`. This supports keeping the diagnostic/metric change but shows exact top-k ranking is still unresolved without a broader target/H3-resolution/modeling change.

- [x] **Step 8: Commit**

```bash
git add analysis/run_two_stage_experiment.py tests/test_two_stage_experiment.py analysis/two_stage_summary_latest.csv analysis/spatial_model_metadata_30m.json analysis/spatial_model_metadata_1h.json analysis/spatial_model_metadata_2h.json
git commit -m "exp: prioritize spatial top-k model selection"
```

Expected: commit succeeds if generated artifacts are intended to be versioned. If artifacts are not versioned, commit only source and tests.

## Task 5: Final Validation and Project Checkpoint

**Files:**
- Read: `analysis/two_stage_summary_latest.csv`
- Read: `analysis/spatial_topk_near_miss_30m_latest.csv`
- Read: `analysis/spatial_topk_near_miss_1h_latest.csv`
- Read: `analysis/spatial_topk_near_miss_2h_latest.csv`
- Update: Notion page `https://app.notion.com/p/3906e655eec381e0b77cc1daf194a227`

**Agent Role:** Reviewer

**Suggested Model Tier:** low-cost

**Why This Tier:** This is command execution, artifact reading, and concise checkpoint writing.

**Inputs Needed:** Final generated artifacts and this task section.

**Expected Output:** Verification evidence and a Notion checkpoint with metrics, interpretation, and the next recommendation.

**Review Gate:** Main agent confirms commands passed, checkpoint was written, and residual risks are explicit.

- [x] **Step 1: Run full focused suite**

Run:

```bash
.venv-ghost/bin/python -m pytest tests/test_spatial_sampling.py tests/test_model_iteration.py tests/test_multi_horizon_iteration.py tests/test_two_stage_experiment.py tests/test_engineered_ranking_features.py tests/test_spatial_ranking_diagnostics.py tests/test_ranking_metrics.py -q -p no:cacheprovider --basetemp .pytest_tmp_spatial_final
```

Expected: PASS. If `tests/test_ranking_metrics.py` did not exist before Task 3, include it only after Task 3 creates it.

- [x] **Step 2: Run full two-stage experiment**

Run:

```bash
.venv-ghost/bin/python analysis/run_two_stage_experiment.py
```

Expected: command completes in under 5 minutes on the rebuilt local database.

- [x] **Step 3: Regenerate near-miss diagnostics**

Run:

```bash
.venv-ghost/bin/python analysis/diagnose_spatial_ranking.py --predictions analysis/spatial_zone_predictions_30m_latest.csv --horizon 30m --k 50
.venv-ghost/bin/python analysis/diagnose_spatial_ranking.py --predictions analysis/spatial_zone_predictions_1h_latest.csv --horizon 1h --k 50
.venv-ghost/bin/python analysis/diagnose_spatial_ranking.py --predictions analysis/spatial_zone_predictions_2h_latest.csv --horizon 2h --k 50
```

Expected: all commands exit `0`.

- [x] **Step 4: Write Notion checkpoint**

Append a concise checkpoint to `Ghost Sweep Project Memory` with:

```markdown
# Checkpoint: spatial ranking diagnostic iteration

Date: 2026-07-01

Summary: Stage 2 exact top-k performance was analyzed with exact, adjacent-cell, district, and region hit diagnostics.

Verification:
- Focused tests: `.venv-ghost/bin/python -m pytest ...` passed with the final pass count from Step 1.
- Full two-stage run: completed
- Diagnostics generated: `analysis/spatial_topk_near_miss_30m_latest.csv`, `analysis/spatial_topk_near_miss_1h_latest.csv`, `analysis/spatial_topk_near_miss_2h_latest.csv`

Metrics:
- Read `analysis/two_stage_summary_latest.csv` and the three `analysis/spatial_topk_near_miss_*_latest.csv` files. Write one bullet each for 30m, 1h, and 2h containing exact precision@50, neighbor hit@50, spatial AP, and top-decile lift as decimal values.

Decision:
- Name exactly one primary next metric: exact precision@50, neighbor hit@50, or district hit@50. Follow it with one sentence explaining the choice from the diagnostics.

Next recommendation:
- Write one sentence naming the next source file, test file, and model change to attempt.
```

- [x] **Step 5: Final git status**

Run:

```bash
git status --short --branch
```

Expected: only intentional source/test/artifact changes remain.

## Self-Review

- **Spec coverage:** The plan covers the current data science concern: zero spatial precision@20/@50 despite nonzero 30m top-decile lift. It adds diagnostics before changing the model, then adds metrics and one controlled model-selection experiment.
- **Placeholder scan:** The plan contains no vague implementation placeholders. Every task has exact files, commands, expected outputs, and concrete snippets where code behavior changes.
- **Type consistency:** Function names are consistent across tests and implementation snippets: `summarize_topk_near_misses`, `near_miss_hit_rate_at_k`, and `_select_model`.
- **Dependency ordering:** Task 1 is read-only context; Task 2 creates diagnostics; Task 3 adds metrics; Task 4 uses those outputs for one modeling change; Task 5 validates and checkpoints.
- **Model-tier fit:** Low-cost work is bounded to artifact inspection and verification. Mid-tier work handles focused scripts/tests. High-capability work is reserved for metric design because it changes model interpretation.
- **Agent independence:** Each task lists enough file paths, commands, and expected outputs for a fresh worker to execute without hidden context.
- **Cost fit:** No task gives a worker the full repo unless not needed; each task limits file scope and escalation triggers.
- **Parallel safety:** Task 1 can run independently. Task 2 and Task 3 should not run in parallel because Task 3 can use Task 2 diagnostics. Task 4 depends on Tasks 2 and 3. Task 5 is final-only.
- **Review gates:** Every task has a concrete pass/fail checkpoint before the next task starts.

Plan complete and saved to `docs/superpowers/plans/2026-07-01-spatial-ranking-diagnostics-and-improvement.md`.

Recommended execution: Subagent-Driven. I dispatch focused agents by task, use cheaper tiers for bounded read/review/mechanical work, reserve high-capability reasoning for architecture and integration review, and checkpoint after each diff.

Alternative execution: Inline Execution. I execute the plan in this session with the same checkpoints, but with less parallelism.
