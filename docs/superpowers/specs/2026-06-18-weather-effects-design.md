# Ghost Sweep — Weather Effects on Enforcement Activity

**Date:** 2026-06-18
**Status:** Draft
**Scope:** Phase 1 — persisted weather ingestion, district-mapped weather facts, and dashboard timeline overlay

---

## 1. Problem Statement

Ghost Sweep currently tracks community-reported traffic warden activity by time and place, but it has no way to explain whether weather conditions influence that activity. The goal of this feature is to add a weather analysis layer that lets a user select a police district and inspect how enforcement activity changes alongside weather conditions such as temperature, rainfall, typhoon signals, and rainstorm signals over time.

This is an exploratory analytics feature first, not a prediction feature. The output for v1 is a selectable district timeline overlay in the dashboard, backed by persisted weather data that can later be reused for model features.

---

## 2. Approved Scope

The following decisions were made during brainstorming and are locked for v1:

- **Product scope:** dashboard + stored weather dataset + model-ready data shape
- **Time granularity target:** hourly-capable analysis structure
- **Spatial matching:** nearest station by police district
- **Enforcement series:** consolidated event count, not raw alert count
- **Backfill target:** match Ghost Sweep history only
- **Dashboard view:** one police district at a time, selected by the user
- **Weather analysis output:** timeline overlay only, no summary scores or correlation analytics
- **Weather refresh cadence:** sync weather collection to the existing approximately 2-hour Ghost Sweep refresh cadence

Important implication: the storage and joins are designed at hourly resolution, but **v1 weather observations will be sampled at the collection cadence** unless a clean HKO historical or interval feed is available for a specific metric.

---

## 3. Objectives and Non-Goals

### 3.1 Objectives

1. Persist weather observations in SQLite alongside existing Ghost Sweep data.
2. Map weather measurements to police districts through explicit station and rainfall mappings.
3. Provide a dashboard section where the user selects a district, date range, and weather metric and sees an enforcement-vs-weather timeline.
4. Create a reusable weather fact layer for future feature engineering and modeling.

### 3.2 Non-Goals

- No all-district weather matrix or multi-district compare view in v1.
- No correlation coefficients, lag analysis, regression output, or “weather effect score” in v1.
- No direct browser-to-HKO API calls.
- No change to the core alert scraping logic beyond orchestrating a weather sync in the same refresh run.

---

## 4. Data Sources

### 4.1 Primary Source: Hong Kong Observatory Open Data

The feature will use HKO open data endpoints and related HKO/data.gov.hk datasets where available.

**Confirmed usable live endpoints:**

- `rhrread` — current weather report style payload containing station temperatures, humidity, and district-level rainfall observations
- `warnsum` — active warning summary containing warning types, issue times, update times, and active warning state

### 4.2 Weather Variables in Scope

Required for v1 dashboard support:

- air temperature
- rainfall amount
- typhoon signal state
- rainstorm signal state

Eligible for storage if available in the same payload with low additional cost:

- relative humidity
- other warning flags carried in the same feed

### 4.3 Historical Data Constraint

HKO live endpoints are sufficient for forward collection, but not every requested metric is guaranteed to have a clean, queryable historical hourly feed through the same API surface. Therefore:

- the **target** backfill window is “match Ghost Sweep history”
- actual historical completeness depends on what HKO exposes cleanly per metric
- where a metric cannot be backfilled reliably, its values will start from rollout onward and older periods will remain missing

The system must preserve that missingness explicitly rather than infer values.

---

## 5. Architecture Overview

Weather will be added as a separate subsystem parallel to the alert/event pipeline.

```text
ghost_listener.py / refresh workflow
    -> existing alert/event pipeline
    -> weather sync step
         -> HKO sources
         -> SQLite weather tables
         -> district weather fact builder
              -> dashboard serialization
              -> future model feature reuse
```

The dashboard will not call HKO directly. It will only consume derived facts already stored locally.

---

## 6. Proposed Components

| File | Responsibility |
|------|----------------|
| `ghost_weather_sources.py` | Fetch and normalize HKO weather and warning payloads |
| `ghost_weather_db.py` | Create and maintain weather tables; upsert normalized records |
| `ghost_weather_mapping.py` | Police district to station/rainfall mapping |
| `ghost_weather_features.py` | Build district-time weather facts and join against consolidated event counts |
| `generate_dashboard.py` | Render the Weather Effects section from persisted derived facts |
| `tests/test_ghost_weather_sources.py` | Weather source normalization tests |
| `tests/test_ghost_weather_mapping.py` | Mapping tests |
| `tests/test_ghost_weather_features.py` | Join and fact-building tests |
| `tests/test_dashboard.py` | Dashboard serialization and control presence tests |

This keeps weather isolated from both the scraper internals and the model code while still making it reusable.

---

## 7. Storage Model

### 7.1 Raw/Normalized Weather Tables

**Table: `weather_station_observations`**

One row per observation timestamp per source station for station-style values.

Suggested fields:

- `observed_at`
- `station_name`
- `temperature_c`
- `humidity_pct`
- `source_payload_time`

**Table: `weather_district_rainfall`**

One row per observation timestamp per HKO rainfall district.

Suggested fields:

- `observed_at`
- `hko_district`
- `rainfall_mm`
- `source_payload_time`

**Table: `weather_warning_intervals`**

One row per warning interval.

Suggested fields:

- `warning_type`
- `signal_code`
- `severity`
- `start_time`
- `end_time`
- `source_update_time`

### 7.2 Mapping Layer

**Static mapping: police district -> weather source mapping**

This maps each Ghost Sweep police district to:

- nearest HKO station for station variables like temperature
- matching HKO district for rainfall

The mapping must live in one explicit place so the assumptions are reviewable and testable.

### 7.3 Derived Fact Table

**Table: `district_weather_facts`**

This is the canonical join layer consumed by the dashboard and future model features.

Suggested fields:

- `district`
- `bucket_start`
- `temperature_c`
- `rainfall_mm`
- `humidity_pct`
- `typhoon_signal_code`
- `rainstorm_signal_code`
- `has_typhoon_signal`
- `has_rainstorm_signal`
- `event_count`

`bucket_start` is the time bucket key used for joins and display. The design targets hourly-capable data, but v1 facts may be sparse because weather collection is synchronized to the approximately 2-hour refresh cadence.

---

## 8. Time-Series and Join Logic

### 8.1 Enforcement Side

The enforcement series is the **consolidated event count** aggregated by:

- police district
- time bucket

This uses the already-cleaned event layer rather than raw alerts.

### 8.2 Weather Side

For each observation timestamp:

- station variables are mapped into district facts through the nearest-station mapping
- rainfall is mapped through the chosen rainfall district mapping
- territory-wide warnings are expanded into per-bucket active flags for every district

### 8.3 Bucket Resolution

The design remains compatible with hourly analysis, but v1 refresh cadence is about 2 hours. Therefore:

- the fact table uses time buckets suitable for hourly dashboards and future feature engineering
- weather values are populated at observed collection timestamps
- missing buckets remain missing/null unless a clean historical or interval data source is available

This avoids inventing hourly weather where the ingestion schedule did not actually capture it.

---

## 9. Ingestion and Refresh Flow

### 9.1 Backfill

Backfill starts from the first Ghost Sweep timestamp and attempts to populate weather history for the same period.

Rules:

- use clean historical HKO sources where available
- do not synthesize missing historical data
- preserve nulls where history is unavailable or partial

### 9.2 Ongoing Refresh

Weather sync runs in the same overall refresh cycle as the Ghost Sweep data collection.

Flow per refresh:

1. collect / update alert data
2. fetch current HKO weather and warning payloads
3. normalize and upsert into weather tables
4. rebuild or incrementally update district weather facts for the new time bucket(s)
5. generate dashboard output

This keeps the operational model simple and aligned with the current repo architecture.

### 9.3 Refresh Trade-Off

Because the chosen refresh cadence is about 2 hours:

- v1 weather facts are good for directional overlay analysis
- they are not guaranteed to be a fully continuous hourly meteorological record
- future work can add denser weather collection without redesigning the storage model

---

## 10. Dashboard UX

### 10.1 New Section

Add a new dashboard section named **Weather Effects**.

### 10.2 Controls

The section includes:

- police district selector
- date range selector
- weather metric selector

Metrics supported in v1:

- temperature
- rainfall
- typhoon signal
- rainstorm signal

### 10.3 Main Visualization

One selected district at a time.

- x-axis: time
- primary series: consolidated event count
- secondary series: selected weather metric

For signal-style weather metrics:

- use shaded regions, step bands, or an equivalent binary-state visual
- do not force warning signals into misleading continuous numeric curves

### 10.4 Behavior Rules

- switching district changes both the enforcement series and the mapped weather source
- switching metric changes only the weather overlay
- missing historical weather must display as gaps rather than filled-in values
- the dashboard must never try to display all districts at once in this weather view

### 10.5 Out of Scope for v1

- side-by-side district comparison
- all-district weather heatmaps
- correlation tables
- lag-analysis overlays
- model prediction overlays in the same section

---

## 11. Failure Handling

- If HKO is unavailable during a refresh, the alert/event pipeline still succeeds.
- Weather ingestion failures must be isolated and logged separately.
- Missing values for one metric must not block storage of other metrics from the same refresh.
- Mapping gaps must be explicit and testable. If a police district lacks a valid mapping, it is excluded from weather analysis until mapped.
- Warning intervals must be stored raw enough to allow deterministic re-expansion into fact rows.

---

## 12. Testing Strategy

### 12.1 Unit Tests

- HKO payload normalization
- station and rainfall mapping correctness
- warning interval expansion into bucketed flags
- event count and weather join logic

### 12.2 Integration Tests

- sample weather payloads + sample consolidated events -> expected district weather facts
- partial weather history -> null values preserved without crashes

### 12.3 Dashboard Tests

- Weather Effects section is present
- district selector is present
- metric selector is present
- serialized data includes weather fact records for the selected district flow

### 12.4 Backfill Smoke Test

- backfill from first Ghost Sweep timestamp to current date
- no duplicate fact rows
- expected district coverage where mappings exist

---

## 13. Risks and Constraints

1. **Historical completeness risk**
   Some requested metrics may not have clean historical hourly coverage through HKO open data.

2. **Cadence mismatch risk**
   The chosen 2-hour sync means v1 weather overlays are analytically useful but not meteorologically complete at hourly resolution.

3. **Mapping risk**
   Police districts and HKO weather/rainfall districts are not identical geographic systems. Mapping quality needs explicit review.

4. **Dashboard complexity risk**
   `generate_dashboard.py` is already large; weather rendering should be added carefully or later split into clearer helper functions during implementation.

---

## 14. Recommended Implementation Shape

Recommended implementation order:

1. Add weather source normalization and DB schema.
2. Add explicit district-to-weather mappings.
3. Add fact-table builder joined to consolidated event counts.
4. Add dashboard controls and overlay chart.
5. Add historical backfill command for the supported metrics.

This order validates data quality before investing in the UI and keeps the dashboard change grounded in stored facts instead of ad hoc API calls.
