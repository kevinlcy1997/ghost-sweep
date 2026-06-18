# Ghost Sweep — Weather Effects on Enforcement Activity

**Date:** 2026-06-18
**Status:** Draft
**Scope:** Phase 1 — forward weather sampling stored in SQLite and rendered as a district timeline overlay in the dashboard

---

## 1. Problem Statement

Ghost Sweep currently shows where and when community-reported traffic warden sightings happen, but it cannot show whether weather conditions line up with those changes in activity. The goal of this feature is to let a user pick a police district and inspect a time series of consolidated event counts against weather signals such as temperature, rainfall, typhoon signal, and rainstorm signal.

This is an exploratory dashboard feature first. It should also leave behind persisted weather samples that later model work can reuse.

---

## 2. Approved Scope

The following decisions are locked for v1:

- **Product scope:** dashboard + stored weather dataset + model-ready data shape
- **Sampling cadence:** weather refresh stays synced to the existing approximately 2-hour Ghost Sweep refresh
- **Spatial matching:** nearest weather station / mapped rainfall district per police district
- **Enforcement series:** consolidated event count, not raw alert count
- **Dashboard view:** one police district at a time
- **Output:** timeline overlay only, no correlation or lag analytics in v1

Important consequence: this is **not** a true hourly weather history feature in v1. It is a district timeline based on weather samples captured on the existing collection cadence.

---

## 3. Objectives and Non-Goals

### 3.1 Objectives

1. Capture weather samples during the normal Ghost Sweep refresh cycle.
2. Persist one district-level weather row per district per weather sample time.
3. Show a selectable district weather/enforcement overlay in the dashboard.
4. Keep the stored data shape reusable for future model features.

### 3.2 Non-Goals

- No historical weather backfill in v1.
- No all-district comparison view.
- No direct browser-to-HKO calls.
- No separate weather scheduler or workflow.
- No regression, correlation score, or derived “weather effect” metric.

---

## 4. Data Source

### 4.1 Primary Source: Hong Kong Observatory Open Data

Use HKO open data endpoints already confirmed reachable.

**Confirmed live endpoints:**

- `rhrread` — current weather report payload containing station temperatures, humidity, and district-level rainfall observations
- `warnsum` — active warning summary containing warning states and timestamps

### 4.2 Variables in Scope

Required in v1:

- air temperature
- rainfall amount
- typhoon signal state
- rainstorm signal state

Nice-to-have if already present in the same feed and cheap to store:

- relative humidity

### 4.3 Historical Data Constraint

V1 collects weather **forward from rollout**. It does not backfill older history.

Reason:

- the user chose sync with the existing 2-hour refresh cadence
- HKO historical hourly availability differs by metric
- backfill adds substantial complexity before the dashboard value is proven

If the overlay proves useful, a one-off historical import can be added later without changing the table shape.

---

## 5. Architecture Overview

Weather should be added as one small helper integrated into the existing refresh path.

```text
ghost_listener.py refresh
    -> existing alert collection
    -> current weather fetch
    -> store district weather samples in SQLite
generate_dashboard.py
    -> read consolidated alerts + stored weather samples
    -> render one-district weather overlay
```

The dashboard reads only local data. It never calls HKO directly.

---

## 6. Proposed Components

| File | Responsibility |
|------|----------------|
| `ghost_weather.py` | Fetch HKO weather/warning payloads and turn them into district weather samples |
| `ghost_db.py` | Add weather sample table and insert/query methods |
| `ghost_districts.py` | Extend district metadata with weather station / rainfall district mapping |
| `ghost_listener.py` | Store one weather sample batch during each refresh cycle |
| `generate_dashboard.py` | Add Weather Effects section and query/render stored samples |
| `tests/test_ghost_weather.py` | Weather fetch/normalize/mapping tests |
| `tests/test_dashboard.py` | Dashboard weather section tests |

This is intentionally smaller than the earlier design: one new Python module, existing DB layer extended, existing district metadata extended, existing dashboard extended.

---

## 7. Storage Model

### 7.1 Weather Samples Table

Add a single table to SQLite:

**Table: `weather_samples`**

One row per police district per collection timestamp.

Suggested fields:

- `district`
- `observed_at`
- `temperature_c`
- `rainfall_mm`
- `humidity_pct`
- `typhoon_signal_code`
- `rainstorm_signal_code`
- `has_typhoon_signal`
- `has_rainstorm_signal`

Suggested primary key:

- `(district, observed_at)`

This table replaces the multi-table raw + derived design from the first draft. For v1, one stored district-level sample table is enough.

### 7.2 Mapping Strategy

Do not create a standalone mapping file in v1.

Instead, extend the existing `DISTRICT_STATIONS` structure in `ghost_districts.py` with optional metadata such as:

- `weather_station`
- `rainfall_district`

That keeps all district-related lookup data in one place.

---

## 8. Time-Series Join Logic

### 8.1 Enforcement Series

Use **consolidated event counts**.

For each stored weather sample timestamp `T`, compute the selected district's event count over the matching recent window:

- recommended v1 window: events in the preceding 2 hours ending at `T`

This keeps the event series aligned with the actual weather sample cadence instead of pretending the system has continuous hourly weather.

### 8.2 Weather Series

For each refresh:

- map station variables like temperature via the configured district weather station
- map rainfall via the configured district rainfall region
- write active signal codes/flags directly onto each district sample row

No separate warning interval table is needed in v1.

### 8.3 Missing Data Rule

If a weather field is unavailable during a sample:

- store null for that field
- keep the row
- show a gap for that metric in the dashboard

Do not infer or fill missing weather values in v1.

---

## 9. Ingestion and Refresh Flow

Weather collection runs inside the existing Ghost Sweep refresh.

Per refresh:

1. collect / update alert data as today
2. fetch current HKO weather and warning payloads
3. convert them into district-level weather sample rows
4. upsert those rows into `weather_samples`
5. generate dashboard output from alerts + stored weather samples

This keeps operations boring and avoids new schedulers, jobs, or workflows.

---

## 10. Dashboard UX

### 10.1 New Section

Add a section named **Weather Effects**.

### 10.2 Controls

The section includes:

- police district selector
- weather metric selector
- simple date-range selector or preset window selector

Metrics in v1:

- temperature
- rainfall
- typhoon signal
- rainstorm signal

### 10.3 Main Visualization

One district at a time.

- x-axis: weather sample timestamps
- primary series: consolidated event count for the matching 2-hour window
- secondary series: selected weather metric

For signal-style metrics:

- use bands, markers, or step state, not a fake continuous curve

### 10.4 Out of Scope for v1

- side-by-side district compare
- all-district matrix view
- computed correlation tables
- historical backfill UI

---

## 11. Failure Handling

- If HKO is unavailable, alert collection still succeeds.
- Weather fetch failures should not block JSON/DB alert updates.
- If one weather field is missing, store the rest of the row.
- If a district lacks mapping metadata, skip its weather row and log it.

---

## 12. Testing Strategy

### 12.1 Unit Tests

- weather payload normalization from `rhrread` and `warnsum`
- district mapping from weather payloads to stored district rows
- DB insert/upsert behavior for `weather_samples`

### 12.2 Integration Tests

- weather sample rows can be created from a mock HKO payload set
- dashboard can read stored weather samples and produce the Weather Effects section

### 12.3 Dashboard Tests

- Weather Effects section exists
- district selector exists
- weather metric selector exists
- serialized weather data is present in generated HTML

---

## 13. Risks and Constraints

1. **Cadence limitation**
   The chosen 2-hour sync means the overlay is sampled, not fully hourly.

2. **Mapping quality**
   Police districts do not perfectly match HKO station and rainfall regions.

3. **Dashboard size**
   `generate_dashboard.py` is already large, so the weather section should be added with minimal extra structure and only split further if it becomes unreadable.

---

## 14. Recommended Implementation Shape

Recommended order:

1. Extend `ghost_db.py` with `weather_samples` storage.
2. Add district weather mapping data to `ghost_districts.py`.
3. Add `ghost_weather.py` to fetch and flatten HKO payloads into district rows.
4. Call weather sync from `ghost_listener.py` during normal refresh.
5. Add Weather Effects controls and chart to `generate_dashboard.py`.

This is the shortest path that delivers user-visible value, stores reusable data, and avoids building raw weather pipelines we do not yet need.
