# 2026-07-11 H3 Overlay And Session Handoff

## Current objective

Hand this repo to another coding agent with the latest accepted work pushed on `main`, plus enough context to continue the unfinished H3 overlay fix without re-discovering the root cause.

## What was completed this session

### 1. Analysis HTML outputs were grouped under `analysis/html/`

This was completed, committed, and merged on `main`.

Key changes:

- `analysis/h3_scale_overlay.html` moved to `analysis/html/h3_scale_overlay.html`
- generators now write HTML reports into `analysis/html/` instead of cluttering `analysis/`
- artifact manifest references were updated to `analysis/html/...`
- focused selectors still passed after the move

Primary files changed for that work:

- `analysis/build_dashboard_manifest.py`
- `analysis/build_spotfire_dashboard.py`
- `analysis/make_h3_scale_overlay.py`
- `analysis/make_zone_map.py`
- `analysis/make_zone_model_visuals.py`
- `analysis/run_ml_experiment.py`
- `analysis/run_model_iteration.py`
- `analysis/run_multi_horizon_experiment.py`
- `analysis/run_resolution_comparison.py`
- `analysis/run_two_stage_experiment.py`
- `analysis/run_zone_ranking_experiment.py`
- `analysis/simulate_real_location_risk.py`
- `tests/test_ci.py`
- `WORKLOG.md`
- `WORKLOG_INDEX.md`

### 2. Visual QA was run on the H3 overlay page

The user reported that the H3 overlap page had no Hong Kong map reference. That report is correct.

Evidence gathered:

- browser snapshot showed the SVG contains only:
  - one background `rect`
  - three `text` labels
  - three `<g>` grid layers (`res-7`, `res-8`, `res-9`)
- no coastline path
- no district outline
- no land mask
- no basemap reference layer

Relevant screenshots created locally during investigation:

- `h3-overlay-initial.png`
- `h3-overlay-res8-before-toggle.png`
- `h3-overlay-res7-after-toggle.png`

These screenshots were for debugging evidence only and should not be relied on as repo artifacts.

## Root cause found

This is not only a styling issue.

The deeper problem is that the overlay data itself is falling back to a plain bounding-box coverage grid instead of a Hong Kong-shaped coverage surface.

### Root cause chain

1. `analysis/make_h3_scale_overlay.py` loads coverage polygons from:
   - `analysis/geo/hk_h3_coverage_res7.geojson`
   - `analysis/geo/hk_h3_coverage_res8.geojson`
   - `analysis/geo/hk_h3_coverage_res9.geojson`

2. Those files are produced by `analysis/build_hk_coverage_grid.py`.

3. `analysis/build_hk_coverage_grid.py` tries to load official HK district polygons from:
   - `analysis/geo/hksar_18_district_boundary.json`

4. That file is currently missing in this repo checkout.

5. Because the boundary file is missing, `load_hk_boundary_geometries()` returns an empty list.

6. Then `sampled_hk_h3_zones()` falls back to filling the raw `HK_BOUNDS` rectangle with H3 cells:

```python
    polygon = h3.LatLngPoly(
        [
            (HK_BOUNDS["lat_min"], HK_BOUNDS["lng_min"]),
            (HK_BOUNDS["lat_min"], HK_BOUNDS["lng_max"]),
            (HK_BOUNDS["lat_max"], HK_BOUNDS["lng_max"]),
            (HK_BOUNDS["lat_max"], HK_BOUNDS["lng_min"]),
        ]
    )
    return set(h3.polygon_to_cells(polygon, resolution))
```

7. Result: the overlay is visually a neat rectangle of hexes, not a Hong Kong-shaped coverage surface.

### Important implication

Even if you add a coastline or district outline layer to the page, the current overlay polygons are still wrong-shaped because the input coverage artifacts were generated from the rectangular fallback.

So the fix should happen in this order:

1. restore a real HK boundary source file
2. regenerate the coverage GeoJSONs
3. then improve the overlay page to include a visible Hong Kong reference layer

## Files to read first

Read these in this order:

1. `analysis/make_h3_scale_overlay.py`
2. `analysis/build_hk_coverage_grid.py`
3. `ghost_districts.py`
4. `tests/test_hk_coverage_grid.py`
5. `analysis/html/h3_scale_overlay.html`
6. `WORKLOG.md` entry `2026-07-10 H3 scale overlay`
7. `WORKLOG.md` entry `2026-07-10 Analysis HTML parent folder`

## Commands already run during this investigation

### HTML reorganization verification

```powershell
rtk python analysis\build_dashboard_manifest.py
rtk python analysis\make_h3_scale_overlay.py
rtk pytest tests\test_ci.py -q -k dashboard_manifest_tracks_model_artifacts -p no:cacheprovider --basetemp .pytest_tmp_manifest_html
rtk pytest tests\test_h3_scale_overlay.py tests\test_spotfire_dashboard.py -q -p no:cacheprovider --basetemp .pytest_tmp_html_reports
```

Results:

- `tests/test_ci.py -k dashboard_manifest_tracks_model_artifacts` -> `1 passed`
- `tests/test_h3_scale_overlay.py tests/test_spotfire_dashboard.py` -> `2 passed`

### Visual QA investigation

Local browser serving was needed because `file://` access was blocked.

```powershell
rtk python -m http.server 8766 --bind 127.0.0.1
```

The overlay was then opened in the browser at:

```text
http://127.0.0.1:8766/analysis/html/h3_scale_overlay.html
```

### Root-cause checks

```powershell
Test-Path analysis\geo\hksar_18_district_boundary.json
Get-Item analysis\geo\hk_h3_coverage_res8.geojson
```

Observed:

- `analysis/geo/hksar_18_district_boundary.json` -> missing
- `analysis/geo/hk_h3_coverage_res8.geojson` -> present

## Current repo state

Branch:

```text
main
```

There were unrelated pre-existing untracked files in the repo during this session, including:

- `.github/copilot-instructions.md`
- `.playwright-mcp/`
- several `docs/superpowers/plans/...` files
- some local PNGs

Do not assume those were created by this task.

There is also an untracked local artifact:

- `analysis/html/ghost_map.html`

It came from moving the older top-level `analysis/ghost_map.html` into the new HTML folder locally. It was not needed for the committed repo change and was not relied on for validation.

## What is still unfinished

The user’s live concern is still open:

> the h3 grid overlay does not have an HK map for reference

That issue is only diagnosed, not fixed.

## Recommended next steps

### Minimum correct path

1. Restore the missing Hong Kong district boundary file expected by:
   - `analysis/build_hk_coverage_grid.py`
   - `ghost_districts.py`

2. Regenerate:
   - `analysis/geo/hk_h3_coverage_res7.geojson`
   - `analysis/geo/hk_h3_coverage_res8.geojson`
   - `analysis/geo/hk_h3_coverage_res9.geojson`

3. Confirm the regenerated polygons are Hong Kong-shaped instead of rectangular.

4. Update `analysis/make_h3_scale_overlay.py` to add a simple HK reference layer. Keep it lazy:
   - district outline or coastline stroke is enough
   - do not add a tile server or JS map dependency unless the user explicitly asks

5. Regenerate `analysis/html/h3_scale_overlay.html`.

6. Re-run:

```powershell
rtk pytest tests\test_h3_scale_overlay.py tests\test_hk_coverage_grid.py -q -p no:cacheprovider --basetemp .pytest_tmp_h3_overlay_fix
```

7. Re-run visual QA against the regenerated page.

### Best lazy implementation shape

Do not jump straight to a live map library.

The likely smallest good fix is:

- use the official HK boundary GeoJSON already expected by the repo
- project those polygons into the same SVG space as the H3 cells
- render a light district/territory outline behind the selected grid layer

That preserves the current static self-contained HTML design while giving real geographic reference.

## Suggested resume prompt

```text
Continue the H3 overlay fix in Ghost Sweep. Read docs/transfer/2026-07-11-h3-overlay-and-session-handoff.md first, then inspect analysis/make_h3_scale_overlay.py and analysis/build_hk_coverage_grid.py. Root cause already found: analysis/geo/hksar_18_district_boundary.json is missing, so coverage falls back to a rectangular HK_BOUNDS grid. Restore the boundary-backed coverage, regenerate the res7-res9 GeoJSONs, then add a simple HK outline layer to the static overlay page and verify with tests plus visual QA.
```
