# HK 200m-Class Grid Resolution Plan

I'm using the cost-aware planning skill. This plan changes Ghost Sweep from the current res8 city/block-ish grid to a res9 200m-class Hong Kong grid for dashboard analysis and downstream modeling.

## Decision

Use H3 resolution 9 as the default fixed HK coverage grid.

Local H3 library sizes:

| Resolution | Average edge length | Fit |
| --- | ---: | --- |
| 8 | 531.4m | Too coarse for crowded HK streets |
| 9 | 200.8m | Best practical 200m-class zone |
| 10 | 75.9m | Strictly smaller, but much heavier and likely sparse |

The recommended interpretation is "around 200m practical radius/edge scale", not strict mathematical less-than-200m. If strict `<= 200m` is required, res10 is the first clearly safe option, but it will create roughly seven times more cells than res9 and around forty-nine times more cells than res8.

## Agent and Model Routing

| Workstream | Agent Role | Suggested Model Tier | Cost Control | Review Gate |
| --- | --- | --- | --- | --- |
| Resolution/config change | Builder | mid-tier | Small deterministic code changes inline | Tests assert default resolution and edge scale |
| Artifact regeneration | Deterministic runner | low-cost | Run existing scripts, no model calls | Generated res9 grid/marts exist and have non-zero events |
| Dashboard/API update | Builder | mid-tier | Reuse existing service and endpoint code | API summary and GeoJSON report res9 |
| Verification | Reviewer | low-cost | Focused tests, full suite, live API smoke | Report counts and local URL |

## Implementation Steps

1. Add tests proving the default grid resolution is 9 and the chosen H3 edge length is 200m-class.
2. Update `ghost_zones.DEFAULT_H3_RESOLUTION` from 8 to 9.
3. Update service/dashboard constants and artifact paths to use res9 outputs by default.
4. Regenerate fixed HK coverage artifacts:
   - `analysis/geo/hk_h3_coverage_res9.csv`
   - `analysis/geo/hk_h3_coverage_res9.geojson`
5. Regenerate fixed-grid feature marts:
   - `analysis/data_discovery/zone_sparsity_profile_res9.csv`
   - `analysis/data_discovery/zone_hour_profile_res9.csv`
   - `analysis/data_discovery/zone_daily_profile_res9.csv`
   - `analysis/data_discovery/zone_recency_profile_res9.csv`
   - `analysis/data_discovery/zone_neighbor_context_res9.csv`
6. Update dashboard manifest and service tests for res9.
7. Restart the dashboard service and smoke-check:
   - `/api/summary`
   - `/api/grid.geojson?min_events=1&limit=5`
   - `http://127.0.0.1:8765/`

## Expected Outcome

The map should show a much finer HK street/block-scale overlay. Compared with res8, each zone is roughly seven times smaller by area, making it more appropriate for dense Hong Kong streets while staying feasible for browser rendering and model evaluation.

## Approval Gate

Approve res9 for the 200m-class implementation, or say "strict 200m" and use res10 instead.
