"""Local full-stack dashboard service for Ghost Sweep analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import h3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghost_zones import DEFAULT_H3_RESOLUTION

ANALYSIS_DIR = ROOT / "analysis"
RESOLUTION = DEFAULT_H3_RESOLUTION
COVERAGE_MODE = "road_access"

PATHS = {
    "coverage": ANALYSIS_DIR / "geo" / f"hk_h3_road_coverage_res{RESOLUTION}.csv",
    "coverage_geojson": ANALYSIS_DIR / "geo" / f"hk_h3_road_coverage_res{RESOLUTION}.geojson",
    "sparsity": ANALYSIS_DIR / "data_discovery" / f"zone_sparsity_profile_road_res{RESOLUTION}.csv",
    "hour": ANALYSIS_DIR / "data_discovery" / f"zone_hour_profile_road_res{RESOLUTION}.csv",
    "day": ANALYSIS_DIR / "data_discovery" / f"zone_daily_profile_road_res{RESOLUTION}.csv",
    "manifest": ANALYSIS_DIR / "dashboard_manifest_latest.json",
    "two_stage_summary": ANALYSIS_DIR / "two_stage_summary_latest.csv",
    "multi_horizon_summary": ANALYSIS_DIR / "multi_horizon_summary_latest.csv",
    "activity_predictions_30m": ANALYSIS_DIR / "activity_predictions_30m_latest.csv",
    "activity_predictions_1h": ANALYSIS_DIR / "activity_predictions_1h_latest.csv",
    "activity_predictions_2h": ANALYSIS_DIR / "activity_predictions_2h_latest.csv",
    "spatial_predictions_30m": ANALYSIS_DIR / "spatial_zone_predictions_30m_latest.csv",
    "spatial_predictions_1h": ANALYSIS_DIR / "spatial_zone_predictions_1h_latest.csv",
    "spatial_predictions_2h": ANALYSIS_DIR / "spatial_zone_predictions_2h_latest.csv",
    "predictions_30m": ANALYSIS_DIR / "iterated_zone_predictions_30m_latest.csv",
    "predictions_1h": ANALYSIS_DIR / "iterated_zone_predictions_1h_latest.csv",
    "predictions_2h": ANALYSIS_DIR / "iterated_zone_predictions_2h_latest.csv",
}


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@lru_cache(maxsize=64)
def _read_csv_cached(path_text: str, mtime: float) -> tuple[dict[str, str], ...]:
    path = Path(path_text)
    if not path.exists():
        return tuple()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    return list(_read_csv_cached(str(path), _mtime(path)))


@lru_cache(maxsize=16)
def _read_json_cached(path_text: str, mtime: float) -> dict[str, object]:
    path = Path(path_text)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_obj(path: Path) -> dict[str, object]:
    return _read_json_cached(str(path), _mtime(path))


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def query_value(query: dict[str, list[str]], key: str, default: str = "") -> str:
    return query.get(key, [default])[0]


def normalize_horizon(horizon: str) -> str:
    return horizon if horizon in {"30m", "1h", "2h"} else "30m"


def horizon_label(minutes: int) -> str:
    return {30: "30m", 60: "1h", 120: "2h"}.get(minutes, f"{minutes}m")


def read_prediction_rows(horizon: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    normalized_horizon = normalize_horizon(horizon)
    preferred = read_csv_rows(PATHS[f"spatial_predictions_{normalized_horizon}"])
    source_rows = preferred or read_csv_rows(PATHS[f"predictions_{normalized_horizon}"])
    for row in source_rows:
        normalized = dict(row)
        normalized["score"] = number(normalized.get("score"))
        if "probability" in normalized and str(normalized.get("probability", "")) != "":
            normalized["probability"] = number(normalized.get("probability"))
        else:
            normalized["probability"] = normalized["score"]
        if "activity_probability" in normalized:
            normalized["activity_probability"] = number(normalized.get("activity_probability"))
        if "spatial_probability" in normalized:
            normalized["spatial_probability"] = number(normalized.get("spatial_probability"))
        if "rank" in normalized and str(normalized.get("rank", "")) != "":
            normalized["rank"] = integer(normalized.get("rank"))
        rows.append(normalized)
    return rows


def prediction_zone_key(row: dict[str, object]) -> str:
    return str(row.get("zone_id") or row.get("h3_zone") or "")


def prediction_sort_key(row: dict[str, object], index: int) -> tuple[object, ...]:
    target_time = str(row.get("target_time") or "")
    rank_value = integer(row.get("rank"), 10**9)
    score_value = number(row.get("score"))
    return (target_time, -rank_value, score_value, index)


def prediction_rows_by_zone(horizon: str) -> dict[str, dict[str, object]]:
    best_rows: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {}
    for index, row in enumerate(read_prediction_rows(horizon)):
        zone_key = prediction_zone_key(row)
        if not zone_key:
            continue
        sort_key = prediction_sort_key(row, index)
        current = best_rows.get(zone_key)
        if current is None or sort_key > current[0]:
            best_rows[zone_key] = (sort_key, row)
    return {zone: row for zone, (_, row) in best_rows.items()}


def merged_zone_rows() -> list[dict[str, object]]:
    sparsity = {row["h3_zone"]: row for row in read_csv_rows(PATHS["sparsity"])}
    rows: list[dict[str, object]] = []
    for coverage in read_csv_rows(PATHS["coverage"]):
        merged = dict(coverage)
        coverage_observed = integer(coverage.get("has_observed_history"))
        sparse = sparsity.get(coverage["h3_zone"], {})
        merged.update(sparse)
        if coverage_observed:
            merged["has_observed_history"] = 1
            merged["is_zero_history"] = 0
        for key in (
            "h3_resolution",
            "has_observed_history",
            "is_zero_history",
            "event_count",
            "district_event_count",
            "region_event_count",
            "has_drivable_road",
            "road_source_mismatch",
            "road_segment_count",
        ):
            if key in merged:
                merged[key] = integer(merged[key])
        for key in ("zone_lat", "zone_lng", "log_event_count"):
            if key in merged:
                merged[key] = number(merged[key])
        rows.append(merged)
    return rows


def paginate(rows: list[dict[str, object]], query: dict[str, list[str]]) -> dict[str, object]:
    limit = max(1, min(integer(query_value(query, "limit", "250"), 250), 5000))
    offset = max(0, integer(query_value(query, "offset", "0"), 0))
    return {
        "total": len(rows),
        "limit": limit,
        "offset": offset,
        "rows": rows[offset : offset + limit],
    }


def api_summary() -> dict[str, object]:
    rows = merged_zone_rows()
    observed = sum(1 for row in rows if integer(row.get("has_observed_history")) == 1)
    events = sum(integer(row.get("event_count")) for row in rows)
    manifest = read_json_obj(PATHS["manifest"])
    artifact_groups = manifest.get("artifact_groups", {})
    return {
        "h3_resolution": RESOLUTION,
        "coverage_mode": COVERAGE_MODE,
        "average_hex_edge_m": round(h3.average_hexagon_edge_length(RESOLUTION, "m"), 1),
        "coverage_cells": len(rows),
        "observed_cells": observed,
        "zero_history_cells": len(rows) - observed,
        "event_count": events,
        "road_access_cells": sum(1 for row in rows if integer(row.get("has_drivable_road")) == 1),
        "road_source_mismatch_cells": sum(
            1 for row in rows if integer(row.get("road_source_mismatch")) == 1
        ),
        "regions": sorted({str(row.get("region", "")) for row in rows if row.get("region")}),
        "districts": sorted({str(row.get("district", "")) for row in rows if row.get("district")}),
        "artifact_groups": len(artifact_groups) if isinstance(artifact_groups, dict) else 0,
    }


def api_coverage(query: dict[str, list[str]]) -> dict[str, object]:
    region = query_value(query, "region", "All")
    district = query_value(query, "district", "All")
    zone = query_value(query, "zone", "")
    min_events = integer(query_value(query, "min_events", "0"), 0)

    rows = []
    for row in merged_zone_rows():
        if region != "All" and row.get("region") != region:
            continue
        if district != "All" and row.get("district") != district:
            continue
        if zone and zone not in str(row.get("h3_zone", "")):
            continue
        if integer(row.get("event_count")) < min_events:
            continue
        rows.append(row)
    rows.sort(key=lambda item: integer(item.get("event_count")), reverse=True)
    return paginate(rows, query)


def api_timeseries(query: dict[str, list[str]]) -> dict[str, object]:
    grain = query_value(query, "grain", "hour")
    source = "day" if grain == "day" else "hour"
    key = "day_of_week" if source == "day" else "hour"
    buckets: dict[int, int] = {}
    for row in read_csv_rows(PATHS[source]):
        bucket = integer(row.get(key))
        buckets[bucket] = buckets.get(bucket, 0) + integer(row.get("event_count"))
    max_bucket = 7 if source == "day" else 24
    return {
        "grain": source,
        "rows": [
            {key: bucket, "event_count": buckets.get(bucket, 0)}
            for bucket in range(max_bucket)
        ],
    }


def api_predictions(query: dict[str, list[str]]) -> dict[str, object]:
    horizon = normalize_horizon(query_value(query, "horizon", "30m"))
    rows = read_prediction_rows(horizon)
    return {"horizon": horizon, **paginate(rows, query)}


def api_model_metrics() -> dict[str, object]:
    two_stage_rows = read_csv_rows(PATHS["two_stage_summary"])
    if two_stage_rows:
        rows = []
        for row in two_stage_rows:
            activity_metadata_path = resolve_path(str(row.get("activity_metadata_path", "")))
            spatial_metadata_path = resolve_path(str(row.get("spatial_metadata_path", "")))
            rows.append(
                {
                    "horizon": row.get("horizon") or horizon_label(integer(row.get("horizon_minutes"))),
                    "horizon_minutes": integer(row.get("horizon_minutes")),
                    "model_family": "two_stage",
                    "activity_model": row.get("activity_model", ""),
                    "spatial_model": row.get("spatial_model", ""),
                    "activity_average_precision": number(row.get("activity_average_precision")),
                    "activity_roc_auc": number(row.get("activity_roc_auc")),
                    "activity_brier_score": number(row.get("activity_brier_score")),
                    "activity_holdout_rows": integer(row.get("activity_holdout_rows")),
                    "activity_holdout_positives": integer(row.get("activity_holdout_positives")),
                    "activity_holdout_start": row.get("activity_holdout_start", ""),
                    "activity_holdout_end": row.get("activity_holdout_end", ""),
                    "spatial_precision_at_20": number(row.get("spatial_precision_at_20")),
                    "spatial_precision_at_50": number(row.get("spatial_precision_at_50")),
                    "spatial_average_precision": number(row.get("spatial_average_precision")),
                    "spatial_top_decile_lift": number(row.get("spatial_top_decile_lift")),
                    "spatial_holdout_rows": integer(row.get("spatial_holdout_rows")),
                    "spatial_holdout_positives": integer(row.get("spatial_holdout_positives")),
                    "spatial_holdout_start": row.get("spatial_holdout_start", ""),
                    "spatial_holdout_end": row.get("spatial_holdout_end", ""),
                    "activity_metadata_path": str(row.get("activity_metadata_path", "")),
                    "spatial_metadata_path": str(row.get("spatial_metadata_path", "")),
                    "activity_predictions_path": str(row.get("activity_predictions_path", "")),
                    "predictions_path": str(row.get("predictions_path", "")),
                    "activity_metadata": read_json_obj(activity_metadata_path)
                    if activity_metadata_path.exists()
                    else {},
                    "spatial_metadata": read_json_obj(spatial_metadata_path)
                    if spatial_metadata_path.exists()
                    else {},
                }
            )
        rows.sort(key=lambda item: integer(item.get("horizon_minutes")))
        return {"total": len(rows), "rows": rows}

    rows = []
    for row in read_csv_rows(PATHS["multi_horizon_summary"]):
        minutes = integer(row.get("horizon_minutes"))
        metadata_path = resolve_path(str(row.get("metadata_path", "")))
        metadata = read_json_obj(metadata_path) if metadata_path.exists() else {}
        rows.append(
            {
                "horizon": horizon_label(minutes),
                "horizon_minutes": minutes,
                "target": row.get("target", ""),
                "chosen_model": row.get("chosen_model", ""),
                "median_precision_at_20": number(row.get("median_precision_at_20")),
                "median_average_precision": number(row.get("median_average_precision")),
                "median_top_decile_lift": number(row.get("median_top_decile_lift")),
                "holdout_precision_at_20": number(row.get("holdout_precision_at_20")),
                "holdout_precision_at_50": number(row.get("holdout_precision_at_50")),
                "holdout_recall_at_20": number(row.get("holdout_recall_at_20")),
                "holdout_top_decile_lift": number(row.get("holdout_top_decile_lift")),
                "holdout_average_precision": number(row.get("holdout_average_precision")),
                "holdout_roc_auc": number(row.get("holdout_roc_auc")),
                "metadata_path": str(row.get("metadata_path", "")),
                "predictions_path": str(row.get("predictions_path", "")),
                "metadata": metadata,
            }
        )
    rows.sort(key=lambda item: integer(item.get("horizon_minutes")))
    return {"total": len(rows), "rows": rows}


def api_artifacts() -> dict[str, object]:
    manifest = read_json_obj(PATHS["manifest"])
    groups = manifest.get("artifact_groups", {})
    rows = []
    if isinstance(groups, dict):
        for group, artifacts in groups.items():
            if not isinstance(artifacts, list):
                continue
            for artifact in artifacts:
                row = {"group": group}
                if isinstance(artifact, dict):
                    row.update(artifact)
                rows.append(row)
    return {"rows": rows, "total": len(rows)}


def api_grid_geojson(query: dict[str, list[str]]) -> dict[str, object]:
    geojson = read_json_obj(PATHS["coverage_geojson"])
    if not geojson:
        return {"type": "FeatureCollection", "features": []}

    region = query_value(query, "region", "All")
    district = query_value(query, "district", "All")
    zone = query_value(query, "zone", "")
    min_events = integer(query_value(query, "min_events", "0"), 0)
    horizon = normalize_horizon(query_value(query, "horizon", "30m"))
    rows = []
    for row in merged_zone_rows():
        if region != "All" and row.get("region") != region:
            continue
        if district != "All" and row.get("district") != district:
            continue
        if zone and zone not in str(row.get("h3_zone", "")):
            continue
        if integer(row.get("event_count")) < min_events:
            continue
        rows.append(row)
    rows_by_zone = {str(row["h3_zone"]): row for row in rows}
    predictions_by_zone = prediction_rows_by_zone(horizon)

    features = []
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        zone_id = str(properties.get("h3_zone", ""))
        row = rows_by_zone.get(zone_id)
        if row is None:
            continue
        merged = dict(feature)
        prediction = dict(predictions_by_zone.get(zone_id, {}))
        merged["properties"] = {**properties, **row, **prediction}
        merged["properties"]["zone_id"] = zone_id
        if "probability" not in merged["properties"] and "score" in prediction:
            merged["properties"]["probability"] = number(prediction.get("score"))
        features.append(merged)

    return {
        "type": "FeatureCollection",
        "features": features,
        "total": len(features),
    }


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ghost Sweep Service Dashboard</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root { --bg:#f4f6f8; --panel:#fff; --ink:#18202a; --muted:#667085; --line:#d0d7e2; --accent:#0f766e; --hot:#b45309; }
* { box-sizing: border-box; }
body { margin:0; font:13px/1.4 "Segoe UI", Arial, sans-serif; color:var(--ink); background:var(--bg); }
header { height:52px; padding:0 16px; display:flex; align-items:center; justify-content:space-between; background:#26313f; color:white; }
h1 { margin:0; font-size:16px; }
.shell { display:grid; grid-template-columns:250px 1fr; min-height:calc(100vh - 52px); }
aside { padding:12px; border-right:1px solid var(--line); background:#e9eef4; }
main { padding:12px; }
label { display:block; margin:10px 0 4px; color:var(--muted); font-size:11px; text-transform:uppercase; }
select,input { width:100%; height:30px; border:1px solid #b8c2cf; border-radius:3px; padding:4px 8px; background:white; }
.kpis { display:grid; grid-template-columns:repeat(6,minmax(120px,1fr)); gap:8px; margin-bottom:10px; }
.kpi,.panel { background:var(--panel); border:1px solid var(--line); border-radius:4px; }
.kpi { padding:10px; min-height:68px; }
.kpi strong { display:block; font-size:22px; }
.kpi span { color:var(--muted); font-size:11px; }
.grid { display:grid; grid-template-columns:1.25fr 1fr; gap:10px; }
.panel h2 { margin:0; padding:9px 10px; font-size:13px; border-bottom:1px solid var(--line); background:#f8fafc; }
.body { padding:10px; }
.map { height:520px; position:relative; background:#dbe4ed; border:1px solid #d7dee8; overflow:hidden; }
.leaflet-container { font:12px/1.3 "Segoe UI", Arial, sans-serif; }
.maplegend { display:flex; align-items:center; gap:10px; margin-top:8px; color:var(--muted); font-size:12px; }
.swatch { width:30px; height:10px; border:1px solid #9aa5b1; display:inline-block; }
.swatch.zero { background:#7f8ea3; opacity:.35; }
.swatch.hot { background:#b45309; opacity:.82; }
.zoneDetail { margin-top:8px; padding:8px; border:1px solid #d8dee7; background:#f8fafc; min-height:60px; }
.barrow { display:grid; grid-template-columns:54px 1fr 48px; gap:8px; align-items:center; margin:5px 0; }
.bar { height:12px; background:#e5e7eb; border:1px solid #d1d5db; }
.fill { height:100%; background:var(--accent); }
.tablewrap { max-height:320px; overflow:auto; border:1px solid #e1e7ef; }
table { width:100%; border-collapse:collapse; font-size:12px; }
th,td { padding:6px 7px; border-bottom:1px solid #e5eaf0; white-space:nowrap; text-align:left; }
th { position:sticky; top:0; background:#f8fafc; }
.small { color:var(--muted); font-size:12px; }
@media(max-width:1100px){.shell{grid-template-columns:1fr}.grid{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}aside{border-right:0;border-bottom:1px solid var(--line)}} 
</style>
</head>
<body>
<header><h1>Ghost Sweep Service Dashboard</h1><div>API-rendered fixed-grid analytics</div></header>
<div class="shell">
<aside>
  <label>Region</label><select id="region"></select>
  <label>District</label><select id="district"></select>
  <label>Horizon</label><select id="horizon"><option value="30m">30 min</option><option value="1h">1 hour</option><option value="2h">2 hours</option></select>
  <label>Minimum Events</label><input id="minEvents" type="number" value="1" min="0">
  <label>Zone Search</label><input id="zone" placeholder="h3 zone id">
  <p class="small">This page streams bounded API responses from the local service instead of embedding all data in the HTML.</p>
</aside>
<main>
  <section class="kpis">
    <div class="kpi"><strong id="cells">...</strong><span>fixed cells</span></div>
    <div class="kpi"><strong id="edgeScale">...</strong><span>avg H3 edge</span></div>
    <div class="kpi"><strong id="observed">...</strong><span>observed cells</span></div>
    <div class="kpi"><strong id="zero">...</strong><span>zero-history cells</span></div>
    <div class="kpi"><strong id="events">...</strong><span>events</span></div>
    <div class="kpi"><strong id="visible">...</strong><span>visible zones</span></div>
  </section>
  <section class="grid">
    <div class="panel"><h2>Model Metrics</h2><div class="body"><div class="tablewrap"><table id="modelMetrics"></table></div></div></div>
    <div class="panel"><h2>Hong Kong H3 Grid Overlay</h2><div class="body"><div id="map" class="map"></div><div class="maplegend"><span><i class="swatch zero"></i> zero-history cell</span><span><i class="swatch hot"></i> observed-history / higher event count</span></div><div id="zoneDetail" class="zoneDetail small">Click a hex cell to inspect zone, district, event count, and coordinates.</div></div></div>
    <div class="panel"><h2>Hourly Pattern</h2><div class="body" id="bars"></div></div>
    <div class="panel"><h2>Top Zones</h2><div class="body"><div class="tablewrap"><table id="zones"></table></div></div></div>
    <div class="panel"><h2>Predictions</h2><div class="body"><div class="tablewrap"><table id="predictions"></table></div></div></div>
    <div class="panel"><h2>Artifacts</h2><div class="body"><div class="tablewrap"><table id="artifacts"></table></div></div></div>
  </section>
</main>
</div>
<script>
const $ = id => document.getElementById(id);
async function fetchJson(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(path + ' returned ' + res.status);
  return await res.json();
}
function fmt(n){ return Number(n || 0).toLocaleString(); }
function options(values){ return ['All'].concat(values || []).map(v=>`<option>${v}</option>`).join(''); }
function table(id, rows, cols){
  const el = $(id);
  if (!rows.length) { el.innerHTML = '<tbody><tr><td>No rows</td></tr></tbody>'; return; }
  cols = cols || Object.keys(rows[0]).slice(0, 6);
  el.innerHTML = '<thead><tr>' + cols.map(c=>`<th>${c}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map(r=>'<tr>' + cols.map(c=>`<td>${r[c] == null ? '' : r[c]}</td>`).join('') + '</tr>').join('') + '</tbody>';
}
let map;
let gridLayer;
function initMap(){
  if (map) return;
  map = L.map('map', { preferCanvas:true }).setView([22.3193, 114.1694], 11);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);
}
function colorForProbability(probability, riskBand){
  if (riskBand === 'critical') return '#7f1d1d';
  if (riskBand === 'high') return '#b45309';
  if (riskBand === 'elevated') return '#d97706';
  if (riskBand === 'low') return '#2563eb';
  const n = Number(probability || 0);
  if (n >= 0.8) return '#7f1d1d';
  if (n >= 0.5) return '#b45309';
  if (n >= 0.25) return '#d97706';
  return '#2563eb';
}
function colorForEvents(events){
  const n = Number(events || 0);
  if (n <= 0) return '#7f8ea3';
  if (n < 5) return '#60a5fa';
  if (n < 25) return '#22c55e';
  if (n < 100) return '#f59e0b';
  return '#b45309';
}
function fillColorForFeature(props){
  if (props.probability != null && props.probability !== '') {
    return colorForProbability(Number(props.probability), props.risk_band);
  }
  return colorForEvents(props.event_count);
}
async function drawMap(params){
  initMap();
  const geo = await fetchJson('/api/grid.geojson?horizon=' + $('horizon').value + '&' + params.toString());
  if (gridLayer) gridLayer.remove();
  gridLayer = L.geoJSON(geo, {
    style: feature => {
      const props = feature.properties || {};
      const hasProbability = props.probability != null && props.probability !== '';
      return {
        color: hasProbability ? '#7c2d12' : (Number(props.event_count || 0) > 0 ? '#78350f' : '#475569'),
        weight: hasProbability ? 0.9 : (Number(props.event_count || 0) > 0 ? 0.8 : 0.35),
        fillColor: fillColorForFeature(props),
        fillOpacity: hasProbability ? 0.66 : (Number(props.event_count || 0) > 0 ? 0.62 : 0.16)
      };
    },
    onEachFeature: (feature, layer) => {
      const p = feature.properties;
      const detail = p.probability != null && p.probability !== '' ? ` · p=${Number(p.probability).toFixed(3)}${p.risk_band ? ' · ' + p.risk_band : ''}` : '';
      layer.bindTooltip(`${p.district || 'Unknown'} · ${p.event_count || 0} events${detail}`, { sticky:true });
      layer.on('click', () => {
        $('zoneDetail').innerHTML = `<strong>${p.h3_zone}</strong><br>${p.district || 'Unknown'} · ${p.region || 'Unknown'}<br>Events: ${p.event_count || 0}${p.probability != null && p.probability !== '' ? ` · probability: ${Number(p.probability).toFixed(3)}${p.risk_band ? ' · ' + p.risk_band : ''}` : ''} · drivable road: ${p.has_drivable_road || 0} · mismatch: ${p.road_source_mismatch || 0}<br>Nearest road: ${p.nearest_road_m || 'n/a'}m ${p.nearest_road_name ? '· ' + p.nearest_road_name : ''}<br>Centroid: ${Number(p.zone_lat).toFixed(5)}, ${Number(p.zone_lng).toFixed(5)}`;
      });
    }
  }).addTo(map);
  if (geo.features.length) map.fitBounds(gridLayer.getBounds(), { padding:[16,16] });
}
async function render(){
  const summary = await fetchJson('/api/summary');
  $('cells').textContent = fmt(summary.coverage_cells);
  $('edgeScale').textContent = Math.round(summary.average_hex_edge_m) + 'm';
  $('observed').textContent = fmt(summary.observed_cells);
  $('zero').textContent = fmt(summary.zero_history_cells);
  $('events').textContent = fmt(summary.event_count);
  if (!$('region').options.length) $('region').innerHTML = options(summary.regions);
  if (!$('district').options.length) $('district').innerHTML = options(summary.districts);

  const modelMetrics = await fetchJson('/api/model-metrics');
  table('modelMetrics', modelMetrics.rows, ['horizon','chosen_model','median_top_decile_lift','median_average_precision','holdout_top_decile_lift','holdout_average_precision']);

  const params = new URLSearchParams({ region:$('region').value || 'All', district:$('district').value || 'All', min_events:$('minEvents').value || '1', zone:$('zone').value || '', limit:'600' });
  const coverage = await fetchJson('/api/coverage?' + params.toString());
  $('visible').textContent = fmt(coverage.total);
  const mapParams = new URLSearchParams(params);
  mapParams.delete('limit');
  await drawMap(mapParams);
  table('zones', coverage.rows.slice(0,80), ['h3_zone','district','region','event_count','has_drivable_road','road_source_mismatch']);

  const ts = await fetchJson('/api/timeseries?grain=hour');
  const max = Math.max(1, ...ts.rows.map(r=>Number(r.event_count)));
  $('bars').innerHTML = ts.rows.map(r=>`<div class="barrow"><span>${r.hour}:00</span><div class="bar"><div class="fill" style="width:${Number(r.event_count)/max*100}%"></div></div><span>${fmt(r.event_count)}</span></div>`).join('');

  const pred = await fetchJson('/api/predictions?horizon=' + $('horizon').value + '&limit=80');
  table('predictions', pred.rows);
  const artifacts = await fetchJson('/api/artifacts');
  table('artifacts', artifacts.rows, ['group','path','exists','row_count','modified_at']);
}
['region','district','horizon','minEvents','zone'].forEach(id => $(id).addEventListener('input', () => render().catch(err => alert(err.message))));
render().catch(err => alert(err.message));
</script>
</body>
</html>"""


def json_response(payload: object, status: int = 200) -> tuple[int, dict[str, str], str]:
    return (
        status,
        {"Content-Type": "application/json; charset=utf-8"},
        json.dumps(payload, ensure_ascii=False),
    )


def geojson_response(payload: object, status: int = 200) -> tuple[int, dict[str, str], str]:
    return (
        status,
        {"Content-Type": "application/geo+json; charset=utf-8"},
        json.dumps(payload, ensure_ascii=False),
    )


def html_response(body: str, status: int = 200) -> tuple[int, dict[str, str], str]:
    return status, {"Content-Type": "text/html; charset=utf-8"}, body


def dispatch(method: str, target: str) -> tuple[int, dict[str, str], str]:
    if method != "GET":
        return json_response({"error": "method_not_allowed"}, 405)

    parsed = urlparse(target)
    query = parse_qs(parsed.query)
    path = parsed.path
    if path == "/":
        return html_response(DASHBOARD_HTML)
    if path == "/api/summary":
        return json_response(api_summary())
    if path == "/api/coverage":
        return json_response(api_coverage(query))
    if path == "/api/timeseries":
        return json_response(api_timeseries(query))
    if path == "/api/predictions":
        return json_response(api_predictions(query))
    if path == "/api/model-metrics":
        return json_response(api_model_metrics())
    if path == "/api/artifacts":
        return json_response(api_artifacts())
    if path == "/api/grid.geojson":
        return geojson_response(api_grid_geojson(query))
    return json_response({"error": "not_found", "path": path}, 404)


class DashboardRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        status, headers, body = dispatch("GET", self.path)
        encoded = body.encode("utf-8")
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), DashboardRequestHandler)
    print(f"Ghost Sweep dashboard service running at http://{host}:{port}/")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
