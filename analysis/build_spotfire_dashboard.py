"""Build a static Spotfire-like dashboard for HK fixed-grid model analysis."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghost_zones import DEFAULT_H3_RESOLUTION

OUTPUT_PATH = ROOT / "analysis" / "spotfire_dashboard.html"
RESOLUTION = DEFAULT_H3_RESOLUTION


def read_csv(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit else rows


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_dashboard_data() -> dict[str, object]:
    return {
        "coverage": read_csv(ROOT / f"analysis/geo/hk_h3_coverage_res{RESOLUTION}.csv"),
        "sparsity": read_csv(ROOT / f"analysis/data_discovery/zone_sparsity_profile_res{RESOLUTION}.csv"),
        "hourProfile": read_csv(ROOT / f"analysis/data_discovery/zone_hour_profile_res{RESOLUTION}.csv"),
        "dayProfile": read_csv(ROOT / f"analysis/data_discovery/zone_daily_profile_res{RESOLUTION}.csv"),
        "neighborContext": read_csv(ROOT / f"analysis/data_discovery/zone_neighbor_context_res{RESOLUTION}.csv"),
        "multiHorizon": read_csv(ROOT / "analysis/multi_horizon_summary_latest.csv"),
        "predictions30m": read_csv(ROOT / "analysis/iterated_zone_predictions_30m_latest.csv", 500),
        "predictions1h": read_csv(ROOT / "analysis/iterated_zone_predictions_1h_latest.csv", 500),
        "predictions2h": read_csv(ROOT / "analysis/iterated_zone_predictions_2h_latest.csv", 500),
        "manifest": read_json(ROOT / "analysis/dashboard_manifest_latest.json"),
    }


def render_html(data: dict[str, object]) -> str:
    payload = json.dumps(data)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ghost Sweep HK Fixed-Grid Analytics</title>
<style>
:root {{
  --bg: #f3f5f7;
  --panel: #ffffff;
  --ink: #18202a;
  --muted: #5d6978;
  --line: #cfd7e2;
  --accent: #0f766e;
  --accent2: #b45309;
  --danger: #b91c1c;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font: 13px/1.4 "Segoe UI", Arial, sans-serif; color: var(--ink); background: var(--bg); }}
header {{ height: 52px; display: flex; align-items: center; justify-content: space-between; padding: 0 16px; background: #26313f; color: #fff; border-bottom: 1px solid #111827; }}
header h1 {{ margin: 0; font-size: 16px; font-weight: 650; letter-spacing: 0; }}
header .meta {{ color: #cbd5e1; font-size: 12px; }}
.shell {{ display: grid; grid-template-columns: 250px 1fr; min-height: calc(100vh - 52px); }}
aside {{ border-right: 1px solid var(--line); background: #e9eef4; padding: 12px; overflow: auto; }}
main {{ padding: 12px; overflow: auto; }}
.filter {{ margin-bottom: 12px; }}
label {{ display: block; font-size: 11px; color: var(--muted); margin-bottom: 4px; text-transform: uppercase; }}
select, input {{ width: 100%; height: 30px; border: 1px solid #b8c2cf; border-radius: 3px; background: #fff; padding: 4px 8px; color: var(--ink); }}
.kpis {{ display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 8px; margin-bottom: 10px; }}
.kpi, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 4px; }}
.kpi {{ padding: 10px; min-height: 70px; }}
.kpi .v {{ font-size: 22px; font-weight: 700; }}
.kpi .t {{ color: var(--muted); font-size: 11px; }}
.grid {{ display: grid; grid-template-columns: 1.25fr 1fr; gap: 10px; }}
.panel {{ min-height: 260px; overflow: hidden; }}
.panel h2 {{ margin: 0; padding: 9px 10px; font-size: 13px; border-bottom: 1px solid var(--line); background: #f8fafc; }}
.panel-body {{ padding: 10px; }}
.map {{ height: 420px; position: relative; background: #eef2f6; border: 1px solid #d7dee8; overflow: hidden; }}
.cell {{ position: absolute; width: 5px; height: 5px; border-radius: 1px; background: #9aa8b8; opacity: .55; }}
.cell.active {{ background: var(--accent2); opacity: .9; }}
.cell.selected {{ outline: 2px solid #111827; z-index: 3; }}
.barrow {{ display: grid; grid-template-columns: 54px 1fr 44px; align-items: center; gap: 8px; margin: 5px 0; }}
.bar {{ height: 12px; background: #e5e7eb; border: 1px solid #d1d5db; }}
.fill {{ height: 100%; background: var(--accent); }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th, td {{ padding: 6px 7px; border-bottom: 1px solid #e5eaf0; text-align: left; white-space: nowrap; }}
th {{ position: sticky; top: 0; background: #f8fafc; color: #334155; }}
.table-wrap {{ max-height: 310px; overflow: auto; border: 1px solid #e1e7ef; }}
.tabs {{ display: flex; gap: 6px; margin-bottom: 8px; }}
.tabs button {{ border: 1px solid #b8c2cf; background: #fff; border-radius: 3px; padding: 5px 9px; cursor: pointer; }}
.tabs button.on {{ background: #26313f; color: #fff; }}
.small {{ color: var(--muted); font-size: 12px; }}
@media (max-width: 1100px) {{
  .shell {{ grid-template-columns: 1fr; }}
  aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
  .kpis {{ grid-template-columns: repeat(2, 1fr); }}
  .grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header>
  <h1>Ghost Sweep HK Fixed-Grid Analytics</h1>
  <div class="meta">Coverage grid · feature discovery · multi-horizon evaluation · artifact trace</div>
</header>
<div class="shell">
<aside>
  <div class="filter"><label>Region</label><select id="region"></select></div>
  <div class="filter"><label>District</label><select id="district"></select></div>
  <div class="filter"><label>Horizon</label><select id="horizon"><option value="30m">30 min</option><option value="1h">1 hour</option><option value="2h">2 hours</option></select></div>
  <div class="filter"><label>Zone Search</label><input id="zoneSearch" placeholder="h3 zone id"></div>
  <div class="filter"><label>Minimum Events</label><input id="minEvents" type="number" min="0" value="0"></div>
  <p class="small">Filters cross-select the map, distributions, top zones, predictions, and artifact tables. Zero-history cells stay visible so coverage gaps are analyzable.</p>
</aside>
<main>
  <section class="kpis">
    <div class="kpi"><div class="v" id="kZones">0</div><div class="t">fixed HK cells</div></div>
    <div class="kpi"><div class="v" id="kActive">0</div><div class="t">observed-history cells</div></div>
    <div class="kpi"><div class="v" id="kZero">0%</div><div class="t">zero-history share</div></div>
    <div class="kpi"><div class="v" id="kEvents">0</div><div class="t">events in filter</div></div>
    <div class="kpi"><div class="v" id="kPeakHour">--</div><div class="t">peak hour</div></div>
    <div class="kpi"><div class="v" id="kHorizon">--</div><div class="t">selected horizon rows</div></div>
  </section>
  <section class="grid">
    <div class="panel"><h2>HK H3 Coverage Map</h2><div class="panel-body"><div id="map" class="map"></div></div></div>
    <div class="panel"><h2>Temporal Pattern</h2><div class="panel-body"><div class="tabs"><button class="on" data-tab="hour">Hour</button><button data-tab="day">Day</button></div><div id="temporal"></div></div></div>
    <div class="panel"><h2>Top Event Zones</h2><div class="panel-body"><div class="table-wrap"><table id="zoneTable"></table></div></div></div>
    <div class="panel"><h2>Model Evaluation and Predictions</h2><div class="panel-body"><div id="modelSummary"></div><div class="table-wrap" style="margin-top:10px"><table id="predictionTable"></table></div></div></div>
    <div class="panel"><h2>Neighbor Context</h2><div class="panel-body"><div class="table-wrap"><table id="neighborTable"></table></div></div></div>
    <div class="panel"><h2>Traceable Artifacts</h2><div class="panel-body"><div class="table-wrap"><table id="artifactTable"></table></div></div></div>
  </section>
</main>
</div>
<script>
const DATA = {payload};
const $ = id => document.getElementById(id);
const num = v => Number(v || 0);
let tab = "hour";

function unique(values) {{ return [...new Set(values.filter(Boolean))].sort(); }}
function optionList(values) {{ return ["All", ...unique(values)].map(v => `<option>${{v}}</option>`).join(""); }}
function fmt(n) {{ return Math.round(n).toLocaleString(); }}

function joinedRows() {{
  const sparse = new Map(DATA.sparsity.map(r => [r.h3_zone, r]));
  return DATA.coverage.map(c => Object.assign({{}}, c, sparse.get(c.h3_zone) || {{}}));
}}

function filteredRows() {{
  const region = $("region").value;
  const district = $("district").value;
  const q = $("zoneSearch").value.trim();
  const minEvents = num($("minEvents").value);
  return joinedRows().filter(r =>
    (region === "All" || r.region === region) &&
    (district === "All" || r.district === district) &&
    (!q || r.h3_zone.includes(q)) &&
    num(r.event_count) >= minEvents
  );
}}

function renderMap(rows) {{
  const el = $("map");
  const lats = DATA.coverage.map(r => num(r.zone_lat));
  const lngs = DATA.coverage.map(r => num(r.zone_lng));
  const minLat = Math.min(...lats), maxLat = Math.max(...lats), minLng = Math.min(...lngs), maxLng = Math.max(...lngs);
  el.innerHTML = rows.map(r => {{
    const x = ((num(r.zone_lng) - minLng) / (maxLng - minLng)) * 96 + 2;
    const y = (1 - ((num(r.zone_lat) - minLat) / (maxLat - minLat))) * 92 + 4;
    const active = num(r.event_count) > 0 ? " active" : "";
    return `<span class="cell${{active}}" title="${{r.h3_zone}} · ${{r.district}} · events ${{r.event_count || 0}}" style="left:${{x}}%;top:${{y}}%"></span>`;
  }}).join("");
}}

function renderBars(rows) {{
  const zones = new Set(rows.map(r => r.h3_zone));
  const source = tab === "hour" ? DATA.hourProfile : DATA.dayProfile;
  const key = tab === "hour" ? "hour" : "day_of_week";
  const counts = new Map();
  source.forEach(r => {{ if (zones.has(r.h3_zone)) counts.set(r[key], (counts.get(r[key]) || 0) + num(r.event_count)); }});
  const max = Math.max(1, ...counts.values());
  $("temporal").innerHTML = [...counts.entries()].sort((a,b)=>num(a[0])-num(b[0])).map(([k,v]) =>
    `<div class="barrow"><span>${{tab === "hour" ? k + ":00" : ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][k]}}</span><div class="bar"><div class="fill" style="width:${{v / max * 100}}%"></div></div><span>${{fmt(v)}}</span></div>`
  ).join("");
  const peak = [...counts.entries()].sort((a,b)=>b[1]-a[1])[0];
  $("kPeakHour").textContent = peak ? (tab === "hour" ? `${{peak[0]}}:00` : ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][peak[0]]) : "--";
}}

function table(el, cols, rows) {{
  el.innerHTML = `<thead><tr>${{cols.map(c=>`<th>${{c.label}}</th>`).join("")}}</tr></thead><tbody>` +
    rows.map(r => `<tr>${{cols.map(c=>`<td>${{c.f ? c.f(r) : (r[c.key] == null ? "" : r[c.key])}}</td>`).join("")}}</tr>`).join("") + "</tbody>";
}}

function renderTables(rows) {{
  const topZones = [...rows].sort((a,b)=>num(b.event_count)-num(a.event_count)).slice(0, 80);
  table($("zoneTable"), [
    {{key:"h3_zone", label:"Zone"}},
    {{key:"district", label:"District"}},
    {{key:"region", label:"Region"}},
    {{key:"event_count", label:"Events"}},
    {{key:"is_zero_history", label:"Zero"}}
  ], topZones);
  const horizon = $("horizon").value;
  const pred = DATA[horizon === "30m" ? "predictions30m" : horizon === "1h" ? "predictions1h" : "predictions2h"] || [];
  $("kHorizon").textContent = fmt(pred.length);
  table($("predictionTable"), Object.keys(pred[0] || {{h3_zone:"", risk_score:""}}).slice(0, 6).map(k => ({{key:k, label:k}})), pred.slice(0, 60));
  const zones = new Set(rows.map(r => r.h3_zone));
  table($("neighborTable"), [
    {{key:"h3_zone", label:"Zone"}},
    {{key:"fixed_neighbor_count", label:"Neighbors"}},
    {{key:"observed_neighbor_count", label:"Observed Nbrs"}},
    {{key:"neighbor_event_count", label:"Nbr Events"}},
    {{key:"neighbor_observed_share", label:"Nbr Obs Share"}}
  ], DATA.neighborContext.filter(r => zones.has(r.h3_zone)).slice(0, 80));
}}

function renderModelSummary() {{
  const rows = DATA.multiHorizon || [];
  if (!rows.length) {{ $("modelSummary").innerHTML = "<p class='small'>No multi-horizon summary artifact found.</p>"; return; }}
  const cols = Object.keys(rows[0]).slice(0, 8).map(k => ({{key:k, label:k}}));
  $("modelSummary").innerHTML = "<div class='table-wrap' style='max-height:150px'><table id='modelTable'></table></div>";
  table(document.getElementById("modelTable"), cols, rows);
}}

function renderArtifacts() {{
  const groups = DATA.manifest.artifact_groups || {{}};
  const rows = Object.entries(groups).flatMap(([group, items]) => items.map(item => Object.assign({{group}}, item)));
  table($("artifactTable"), [
    {{key:"group", label:"Group"}},
    {{key:"path", label:"Path"}},
    {{key:"exists", label:"Exists"}},
    {{key:"row_count", label:"Rows"}},
    {{key:"modified_at", label:"Modified"}}
  ], rows);
}}

function render() {{
  const rows = filteredRows();
  const active = rows.filter(r => num(r.event_count) > 0).length;
  const events = rows.reduce((s,r)=>s+num(r.event_count),0);
  $("kZones").textContent = fmt(rows.length);
  $("kActive").textContent = fmt(active);
  $("kZero").textContent = rows.length ? Math.round((rows.length - active) / rows.length * 100) + "%" : "0%";
  $("kEvents").textContent = fmt(events);
  renderMap(rows);
  renderBars(rows);
  renderTables(rows);
  renderModelSummary();
  renderArtifacts();
}}

function init() {{
  $("region").innerHTML = optionList(DATA.coverage.map(r => r.region));
  $("district").innerHTML = optionList(DATA.coverage.map(r => r.district));
  ["region","district","horizon","zoneSearch","minEvents"].forEach(id => $(id).addEventListener("input", render));
  document.querySelectorAll(".tabs button").forEach(btn => btn.addEventListener("click", () => {{
    document.querySelectorAll(".tabs button").forEach(b => b.classList.remove("on"));
    btn.classList.add("on");
    tab = btn.dataset.tab;
    render();
  }}));
  render();
}}
init();
</script>
</body>
</html>"""


def write_dashboard(path: Path = OUTPUT_PATH) -> Path:
    path.write_text(render_html(build_dashboard_data()), encoding="utf-8")
    return path


if __name__ == "__main__":
    output = write_dashboard()
    print(f"Wrote dashboard to {output.relative_to(ROOT)}")
