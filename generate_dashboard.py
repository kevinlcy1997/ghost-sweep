#!/usr/bin/env python3
"""Generate a static HTML dashboard from ghost_alerts.json."""

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

# District station coordinates for mapping
DISTRICT_STATIONS = {
    "Eastern": {"lat": 22.2870, "lng": 114.2190, "region": "Hong Kong Island"},
    "Wan Chai": {"lat": 22.2780, "lng": 114.1720, "region": "Hong Kong Island"},
    "Central": {"lat": 22.2816, "lng": 114.1585, "region": "Hong Kong Island"},
    "Western": {"lat": 22.2870, "lng": 114.1420, "region": "Hong Kong Island"},
    "Wong Tai Sin": {"lat": 22.3420, "lng": 114.1930, "region": "Kowloon East"},
    "Kwun Tong": {"lat": 22.3130, "lng": 114.2250, "region": "Kowloon East"},
    "Tseung Kwan O": {"lat": 22.3170, "lng": 114.2590, "region": "Kowloon East"},
    "Sau Mau Ping": {"lat": 22.3290, "lng": 114.2320, "region": "Kowloon East"},
    "Yau Tsim": {"lat": 22.2980, "lng": 114.1720, "region": "Kowloon West"},
    "Mong Kok": {"lat": 22.3193, "lng": 114.1694, "region": "Kowloon West"},
    "Sham Shui Po": {"lat": 22.3310, "lng": 114.1590, "region": "Kowloon West"},
    "Kowloon City": {"lat": 22.3280, "lng": 114.1870, "region": "Kowloon West"},
    "Tai Po": {"lat": 22.4510, "lng": 114.1680, "region": "New Territories North"},
    "Tuen Mun": {"lat": 22.3910, "lng": 113.9770, "region": "New Territories North"},
    "Yuen Long": {"lat": 22.4440, "lng": 114.0220, "region": "New Territories North"},
    "Border": {"lat": 22.5030, "lng": 114.1280, "region": "New Territories North"},
    "Tsuen Wan": {"lat": 22.3710, "lng": 114.1140, "region": "New Territories South"},
    "Kwai Tsing": {"lat": 22.3560, "lng": 114.1300, "region": "New Territories South"},
    "Sha Tin": {"lat": 22.3810, "lng": 114.1880, "region": "New Territories South"},
    "Airport": {"lat": 22.3080, "lng": 113.9185, "region": "New Territories South"},
    "Lantau": {"lat": 22.2660, "lng": 113.9430, "region": "New Territories South"},
}

REGION_COLORS = {
    "Hong Kong Island": "#e74c3c",
    "Kowloon West": "#3498db",
    "Kowloon East": "#2ecc71",
    "New Territories North": "#f39c12",
    "New Territories South": "#9b59b6",
}

import math

def _haversine_m(lat1, lng1, lat2, lng2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_district(lat, lng):
    best_dist = float("inf")
    best_name, best_region = "", ""
    for name, info in DISTRICT_STATIONS.items():
        d = _haversine_m(lat, lng, info["lat"], info["lng"])
        if d < best_dist:
            best_dist = d
            best_name = name
            best_region = info["region"]
    return best_name, best_region


def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    alerts = []
    for rid, rec in data.get("alerts", {}).items():
        lat = float(rec.get("lat", 0))
        lng = float(rec.get("lng", 0))
        if not (22.1 < lat < 22.6 and 113.8 < lng < 114.5):
            continue
        district, region = get_district(lat, lng)
        create_dt = rec.get("create_dt", "")
        try:
            dt = datetime.strptime(create_dt, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            dt = None
        alerts.append({
            "id": rid,
            "lat": lat, "lng": lng,
            "address": rec.get("address", ""),
            "district": district, "region": region,
            "create_dt": create_dt,
            "dt": dt,
            "hour": dt.hour if dt else 0,
            "dow": dt.strftime("%A") if dt else "Unknown",
            "upvote": int(rec.get("upvote", 0)),
            "downvote": int(rec.get("downvote", 0)),
        })
    return alerts, data.get("meta", {})


def deduplicate_alerts(alerts):
    """Consolidate near-duplicate alerts (within 20m and 15min) into single events."""
    if not alerts:
        return alerts
    sorted_alerts = sorted(alerts, key=lambda a: a["create_dt"])
    clusters = []
    for alert in sorted_alerts:
        merged = False
        for cluster in clusters:
            # Check time distance
            if alert["dt"] and cluster["latest_dt"]:
                time_diff = (alert["dt"] - cluster["latest_dt"]).total_seconds() / 60.0
                if time_diff > 15:
                    continue
            else:
                continue
            # Check spatial distance
            dist = _haversine_m(alert["lat"], alert["lng"], cluster["lat"], cluster["lng"])
            if dist <= 20:
                # Merge: update centroid, keep best address
                n = cluster["count"]
                cluster["lat"] = (cluster["lat"] * n + alert["lat"]) / (n + 1)
                cluster["lng"] = (cluster["lng"] * n + alert["lng"]) / (n + 1)
                cluster["count"] += 1
                cluster["latest_dt"] = alert["dt"]
                if alert["upvote"] > cluster.get("upvote", 0):
                    cluster["address"] = alert["address"]
                    cluster["upvote"] = alert["upvote"]
                merged = True
                break
        if not merged:
            clusters.append({
                "lat": alert["lat"], "lng": alert["lng"],
                "address": alert["address"],
                "district": alert["district"], "region": alert["region"],
                "create_dt": alert["create_dt"], "dt": alert["dt"],
                "latest_dt": alert["dt"],
                "hour": alert["hour"], "dow": alert["dow"],
                "upvote": alert["upvote"], "downvote": alert["downvote"],
                "count": 1, "id": alert["id"],
            })
    return clusters


def compute_stats(alerts):
    stats = {}
    stats["total"] = len(alerts)
    stats["districts"] = Counter(a["district"] for a in alerts)
    stats["regions"] = Counter(a["region"] for a in alerts)

    # Hourly distribution
    stats["hourly"] = Counter(a["hour"] for a in alerts)

    # Day of week
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    stats["dow"] = {d: 0 for d in dow_order}
    for a in alerts:
        if a["dow"] in stats["dow"]:
            stats["dow"][a["dow"]] += 1

    # Hour x DOW heatmap
    heatmap = defaultdict(lambda: defaultdict(int))
    for a in alerts:
        if a["dow"] != "Unknown":
            heatmap[a["dow"]][a["hour"]] += 1
    stats["heatmap"] = heatmap

    # Top addresses (global, kept for backward compat)
    addr_counts = Counter(a["address"] for a in alerts if a["address"])
    stats["top_addresses"] = addr_counts.most_common(15)

    # Top addresses grouped by district — top 5 per district with last 5 records each
    by_district_addr = defaultdict(lambda: defaultdict(list))
    for a in alerts:
        if a["address"]:
            by_district_addr[a["district"]][a["address"]].append(a)
    stats["top_by_district"] = {}
    for district in sorted(by_district_addr.keys()):
        addr_entries = []
        for addr, records in sorted(by_district_addr[district].items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            recent = sorted([r for r in records if r.get("dt")], key=lambda r: r["dt"], reverse=True)[:5]
            addr_entries.append({"address": addr, "count": len(records), "recent": recent})
        stats["top_by_district"][district] = addr_entries

    # Date range
    dates = [a["dt"] for a in alerts if a["dt"]]
    stats["first_dt"] = min(dates).strftime("%Y-%m-%d %H:%M") if dates else "N/A"
    stats["last_dt"] = max(dates).strftime("%Y-%m-%d %H:%M") if dates else "N/A"
    stats["days_span"] = (max(dates) - min(dates)).days + 1 if len(dates) > 1 else 1

    # Recent alerts (global)
    sorted_alerts = sorted([a for a in alerts if a["dt"]], key=lambda x: x["dt"], reverse=True)
    stats["recent"] = sorted_alerts[:20]

    # Recent alerts grouped by district
    recent_by_district = defaultdict(list)
    for a in sorted_alerts:
        recent_by_district[a["district"]].append(a)
    stats["recent_by_district"] = {d: recs[:10] for d, recs in recent_by_district.items()}

    return stats


def generate_html(alerts, stats, meta):
    # Prepare data for JS — aggregate to grid cells for heatmap density
    heatmap_points = json.dumps([[a["lat"], a["lng"], a.get("count", 1)] for a in alerts])

    # Grid-based density for circle markers (0.005° grid = ~500m)
    grid = {}
    for a in alerts:
        key = (round(a["lat"] / 0.005) * 0.005, round(a["lng"] / 0.005) * 0.005)
        grid[key] = grid.get(key, 0) + a.get("count", 1)
    # Convert to JSON array: [lat, lng, count]
    grid_cells = json.dumps([[lat, lng, count] for (lat, lng), count in grid.items()])
    max_density = max(grid.values()) if grid else 1

    # Fixed HK center
    hk_center = json.dumps([22.36, 114.11])

    district_labels = json.dumps(sorted(stats["districts"].keys(), key=lambda d: stats["districts"][d], reverse=True))
    district_values = json.dumps([stats["districts"][d] for d in sorted(stats["districts"].keys(), key=lambda d: stats["districts"][d], reverse=True)])
    district_colors = json.dumps([REGION_COLORS.get(DISTRICT_STATIONS.get(d, {}).get("region", ""), "#999") for d in sorted(stats["districts"].keys(), key=lambda d: stats["districts"][d], reverse=True)])

    region_labels = json.dumps(sorted(stats["regions"].keys(), key=lambda r: stats["regions"][r], reverse=True))
    region_values = json.dumps([stats["regions"][r] for r in sorted(stats["regions"].keys(), key=lambda r: stats["regions"][r], reverse=True)])
    region_colors = json.dumps([REGION_COLORS.get(r, "#999") for r in sorted(stats["regions"].keys(), key=lambda r: stats["regions"][r], reverse=True)])

    hourly_labels = json.dumps(list(range(24)))
    hourly_values = json.dumps([stats["hourly"].get(h, 0) for h in range(24)])

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_labels = json.dumps(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    dow_values = json.dumps([stats["dow"].get(d, 0) for d in dow_order])

    # Hour x DOW heatmap data
    heatmap_data = []
    for di, day in enumerate(dow_order):
        for h in range(24):
            val = stats["heatmap"].get(day, {}).get(h, 0)
            if val > 0:
                heatmap_data.append({"x": h, "y": di, "v": val})
    heatmap_json = json.dumps(heatmap_data)

    # Top addresses by district — HTML
    addr_by_district_html = ""
    for district in sorted(stats["top_by_district"].keys()):
        entries = stats["top_by_district"][district]
        if not entries:
            continue
        region = DISTRICT_STATIONS.get(district, {}).get("region", "")
        color = REGION_COLORS.get(region, "#999")
        addr_by_district_html += f'<div class="district-group"><h3 style="color:{color};margin:1rem 0 0.5rem">● {district} <span style="color:#8b949e;font-size:0.8rem">({region})</span></h3>\n'
        for entry in entries:
            safe_addr = entry["address"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            addr_by_district_html += f'<details><summary style="cursor:pointer;padding:4px 0"><b>{safe_addr}</b> <span style="color:#f39c12">({entry["count"]} sightings)</span></summary>\n'
            addr_by_district_html += '<table style="margin:4px 0 8px 1rem"><thead><tr><th>Time</th><th>Upvotes</th></tr></thead><tbody>\n'
            for r in entry["recent"]:
                addr_by_district_html += f'<tr><td>{r["create_dt"]}</td><td>{r.get("upvote", 0)}</td></tr>\n'
            addr_by_district_html += '</tbody></table></details>\n'
        addr_by_district_html += '</div>\n'

    # Recent alerts grouped by district — HTML
    recent_by_district_html = ""
    for district in sorted(stats["recent_by_district"].keys()):
        records = stats["recent_by_district"][district]
        if not records:
            continue
        region = DISTRICT_STATIONS.get(district, {}).get("region", "")
        color = REGION_COLORS.get(region, "#999")
        recent_by_district_html += f'<h3 style="color:{color};margin:1rem 0 0.5rem">● {district} <span style="color:#8b949e;font-size:0.8rem">({region})</span></h3>\n'
        recent_by_district_html += '<table><thead><tr><th>Time</th><th>Address</th></tr></thead><tbody>\n'
        for a in records:
            safe_addr = a["address"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            recent_by_district_html += f'<tr><td style="white-space:nowrap">{a["create_dt"]}</td><td>{safe_addr}</td></tr>\n'
        recent_by_district_html += '</tbody></table>\n'

    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>走鬼 Ghost Sweep Dashboard</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; }}
.header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 2rem; text-align: center; border-bottom: 2px solid #f39c12; }}
.header h1 {{ font-size: 2rem; color: #f39c12; margin-bottom: 0.5rem; }}
.header .subtitle {{ color: #8b949e; font-size: 0.9rem; }}
.stats-bar {{ display: flex; justify-content: center; gap: 2rem; padding: 1rem; background: #161b22; flex-wrap: wrap; }}
.stat-card {{ text-align: center; padding: 1rem 2rem; background: #21262d; border-radius: 8px; min-width: 140px; }}
.stat-card .num {{ font-size: 1.8rem; font-weight: bold; color: #f39c12; }}
.stat-card .label {{ font-size: 0.8rem; color: #8b949e; margin-top: 4px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; padding: 1rem; max-width: 1400px; margin: 0 auto; }}
.full {{ grid-column: 1 / -1; }}
.card {{ background: #161b22; border-radius: 8px; padding: 1.5rem; border: 1px solid #30363d; }}
.card h2 {{ font-size: 1.1rem; color: #f0f6fc; margin-bottom: 1rem; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }}
#heatmap {{ height: 500px; border-radius: 8px; }}
canvas {{ max-height: 350px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #21262d; }}
th {{ color: #f39c12; font-weight: 600; }}
td.num {{ text-align: right; font-weight: bold; color: #f39c12; }}
.legend {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; font-size: 0.8rem; }}
.legend span {{ display: flex; align-items: center; gap: 4px; }}
.legend .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
@media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<div class="header">
  <h1>走鬼 Ghost Sweep</h1>
  <div class="subtitle">Hong Kong Enforcement Officer Activity Dashboard</div>
  <div class="subtitle" style="margin-top:8px">Data: {stats["first_dt"]} → {stats["last_dt"]} · Updated: {now}</div>
</div>

<div class="stats-bar">
  <div class="stat-card"><div class="num">{stats["total"]}</div><div class="label">Total Sightings</div></div>
  <div class="stat-card"><div class="num">{len(stats["districts"])}</div><div class="label">Active Districts</div></div>
  <div class="stat-card"><div class="num">{len(stats["regions"])}</div><div class="label">Police Regions</div></div>
  <div class="stat-card"><div class="num">{stats["days_span"]}</div><div class="label">Days Tracked</div></div>
</div>

<div class="grid">

  <div class="card full">
    <h2>Heatmap — Alert Density Across Hong Kong</h2>
    <div class="legend">
      <span><span class="dot" style="background:#e74c3c"></span> HK Island</span>
      <span><span class="dot" style="background:#3498db"></span> Kowloon West</span>
      <span><span class="dot" style="background:#2ecc71"></span> Kowloon East</span>
      <span><span class="dot" style="background:#f39c12"></span> NT North</span>
      <span><span class="dot" style="background:#9b59b6"></span> NT South</span>
    </div>
    <div id="heatmap"></div>
  </div>

  <div class="card">
    <h2>Sightings by Police District</h2>
    <canvas id="districtChart"></canvas>
  </div>

  <div class="card">
    <h2>Share by Police Region</h2>
    <canvas id="regionChart"></canvas>
  </div>

  <div class="card">
    <h2>Activity by Hour of Day</h2>
    <canvas id="hourlyChart"></canvas>
  </div>

  <div class="card">
    <h2>Activity by Day of Week</h2>
    <canvas id="dowChart"></canvas>
  </div>

  <div class="card">
    <h2>Top Hotspot Addresses by District</h2>
    <div style="max-height:500px;overflow-y:auto">{addr_by_district_html}</div>
  </div>

  <div class="card">
    <h2>Latest Alerts by District</h2>
    <div style="max-height:500px;overflow-y:auto">{recent_by_district_html}</div>
  </div>

</div>

<script>
// Heatmap — using native Leaflet CircleMarkers (no canvas sync issues)
setTimeout(function() {{
  const map = L.map('heatmap').setView({hk_center}, 11);
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution: '&copy; OSM &copy; CARTO', subdomains: 'abcd', maxZoom: 19
  }}).addTo(map);

  // Grid-based density circles — each cell is ~500m, color/size by density
  const cells = {grid_cells};
  const maxD = {max_density};
  function getColor(d) {{
    const t = Math.min(d / Math.max(maxD * 0.6, 1), 1);
    if (t < 0.25) return '#0066ff';
    if (t < 0.5) return '#00cccc';
    if (t < 0.75) return '#ffcc00';
    return '#ff3300';
  }}
  cells.forEach(function(c) {{
    const intensity = Math.min(c[2] / Math.max(maxD * 0.5, 1), 1);
    L.circleMarker([c[0], c[1]], {{
      radius: 6 + intensity * 18,
      fillColor: getColor(c[2]),
      fillOpacity: 0.25 + intensity * 0.45,
      color: getColor(c[2]),
      weight: 0.5,
      opacity: 0.6
    }}).bindPopup('<b>' + c[2] + ' sightings</b><br>Grid: ' + c[0].toFixed(3) + ', ' + c[1].toFixed(3)).addTo(map);
  }});
  map.invalidateSize();
}}, 200);

// District bar chart
new Chart(document.getElementById('districtChart'), {{
  type: 'bar',
  data: {{ labels: {district_labels}, datasets: [{{ data: {district_values}, backgroundColor: {district_colors}, borderWidth: 0 }}] }},
  options: {{ indexAxis: 'y', plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }}, y: {{ ticks: {{ color: '#c9d1d9', font: {{ size: 10 }} }}, grid: {{ display: false }} }} }} }}
}});

// Region pie chart
new Chart(document.getElementById('regionChart'), {{
  type: 'doughnut',
  data: {{ labels: {region_labels}, datasets: [{{ data: {region_values}, backgroundColor: {region_colors}, borderWidth: 2, borderColor: '#161b22' }}] }},
  options: {{ plugins: {{ legend: {{ position: 'bottom', labels: {{ color: '#c9d1d9', font: {{ size: 11 }} }} }} }} }}
}});

// Hourly chart
new Chart(document.getElementById('hourlyChart'), {{
  type: 'bar',
  data: {{ labels: {hourly_labels}, datasets: [{{ data: {hourly_values}, backgroundColor: '#3498db', borderWidth: 0 }}] }},
  options: {{ plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ title: {{ display: true, text: 'Hour', color: '#8b949e' }}, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }}, y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }} }} }}
}});

// Day of week chart
new Chart(document.getElementById('dowChart'), {{
  type: 'bar',
  data: {{ labels: {dow_labels}, datasets: [{{ data: {dow_values}, backgroundColor: ['#3498db','#3498db','#3498db','#3498db','#3498db','#e74c3c','#e74c3c'], borderWidth: 0 }}] }},
  options: {{ plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }}, y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }} }} }}
}});
</script>

<div style="text-align:center;padding:2rem;color:#484f58;font-size:0.75rem;">
  走鬼 Ghost Sweep · <a href="https://github.com/kevinlcy1997/ghost-sweep" style="color:#58a6ff">GitHub</a> · Auto-updated every 5 minutes
</div>

</body>
</html>"""
    return html


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else "ghost_alerts.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "docs"

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {json_path}...")
    alerts, meta = load_data(json_path)
    print(f"  {len(alerts)} valid raw alerts")

    alerts = deduplicate_alerts(alerts)
    print(f"  {len(alerts)} events after consolidation (20m/15min)")

    stats = compute_stats(alerts)
    print(f"  {len(stats['districts'])} districts, {len(stats['regions'])} regions")
    print(f"  Date range: {stats['first_dt']} → {stats['last_dt']}")

    html = generate_html(alerts, stats, meta)
    output_path = os.path.join(output_dir, "index.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Dashboard written to {output_path}")


if __name__ == "__main__":
    main()
