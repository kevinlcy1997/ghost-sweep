"""Create a fully local SVG/HTML map overlay for the zone risk forecast."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GEOJSON_PATH = ROOT / "ghost_zone_forecast.geojson"
OUT_PATH = ROOT / "analysis" / "zone_forecast_map.html"
CHOICE_PATH = ROOT / "analysis" / "resolution_choice_latest.json"

WIDTH = 980
HEIGHT = 730
PAD = 44


def _project(lng: float, lat: float, bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    min_lng, max_lng, min_lat, max_lat = bounds
    x = PAD + (lng - min_lng) / (max_lng - min_lng) * (WIDTH - PAD * 2)
    y = HEIGHT - PAD - (lat - min_lat) / (max_lat - min_lat) * (HEIGHT - PAD * 2)
    return x, y


def _color(score: float) -> str:
    if score >= 0.75:
        return "#991b1b"
    if score >= 0.55:
        return "#dc2626"
    if score >= 0.35:
        return "#f97316"
    if score >= 0.18:
        return "#fbbf24"
    return "#fee2e2"


def _polygon_points(coords: list[list[float]], bounds: tuple[float, float, float, float]) -> str:
    points = [_project(float(lng), float(lat), bounds) for lng, lat in coords]
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _resolve_geojson_path() -> tuple[Path, dict]:
    if CHOICE_PATH.exists():
        choice = json.loads(CHOICE_PATH.read_text(encoding="utf-8"))
        selected = ROOT / choice["geojson_path"]
        if selected.exists():
            return selected, choice
    return GEOJSON_PATH, {}


def main() -> None:
    geojson_path, choice = _resolve_geojson_path()
    geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = geojson.get("features", [])
    all_points = [
        point
        for feature in features
        for ring in feature["geometry"]["coordinates"]
        for point in ring
    ]
    lngs = [float(point[0]) for point in all_points]
    lats = [float(point[1]) for point in all_points]
    lng_pad = (max(lngs) - min(lngs)) * 0.08
    lat_pad = (max(lats) - min(lats)) * 0.08
    bounds = (min(lngs) - lng_pad, max(lngs) + lng_pad, min(lats) - lat_pad, max(lats) + lat_pad)

    sorted_features = sorted(features, key=lambda item: item["properties"]["risk_rank"])
    scores = [float(f["properties"].get("score", 0)) for f in features]

    polygon_markup = []
    for feature in sorted_features:
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"][0]
        score = float(props.get("score", 0))
        rank = int(props["risk_rank"])
        stroke_width = 2.6 if rank <= 20 else 1.1
        polygon_markup.append(
            f'<polygon class="zone" id="zone-{rank}" '
            f'data-rank="{rank}" '
            f'data-score="{score:.3f}" '
            f'data-district="{html.escape(str(props["district"]))}" '
            f'data-region="{html.escape(str(props["region"]))}" '
            f'data-recent="{int(props["recent_events_24h"])}" '
            f'points="{_polygon_points(coords, bounds)}" '
            f'fill="{_color(score)}" fill-opacity="{max(0.28, min(0.82, score)):.2f}" '
            f'stroke="{_color(score)}" stroke-width="{stroke_width}" />'
        )

    top_list = []
    for feature in sorted_features[:12]:
        p = feature["properties"]
        top_list.append(
            "<li>"
            f'<button data-jump="{int(p["risk_rank"])}">#{int(p["risk_rank"])} '
            f'{html.escape(str(p["district"]))} · {float(p["score"]):.3f}</button>'
            f"<span>{html.escape(str(p['region']))} · recent 24h: {int(p['recent_events_24h'])}</span>"
            "</li>"
        )

    if choice:
        subtitle = (
            f"Selected H3 res {choice['resolution']} from resolution comparison. "
            f"Precision@20 {float(choice['precision_at_20']):.3f}, "
            f"top-decile lift {float(choice['top_decile_lift']):.3f}, "
            f"one-off-zone rate {float(choice['one_off_zone_rate']):.3f}."
        )
    else:
        subtitle = f"Fully local SVG overlay from {geojson_path.name}. No internet tiles are required."

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ghost Sweep Zone Forecast Map</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #667085;
      --paper: #f6f8fb;
      --panel: #ffffff;
      --line: #d7e0ec;
      --navy: #111827;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: var(--ink); background: var(--paper); }}
    main {{ min-height: 100vh; display: grid; grid-template-columns: 360px 1fr; }}
    aside {{ background: var(--panel); border-right: 1px solid var(--line); padding: 24px 22px; overflow: auto; }}
    h1 {{ margin: 0 0 8px; font-size: 27px; line-height: 1.08; letter-spacing: 0; }}
    .sub {{ color: var(--muted); font-size: 14px; line-height: 1.45; margin-bottom: 20px; }}
    .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px; }}
    .stat {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfcfe; }}
    .stat strong {{ display: block; font-size: 24px; line-height: 1; margin-bottom: 6px; }}
    .stat span {{ font-size: 12px; color: var(--muted); }}
    .legend {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin: 16px 0 20px; }}
    .legend-row {{ display: grid; grid-template-columns: 28px 1fr; gap: 9px; align-items: center; margin: 8px 0; color: #344054; font-size: 13px; }}
    .swatch {{ width: 28px; height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,.1); }}
    ol {{ margin: 0; padding-left: 22px; }}
    li {{ margin-bottom: 11px; color: #344054; line-height: 1.35; font-size: 13px; }}
    li button {{ display: block; appearance: none; background: none; border: 0; color: var(--ink); font: inherit; font-weight: 700; cursor: pointer; padding: 0 0 2px; text-align: left; }}
    li span {{ color: var(--muted); }}
    .map-wrap {{ position: relative; min-height: 100vh; padding: 28px; background: linear-gradient(180deg, #eaf1f8, #dbe6ef); }}
    .map-card {{ height: calc(100vh - 56px); min-height: 690px; background: #eef4f9; border: 1px solid #cad6e2; border-radius: 8px; overflow: hidden; position: relative; }}
    svg {{ width: 100%; height: 100%; display: block; }}
    .zone {{ cursor: pointer; transition: fill-opacity .12s ease, stroke-width .12s ease; }}
    .zone:hover, .zone.active {{ fill-opacity: .92; stroke: #111827; stroke-width: 3.2; }}
    .water-label {{ fill: #7b8da3; font-size: 18px; letter-spacing: 0; opacity: .62; }}
    .axis-label {{ fill: #667085; font-size: 12px; }}
    .tooltip {{ position: absolute; min-width: 240px; pointer-events: none; background: white; border: 1px solid #cbd5e1; box-shadow: 0 14px 36px rgba(15,23,42,.16); border-radius: 8px; padding: 12px 14px; display: none; }}
    .tooltip strong {{ display: block; margin-bottom: 7px; }}
    .tooltip div {{ margin-top: 4px; color: #344054; font-size: 13px; }}
    .note {{ margin-top: 18px; padding: 12px; border-radius: 8px; color: #344054; background: #fff7df; border: 1px solid #f0d98c; font-size: 13px; line-height: 1.4; }}
    @media (max-width: 860px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .map-wrap {{ min-height: 620px; padding: 16px; }}
      .map-card {{ height: 70vh; min-height: 560px; }}
    }}
  </style>
</head>
<body>
  <main>
    <aside>
      <h1>Hong Kong Zone Risk Map</h1>
    <div class="sub">{subtitle} Source: <code>{geojson_path.name}</code>.</div>
      <div class="stat-grid">
        <div class="stat"><strong>{len(features)}</strong><span>scored zones</span></div>
        <div class="stat"><strong>{max(scores):.3f}</strong><span>max score</span></div>
        <div class="stat"><strong>{min(scores):.3f}</strong><span>min score</span></div>
        <div class="stat"><strong>2h</strong><span>forecast window</span></div>
      </div>
      <div class="legend">
        <strong>Risk score</strong>
        <div class="legend-row"><div class="swatch" style="background:#fee2e2"></div><div>Lower ranked zones</div></div>
        <div class="legend-row"><div class="swatch" style="background:#f97316"></div><div>Elevated watch zones</div></div>
        <div class="legend-row"><div class="swatch" style="background:#991b1b"></div><div>Highest ranked zones</div></div>
      </div>
      <h2 style="font-size:16px;margin:0 0 12px">Top 12 watchlist</h2>
      <ol>{''.join(top_list)}</ol>
      <div class="note">The polygons are the scored H3 zones, not the full Hong Kong land boundary. Use this as a watchlist overlay for model-ranked attention.</div>
    </aside>
    <section class="map-wrap">
      <div class="map-card">
        <svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="Hong Kong H3 zone risk overlay">
          <rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#e4edf5" />
          <text x="52" y="68" class="water-label">Hong Kong zone forecast overlay</text>
          <text x="52" y="{HEIGHT - 35}" class="axis-label">west</text>
          <text x="{WIDTH - 92}" y="{HEIGHT - 35}" class="axis-label">east</text>
          {''.join(polygon_markup)}
        </svg>
        <div id="tooltip" class="tooltip"></div>
      </div>
    </section>
  </main>
  <script>
    const tooltip = document.getElementById('tooltip');
    const zones = document.querySelectorAll('.zone');
    function showTooltip(zone, event) {{
      tooltip.innerHTML = `<strong>Rank #${{zone.dataset.rank}} · ${{zone.dataset.district}}</strong>
        <div>Score: ${{zone.dataset.score}}</div>
        <div>Region: ${{zone.dataset.region}}</div>
        <div>Recent events 24h: ${{zone.dataset.recent}}</div>`;
      tooltip.style.display = 'block';
      const rect = document.querySelector('.map-card').getBoundingClientRect();
      tooltip.style.left = `${{event.clientX - rect.left + 16}}px`;
      tooltip.style.top = `${{event.clientY - rect.top + 16}}px`;
    }}
    zones.forEach(zone => {{
      zone.addEventListener('mousemove', event => showTooltip(zone, event));
      zone.addEventListener('mouseleave', () => tooltip.style.display = 'none');
      zone.addEventListener('click', () => {{
        zones.forEach(z => z.classList.remove('active'));
        zone.classList.add('active');
      }});
    }});
    document.querySelectorAll('[data-jump]').forEach(button => {{
      button.addEventListener('click', () => {{
        const zone = document.getElementById(`zone-${{button.dataset.jump}}`);
        if (!zone) return;
        zones.forEach(z => z.classList.remove('active'));
        zone.classList.add('active');
        zone.scrollIntoView({{ behavior: 'smooth', block: 'center', inline: 'center' }});
      }});
    }});
  </script>
</body>
</html>
"""
    OUT_PATH.write_text(html_doc, encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
