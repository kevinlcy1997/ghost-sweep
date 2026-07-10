"""Create a local HTML overlay that compares HK H3 grid scales."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.build_hk_coverage_grid import build_hk_coverage_grid, write_coverage_artifacts


GEO_DIR = ROOT / "analysis" / "geo"
OUT_PATH = ROOT / "analysis" / "h3_scale_overlay.html"
RESOLUTIONS = (7, 8, 9)
DEFAULT_RESOLUTION = 8
WIDTH = 980
HEIGHT = 760
PAD = 36


def _project(
    lng: float,
    lat: float,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    min_lng, max_lng, min_lat, max_lat = bounds
    x = PAD + (lng - min_lng) / (max_lng - min_lng) * (WIDTH - PAD * 2)
    y = HEIGHT - PAD - (lat - min_lat) / (max_lat - min_lat) * (HEIGHT - PAD * 2)
    return x, y


def _polygon_points(coords: list[list[float]], bounds: tuple[float, float, float, float]) -> str:
    points = [_project(float(lng), float(lat), bounds) for lng, lat in coords]
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _ensure_geojson_path(resolution: int) -> Path:
    path = GEO_DIR / f"hk_h3_coverage_res{resolution}.geojson"
    if path.exists():
        return path
    rows = build_hk_coverage_grid(resolution=resolution)
    _, path = write_coverage_artifacts(rows, resolution=resolution)
    return path


def _load_datasets() -> tuple[dict[int, dict], tuple[float, float, float, float]]:
    datasets: dict[int, dict] = {}
    all_points: list[list[float]] = []
    for resolution in RESOLUTIONS:
        geojson = json.loads(_ensure_geojson_path(resolution).read_text(encoding="utf-8"))
        features = geojson.get("features", [])
        polygons = [feature["geometry"]["coordinates"][0] for feature in features]
        for polygon in polygons:
            all_points.extend(polygon)
        datasets[resolution] = {
            "cell_count": len(features),
            "edge_m": round(h3.average_hexagon_edge_length(resolution, "m"), 1),
            "polygons": polygons,
        }

    lngs = [float(point[0]) for point in all_points]
    lats = [float(point[1]) for point in all_points]
    lng_pad = (max(lngs) - min(lngs)) * 0.04
    lat_pad = (max(lats) - min(lats)) * 0.04
    bounds = (min(lngs) - lng_pad, max(lngs) + lng_pad, min(lats) - lat_pad, max(lats) + lat_pad)
    return datasets, bounds


def build_overlay_html(
    datasets: dict[int, dict],
    bounds: tuple[float, float, float, float],
    default_resolution: int = DEFAULT_RESOLUTION,
) -> str:
    groups: list[str] = []
    buttons: list[str] = []
    stats: list[str] = []
    for resolution in RESOLUTIONS:
        dataset = datasets[resolution]
        polygons = "".join(
            f'<polygon points="{_polygon_points(coords, bounds)}" />'
            for coords in dataset["polygons"]
        )
        display = "inline" if resolution == default_resolution else "none"
        active = "active" if resolution == default_resolution else ""
        groups.append(
            f'<g id="res-{resolution}" class="grid-layer" data-resolution="{resolution}" style="display:{display}">{polygons}</g>'
        )
        buttons.append(
            f'<button type="button" class="toggle {active}" data-resolution="{resolution}">H3 res {resolution}</button>'
        )
        stats.append(
            f'"{resolution}":{{cells:"{dataset["cell_count"]:,}",edge:"{dataset["edge_m"]:,.1f} m"}}'
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HK H3 Grid Scale Overlay</title>
  <style>
    :root {{
      --bg: #eef4fa;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d8e3ef;
      --accent: #0f766e;
      --accent-soft: #ccfbf1;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: var(--bg); color: var(--ink); }}
    main {{ display: grid; grid-template-columns: 320px 1fr; min-height: 100vh; }}
    aside {{ background: var(--panel); border-right: 1px solid var(--line); padding: 24px; }}
    h1 {{ margin: 0 0 10px; font-size: 28px; }}
    .sub {{ color: var(--muted); font-size: 14px; line-height: 1.45; margin-bottom: 18px; }}
    .toggles {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 18px; }}
    .toggle {{ border: 1px solid var(--line); background: #f8fbff; color: var(--ink); padding: 9px 12px; border-radius: 999px; cursor: pointer; font-weight: 700; }}
    .toggle.active {{ border-color: var(--accent); background: var(--accent-soft); color: #115e59; }}
    .cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 18px; }}
    .card {{ border: 1px solid var(--line); border-radius: 10px; background: #fbfdff; padding: 12px; }}
    .card strong {{ display: block; font-size: 24px; margin-bottom: 5px; }}
    .card span {{ color: var(--muted); font-size: 12px; }}
    .note {{ border: 1px solid #f1d7a7; background: #fff7df; border-radius: 10px; padding: 12px; font-size: 13px; line-height: 1.45; }}
    .canvas {{ padding: 24px; }}
    .frame {{ height: calc(100vh - 48px); min-height: 700px; background: linear-gradient(180deg, #e7eff7, #dbe7f1); border: 1px solid #c7d6e4; border-radius: 10px; overflow: hidden; }}
    svg {{ width: 100%; height: 100%; display: block; }}
    .grid-layer polygon {{ fill: rgba(15,118,110,.08); stroke: rgba(15,118,110,.55); stroke-width: 0.85; }}
    .water-label {{ fill: #7a8ea7; font-size: 18px; opacity: .72; }}
    .axis-label {{ fill: #667085; font-size: 12px; }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .frame {{ height: 70vh; min-height: 560px; }}
    }}
  </style>
</head>
<body>
  <main>
    <aside>
      <h1>HK H3 Grid Scale</h1>
      <div class="sub">Single-canvas overlay for comparing how coarse or fine the Hong Kong H3 tessellation looks at resolutions 7, 8, and 9.</div>
      <div class="toggles">{''.join(buttons)}</div>
      <div class="cards">
        <div class="card"><strong id="cellCount">{datasets[default_resolution]["cell_count"]:,}</strong><span>cells shown</span></div>
        <div class="card"><strong id="edgeLength">{datasets[default_resolution]["edge_m"]:,.1f} m</strong><span>avg hex edge</span></div>
      </div>
      <div class="note">
        Bigger cells (res 7) make the map less sparse and easier for the model to learn, but they blur location detail.
        Smaller cells (res 9) localize better, but create a much sparser ranking problem.
      </div>
    </aside>
    <section class="canvas">
      <div class="frame">
        <svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="Hong Kong H3 grid scale overlay">
          <rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#e7eff7" />
          <text x="46" y="64" class="water-label">Hong Kong H3 coverage overlay</text>
          <text x="46" y="{HEIGHT - 32}" class="axis-label">west</text>
          <text x="{WIDTH - 92}" y="{HEIGHT - 32}" class="axis-label">east</text>
          {''.join(groups)}
        </svg>
      </div>
    </section>
  </main>
  <script>
    const stats = {{{",".join(stats)}}};
    const buttons = [...document.querySelectorAll('.toggle')];
    const layers = [...document.querySelectorAll('.grid-layer')];
    const cellCount = document.getElementById('cellCount');
    const edgeLength = document.getElementById('edgeLength');
    function setResolution(resolution) {{
      buttons.forEach(button => button.classList.toggle('active', button.dataset.resolution === resolution));
      layers.forEach(layer => {{
        layer.style.display = layer.dataset.resolution === resolution ? 'inline' : 'none';
      }});
      cellCount.textContent = stats[resolution].cells;
      edgeLength.textContent = stats[resolution].edge;
    }}
    buttons.forEach(button => {{
      button.addEventListener('click', () => setResolution(button.dataset.resolution));
    }});
    setResolution('{default_resolution}');
  </script>
</body>
</html>
"""


def write_overlay(path: Path = OUT_PATH) -> Path:
    datasets, bounds = _load_datasets()
    path.write_text(build_overlay_html(datasets, bounds), encoding="utf-8")
    return path


def main() -> None:
    print(write_overlay())


if __name__ == "__main__":
    main()
