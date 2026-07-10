from __future__ import annotations

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "analysis" / "reports"
REPORT_FILENAME = "current-model-data-analysis.html"


def output_path_for_date(root_dir: Path, report_date: str) -> Path:
    return Path(root_dir) / "analysis" / "reports" / report_date / REPORT_FILENAME


def build_report_payload(**sections):
    return sections


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_report_html(payload: dict) -> str:
    raw = payload["raw"]
    reconstructed = payload["reconstructed_geo"]
    stage1_30m = payload["stage1"]["30m"]
    stage2_30m = payload["stage2"]["30m"]
    writeup = payload["writeup"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Current Model Data Analysis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f6f8fb; color: #172033; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .card, .panel {{ background: #fff; border: 1px solid #dbe4ef; border-radius: 10px; padding: 16px; }}
    .metric {{ font-size: 28px; font-weight: 700; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #e6edf5; }}
    h1, h2, h3 {{ margin-top: 0; }}
  </style>
</head>
<body>
<main>
  <section class="panel">
    <h1>Current Model Data Analysis</h1>
    <p>Static snapshot of the current accepted model design: Stage 1 city-activity gating plus Stage 2 spatial ranking.</p>
  </section>
  <section class="grid">
    <div class="card"><div>Raw alert count</div><div class="metric">{raw["total_alerts"]}</div></div>
    <div class="card"><div>Usable enriched events</div><div class="metric">{reconstructed["usable_events"]}</div></div>
    <div class="card"><div>Active H3 zones</div><div class="metric">{reconstructed["unique_zones"]}</div></div>
    <div class="card"><div>Stage 1 30m positive rate</div><div class="metric">{_pct(stage1_30m["positive_rate"])}</div></div>
    <div class="card"><div>Stage 2 30m positive rate</div><div class="metric">{_pct(stage2_30m["positive_rate"])}</div></div>
    <div class="card"><div>Stage 2 sampled neg:pos</div><div class="metric">{stage2_30m["sampled_neg_to_pos_ratio"]:.1f}:1</div></div>
  </section>
  <section class="panel">
    <h2>Dashboard summary</h2>
    <p>Top district: {escape(reconstructed["top_districts"][0][0])}</p>
    <p>Top region: {escape(reconstructed["top_regions"][0][0])}</p>
    <p>Most complete raw fields: lat/lng/create_dt; district/region missing in raw payload.</p>
  </section>
  <section class="panel">
    <h2>Full write-up</h2>
    <h3>Executive read</h3>
    <p>{escape(writeup["executive_read"])}</p>
    <h3>What / So What / Now What</h3>
    <p><strong>What:</strong> {escape(writeup["what_so_what_now_what"]["what"])}</p>
    <p><strong>So What:</strong> {escape(writeup["what_so_what_now_what"]["so_what"])}</p>
    <p><strong>Now What:</strong> {escape(writeup["what_so_what_now_what"]["now_what"])}</p>
  </section>
</main>
</body>
</html>"""


def write_report(payload: dict, root_dir: Path | None = None, report_date: str = "2026-07-10") -> Path:
    root = Path(root_dir) if root_dir is not None else ROOT
    out_path = output_path_for_date(root, report_date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report_html(payload), encoding="utf-8")
    return out_path
