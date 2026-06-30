"""Generate a lightweight Hong Kong zone geography survey."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghost_db import GhostDB
from ghost_ranking_features import enrich_events_with_zones


OUTPUT_DIR = ROOT / "analysis" / "geo"
SUMMARY_CSV = OUTPUT_DIR / "hk_zone_summary.csv"
SUMMARY_HTML = OUTPUT_DIR / "hk_zone_summary.html"


def build_zone_summary(resolution: int = 8) -> pd.DataFrame:
    events = enrich_events_with_zones(GhostDB(str(ROOT / "ghost_alerts.db")).get_all_events())
    df = pd.DataFrame(events)
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(["h3_zone", "district", "region"], dropna=False)
        .agg(
            events=("h3_zone", "size"),
            first_seen=("create_dt", "min"),
            last_seen=("create_dt", "max"),
            zone_lat=("zone_lat", "first"),
            zone_lng=("zone_lng", "first"),
            unique_grid_cells=("grid_cell", "nunique"),
        )
        .reset_index()
        .sort_values(["events", "last_seen"], ascending=[False, False])
    )
    district_totals = df.groupby("district")["h3_zone"].nunique().rename("district_active_zones")
    summary = summary.merge(district_totals, on="district", how="left")
    summary["is_one_off_zone"] = summary["events"].eq(1).astype(int)
    return summary


def write_html(summary: pd.DataFrame) -> None:
    top = summary.head(25).copy()
    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>HK Zone Survey</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #dfe5ef; padding: 8px; text-align: left; font-size: 13px; }}
th {{ background: #eef3f9; }}
.metric {{ display: inline-block; margin-right: 24px; font-size: 18px; }}
</style></head><body>
<h1>Hong Kong Zone Survey</h1>
<p class="metric"><strong>{len(summary)}</strong> active H3 zones</p>
<p class="metric"><strong>{int(summary["events"].sum())}</strong> events</p>
<p class="metric"><strong>{int(summary["is_one_off_zone"].sum())}</strong> one-off zones</p>
<h2>Top Recurring Zones</h2>
{top.to_html(index=False)}
</body></html>"""
    SUMMARY_HTML.write_text(html_doc, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_zone_summary()
    if summary.empty:
        raise RuntimeError("No events available for geography survey")
    summary.to_csv(OUT_DIR / f"hk_zone_summary_res{resolution}.csv", index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    write_html(summary)
    print(SUMMARY_HTML)


if __name__ == "__main__":
    main()


