"""Render a visual explainer for the HK zone-ranking model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "zone_model_visual_explainer.png"
REPORT = ROOT / "analysis" / "zone_ranking_report_20260627_222855.html"
SURVEY = ROOT / "analysis" / "geo" / "hk_zone_summary.csv"
GEOJSON = ROOT / "ghost_zone_forecast.geojson"


PALETTE = {
    "ink": "#172033",
    "muted": "#667085",
    "line": "#CBD5E1",
    "paper": "#F7F9FC",
    "white": "#FFFFFF",
    "blue": "#2B6CB0",
    "amber": "#D97706",
    "green": "#059669",
    "red": "#DC2626",
    "teal": "#0F766E",
}


def box(ax, x, y, w, h, title, body, color):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=color,
        facecolor=PALETTE["white"],
    )
    ax.add_patch(patch)
    ax.text(x + 0.025, y + h - 0.045, title, ha="left", va="top", fontsize=13, weight="bold", color=color)
    ax.text(x + 0.025, y + h - 0.092, body, ha="left", va="top", fontsize=9.3, color=PALETTE["ink"], linespacing=1.22)


def arrow(ax, x1, y1, x2, y2, color=PALETTE["line"]):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, shrinkA=5, shrinkB=5),
    )


def hexagon(cx, cy, r):
    import math

    return [(cx + r * math.cos(math.pi / 6 + i * math.pi / 3), cy + r * math.sin(math.pi / 6 + i * math.pi / 3)) for i in range(6)]


def metric(ax, x, y, label, value, color):
    ax.text(x, y, value, ha="left", va="top", fontsize=23, weight="bold", color=color)
    ax.text(x, y - 0.045, label, ha="left", va="top", fontsize=9, color=PALETTE["muted"])


def main() -> None:
    metrics = pd.read_html(REPORT)[0]
    best = metrics.sort_values(["precision_at_20", "average_precision"], ascending=False).iloc[0]
    survey = pd.read_csv(SURVEY)
    geo = json.loads(GEOJSON.read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(16, 10), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(PALETTE["paper"])

    ax.text(0.055, 0.945, "Ghost Sweep: Hong Kong Zone-Ranking Model", fontsize=25, weight="bold", color=PALETTE["ink"])
    ax.text(
        0.055,
        0.91,
        "How raw alerts become a 2-hour ranked risk forecast for map-ready Hong Kong zones",
        fontsize=12,
        color=PALETTE["muted"],
    )

    # Left panel: construction.
    ax.text(0.055, 0.845, "1  Feature Construction", fontsize=15, weight="bold", color=PALETTE["ink"])
    box(
        ax,
        0.055,
        0.69,
        0.22,
        0.13,
        "Raw alerts",
        f"{int(survey.events.sum()):,} cleaned events\nlat/lng, time, district\nsalvaged from JSON into SQLite",
        PALETTE["blue"],
    )
    box(
        ax,
        0.055,
        0.525,
        0.22,
        0.13,
        "H3 zone layer",
        f"{len(survey):,} active H3 zones\n21 HK districts represented\nzone centroid + polygon",
        PALETTE["teal"],
    )
    box(
        ax,
        0.055,
        0.36,
        0.22,
        0.13,
        "Training rows",
        "zone_id + target_time\npast-only features\nfuture-only label",
        PALETTE["amber"],
    )
    arrow(ax, 0.165, 0.685, 0.165, 0.66)
    arrow(ax, 0.165, 0.52, 0.165, 0.495)

    for idx, (cx, cy, c) in enumerate(
        [
            (0.318, 0.61, PALETTE["blue"]),
            (0.352, 0.63, PALETTE["teal"]),
            (0.386, 0.61, PALETTE["amber"]),
            (0.352, 0.57, PALETTE["green"]),
            (0.386, 0.55, PALETTE["line"]),
            (0.318, 0.55, PALETTE["line"]),
        ]
    ):
        ax.add_patch(Polygon(hexagon(cx, cy, 0.025), closed=True, facecolor=c, alpha=0.22 + idx * 0.04, edgecolor=c, lw=1.2))
    ax.text(0.302, 0.505, "Sparse grid -> actionable\nzone candidates", fontsize=10, color=PALETTE["muted"])

    # Middle panel: training.
    ax.text(0.39, 0.845, "2  Model Training", fontsize=15, weight="bold", color=PALETTE["ink"])
    box(
        ax,
        0.39,
        0.69,
        0.255,
        0.13,
        "Target",
        "alert_next_2h = 1\nif the zone has at least one alert\nin the next two hours",
        PALETTE["red"],
    )
    box(
        ax,
        0.39,
        0.525,
        0.255,
        0.13,
        "Feature families",
        "recent zone activity\nrecent district activity\nhour/day pattern + urban core",
        PALETTE["blue"],
    )
    box(
        ax,
        0.39,
        0.36,
        0.255,
        0.13,
        "Model selection",
        "historical baseline\nlogistic regression\nrandom forest + LightGBM",
        PALETTE["green"],
    )
    arrow(ax, 0.518, 0.685, 0.518, 0.66)
    arrow(ax, 0.518, 0.52, 0.518, 0.495)

    # Right panel: result.
    ax.text(0.705, 0.845, "3  What The Result Says", fontsize=15, weight="bold", color=PALETTE["ink"])
    box(
        ax,
        0.705,
        0.69,
        0.24,
        0.13,
        "Best model",
        f"{best['model']}\nrank zones by near-term risk\nlogged in MLflow",
        PALETTE["green"],
    )
    box(
        ax,
        0.705,
        0.525,
        0.24,
        0.13,
        "Map output",
        f"{len(geo['features'])} GeoJSON features\nscore + rank + district\nready for heatmap display",
        PALETTE["teal"],
    )
    box(
        ax,
        0.705,
        0.36,
        0.24,
        0.13,
        "Interpretation",
        "Use top-K zones as a watchlist.\nThe score is a ranking signal,\nnot a certainty claim.",
        PALETTE["amber"],
    )

    # Metric strip.
    strip = FancyBboxPatch((0.055, 0.12), 0.89, 0.125, boxstyle="round,pad=0.02,rounding_size=0.025", linewidth=0, facecolor=PALETTE["ink"])
    ax.add_patch(strip)
    metric(ax, 0.085, 0.215, "Precision@20", f"{best['precision_at_20']:.2f}", "#86EFAC")
    metric(ax, 0.27, 0.215, "Top-decile lift", f"{best['top_decile_lift']:.2f}x", "#FCD34D")
    metric(ax, 0.455, 0.215, "Average precision", f"{best['average_precision']:.3f}", "#93C5FD")
    metric(ax, 0.64, 0.215, "ROC-AUC", f"{best['roc_auc']:.3f}", "#C4B5FD")
    metric(ax, 0.80, 0.215, "Active zones", f"{len(survey):,}", "#5EEAD4")

    ax.text(
        0.055,
        0.06,
        "Reading the model: it is strongest as an operations ranking tool. Precision@20 means the top twenty zones are the primary product surface; "
        "low recall@20 is expected because Hong Kong still has many active zones outside a short watchlist.",
        fontsize=10.5,
        color=PALETTE["muted"],
    )

    fig.savefig(OUT, bbox_inches="tight", facecolor=PALETTE["paper"])
    print(OUT)


if __name__ == "__main__":
    main()
