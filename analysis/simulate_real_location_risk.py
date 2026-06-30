from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghost_zones import DEFAULT_H3_RESOLUTION, compute_h3_zone
ITERATED_PREDICTIONS_PATH = ROOT / "analysis" / "iterated_zone_predictions_latest.csv"
DEFAULT_PREDICTIONS_PATH = ROOT / "analysis" / "zone_predictions_res8_latest.csv"
DEFAULT_OUTPUT_PATH = ROOT / "analysis" / "real_location_simulation_latest.json"
DEFAULT_CSV_PATH = ROOT / "analysis" / "real_location_simulation_latest.csv"
DEFAULT_HTML_PATH = ROOT / "analysis" / "real_location_simulation_report.html"
HORIZON_PREDICTION_PATHS = {
    "30m": ROOT / "analysis" / "iterated_zone_predictions_30m_latest.csv",
    "1h": ROOT / "analysis" / "iterated_zone_predictions_1h_latest.csv",
    "2h": ROOT / "analysis" / "iterated_zone_predictions_2h_latest.csv",
}


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def _risk_band(score: float, percentile: float) -> str:
    if percentile >= 0.9 or score >= 0.75:
        return "high"
    if percentile >= 0.6 or score >= 0.4:
        return "medium"
    return "low"


def _load_predictions(path: Path | None = None) -> pd.DataFrame:
    if path is not None:
        return pd.read_csv(path)
    if ITERATED_PREDICTIONS_PATH.exists():
        return pd.read_csv(ITERATED_PREDICTIONS_PATH)
    return pd.read_csv(DEFAULT_PREDICTIONS_PATH)


def _select_snapshot(predictions: pd.DataFrame, target_time: str | pd.Timestamp) -> pd.DataFrame:
    frame = predictions.copy()
    frame["target_time"] = pd.to_datetime(frame["target_time"])
    requested = pd.Timestamp(target_time)
    if requested in set(frame["target_time"]):
        return frame.loc[frame["target_time"] == requested].copy()
    nearest_time = min(frame["target_time"].unique(), key=lambda value: abs(value - requested))
    return frame.loc[frame["target_time"] == nearest_time].copy()


def simulate_location_risk(
    lat: float,
    lng: float,
    target_time: str | pd.Timestamp,
    predictions: pd.DataFrame | None = None,
    resolution: int = DEFAULT_H3_RESOLUTION,
    top_n: int = 5,
) -> dict[str, Any]:
    if predictions is None:
        predictions = _load_predictions()

    zone_id = compute_h3_zone(lat, lng, resolution=resolution)
    snapshot = _select_snapshot(predictions, target_time)
    if snapshot.empty:
        raise ValueError("No prediction rows are available for simulation.")

    snapshot = snapshot.sort_values("score", ascending=False).reset_index(drop=True)
    snapshot["rank"] = snapshot.index + 1
    matched = snapshot.loc[snapshot["zone_id"] == zone_id]
    if matched.empty:
        if {"zone_lat", "zone_lng"}.issubset(snapshot.columns):
            distances = snapshot.apply(
                lambda row: _haversine_km(lat, lng, row["zone_lat"], row["zone_lng"]),
                axis=1,
            )
            selected = snapshot.loc[distances.idxmin()].copy()
            selected["distance_km"] = float(distances.loc[distances.idxmin()])
        else:
            selected = snapshot.iloc[-1].copy()
            selected["distance_km"] = None
    else:
        selected = matched.iloc[0].copy()
        selected["distance_km"] = 0.0

    total = int(len(snapshot))
    rank = int(selected["rank"])
    percentile = 1.0 if total == 1 else 1.0 - ((rank - 1) / (total - 1))
    score = float(selected["score"])
    band = _risk_band(score, percentile)

    hot_zones: list[dict[str, Any]] = []
    for _, row in snapshot.head(top_n).iterrows():
        distance_km = None
        if {"zone_lat", "zone_lng"}.issubset(snapshot.columns):
            distance_km = _haversine_km(lat, lng, row["zone_lat"], row["zone_lng"])
        hot_zones.append(
            {
                "zone_id": row["zone_id"],
                "district": row.get("district"),
                "region": row.get("region"),
                "score": float(row["score"]),
                "rank": int(row["rank"]),
                "distance_km": None if distance_km is None else round(float(distance_km), 3),
            }
        )

    district = selected.get("district")
    summary = (
        f"{district} is ranked {rank} of {total} zones for {pd.Timestamp(selected['target_time'])}. "
        f"Risk band is {band} with score {score:.3f}."
    )
    return {
        "input_lat": float(lat),
        "input_lng": float(lng),
        "requested_target_time": str(pd.Timestamp(target_time)),
        "matched_target_time": str(pd.Timestamp(selected["target_time"])),
        "zone_id": zone_id,
        "matched_zone_id": selected["zone_id"],
        "district": district,
        "region": selected.get("region"),
        "score": score,
        "rank": rank,
        "total_zones": total,
        "rank_percentile": float(percentile),
        "risk_band": band,
        "distance_to_matched_zone_km": selected["distance_km"],
        "nearest_hot_zones": hot_zones,
        "summary": summary,
    }


def _load_predictions_by_horizon() -> dict[str, pd.DataFrame]:
    return {
        horizon: pd.read_csv(path)
        for horizon, path in HORIZON_PREDICTION_PATHS.items()
        if path.exists()
    }


def simulate_location_multi_horizon(
    lat: float,
    lng: float,
    target_time: str | pd.Timestamp,
    predictions_by_horizon: dict[str, pd.DataFrame] | None = None,
    resolution: int = DEFAULT_H3_RESOLUTION,
) -> dict[str, Any]:
    if predictions_by_horizon is None:
        predictions_by_horizon = _load_predictions_by_horizon()
    if not predictions_by_horizon:
        raise ValueError("No horizon prediction files are available.")

    ordered = ["30m", "1h", "2h"]
    horizons = {
        horizon: simulate_location_risk(
            lat=lat,
            lng=lng,
            target_time=target_time,
            predictions=predictions_by_horizon[horizon],
            resolution=resolution,
        )
        for horizon in ordered
        if horizon in predictions_by_horizon
    }
    return {
        "input_lat": float(lat),
        "input_lng": float(lng),
        "target_time": str(pd.Timestamp(target_time)),
        "horizons": horizons,
    }


def run_default_simulations() -> list[dict[str, Any]]:
    scenarios = [
        ("Mong Kok", 22.3154, 114.1698),
        ("Tsim Sha Tsui", 22.2988, 114.1722),
        ("Central", 22.2819, 114.1589),
        ("Causeway Bay", 22.2797, 114.1850),
        ("Sha Tin", 22.3820, 114.1880),
        ("Tung Chung", 22.2893, 113.9415),
    ]
    predictions_by_horizon = _load_predictions_by_horizon()
    predictions = predictions_by_horizon.get("2h", _load_predictions())
    latest_time = pd.to_datetime(predictions["target_time"]).max()
    hottest_time = pd.to_datetime(predictions.sort_values("score", ascending=False).iloc[0]["target_time"])
    results = []
    for label, target_time in [("latest", latest_time), ("hottest", hottest_time)]:
        for name, lat, lng in scenarios:
            result = simulate_location_multi_horizon(
                lat=lat,
                lng=lng,
                target_time=target_time,
                predictions_by_horizon=predictions_by_horizon or None,
            )
            result["scenario"] = name
            result["time_scenario"] = label
            results.append(result)
    DEFAULT_OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary_rows = [
        {
            "time_scenario": result["time_scenario"],
            "scenario": result["scenario"],
            "matched_target_time": next(iter(result["horizons"].values()))["matched_target_time"],
            "risk_30m": result["horizons"].get("30m", {}).get("risk_band"),
            "score_30m": result["horizons"].get("30m", {}).get("score"),
            "rank_30m": result["horizons"].get("30m", {}).get("rank"),
            "risk_1h": result["horizons"].get("1h", {}).get("risk_band"),
            "score_1h": result["horizons"].get("1h", {}).get("score"),
            "rank_1h": result["horizons"].get("1h", {}).get("rank"),
            "risk_2h": result["horizons"].get("2h", {}).get("risk_band"),
            "score_2h": result["horizons"].get("2h", {}).get("score"),
            "rank_2h": result["horizons"].get("2h", {}).get("rank"),
            "district": result["horizons"].get("2h", next(iter(result["horizons"].values()))).get("district"),
            "summary": " | ".join(
                f"{horizon}: {value['risk_band']} {value['score']:.3f}"
                for horizon, value in result["horizons"].items()
            ),
        }
        for result in results
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DEFAULT_CSV_PATH, index=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Ghost Sweep Location Simulation</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; }}
    h1 {{ margin-bottom: 8px; }}
    .note {{ padding: 14px; border: 1px solid #d9e2ef; border-radius: 8px; background: #f7f9fc; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ef; padding: 8px; text-align: right; }}
    th {{ background: #eef3f9; }}
    td:nth-child(1), td:nth-child(2), td:nth-child(3), td:nth-child(4), td:nth-child(5), td:nth-child(6), td:nth-child(11) {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Ghost Sweep Location Simulation</h1>
  <div class="note">Runs named Hong Kong locations against the latest iterated model predictions. The latest snapshot checks quiet-period behavior; the hottest snapshot checks whether known dense urban cores rise when the model expects activity.</div>
  {summary.to_html(index=False)}
</body>
</html>
"""
    DEFAULT_HTML_PATH.write_text(html, encoding="utf-8")
    return results


def main() -> None:
    results = run_default_simulations()
    for result in results:
        horizons = result["horizons"]
        display = " ".join(
            f"{horizon}={value['risk_band']}:{value['score']:.3f}"
            for horizon, value in horizons.items()
        )
        print(
            f"{result['time_scenario']} | {result['scenario']}: {display}"
        )
    print(DEFAULT_OUTPUT_PATH)
    print(DEFAULT_HTML_PATH)


if __name__ == "__main__":
    main()
