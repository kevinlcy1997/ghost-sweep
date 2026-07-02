from __future__ import annotations

import argparse
from pathlib import Path

import h3
import pandas as pd


def _score_column(frame: pd.DataFrame) -> str:
    if "spatial_probability" in frame:
        return "spatial_probability"
    if "probability" in frame:
        return "probability"
    if "score" in frame:
        return "score"
    raise ValueError("Prediction frame must include spatial_probability, probability, or score.")


def _label_column(frame: pd.DataFrame) -> str:
    if "actual" in frame:
        return "actual"
    if "target" in frame:
        return "target"
    raise ValueError("Prediction frame must include actual or target label column.")


def _h3_neighbors(zone_id: str, radius: int) -> set[str]:
    return set(h3.grid_disk(zone_id, radius)) - {zone_id}


def summarize_topk_near_misses(
    predictions: pd.DataFrame,
    k: int = 50,
    neighbor_lookup: dict[str, set[str]] | None = None,
    ring2_lookup: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    score_col = _score_column(predictions)
    label_col = _label_column(predictions)
    neighbor_lookup = neighbor_lookup or {}
    ring2_lookup = ring2_lookup or {}
    rows: list[dict[str, object]] = []

    for target_time, group in predictions.groupby("target_time", sort=True):
        positives = group[group[label_col].astype(int) == 1]
        positive_zones = set(positives["zone_id"].astype(str))
        positive_districts = set(positives.get("district", pd.Series(dtype=str)).astype(str))
        positive_regions = set(positives.get("region", pd.Series(dtype=str)).astype(str))
        top = group.sort_values(score_col, ascending=False).head(k)

        exact_hits = 0
        ring1_hits = 0
        ring2_hits = 0
        district_hits = 0
        region_hits = 0
        for row in top.itertuples(index=False):
            zone_id = str(getattr(row, "zone_id"))
            exact = zone_id in positive_zones
            ring1_zones = neighbor_lookup[zone_id] if zone_id in neighbor_lookup else _h3_neighbors(zone_id, 1)
            ring2_zones = ring2_lookup[zone_id] if zone_id in ring2_lookup else _h3_neighbors(zone_id, 2)
            ring1 = bool(ring1_zones & positive_zones)
            ring2 = bool(ring2_zones & positive_zones)
            district = str(getattr(row, "district", "")) in positive_districts
            region = str(getattr(row, "region", "")) in positive_regions
            exact_hits += int(exact)
            ring1_hits += int(exact or ring1)
            ring2_hits += int(exact or ring1 or ring2)
            district_hits += int(district)
            region_hits += int(region)

        rows.append(
            {
                "target_time": target_time,
                "k": k,
                "positives": int(len(positives)),
                "exact_hits": exact_hits,
                "ring1_hits": ring1_hits,
                "ring2_hits": ring2_hits,
                "district_hits": district_hits,
                "region_hits": region_hits,
                "exact_precision": exact_hits / k if k else 0.0,
                "ring1_precision": ring1_hits / k if k else 0.0,
                "ring2_precision": ring2_hits / k if k else 0.0,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose exact and near-miss spatial top-k ranking.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--horizon", required=True)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    summary = summarize_topk_near_misses(predictions, k=args.k)
    output = Path(args.output or f"analysis/spatial_topk_near_miss_{args.horizon}_latest.csv")
    summary.to_csv(output, index=False)
    print(summary.drop(columns=["target_time"]).mean(numeric_only=True).to_string())
    print(output)


if __name__ == "__main__":
    main()
