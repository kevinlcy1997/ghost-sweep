# ghost_predict.py
"""CLI entry point for Ghost Sweep prediction system."""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from ghost_db import GhostDB
from ghost_clean import consolidate_events
from ghost_districts import assign_district_to_events
from ghost_features import build_features, build_training_data
from ghost_model import GhostModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ghost_predict")

DB_PATH = "ghost_alerts.db"
MODEL_PATH = "models/model_latest.joblib"
FORECAST_PATH = "ghost_forecast.json"


def cmd_clean(args):
    db = GhostDB(DB_PATH)
    log.info("Fetching raw sightings for cleaning...")
    sightings = db.get_unprocessed_sightings()
    log.info("  %d raw sightings", len(sightings))
    events = consolidate_events(sightings)
    events = assign_district_to_events(events)
    log.info("  %d events after consolidation", len(events))
    db.execute("DELETE FROM events")
    db._conn.commit()
    db.insert_events(events)
    log.info("  Events table rebuilt with %d records", len(events))
    db.close()


def cmd_train(args):
    db = GhostDB(DB_PATH)
    days = db.count_days_collected()
    model = GhostModel()
    gate = model.check_data_gate(days)
    if not gate["ready"]:
        log.warning("Insufficient data: %d/%d days. Need %d more.", gate["days_collected"], gate["days_needed"], gate["days_remaining"])
        db.close()
        sys.exit(1)
    events = db.get_all_events()
    db.close()
    if not events:
        log.error("No events. Run 'clean' first.")
        sys.exit(1)
    log.info("Building training data from %d events...", len(events))
    train_df = build_training_data(events)
    log.info("  Training samples: %d", len(train_df))
    metrics = model.train(train_df)
    log.info("Training complete: AUC=%.4f F1=%.4f Trees=%d", metrics["auc_roc"], metrics["f1"], metrics["n_estimators"])


def cmd_forecast(args):
    hours = args.hours
    if not Path(MODEL_PATH).exists():
        log.error("No trained model. Run 'train' first.")
        sys.exit(1)
    db = GhostDB(DB_PATH)
    events = db.get_all_events()
    db.close()
    model = GhostModel(MODEL_PATH)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    log.info("Generating %dh forecast...", hours)
    features_df = build_features(events, now, hours)
    if features_df.empty:
        log.warning("No active cells.")
        return
    predictions = model.predict(features_df)
    features_df["probability"] = predictions
    features_df["risk"] = features_df["probability"].apply(lambda p: "high" if p >= 0.7 else ("medium" if p >= 0.4 else "low"))
    features_df = features_df.sort_values("probability", ascending=False)

    dt_24h_ago = now - timedelta(hours=24)
    recent_events_by_cell = {}
    for ev in events:
        if ev.get("create_dt", "") >= dt_24h_ago.strftime("%Y-%m-%d %H:%M:%S"):
            cell = ev.get("grid_cell", "")
            recent_events_by_cell.setdefault(cell, []).append({
                "lat": ev["lat"], "lng": ev["lng"],
                "address": ev.get("address", ""), "create_dt": ev.get("create_dt", ""),
                "report_count": ev.get("report_count", 1),
            })

    cells_output = []
    for _, row in features_df.iterrows():
        cell_id = row["grid_cell"]
        parts = cell_id.split("_")
        lat = float(parts[0]) if len(parts) == 2 else 0
        lng = float(parts[1]) if len(parts) == 2 else 0
        cell_out = {"cell": cell_id, "lat": lat, "lng": lng, "district": row.get("district", ""),
                    "region": row.get("region", ""), "probability": round(row["probability"], 4), "risk": row["risk"]}
        if row["risk"] == "high" and cell_id in recent_events_by_cell:
            cell_out["recent_events"] = recent_events_by_cell[cell_id][:5]
        cells_output.append(cell_out)

    forecast = {"generated_at": now.isoformat(), "forecast_window": f"{hours}h", "cells": cells_output}
    with open(FORECAST_PATH, "w", encoding="utf-8") as f:
        json.dump(forecast, f, ensure_ascii=False, indent=2)
    high_count = sum(1 for c in cells_output if c["risk"] == "high")
    log.info("Forecast saved: %d cells, %d high-risk", len(cells_output), high_count)


def cmd_stats(args):
    db = GhostDB(DB_PATH)
    sighting_count = db.execute("SELECT COUNT(*) FROM sightings").fetchone()[0]
    event_count = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    days = db.count_days_collected()
    cycles = db.execute("SELECT COUNT(*) FROM poll_cycles").fetchone()[0]
    print(f"Ghost Sweep \u2014 Data Statistics")
    print(f"{'\u2500' * 40}")
    print(f"Days collected:   {days}")
    print(f"Poll cycles:      {cycles}")
    print(f"Raw sightings:    {sighting_count}")
    print(f"Cleaned events:   {event_count}")
    print(f"Model exists:     {'Yes' if Path(MODEL_PATH).exists() else 'No'}")
    model = GhostModel()
    gate = model.check_data_gate(days)
    if gate["ready"]:
        print("Training gate:    READY")
    else:
        print(f"Training gate:    {gate['days_remaining']} more days needed")
    db.close()


def cmd_districts(args):
    db = GhostDB(DB_PATH)
    rows = db.execute("SELECT district, region, COUNT(*) as c FROM events WHERE district!='' GROUP BY district, region ORDER BY c DESC").fetchall()
    db.close()
    if not rows:
        print("No events yet.")
        return
    print(f"{'District':<20} {'Region':<25} {'Events':>8}")
    print(f"{'\u2500'*20} {'\u2500'*25} {'\u2500'*8}")
    for row in rows:
        print(f"{row[0]:<20} {row[1]:<25} {row[2]:>8}")


def cmd_migrate(args):
    db = GhostDB(DB_PATH)
    count = db.migrate_from_json(args.json_file)
    log.info("Migrated %d alerts from %s", count, args.json_file)
    db.close()


def main():
    parser = argparse.ArgumentParser(description="Ghost Sweep Prediction CLI")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("clean", help="Run event consolidation")
    sub.add_parser("train", help="Train model")
    fp = sub.add_parser("forecast", help="Generate forecast")
    fp.add_argument("--hours", type=int, default=1)
    sub.add_parser("stats", help="Show statistics")
    sub.add_parser("districts", help="Per-district summary")
    mp = sub.add_parser("migrate", help="Import JSON to SQLite")
    mp.add_argument("json_file", nargs="?", default="ghost_alerts.json")

    args = parser.parse_args()
    cmds = {"clean": cmd_clean, "train": cmd_train, "forecast": cmd_forecast,
            "stats": cmd_stats, "districts": cmd_districts, "migrate": cmd_migrate}
    if args.command in cmds:
        cmds[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
