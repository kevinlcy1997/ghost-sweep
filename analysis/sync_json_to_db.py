"""Sync ghost_alerts.json into ghost_alerts.db using the project DB layer."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghost_db import GhostDB

JSON_PATH = ROOT / "ghost_alerts.json"
DB_PATH = ROOT / "ghost_alerts.db"
VERIFY_PATH = ROOT / "analysis" / "sync_verify.txt"


def _find_matching_brace(text: str, open_pos: int) -> int:
    depth = 0
    in_string = False
    escaped = False

    for idx in range(open_pos, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx

    raise ValueError("No matching closing brace found")


def salvage_alert_records(raw_json: str) -> list[dict]:
    """Extract complete alert objects from a damaged ghost_alerts.json file."""
    alerts_match = re.search(r'"alerts"\s*:\s*{', raw_json)
    if not alerts_match:
        return []

    pos = alerts_match.end()
    records: list[dict] = []
    decoder = json.JSONDecoder(strict=False)
    key_pattern = re.compile(r'\s*,?\s*"([^"]+)"\s*:\s*{', re.DOTALL)

    while True:
        match = key_pattern.match(raw_json, pos)
        if not match:
            break
        obj_start = match.end() - 1
        try:
            obj_end = _find_matching_brace(raw_json, obj_start)
            record = decoder.decode(raw_json[obj_start : obj_end + 1])
        except (ValueError, json.JSONDecodeError):
            break
        if isinstance(record, dict):
            record.setdefault("alert_record_id", match.group(1))
            records.append(record)
        pos = obj_end + 1

    return records


def main() -> None:
    raw_json = JSON_PATH.read_text(encoding="utf-8")
    loader = "json"
    try:
        decoder = json.JSONDecoder(strict=False)
        data, end_pos = decoder.raw_decode(raw_json)
        alerts = data["alerts"]
        records = list(alerts.values()) if isinstance(alerts, dict) else alerts
    except json.JSONDecodeError as exc:
        loader = f"salvage_after_decode_error_line_{exc.lineno}"
        records = salvage_alert_records(raw_json)
        end_pos = exc.pos
    if not records:
        raise RuntimeError("No alert records found in ghost_alerts.json")

    db = GhostDB(str(DB_PATH))
    inserted = db.insert_sightings(records)
    cur = db._conn.cursor()

    sightings_count = cur.execute("SELECT COUNT(*) FROM sightings").fetchone()[0]
    events_count = cur.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    poll_cycles_count = cur.execute("SELECT COUNT(*) FROM poll_cycles").fetchone()[0]
    min_dt, max_dt, distinct_ids = cur.execute(
        "SELECT MIN(create_dt), MAX(create_dt), COUNT(DISTINCT alert_record_id) "
        "FROM sightings"
    ).fetchone()

    VERIFY_PATH.write_text(
        "\n".join(
            [
                f"json_records={len(records)}",
                f"json_loader={loader}",
                f"json_first_object_end_pos={end_pos}",
                f"json_trailing_chars_ignored={len(raw_json) - end_pos}",
                f"insert_method_return={inserted}",
                f"sightings={sightings_count}",
                f"events={events_count}",
                f"poll_cycles={poll_cycles_count}",
                f"distinct_alert_record_ids={distinct_ids}",
                f"min_create_dt={min_dt}",
                f"max_create_dt={max_dt}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
