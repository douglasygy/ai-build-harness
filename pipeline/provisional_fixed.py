"""Builder ALREADY_FIXED claims (verified quote) remembered across invocations, so a rerun does not
rebuild a unit only to re-claim the same fix. Consumed (cleared) when the reviewer judges them."""
from __future__ import annotations

import json
from pathlib import Path

FILENAME = ".provisional_fixed.json"


def load_provisional(project_dir) -> dict:
    try:
        data = json.loads((Path(project_dir) / FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_provisional(project_dir, claims: list) -> None:
    if not claims:
        return
    data = load_provisional(project_dir)
    for c in claims:
        data[c["id"]] = {"finding": c["finding"], "quote": c["quote"]}
    (Path(project_dir) / FILENAME).write_text(json.dumps(data, indent=2), encoding="utf-8")


def clear_provisional(project_dir) -> None:
    try:
        (Path(project_dir) / FILENAME).unlink()
    except OSError:
        pass
