"""Per-project team roster sidecar: records which additional agents (beyond the permanent core
four) Douglas has approved for THIS project specifically - never the global roster. See AGENTS.md
"Team composition gate (PM + Architect)" and pipeline/review_team_composition.py, which produces
the proposal this records the approved subset of.

This file only records WHAT was approved, for auditability and so a resumed session doesn't need to
re-ask. It is not itself a model-config source - an approved addition's actual model still goes into
the project's own models.yaml under `roles`, the same file every other per-project override lives
in, so config_loader.load_role() picks it up with no special-casing for team-composition roles.
"""
from __future__ import annotations

import json
from pathlib import Path


def _roster_path(project_dir) -> Path:
    return Path(project_dir) / ".team_roster.json"


def save_team_roster(project_dir, approved_agents: list) -> None:
    """`approved_agents` is a list of {"role_name", "reason", "recommended_model", "approved_on"}
    entries - the subset of a review_team_composition.py proposal Douglas actually approved.
    """
    path = _roster_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(approved_agents, indent=2), encoding="utf-8")


def load_team_roster(project_dir) -> list:
    path = _roster_path(project_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []
