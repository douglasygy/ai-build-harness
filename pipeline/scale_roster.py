"""Per-project scale-adaptive roster marker: durably records which of PM/Architect Douglas has
decided are part of THIS project - the thing that was previously only ever a conversation, never
written down anywhere a script could check. See AGENTS.md "Wiring PM/Architect into the build".

Distinct from pipeline/team_roster.py: that module records the OUTCOME of the team-composition
bounce (additional agents PM/Architect themselves proposed and Douglas approved - DevOps, a second
builder, a second reviewer). This one is simpler and one level up - whether PM and/or Architect
exist for this project at all, decided by Douglas directly, never by a bounce (they can't propose
their own existence).

Written by Agent 1 (Claude Code) directly, at session start when Douglas decides the roster - same
manual-write-after-human-decision pattern as SPEC.md itself. This file's only purpose is to let
pipeline/build_and_review.py enforce that an approved PM/Architect decision file actually exists
before a build proceeds, for any role this project has opted into; it is never itself a source of
model configuration (that's still the project's own models.yaml, as everywhere else).
"""
from __future__ import annotations

import json
from pathlib import Path

ROSTER_FILENAME = ".scale_roster.json"
PM_EPICS_FILENAME = ".pm_epics.json"
ARCHITECT_DECISIONS_FILENAME = ".architect_decisions.json"


def roster_path(project_dir) -> Path:
    return Path(project_dir) / ROSTER_FILENAME


def load_roster(project_dir) -> dict:
    path = roster_path(project_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_roster(project_dir, roster: dict) -> None:
    path = roster_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(roster, indent=2), encoding="utf-8")


def pm_epics_path(project_dir) -> Path:
    return Path(project_dir) / PM_EPICS_FILENAME


def architect_decisions_path(project_dir) -> Path:
    return Path(project_dir) / ARCHITECT_DECISIONS_FILENAME


def load_python_build_units(project_dir) -> list:
    """The first "python"-platform workflow's own build_units from .architect_decisions.json
    (2026-09-28, Douglas's direct instruction - replaces PM-epic/checkpoint stage gating for the
    python path). [] if there's no Architect decision, no python workflow in it, or it declared
    none - all of which mean "this project doesn't use unit decomposition," not an error, matching
    the scale-adaptive philosophy (a small project needs none of this machinery). Shared by
    builder_node.py (actually building the units) and build_and_review.py (resolving --until-unit
    and reporting how many units exist) so neither drifts from the other's idea of what's declared.

    Known simplification, not silently pretended otherwise: a hybrid system with more than one
    python-platform workflow isn't fully supported by this pipeline yet regardless (there's no
    python equivalent of n8n's own --workflow selector today) - this picks the FIRST python
    workflow it finds, the same limitation that already exists elsewhere, not one introduced here.
    """
    if not project_dir:
        return []
    path = architect_decisions_path(project_dir)
    if not path.exists():
        return []
    try:
        decisions = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    workflows = ((decisions.get("system_shape") or {}).get("workflows")) or []
    for wf in workflows:
        if isinstance(wf, dict) and wf.get("platform") == "python":
            return wf.get("build_units") or []
    return []


def missing_prerequisites(project_dir) -> list:
    """Returns a list of human-readable problems if this project's roster requires an approved
    PM/Architect decision file that doesn't exist yet - empty if there's nothing to enforce (the
    default: no roster file, or a roster with neither role marked True).
    """
    roster = load_roster(project_dir)
    problems = []
    if roster.get("pm") and not pm_epics_path(project_dir).exists():
        problems.append(
            f"This project's roster includes PM, but {pm_epics_path(project_dir)} does not exist yet. "
            f"Run `python -m pipeline.review_pm --spec <spec> --project-dir {project_dir}`, get Douglas's "
            f"approval, and write the approved result there before building."
        )
    if roster.get("architect") and not architect_decisions_path(project_dir).exists():
        problems.append(
            f"This project's roster includes Architect, but {architect_decisions_path(project_dir)} does "
            f"not exist yet. Run `python -m pipeline.review_architect --spec <spec> --project-dir "
            f"{project_dir}`, get Douglas's approval, and write the approved result there before building."
        )
    return problems
