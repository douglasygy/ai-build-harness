"""Which of an Architect-approved system_shape's n8n workflows a given build/deploy invocation
targets - the mechanical fix for a real framework gap (reported by a peer session, confirmed
2026-09-26): review_architect.py's schema has always allowed `system_shape.workflows` to list
several SAME-PLATFORM n8n workflows, but every n8n build/deploy path hardcoded a single
`code/n8n/workflow.json` - so Architect could design 3 n8n workflows and the builder would
silently build one, overwriting whichever node happened to write last, with no error anywhere.

Douglas's explicit direction (2026-09-26): build real multi-workflow n8n support, not just a guard
- his n8n projects will genuinely need more than one workflow to cover an end-to-end client
process. This module is the single source of truth for "which workflow, which folder" so that
answer is computed once and threaded through consistently, rather than re-derived per call site.

The legacy shape (code/n8n/workflow.json directly, no subfolder) is preserved EXACTLY for every
project with zero or one n8n-platform workflow in its Architect decision - which is every existing
project today, and any future project that only ever needed one n8n workflow. Nothing changes for
that case no matter what flags are passed. The nested shape (code/n8n/<slug>/workflow.json) only
ever applies once a project's own approved Architect decision genuinely names more than one.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .scale_roster import architect_decisions_path


def slugify(name: str) -> str:
    """Filesystem/CLI-safe form of a workflow name - also what Douglas types after --workflow."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return slug or "workflow"


def n8n_workflows(project_dir) -> list:
    """Every system_shape workflow entry whose platform is n8n, from this project's approved
    Architect decision - empty if there's no Architect decision file, no system_shape, or none of
    its workflows are n8n. Never raises: a malformed/missing file degrades to "no constraint",
    since validating that file's own shape is Architect approval's job, not this function's.
    """
    path = architect_decisions_path(project_dir)
    if not path.exists():
        return []
    try:
        decisions = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(decisions, dict):
        return []
    workflows = ((decisions.get("system_shape") or {}).get("workflows")) or []
    return [w for w in workflows if isinstance(w, dict) and w.get("platform") == "n8n" and w.get("name")]


def resolve_workflow_slug(project_dir, workflow_arg: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Returns (slug, error) - exactly one is not None.

    slug is None (use the legacy flat code/n8n/workflow.json path, completely unchanged) whenever
    this project's Architect decision names at most one n8n workflow - regardless of whether
    --workflow was passed, so an extra/harmless flag on a single-workflow project never changes
    anything. Once it names more than one, --workflow becomes required: omitting it, or passing one
    that doesn't match any named n8n workflow, is a clear, named error - never a guess at which one
    was meant.
    """
    workflows = n8n_workflows(project_dir)
    if len(workflows) <= 1:
        return None, None

    names = [w["name"] for w in workflows]
    if not workflow_arg:
        return None, (
            f"This project's approved Architect decision specifies {len(workflows)} n8n workflows "
            f"({', '.join(names)}) - the n8n builder produces one workflow per invocation, so pass "
            f"--workflow <name> to say which one this build targets."
        )
    for w in workflows:
        if w["name"] == workflow_arg or slugify(w["name"]) == slugify(workflow_arg):
            return slugify(w["name"]), None
    return None, (
        f"--workflow {workflow_arg!r} does not match any of this project's approved n8n workflows: "
        f"{', '.join(names)}."
    )


def n8n_subdir(code_dir, workflow: Optional[str]) -> Path:
    """code/n8n (legacy, every existing project) when workflow is falsy, else code/n8n/<slug> -
    the one place this fork happens, so every path-constructing call site stays a one-line change."""
    base = Path(code_dir) / "n8n"
    return base / workflow if workflow else base
