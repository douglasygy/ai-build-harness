"""Per-project waiver list for the reviewer (2026-09-29, Douglas's direct requirement, relayed via
video-generator-78 - urgent, real incident: a real, already-decided waiver (a symlink-escape edge
case in disk_storage.py, explicitly declined by Douglas) kept coming back as a must_fix finding
every single round, burning a whole run's budget re-litigating something already settled).

Distinct from severity (pipeline/nodes/reviewer_node.py's must_fix/should_fix/nitpick): severity is
about how bad a NEW finding is; a waiver is about a SPECIFIC finding Douglas has ALREADY,
explicitly, permanently decided doesn't need fixing for THIS project - a genuinely real finding at
whatever severity the reviewer would otherwise assign, deliberately accepted as-is. A waived
finding must never come back, regardless of severity.

Only Agent 1 (or a peer session acting on Douglas's own explicit say-so) ever calls add_waiver -
never the reviewer/builder/architect roles themselves, which would let a model quietly suppress its
own inconvenient findings. Every waiver is also durably logged via record_decision, the same as any
other decision Douglas makes.

Matching is done by the REVIEWER ITSELF, not a mechanical text/substring comparison - a waiver's
own wording will never exactly match how a future round happens to phrase the same underlying
issue, and fuzzy/substring matching risks silently waiving something that only superficially
resembles it. The reviewer is shown every waiver's id/text/reason and told plainly not to report
a match at any severity; if it reports one anyway, tagging it with a real waiver id gets it
mechanically dropped regardless (never trusting the instruction alone - same "mechanical check,
not self-report" principle as everything else in this pipeline). A kept problem that was NOT
tagged but whose text substantially overlaps a waiver's own wording is not auto-dropped (a
mechanical text match could waive something that only superficially resembles the real waiver) -
it's only flagged to stderr, so a human can notice a possible miss without risking a false drop.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .decision_log import record_decision

WAIVERS_FILENAME = ".review_waivers.json"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _waivers_path(project_dir) -> Path:
    return Path(project_dir) / WAIVERS_FILENAME


def _slugify(text: str, existing_ids: set) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")[:40] or "waiver"
    if slug not in existing_ids:
        return slug
    n = 2
    while f"{slug}-{n}" in existing_ids:
        n += 1
    return f"{slug}-{n}"


def load_waivers(project_dir) -> list:
    path = _waivers_path(project_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def add_waiver(
    project_dir, text: str, reason: str, waived_by: str = "Douglas",
    id: Optional[str] = None, date: Optional[str] = None,
) -> dict:
    """Appends one waiver and returns the new entry (including its id, auto-generated from `text`
    as a short slug if not given explicitly). Refuses a duplicate `id`. Always durably logged via
    record_decision, same as any other decision Douglas makes - this is a real product decision
    (accept a known limitation as-is), not an implementation detail."""
    waivers = load_waivers(project_dir)
    existing_ids = {w.get("id") for w in waivers}
    if id and id in existing_ids:
        raise ValueError(f"waiver id '{id}' already exists for this project - use a different id or edit it directly")
    entry = {
        "id": id or _slugify(text, existing_ids),
        "text": text,
        "reason": reason,
        "waived_by": waived_by,
        "date": date or datetime.now(timezone.utc).date().isoformat(),
    }
    waivers.append(entry)
    _waivers_path(project_dir).write_text(json.dumps(waivers, indent=2), encoding="utf-8")
    record_decision(
        decision_point=f"Waive reviewer finding for project {Path(project_dir).name}: {text}",
        choice=f"Accepted as a known limitation (waiver id '{entry['id']}') - will not be reported again, at any severity.",
        project=Path(project_dir).name,
        rationale=reason,
        proposed_by=waived_by,
    )
    return entry


def format_waivers_for_prompt(waivers: list) -> str:
    """'' if no waivers - a caller can unconditionally splice this into a prompt without an extra
    empty-check."""
    if not waivers:
        return ""
    lines = [f"- id \"{w.get('id', '?')}\": {w.get('text', '')} (reason: {w.get('reason', '')})" for w in waivers]
    return (
        "DOUGLAS HAS EXPLICITLY ACCEPTED THESE KNOWN LIMITATIONS FOR THIS PROJECT - do not report a "
        "problem that matches one of these, at ANY severity, even if you still find it technically "
        "true. If you are about to report something and realize it matches one of these, simply do "
        "not report it. If you believe a DIFFERENT, genuinely new problem is related to one of these "
        "but is not actually the same thing, report it normally - waivers cover only the exact "
        "accepted limitation below, not everything nearby it:\n" + "\n".join(lines)
    )


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Add or list this project's reviewer waivers.")
    parser.add_argument("action", choices=["add", "list"])
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--text", help="The finding's own text, verbatim or close to it (required for 'add')")
    parser.add_argument("--reason", help="Why Douglas is waiving this (required for 'add')")
    parser.add_argument("--waived-by", default="Douglas")
    parser.add_argument("--id", help="Explicit waiver id (default: auto-generated slug from --text)")
    args = parser.parse_args()

    if args.action == "list":
        print(json.dumps(load_waivers(args.project_dir), indent=2))
        return 0

    if not args.text or not args.reason:
        parser.error("add requires --text and --reason")
    entry = add_waiver(args.project_dir, args.text, args.reason, waived_by=args.waived_by, id=args.id)
    print(json.dumps(entry, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
