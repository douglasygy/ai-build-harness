"""Externally verified findings injected into a build (trading-team, 2026-09-30).

A human or Agent 1 finding (for example from a concept-conformance check) has no other way into the
pipeline. Stored append-only in <project>/.external_findings.jsonl:
  {"id", "timestamp", "source", "unit", "text"}                       - a finding
  {"resolved": id, "timestamp", "by", "evidence"}                     - its resolution
A finding stays OPEN until the reviewer confirms it fixed (with a verbatim code quote, checked
mechanically) or Douglas dismisses it; reviewer silence never clears it.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

FILENAME = ".external_findings.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(project_dir) -> list:
    try:
        lines = (Path(project_dir) / FILENAME).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict):
            out.append(e)
    return out


def _append(project_dir, entry: dict) -> None:
    with open(Path(project_dir) / FILENAME, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def record_finding(project_dir, text: str, source: str = "Douglas", unit: str = "") -> dict:
    text = text.strip()
    entry = {
        "id": "ext-" + hashlib.sha1(text.lower().encode("utf-8")).hexdigest()[:8],
        "timestamp": _now(),
        "source": (source or "Douglas").strip(),
        "unit": (unit or "").strip(),
        "text": text,
    }
    if not any(e.get("id") == entry["id"] and "text" in e for e in _read(project_dir)):
        _append(project_dir, entry)
    return entry


def resolve_finding(project_dir, finding_id: str, by: str, evidence: str = "") -> None:
    _append(project_dir, {"resolved": finding_id, "timestamp": _now(), "by": by, "evidence": evidence})


def load_open_findings(project_dir) -> list:
    entries = _read(project_dir)
    resolved = {e["resolved"] for e in entries if "resolved" in e}
    seen, out = set(), []
    for e in entries:
        if "text" in e and e.get("id") not in resolved and e["id"] not in seen:
            seen.add(e["id"])
            out.append(e)
    return out


def dismiss_matching(project_dir, substrings: list, by: str = "Douglas") -> list:
    """Resolve every open finding whose id equals, or whose text contains, one of the needles;
    returns their ids. Use unmatched_needles() to report needles that hit nothing."""
    needles = [s.lower().strip() for s in substrings if s and s.strip()]
    hit = [
        e["id"] for e in load_open_findings(project_dir)
        if any(n == e["id"].lower() or n in e["text"].lower() for n in needles)
    ]
    for fid in hit:
        resolve_finding(project_dir, fid, by, "dismissed")
    return hit


def unmatched_needles(open_before: list, substrings: list) -> list:
    """Needles that match no finding in `open_before` (a load_open_findings snapshot)."""
    out = []
    for s in substrings:
        n = (s or "").lower().strip()
        if n and not any(n == e["id"].lower() or n in e["text"].lower() for e in open_before):
            out.append(s)
    return out


def finding_line(entry: dict) -> str:
    """The 'node: description' shape every downstream consumer expects."""
    text = entry["text"]
    return f"{entry['unit']}: {text}" if entry.get("unit") else text
