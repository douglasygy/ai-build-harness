"""Build-scoped journal (2026-09-28, Douglas's firm requirement - his own words, verbatim: "FIX THE
PART 2, FORGETTING ITS OWN LAST TRY - DOES THIS EVEN SOUND ACCEPTABLE TO YOU?! CMON!"). Real,
non-hypothetical damage this exists to prevent: a retry with no memory of what it already tried
invented storage_memory.py instead of fixing the file that actually needed fixing - and this happens
on EVERY retry by construction, not an occasional edge case.

Distinct from tech_quirks.md (pipeline/tech_quirks.py) - that is cross-project TECHNICAL facts about
the libraries/platforms this pipeline builds with; this is "what THIS build itself already decided or
tried," scoped to one project, one build. Lives inside the project directory itself (not memory/),
and survives a checkpoint resume - the same "a resume never loses real progress" principle already
applied everywhere else today.

Douglas's own correction to a first, narrower proposal (carry a summary forward to only the next
attempt): "This bring forward memory must last through the build. Not just to next attempt. Be smart
about this one." Concretely: available to every LATER step of the SAME build, not just the
immediate next attempt - a later unit should be able to see a decision an earlier one made; a later
reviewer-round retry should be able to see what an earlier round already tried.

Populated MECHANICALLY, never via a new LLM call - every entry is either the builder's own
already-produced builder_account text, or a mechanical summary of the reviewer's own
already-produced verdict/kept-problems-count - reusing output these roles already generate for other
reasons, never asking anyone to additionally write a journal entry. Every entry is labeled and
treated as an UNVERIFIED SELF-REPORT, the exact same trust posture already applied to
builder_account (see reviewer_node.py's own BUILDER'S OWN ACCOUNT section) - never ground truth.

Bounded by design, deliberately: load_relevant_entries never returns the WHOLE journal - only
entries belonging to the given unit itself, any unit named in its own depends_on, and reviewer-round
entries (always relevant, since a round's feedback isn't scoped to one unit) - never the whole
build's full accumulated log unconditionally, which would grow forever as the build progresses and
reintroduce the exact compounding-resend problem this same day's context-pruning fix
(_prune_stale_exploration_results) already closed, one level up.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_JOURNAL_FILENAME = ".build_journal.jsonl"

# A reviewer round doesn't belong to one specific unit (the reviewer looks at the whole codebase,
# not one unit at a time) - entries tagged with this sentinel are always included by
# load_relevant_entries, regardless of which unit is asking.
REVIEW_ROUND_UNIT = "__review_round__"

# A single misbehaving entry (a model writing an essay instead of "a few sentences") must not blow
# up what every later step gets shown - truncated, not rejected, since a long-but-real account is
# still more useful than none.
_MAX_SUMMARY_CHARS = 800


def _journal_path(project_dir) -> Path:
    return Path(project_dir) / _JOURNAL_FILENAME


def record_journal_entry(project_dir, unit: str, role: str, summary: str) -> None:
    """Appends one entry. `unit` is the build_unit name this concerns (or REVIEW_ROUND_UNIT for a
    reviewer round's own entry). `role` is who produced the underlying text ("builder", "reviewer").
    `summary` must already exist (builder_account, or a mechanical reviewer-verdict summary) - this
    function never generates text, only records it. Silently does nothing on an empty/whitespace
    summary - no entry worth keeping, and callers should not need to special-case "nothing to say"
    themselves before calling this."""
    summary = (summary or "").strip()
    if not summary:
        return
    if len(summary) > _MAX_SUMMARY_CHARS:
        summary = summary[:_MAX_SUMMARY_CHARS] + "... (truncated)"
    path = _journal_path(project_dir)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "unit": unit,
        "role": role,
        "summary": summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def clear_journal(project_dir) -> bool:
    """Deletes this project's journal entirely, for a genuine from-scratch python rebuild (see
    pipeline/nodes/builder_node.py's wipe_python_code, called alongside this from
    build_and_review.py's --fresh-python) - if code_dir gets wiped but the journal survives, a later
    unit would be handed stale entries referencing units that "were built" and code that no longer
    exists. Returns whether a file was actually there to remove."""
    path = _journal_path(project_dir)
    if path.exists():
        path.unlink()
        return True
    return False


def load_all_entries(project_dir) -> list:
    """Every entry ever recorded for this project - callers wanting a bounded, relevant subset
    should use load_relevant_entries below instead; this is exposed mainly for tooling/debugging."""
    path = _journal_path(project_dir)
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def load_relevant_entries(project_dir, unit: str, depends_on: list | None = None) -> list:
    """Only entries for `unit` itself, any unit named in `depends_on`, or a reviewer round
    (REVIEW_ROUND_UNIT, always relevant) - see the module docstring for why this must never be the
    unfiltered whole-journal log."""
    relevant_units = {unit, REVIEW_ROUND_UNIT, *(depends_on or [])}
    return [e for e in load_all_entries(project_dir) if e.get("unit") in relevant_units]


def format_journal_for_prompt(entries: list) -> str:
    """'' if no entries (the common case for an early unit with nothing built before it) - so a
    caller can unconditionally splice this into a prompt without an extra empty-check. Otherwise a
    labeled section framed as an UNVERIFIED self-report, the same trust posture builder_account
    already gets in reviewer_node.py - never presented as ground truth.

    Framing strengthened 2026-09-29 (real incident: a peer's own precisely diagnosed, verified fix
    for image_team_subgraph - remove a specific .compile() call - sat in the journal, correctly
    tagged, and still didn't land across 3 dedicated attempts; also moved earlier in the prompt by
    _format_unit_task, see that function's own comment). The OLD framing read as passive background
    context ("use it... if convenient"); this version says directly that an entry is often the exact
    reason a PRIOR attempt on this same unit failed, without weakening the "unverified, don't just
    trust it" caution - a journal entry can still be a builder's own wrong self-report, and remains
    labeled as exactly that."""
    if not entries:
        return ""
    lines = [f"- [{e.get('unit', '?')}, {e.get('role', '?')}] {e.get('summary', '')}" for e in entries]
    return (
        "BUILD JOURNAL - READ THIS FIRST (findings from earlier attempts/rounds on THIS unit or its "
        "dependencies - often the exact reason a prior attempt failed; engage with each entry "
        "directly rather than re-deriving from scratch. Each entry is that role's OWN UNVERIFIED "
        "account of what it did or found, not independently re-verified ground truth - verify "
        "anything you actually rely on before depending on it):\n" + "\n".join(lines)
    )
