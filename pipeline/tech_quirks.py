"""Repo-native, cross-project shared technical-knowledge ledger (2026-09-28, Douglas's direct
instruction, correcting Agent 1's own first proposal - a per-PROJECT file - after seeing a real
transcript: "this whole harness cannot be running solo, everything it reads and learns in the
process should follow through the build. Not reread and relearn each build run.").

Distinct from decision_log.jsonl (Douglas's OWN decisions about the pipeline) and build_log.jsonl
(what happened in one run) - this is technical lessons ABOUT THE PLATFORMS/LIBRARIES this pipeline
builds WITH (a LangGraph API quirk, an n8n node's real behavior, a vendor API's undocumented
requirement), the kind of thing a human dev team puts in a shared wiki so nobody re-discovers the
same gotcha from scratch. Real evidence this was needed: creative-team was 9 real build invocations
deep, across multiple stages and retries, and a real LangGraph fact (Runtime has no .config) had to
be manually rediscovered and hand-written into SPEC.md by a human/peer session standing in for
exactly this feature - nothing any attempt ever learned survived past the invocation that learned it.

Deliberately CROSS-PROJECT (memory/, not a per-project file, unlike .architect_decisions.json/
.pm_epics.json): Douglas's own examples (LangGraph's SqliteSaver, n8n's Slack node encoding) are
facts about the shared TOOLS this pipeline uses on every project, not one project's own business
logic - a fact worth keeping past the project that discovered it.

Read AND written by every role that can plausibly discover something reusable this way - Douglas's
explicit instruction: "any agent should be able to read and write it... if architect deems something
important, it writes into it, same for review, or even concept" (plus his own original proposal
naming Builder). Wired into: review_concept.py (concept_advocate/business_analyst),
review_architect.py (architect), builder_node.py (both the python-agentic and n8n paths),
reviewer_node.py. NOT wired into pm/team_composition/build_scribe/preference_analyst - those roles
reason about THIS pipeline's own process and Douglas's preferences, not about the target codebase's
technical platforms, so a technical quirks ledger isn't the right shape for what they do.

Real risk this module is deliberately built around, not blind to: a wrong "fact" written once would
be shown to every future build on every project, forever, unless corrected by hand - a much longer
blast radius than any other self-report this pipeline already treats with suspicion (builder_account,
a reviewer's own verdict). Two mitigations, both deliberately lightweight (Douglas's own explicit
"without overcomplicating your architecture"): every entry is attributed to whoever wrote it
(with_tech_quirks's own framing tells every reader - human or model - to treat an entry as an
unverified claim, not verified ground truth, the same posture already used for builder_account), and
nothing here re-verifies a claim before accepting it - correcting a bad entry is a human editing this
one plain-text file, the same as decision-patterns.md already works.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

REPO_ROOT = Path(__file__).resolve().parent.parent
TECH_QUIRKS_PATH = REPO_ROOT / "memory" / "tech-quirks.md"

_HEADER = (
    "# Tech Quirks\n\n"
    "Shared technical lessons about the platforms/libraries this pipeline builds with - "
    "cross-project, appended to by any role (concept, architect, builder, reviewer) that confirms "
    "something reusable. Each entry is that role's own claim at the time it was written, attributed "
    "below it - never independently re-verified before being shown to a later run. A human corrects "
    "or removes a wrong entry by editing this file directly, the same as decision-patterns.md.\n\n"
)


def load_tech_quirks() -> str:
    """Full current file content, or '' if nothing has been recorded yet."""
    if not TECH_QUIRKS_PATH.exists():
        return ""
    return TECH_QUIRKS_PATH.read_text(encoding="utf-8")


def with_tech_quirks(system_prompt: str) -> str:
    """Prepends the current tech-quirks.md content (if any) to a role's own static system prompt,
    clearly marked as unverified claims - not automatic ground truth, the same posture this pipeline
    already applies to builder_account. Call ONCE per run/attempt (not once per turn/call within a
    loop already in progress), so the result is a stable, cacheable prefix across every call that run
    makes - the same "static content first" structure every role's prompt already uses.

    Returns system_prompt completely unchanged if nothing has been recorded yet, so a role with no
    quirks on file (or before this file exists at all) behaves exactly as it did before this existed.
    """
    quirks = load_tech_quirks()
    if not quirks.strip():
        return system_prompt
    return (
        "SHARED TECHNICAL KNOWLEDGE (memory/tech-quirks.md) - lessons other roles or earlier runs "
        "recorded about the libraries/platforms this project uses. Each line below is an UNVERIFIED "
        "CLAIM from whoever wrote it, worth checking against reality if it's actually relevant to "
        "what you're doing now - never assume it's still accurate without a reason to think so:\n\n"
        + quirks.strip() + "\n\n---\n\n" + system_prompt
    )


def record_tech_quirk_entry(lesson: str, source: str) -> None:
    """The mechanical writer every path funnels through (called directly by single-shot roles after
    parsing their own structured JSON output; wrapped as a tool for the agentic builder below).
    Silently does nothing on an empty/whitespace lesson - callers should not call this with nothing
    to say, but this is the one shared choke point, so it's the one place that actually enforces it.
    """
    lesson = lesson.strip()
    if not lesson:
        return
    TECH_QUIRKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not TECH_QUIRKS_PATH.exists():
        TECH_QUIRKS_PATH.write_text(_HEADER, encoding="utf-8")
    date = datetime.now(timezone.utc).date().isoformat()
    with open(TECH_QUIRKS_PATH, "a", encoding="utf-8") as f:
        f.write(f"- {lesson} ({source}, {date})\n")


def make_tech_quirk_tool(source: str):
    """Returns a LangChain tool bound to a fixed `source` label via closure, for the agentic python
    builder only (the one role with a real tool-calling loop) - every entry it writes is
    automatically attributed to whichever attempt's agent loop is holding it. The model itself never
    supplies (or can spoof) the source."""

    @tool
    def record_tech_quirk(lesson: str) -> str:
        """Record a genuinely reusable technical lesson about a LIBRARY, FRAMEWORK, or PLATFORM this
        project depends on - NOT this project's own business logic. The kind of gotcha a human
        developer would put in a shared team wiki so nobody re-discovers it the hard way: a specific
        API's real behavior, a version-specific quirk, an undocumented requirement you had to work
        out by reading source or by trial and error. Every future build on ANY project using this
        pipeline will see this, so only call it for something you've actually confirmed (a real
        error you hit and fixed, a real behavior you verified by running something) - never a guess
        or something you're still unsure about. Shape: '<Topic>: <the gotcha and what to do
        instead>' - one specific, self-contained fact, not a running commentary on this build."""
        if not lesson.strip():
            return "ERROR: empty lesson - nothing recorded."
        record_tech_quirk_entry(lesson, source)
        return "Recorded to the shared tech-quirks ledger (memory/tech-quirks.md) - every future build on this pipeline will see this."

    return record_tech_quirk
