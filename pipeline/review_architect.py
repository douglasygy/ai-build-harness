"""Architect's primary job: decide the things the plan phase currently has to assume on its own -
one workflow or several, where data lives, credential/security architecture, and a system-level
regression-testing strategy across multiple interacting workflows. See AGENTS.md "Scale-adaptive
roles: PM and Architect" and "Wiring PM/Architect into the build".

Single-shot, not a bounce - this is Architect's own structural judgment applied to an already-locked
spec, run once per project before the plan phase.

Usage:
    python -m pipeline.review_architect --spec path/to/SPEC.md [--pm-epics path/to/.pm_epics.json] [--project-dir <dir>]

Prints ONE JSON object to stdout:
    {"summary": "...", "system_shape": {"workflow_count": N, "workflows": [{"name": "...",
    "scope": "..."}]}, "data_placement": "...", "credential_architecture": "...",
    "interface_contracts": [{"between": "...", "payload_shape": "...", "discriminator_field": "...",
    "discriminator_values": ["..."]}], "regression_strategy": "..."}
or, on failure after retries: {"status": "error", "message": "..."}

"interface_contracts" exists because of a real incident: an n8n shell and its LangGraph service,
built in separate builder invocations with no shared pinned contract, each independently invented a
different vocabulary for the same response-type field ("prompt"/"post" vs "text"/"file") - both
halves built and tested clean in isolation, and every real reply silently vanished at runtime. Empty
(`[]`) only when the system genuinely has just one piece talking to nothing else.

Douglas approves this output the same way he approves everything else - nothing here writes it
anywhere. Once approved, Agent 1 (Claude Code) writes it verbatim to
<project-dir>/.architect_decisions.json, which pipeline/nodes/builder_node.py then folds into every
builder call as binding context - see _augment_spec_with_scale_adaptive_context.

Reconsideration (--previous-decision / --reconsideration-reason): a real incident (seo-blog-bot,
documented in that project's own .architect_decisions.json "history") showed a re-run asked to
"reconsider" a prior Architect decision can silently repeat the exact previous answer verbatim,
never engaging the new information at all - Douglas had to notice, restructure the spec to force a
second re-run, and only then got a real reconsideration. These two flags exist so re-review is never
a bare re-ask of the same question: they hand the model its own previous decision plus the specific
new reason to reconsider, AND mechanically check (see reconsideration_engaged) that the new response
actually differs and actually addresses that reason before accepting it - retrying with the failure
stated plainly if not, and surfacing (never silently passing through the stale answer) if it still
hasn't engaged after retrying.

--add-build-units-only (2026-09-28, Douglas's direct instruction): a specific reconsideration mode
for upgrading an EXISTING project's decision file (approved before build_units existed) to the
current schema, without re-litigating anything already approved. A full from-scratch Architect
re-run was considered and rejected for this: an LLM re-run has no memory of the prior reasoning and
could genuinely land on a different platform/architecture answer than what was already approved,
for no reason. Reuses the same --previous-decision/--reconsideration-reason mechanism above (a
sensible default reason is filled in if not given explicitly) plus one additional mechanical check
(only_build_units_changed) that reconsideration_engaged alone doesn't provide: not just that the
response changed and engaged the reason, but that NOTHING besides build_units/summary changed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .config_loader import ConfigError, load_role
from .constants import (
    MAX_AGENT_TURNS,
    MAX_INTERNAL_FIX_ATTEMPTS,
    MAX_TOKENS_PER_ATTEMPT,
    UNIT_MAX_EFFORT_POINTS,
    UNIT_MAX_EXPORTS,
    UNIT_MAX_FILES,
)
from .json_extract import extract_json_object
from .langfuse_setup import new_run_id, run_config
from .model_fallback import attempt_fallback, resolve_llm_for_role
from .retry import with_retries_nonempty
from .tech_quirks import record_tech_quirk_entry, with_tech_quirks
from .token_guard import extract_tokens

_WORD_RE = re.compile(r"[A-Za-z]{4,}")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def _significant_words(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text)}


def build_limits_sheet(project_dir=None) -> str:
    """Text listing every role's configured model and output cap plus the builder's per-unit
    budgets, read live from config so the Architect sizes units against real numbers."""
    lines = []
    for role in ("builder", "reviewer", "architect", "pm"):
        try:
            cfg = load_role(role, project_dir=project_dir)
            lines.append(f"- {role}: {cfg.model}, max output {cfg.max_tokens:,} tokens per response")
        except ConfigError:
            lines.append(f"- {role}: (not configured)")
    lines.append(
        f"- builder budget per attempt: at most {MAX_AGENT_TURNS} tool round trips (a turn may batch "
        f"several file writes) and {MAX_TOKENS_PER_ATTEMPT:,} total tokens, and every turn re-sends "
        "the whole conversation so cost grows each turn"
    )
    lines.append(
        f"- sprint capacity per unit: at most {UNIT_MAX_EFFORT_POINTS} effort points, "
        f"{UNIT_MAX_FILES} files, {UNIT_MAX_EXPORTS} exports"
    )
    return "\n".join(lines)


def validate_build_units(units: list) -> list[str]:
    """Mechanical (no LLM call) check that a build_units decomposition satisfies Douglas's two hard
    requirements (2026-09-28) - see SYSTEM_PROMPT's own instruction for the full framing. Returns a
    list of specific problems, empty if the whole decomposition is valid - same "propose a targeted
    fix, don't trust compliance" pattern as reconsideration_engaged above, applied to a different
    field.

    Cannot check REAL behavioral testability (no code exists yet at this stage, this runs before the
    plan phase even starts) - only the structural floor that's actually checkable from the
    architect's own declaration: a unit with no exports, or whose exports are ALL constants, has no
    behavior a real test could exercise regardless of size; a unit with no verification_hint hasn't
    stated what such a test would even check. Requirement (a)'s actual size/viability judgment still
    needs the architect's own judgment - this is the backstop against an obviously-too-small unit
    slipping through unchecked, not a substitute for that judgment.
    """
    problems: list[str] = []
    if not isinstance(units, list) or not units:
        return problems  # optional - a project/workflow that doesn't need this is not an error
    earlier_names: set = set()
    for i, unit in enumerate(units):
        if not isinstance(unit, dict):
            problems.append(f"build_units[{i}] is not an object")
            continue
        name = str(unit.get("name") or "").strip()
        if not name:
            problems.append(f"build_units[{i}] has no name")
            continue
        if name in earlier_names:
            problems.append(f"build_units[{i}] ('{name}') duplicates an earlier unit's name - every unit name must be unique")

        interface = unit.get("interface")
        exports = interface.get("exports") if isinstance(interface, dict) else None
        if not exports:
            problems.append(
                f"'{name}' has no interface.exports - a unit with nothing to export has no "
                "behavior a real test could exercise; merge it with a neighbor instead of building it alone"
            )
        elif all(isinstance(e, dict) and e.get("kind") == "constant" for e in exports):
            problems.append(
                f"'{name}'s exports are all constants - no real behavior to test; merge it with a "
                "neighbor that has actual logic"
            )

        files = unit.get("files") or []
        if isinstance(files, list) and len(files) > UNIT_MAX_FILES:
            problems.append(
                f"'{name}' lists {len(files)} files; a sprint-sized unit holds at most {UNIT_MAX_FILES} - "
                "split it into smaller progressive units"
            )
        if isinstance(exports, list) and len(exports) > UNIT_MAX_EXPORTS:
            problems.append(
                f"'{name}' declares {len(exports)} exports; a sprint-sized unit holds at most "
                f"{UNIT_MAX_EXPORTS} - split it into smaller progressive units"
            )
        points = unit.get("effort_points")
        if isinstance(points, bool) or not isinstance(points, int) or not 1 <= points <= UNIT_MAX_EFFORT_POINTS:
            problems.append(
                f"'{name}' needs an integer effort_points from 1 to {UNIT_MAX_EFFORT_POINTS} "
                f"(got {points!r}); work that would need more than {UNIT_MAX_EFFORT_POINTS} must be split"
            )

        if not str(unit.get("verification_hint") or "").strip():
            problems.append(
                f"'{name}' has no verification_hint - state what REAL behavior a test of this unit "
                "should check, not just \"does it parse\""
            )

        # Checked against earlier_names BEFORE adding this unit's own name below - a unit "depending
        # on" itself, or on a unit that comes AFTER it in this same list, must fail this check.
        for dep in unit.get("depends_on") or []:
            if dep not in earlier_names:
                problems.append(
                    f"'{name}' depends_on '{dep}', which is not an earlier unit in this list - "
                    "dependencies may only reference units that come before them"
                )

        earlier_names.add(name)

    # Walking skeleton (only enforced from three units up - a tiny build is its own integration).
    if len(units) >= 3:
        flagged = [i for i, u in enumerate(units) if isinstance(u, dict) and u.get("walking_skeleton") is True]
        if len(flagged) != 1 or flagged[0] > 2:
            problems.append(
                "exactly one of the FIRST THREE units must have \"walking_skeleton\": true - the quickest "
                "runnable end-to-end path (real entry point through every stage to real output, external "
                "boundary faked) so every later unit can be tested integrated, not only in isolation "
                f"(found flagged positions: {[i + 1 for i in flagged] or 'none'})"
            )
    return problems


def resolve_until_unit(units: list, until_unit) -> int:
    """Index of the last unit to build, from a name or 1-based number - same semantics as
    build_stages.py's retiring resolve_until_stage (None/""/"all" -> build every unit, an unmatched
    name/number also falls back to building everything rather than guessing). Lives here, not
    build_stages.py, since build_units is this module's own concept - build_stages.py's staging
    functions are being retired for the python path (2026-09-28, Douglas's direct instruction),
    replaced by this unit-scoped equivalent; n8n's own --until-slice is untouched."""
    if until_unit in (None, "", "all"):
        return len(units) - 1
    text = str(until_unit).strip()
    for i, unit in enumerate(units):
        if str(unit.get("name", "")).lower() == text.lower():
            return i
    try:
        n = int(text)
        if 1 <= n <= len(units):
            return n - 1
    except ValueError:
        pass
    return len(units) - 1


def reconsideration_engaged(previous: dict, new: dict, reason: str) -> tuple[bool, str]:
    """Mechanical (no LLM call) check that a reconsideration response actually engaged the new
    information, instead of trusting that asking "please reconsider" was enough - it wasn't, for
    real, once already (see module docstring). Returns (engaged, explanation-if-not).
    """
    prev_text = json.dumps(previous, sort_keys=True)
    new_text = json.dumps(new, sort_keys=True)
    if _normalize(prev_text) == _normalize(new_text):
        return False, "the new response is byte-for-byte identical to the previous decision - nothing was reconsidered at all"

    reason_words = _significant_words(reason) - _significant_words(prev_text)
    if reason_words:
        new_words = _significant_words(new_text)
        overlap = reason_words & new_words
        if not overlap:
            sample = sorted(reason_words)[:8]
            return False, (
                f"the new response never mentions any term from the stated reconsideration reason "
                f"({sample}) anywhere in its answer - it does not appear to have engaged the new "
                f"information at all"
            )
    return True, ""

def _strip_expected_changes(decision: dict) -> dict:
    """A deep copy of `decision` with build_units (per python workflow), total_tokens, and summary
    removed - the fields an "add build_units only" reconsideration is EXPECTED to change. Never
    mutates the original dict - both the previous and new decision may still be needed unchanged
    elsewhere (e.g. printed, or compared again)."""
    stripped = json.loads(json.dumps(decision))  # cheap, dependency-free deep copy
    stripped.pop("total_tokens", None)
    stripped.pop("summary", None)
    workflows = ((stripped.get("system_shape") or {}).get("workflows")) or []
    for wf in workflows:
        if isinstance(wf, dict):
            wf.pop("build_units", None)
    return stripped


def only_build_units_changed(previous: dict, new: dict) -> tuple[bool, str]:
    """Mechanical (no LLM call) check specifically for an "add build_units only" reconsideration
    (2026-09-28, Douglas's direct instruction, for upgrading an existing project's decision file to
    the build_units schema without re-litigating anything already approved).

    reconsideration_engaged above verifies the response CHANGED and ENGAGED the stated reason's own
    words - it does NOT verify that ONLY the intended field changed. Nothing stops a model from also
    quietly revising data_placement's wording, or the platform choice, even when asked only to add
    build_units. For an ordinary reconsideration that's fine (deciding what should change IS the
    point) - for this specific "add-only" mode it's a real risk worth closing mechanically, the same
    "don't trust a polite instruction alone" principle behind everything else built today, not a
    replacement for reconsideration_engaged but an ADDITIONAL check run alongside it.

    Strips build_units/total_tokens/summary from both sides (see _strip_expected_changes) and
    requires everything else to come back byte-identical. Returns (unchanged_elsewhere,
    explanation-if-not).
    """
    prev_stripped = json.dumps(_strip_expected_changes(previous), sort_keys=True)
    new_stripped = json.dumps(_strip_expected_changes(new), sort_keys=True)
    if prev_stripped != new_stripped:
        return False, (
            "something other than build_units/summary changed - this reconsideration is scoped to "
            "ONLY add build_units, not revise any other already-approved decision. Revert every "
            "other field back to exactly what it was in YOUR PREVIOUS DECISION above, and only "
            "add/populate build_units."
        )
    return True, ""


SYSTEM_PROMPT = """You are the Architect for this project. Read the approved spec (and the PM's
epics/stories, if given) and decide what the plan phase would otherwise have to assume on its own:
should this be one workflow/codebase or several genuinely independent ones, where does data live,
what does credential/security architecture need beyond basic per-client isolation, and what
system-level regression-testing strategy fits (relevant once there is more than one workflow
sharing data or credentials). If the spec is small enough that the obvious answer is "one workflow,
nothing unusual," say so plainly rather than inventing structure it doesn't need.

Platform choice (this pipeline supports both n8n workflow builds and Python/code-based builds, e.g.
LangGraph) is one of the things you decide, and it is NOT a binary either/or. Before picking one
platform for the whole project, check whether a hybrid actually fits better: e.g. a thin n8n shell
handling only what n8n is genuinely good at (a trigger, simple delivery/notification) wrapping a
call to a code-based service that owns the parts n8n is weak at (stateful loops, complex branching,
multi-step orchestration) - especially when the project already has always-on hosting for one
platform (an existing n8n instance on a VPS, say) that would otherwise sit unused while a new
standalone process needs its own hosting solved from scratch. A real prior incident: an earlier
Architect run recommended pure Python/LangGraph over n8n solely because of the rework-loop's state
handling, without considering that a hybrid removes that exact objection (the code service owns the
state, n8n never touches it) while also reusing hosting that already exists - the recommendation had
to be redone once this was pointed out. Do not repeat that mistake: always check for a hybrid split
before defaulting to "platform A entirely" or "platform B entirely."

The n8n builder produces exactly ONE n8n workflow per build invocation - there is no mechanism for
one build to emit several. If you recommend more than one n8n-platform workflow, each one is a
SEPARATE build (Agent 1 runs build_and_review.py once per workflow, selected by name), so: (1) give
each n8n workflow in "workflows" a short, distinct "name" - it becomes the literal CLI selector
Agent 1 types, so keep it a plain identifier-friendly phrase (e.g. "Creative Director", not a full
sentence), never two workflows sharing a name; (2) before recommending several n8n workflows, check
whether they genuinely need to be separate n8n processes at all, or whether a hybrid split (one n8n
shell plus a Python/LangGraph service that internally handles what would otherwise be several n8n
workflows' coordination) is the better fit - the same hybrid-first instinct as the platform-choice
paragraph above, now applied within a single platform's own workflow count, not just across
platforms. Real gap this closes: this schema has always allowed "workflow_count" > 1, but the n8n
build path only ever supported exactly one until this was fixed - don't rely on the old assumption
that any number here "just works" without a name Agent 1 can actually select by.

Whenever the system shape has more than one piece that talks to another piece at runtime (a hybrid
n8n-shell-plus-service split, or any multi-workflow system) - name the EXACT wire contract between
them: every field name and, for any status/type/kind discriminator field, every exact string value
it can take and what each value means. Do not leave this for each piece's own builder call to invent
independently - they are separate, isolated builder invocations with no visibility into each other's
choices. Real incident: an n8n shell and its LangGraph service were built in separate sessions,
neither pinned to an explicit contract anywhere (not in the spec, not in a prior Architect run) - the
n8n side's builder invented a response_type of "prompt"/"post" while the service's builder
independently invented "text"/"file" for the exact same field. Both halves built and tested clean in
isolation; every real reply silently vanished at runtime because the n8n side's routing never matched
any value the service actually sent. Name the contract explicitly in "interface_contracts" so both
builder calls receive the SAME pinned vocabulary as binding context, not something each invents.

For every workflow whose "platform" is "python" (never for an "n8n" workflow - n8n already plans and
verifies one node at a time on its own, via a completely separate mechanism untouched by this),
decompose its build into an ORDERED sequence of "build_units" - the python analog of n8n's own
per-node discipline, so a python build gets the same "plan the pieces, verify each one, hand the next
piece a real contract instead of making it rediscover one" property n8n has always had. Real incident
this exists to prevent: a build's own internal retry rewrote a storage module from scratch with no
memory of what another already-written file expected from it, inventing a different class name
(InMemoryStorage vs. InMemoryStorageBackend) for the exact same thing.

Two hard requirements for every unit:
(a) Large enough to support a GENUINELY VIABLE automated check of its own real behavior - not "does
it parse," something that exercises actual logic. If a piece is too small to clear this bar on its
own, merge it with a neighboring piece rather than listing it separately - a mechanical check below
enforces the structural floor of this (real exports, not all constants, a stated verification_hint),
but the actual size/viability judgment is yours to make, not something that check can fully judge in
your place.
(b) An EXPLICIT interface (exported names, kinds, signatures) declared BEFORE any of it is built -
this is what lets a LATER unit be HANDED an EARLIER unit's real shape directly instead of having to
read its file and guess. Every export a later unit needs from an earlier one must actually appear in
that earlier unit's own declared interface - if it's missing, that's a gap in THIS decomposition,
not something the builder should have to invent later.

(c) SPRINT-SIZED, like a Jira sprint with a fixed capacity: each unit is a small, bite-sized story
the builder can finish, test and fix within its tiny per-attempt budget (see the LIMITS SHEET in the
user message for the real numbers). Give every unit an integer "effort_points" (1 = one small file
of simple logic, 2 = a couple of files or moderate logic, 3 = the MOST one unit may hold), and never
exceed the sheet's file and export caps. Prefer MANY small progressive units, each building on the
last (foundation first: models/config, then storage, then logic, then wiring, then entry point),
over a few big ones - smaller units mean fewer errors, cheaper retries and easier bug fixes. This
balances requirement (a): small, but each still needs real testable behavior. Keep your own JSON
compact (short descriptions) so the whole decision fits your own max output.

(d) WALKING SKELETON FIRST (binding). Unit-level tests alone are a blind spot: every unit can pass
its own tests and the assembled system still fails, because nobody ever ran the real pieces together
(real incident, trading-team: units tested only in isolation, never end to end). So the decomposition
must reach a runnable end-to-end path as EARLY as possible. Among the first three units, exactly one
must carry `"walking_skeleton": true`: the quickest possible thin slice that wires the real entry point
through every stage of the system to its real output (thin or hard-coded stage internals are fine,
faking a stage is not - each stage must be a real function/class that later units fill in). The
skeleton unit's verification_hint must describe a real end-to-end run (entry point in, final output
out, with only the external network/SDK boundary faked). Every LATER unit must extend that same
end-to-end test so its own new behavior is exercised through the real assembled path, not only in
isolation; say so in its verification_hint ("extends the end-to-end test: ..."). Foundation units
(models/config) that the skeleton needs may come before it, but never push it past the third unit,
and never leave integration to a final "wiring" unit at the end.

Order units so "depends_on" only ever names units that come BEFORE it in the list (checked
mechanically - a forward or self reference is rejected). List every file this unit is intended to
cover under "files" (a hint the builder can use its own reasonable judgment on for internals, not a
literal requirement) - and "verification_hint": a concrete description of the REAL behavior a test
of this unit should exercise (e.g. "save() then load() of the same key returns equal data", never
"the class can be instantiated" or "does it parse").

Respond with ONLY a JSON object: {"summary": "...", "system_shape": {"workflow_count": N,
"workflows": [{"name": "...", "scope": "...", "platform": "n8n | python | other",
"build_units": [{"name": "...", "purpose": "...", "files": ["..."],
"effort_points": 1, "walking_skeleton": false (true on exactly one of the first three units), "interface": {"exports": [{"name": "...", "kind": "class | function | constant",
"signature": "...", "description": "..."}]}, "depends_on": ["<earlier unit name>", ...],
"verification_hint": "..."}] (only for a "python" workflow - omit or leave empty for "n8n")}]},
"data_placement": "...", "credential_architecture": "...", "interface_contracts": [{"between": "...",
"payload_shape": "...", "discriminator_field": "...", "discriminator_values": ["..."]}] or [] if only
one piece exists, "regression_strategy": "...", "tech_quirks": [...]}

"tech_quirks" is OPTIONAL - a list of short, self-contained strings, each a genuinely reusable
technical lesson about a LIBRARY, FRAMEWORK, or PLATFORM (never this project's own business logic)
you're confident enough about to want every future build on any project using this pipeline to see.
Shape: "<Topic>: <the gotcha and what to do instead>". Leave it empty (or omit it) on the common case
where structural analysis alone didn't surface anything like this - most runs won't.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="Path to the approved SPEC.md")
    parser.add_argument("--pm-epics", help="Path to an approved .pm_epics.json, if PM already ran for this project")
    parser.add_argument(
        "--project-dir",
        help="Project folder, so this project's own models.yaml (if any) overrides the global model config",
    )
    parser.add_argument(
        "--previous-decision",
        help="Path to this project's existing .architect_decisions.json, when this run is a "
        "reconsideration of a prior decision rather than a first pass",
    )
    parser.add_argument(
        "--reconsideration-reason",
        help="The specific new information or objection Douglas raised - required when "
        "--previous-decision is given, since a bare 'please reconsider' is exactly what let the "
        "real incident (see module docstring) slip through unengaged. Optional with "
        "--add-build-units-only - a sensible default reason is used if omitted.",
    )
    parser.add_argument(
        "--add-build-units-only",
        action="store_true",
        help=(
            "2026-09-28, Douglas's direct instruction: for upgrading an existing project's decision "
            "file (approved before build_units existed) to the current schema without re-litigating "
            "anything already approved. Requires --previous-decision. Mechanically enforces (see "
            "only_build_units_changed), on top of the usual reconsideration_engaged check, that "
            "nothing besides build_units/summary changed - a full Architect re-run risks landing on "
            "a genuinely different platform/architecture answer than what was already approved, for "
            "no reason, since a fresh run has no memory of the prior reasoning."
        ),
    )
    args = parser.parse_args()

    if args.add_build_units_only and not args.previous_decision:
        print(json.dumps({"status": "error", "message": "--add-build-units-only requires --previous-decision"}))
        return 1
    if args.add_build_units_only and not args.reconsideration_reason:
        # Sensible, repeatable default for this specific, well-known scenario - the caller doesn't
        # need to retype the same boilerplate every time it upgrades a project's decision file.
        args.reconsideration_reason = (
            "Your previous decision predates the build_units field. Add a build_units decomposition "
            "for each \"python\"-platform workflow only - every other already-approved decision "
            "(platform, data_placement, credential_architecture, interface_contracts, "
            "regression_strategy) stays exactly as it was."
        )

    spec_text = Path(args.spec).read_text(encoding="utf-8")
    user_prompt = (
        f"LIMITS SHEET (size every build unit to fit these; do not exceed them):\n"
        f"{build_limits_sheet(args.project_dir)}\n\nAPPROVED SPEC:\n{spec_text}"
    )
    if args.pm_epics:
        user_prompt += f"\n\nPM'S APPROVED EPICS/STORIES:\n{Path(args.pm_epics).read_text(encoding='utf-8')}"

    previous_decision = None
    if args.previous_decision:
        if not args.reconsideration_reason:
            print(json.dumps({"status": "error", "message": "--previous-decision requires --reconsideration-reason"}))
            return 1
        previous_decision = json.loads(Path(args.previous_decision).read_text(encoding="utf-8"))
        user_prompt += (
            f"\n\nTHIS IS A RECONSIDERATION of a decision you already made, not a fresh first pass.\n"
            f"YOUR PREVIOUS DECISION:\n{json.dumps(previous_decision, indent=2)}\n\n"
            f"THE SPECIFIC NEW INFORMATION OR OBJECTION TO RECONSIDER:\n{args.reconsideration_reason}\n\n"
            f"You MUST explicitly engage this new information - either revise your decision because of "
            f"it and explain what changed and why in \"summary\", or explain in \"summary\" specifically "
            f"why the original decision still stands despite it. Simply repeating your previous answer "
            f"unchanged, or answering as if this were a fresh first pass that never mentions the new "
            f"information, is not acceptable - this exact failure happened for real once already and "
            f"cost Douglas a wasted re-run cycle to catch."
        )
        if args.add_build_units_only:
            user_prompt += (
                "\n\nTHIS RECONSIDERATION IS SCOPED TO ONE THING ONLY: adding build_units. Copy "
                "EVERY OTHER FIELD from YOUR PREVIOUS DECISION above verbatim, unchanged - "
                "system_shape (including workflow platform/scope), data_placement, "
                "credential_architecture, interface_contracts, and regression_strategy must all come "
                "back exactly as they already were. Do not re-evaluate, rephrase, or \"improve\" "
                "anything you weren't asked to touch, even if you'd now word it differently - this "
                "is mechanically checked, and any other change will be rejected and sent back to you."
            )

    run_id = new_run_id()
    llm = resolve_llm_for_role("architect", project_dir=args.project_dir)
    config = run_config(run_id, tags=["architect", "system_shape"])

    result = None
    engagement_problem = ""
    unit_problems: list[str] = []
    scope_problem = ""
    system_prompt = with_tech_quirks(SYSTEM_PROMPT)
    try:
        for _attempt in range(1, MAX_INTERNAL_FIX_ATTEMPTS + 1):
            messages = [("system", system_prompt), ("user", user_prompt)]
            try:
                response = with_retries_nonempty(lambda: llm.invoke(messages, config=config))
            except Exception as exc:
                response = attempt_fallback(
                    "architect", lambda fb_llm: fb_llm.invoke(messages, config=config),
                    project_dir=args.project_dir, primary_error=exc,
                )
            result = extract_json_object(response.content, context="architect response")
            for lesson in result.pop("tech_quirks", None) or []:
                if isinstance(lesson, str) and lesson.strip():
                    record_tech_quirk_entry(lesson, source="architect")
            result["total_tokens"] = extract_tokens(response)

            # Independent of the reconsideration check below - a fresh first pass has no previous
            # decision to reconsider but ITS build_units still need this same mechanical floor.
            workflows = ((result.get("system_shape") or {}).get("workflows")) or []
            unit_problems = []
            for wf in workflows:
                if isinstance(wf, dict) and wf.get("platform") == "python":
                    unit_problems.extend(validate_build_units(wf.get("build_units") or []))

            engaged, engagement_problem = (
                reconsideration_engaged(previous_decision, result, args.reconsideration_reason)
                if previous_decision is not None else (True, "")
            )
            in_scope, scope_problem = (
                only_build_units_changed(previous_decision, result)
                if previous_decision is not None and args.add_build_units_only else (True, "")
            )

            if engaged and in_scope and not unit_problems:
                break

            feedback_parts = []
            if not engaged:
                feedback_parts.append(
                    f"Your previous attempt at this reconsideration failed a mechanical check: "
                    f"{engagement_problem}. Actually engage with the new information stated above "
                    "and reflect it in your answer."
                )
            if not in_scope:
                feedback_parts.append(f"Your previous attempt failed a mechanical check: {scope_problem}")
            if unit_problems:
                feedback_parts.append(
                    "Your build_units decomposition failed a mechanical check:\n"
                    + "\n".join(f"- {p}" for p in unit_problems)
                    + "\nFix the specific unit(s) named above - merge a too-small unit with a "
                    "neighbor, or add what's missing - rather than changing anything that wasn't flagged."
                )
            user_prompt += "\n\n" + "\n\n".join(feedback_parts)
            result = None
    except Exception as exc:  # OpenRouter outage, exhausted retries, malformed model output, etc.
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 1

    if result is None:
        # Ran out of attempts still failing one of the mechanical checks above - surfacing this
        # plainly is the whole point: silently falling back to "pass this through anyway" is exactly
        # the unengaged-repeat failure this mechanism exists to prevent, just moved one step later.
        problems = []
        if engagement_problem:
            problems.append(f"reconsideration not engaged: {engagement_problem}")
        if scope_problem:
            problems.append(f"scope: {scope_problem}")
        if unit_problems:
            problems.append("build_units still invalid: " + "; ".join(unit_problems))
        print(json.dumps({
            "status": "did_not_pass_mechanical_checks",
            "message": f"After {MAX_INTERNAL_FIX_ATTEMPTS} attempts, the Architect's response still "
            f"failed: {' / '.join(problems)}. Bring this to Douglas rather than treating it as approved.",
        }))
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
