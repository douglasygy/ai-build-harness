"""PM epics (still read as binding context - see pipeline/nodes/builder_node.py's
_augment_spec_with_scale_adaptive_context) and the mechanical stub-regression/test-coverage-gap
checks reviewer_node.py runs against whatever's reported built so far.

RETIRED 2026-09-28 (Douglas's direct instruction - checkpoint/stage gating removed for the python
path entirely): this module used to ALSO group PM's epics into checkpoint stages and resolve
--until-stage against them (stages_from_epics, resolve_until_stage, in_scope_and_deferred - all
removed). Architect-led build_units decomposition (pipeline/review_architect.py,
pipeline/nodes/builder_node.py's _build_python_units) replaces that entirely, at finer granularity
than the old 4-stage cap: Douglas's own reasoning was that a properly-decomposed unit sequence
(ordered, contracted, sized for a real behavioral check) already gives the "don't fly blind through
one long unsupervised run" property the stage system existed for, without a separate mechanism
grouping epics into stages on top of it. next_unbuilt_stage_index below is NOT retired - its own
logic only ever read run_result.json's generic "slices" shape (total/built), which
_build_python_units already produces in unit-derived form, so it works unchanged for units too
despite its name; renaming it was not worth the blast radius for a purely cosmetic mismatch.
check_stage_stub_regressions/check_stage_test_coverage_gaps are similarly UNCHANGED and un-retired -
their own "epics_built" parameter is a plain list of names, and reviewer_node.py now passes unit
names instead of epic names, which needs no change to either function's own logic.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .scale_roster import pm_epics_path


def load_pm_epics(project_dir) -> list:
    """This project's approved PM epics, in order - empty if PM never ran for this project (the
    common case; python staging simply does not apply then, same as n8n slicing not applying to a
    project whose plan phase never produced slices)."""
    path = pm_epics_path(project_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    epics = data.get("epics") if isinstance(data, dict) else None
    return [e for e in (epics or []) if isinstance(e, dict) and e.get("name")]


def next_unbuilt_stage_index(project_dir, stage_count: int) -> int:
    """Douglas's low-friction default: re-invoking build_and_review.py with no --until-stage at
    all advances exactly one checkpoint each time, rather than requiring him to compute and pass an
    increasing stage number himself. Reads how many stages the LAST run reported built, from this
    same project's run_result.json - 0 (start at the first stage) if there is no prior run, the
    prior run was not this kind of staged build, or its own stage count no longer matches (e.g. the
    epics were re-planned since), since a stale count could otherwise skip real work.
    """
    result_path = Path(project_dir) / "run_result.json"
    if not result_path.exists():
        return 0
    try:
        prior = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    slices = prior.get("slices") if isinstance(prior, dict) else None
    if not isinstance(slices, dict) or slices.get("total") != stage_count:
        return 0
    built = slices.get("built")
    already_built = len(built) if isinstance(built, list) else 0
    return min(stage_count - 1, already_built)


# Exact, mandated wording the builder used to be told to write, verbatim, whenever it stubbed a
# piece belonging to an epic outside a checkpointed stage's own scope - retired 2026-09-28 alongside
# the stage system itself (build_units' own ordering makes this moot: a unit only ever depends on
# EARLIER, already-built units, never a later, not-yet-built one, so there is no longer anything
# for a unit to defer-and-stub in the first place). check_stage_stub_regressions below is
# UNCHANGED and still real - it still mechanically detects this exact marker string if a builder
# habit or an older project's own carried-forward code still uses it, it just no longer describes
# something the current builder is instructed to write.
def stub_marker(epic_name: str) -> str:
    return f"STUB: out of scope for epic {epic_name!r} - do not implement until that epic is in scope."


_STUB_MATCH_WINDOW = 4


def _has_legacy_stub_mention(lines: list, epic_name: str, window: int = _STUB_MATCH_WINDOW) -> bool:
    """Looser fallback for a stub written before the exact stub_marker() convention existed - real
    incident (reported live, 2026-09-26): the actual stub that motivated this whole check reads
    across two lines ('...subgraph - STUB.\\nOut of scope for this build stage (epic: "...")'),
    semantically identical to stub_marker() but textually different, so an exact match alone missed
    the very case it was built for. Requires "stub" (case-insensitive) and the epic's exact name to
    co-occur within a small sliding window of lines - same file AND close together, not just
    anywhere in a large file, to keep the (accepted) false-positive risk low. A window rather than a
    single line because the real example above already spans two.
    """
    lowered_name = epic_name.lower()
    lowered_lines = [l.lower() for l in lines]
    for i in range(len(lines)):
        joined = " ".join(lowered_lines[i:i + window])
        if "stub" in joined and lowered_name in joined:
            return True
    return False


def check_stage_stub_regressions(code_dir, epics_built: list) -> list:
    """Mechanical (no LLM call): for every epic that's supposed to be built as of this round
    (cumulative - every stage through the current one, not just the epic(s) newly added this
    round), checks whether a stub for it is still present in the actual code - either the exact,
    standardized stub_marker() (high confidence), or a looser same-vicinity "stub" + epic-name
    mention for stubs written before that convention existed (lower confidence - flagged as such,
    since a coincidental co-occurrence is possible). If either matches, that epic's real
    implementation almost certainly does not exist yet, no matter what the build reported - a stub
    written back when that epic was still deferred does not get silently rewritten just because a
    later round claims to have reached it.

    Returns structured findings ({"file", "epic", "confidence", "message"}), not plain strings -
    reviewer_node.py needs the real file/confidence fields, not just prose, to mechanically verify
    the reviewer actually engaged each one (see verify_mechanical_finding_engagement there): a real incident
    (2026-09-26, three consecutive runs, 1M+ tokens) showed this evidence reaching the reviewer
    exactly as designed, and the reviewer simply saying nothing and passing anyway - an unverified
    prose hint alone was not enough to stop that.

    Never raises, never blocks a build on its own by itself - same pattern as
    pipeline/interface_contracts.py's check_interface_contracts: mechanical evidence, acted on
    through the reviewer's own reporting path.
    """
    code_dir = Path(code_dir)
    if not epics_built or not code_dir.exists():
        return []

    findings = []
    for path in sorted(code_dir.rglob("*")):
        if not (path.is_file() and path.suffix in {".py", ".md", ".json", ".yaml", ".yml", ".js"}):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = str(path.relative_to(code_dir))
        for name in epics_built:
            if stub_marker(name) in text:
                findings.append({
                    "file": rel, "epic": name, "confidence": "high",
                    "message": (
                        f"{rel}: contains the exact deferred-stub marker for epic {name!r}, which is "
                        f"reported built as of this round - its real implementation likely does not "
                        f"exist yet, regardless of what else was built."
                    ),
                })
            elif _has_legacy_stub_mention(text.splitlines(), name):
                findings.append({
                    "file": rel, "epic": name, "confidence": "low",
                    "message": (
                        f"{rel}: mentions 'stub' near epic {name!r}'s own name, but not the exact "
                        f"standardized marker (this file likely predates that convention) - LOWER "
                        f"CONFIDENCE than an exact hit, verify the real content before treating this "
                        f"as a genuine gap. Epic {name!r} is reported built as of this round."
                    ),
                })
    return findings


# Real definitions only - a file with none (a docstring-only package __init__.py, a pure re-export
# module) has nothing to test and is deliberately never flagged, no separate size/triviality
# heuristic needed on top of this.
_DEFINITION_RE = re.compile(r"^\s*(?:async\s+)?def\s+\w+|^\s*class\s+\w+", re.MULTILINE)
_TEST_SOURCE_EXCLUDE = {"__pycache__"}

# Real branching/error-handling logic, not import/assignment/docstring lines - used only to RANK
# findings (highest-risk-of-a-silent-bug first), never to exclude one. A thin wrapper/glue module
# still gets flagged, just later than a module with real decision logic in it.
_COMPLEXITY_MARKER_RE = re.compile(r"\b(?:if|elif|for|while|except|raise|assert)\b")

# Real incident this exists to catch (2026-09-27, creative-team): all 22 accumulated coverage gaps
# were force-injected as problems in one round (the mechanical detection/enforcement worked exactly
# as designed), and the builder then tried to write real tests for all 22 modules in that single
# round and hit the 500,000-token run ceiling with nothing saved to disk - zero tests, full budget
# spent. Same root cause and same fix shape as the whole stage-checkpoint feature this sits beside:
# an unbounded amount of work handed to one round is not an achievable task, no matter how correct
# the evidence behind it is. Capping here (not in reviewer_node.py) means the fix is stateless and
# self-correcting for free: this function already re-scans the real code from scratch every round,
# so as soon as a module gets a real test, it naturally drops off next round's findings on its own -
# no separate "already queued" bookkeeping needed. Kept deliberately small relative to the 22-module
# incident (a fraction of it, not half of it) since real test-writing - reading a module, understanding
# its behavior, writing multiple real assertions - is much more expensive per item than a stub check.
_MAX_COVERAGE_FINDINGS_PER_ROUND = 5


def check_stage_test_coverage_gaps(code_dir, epics_built: list) -> list:
    """Mechanical (no LLM call): every real (has an actual function/class definition) non-test
    Python module under code_dir that has neither a same-named test_<module>.py anywhere in the
    tree, nor its own module name mentioned anywhere in any test file's content, is flagged as
    having no apparent test coverage - fed to the reviewer the same way check_stage_stub_regressions
    is, once at least one epic is reported built (a project with no PM epics/staging has nothing to
    check here; this is scoped to staged python builds only, same as the stub check).

    Real incident this exists to catch (2026-09-26/27, creative-team): SPEC.md stated an explicit,
    itemized testing requirement, binding on every stage - across 4 real build rounds and several
    hundred lines of new logic (a genuine SEO/GEO guidance provider, a vendor-agnostic LLM client, a
    real Content Writer + Writer Gatekeeper subgraph), the reviewer passed all four rounds while
    test count stayed flat at the same 14 tests across 3 files the whole time. Same root cause and
    same fix shape as the stub-regression check: mechanical evidence handed to the reviewer, whose
    engagement is then itself mechanically enforced (see reviewer_node.py's verify_mechanical_finding_engagement,
    which this check's findings are also routed through - the two checks share one enforcement path,
    not two parallel ones).

    Deliberately epic-agnostic (does not try to map specific files to specific epics - PM's epic
    schema has no such mapping today, and building one would be a bigger, separate piece of work):
    flags every currently-uncovered real module, every round, until it actually gets a test -
    matching the real, observed failure shape, where the gap accumulates silently across MULTIPLE
    stages rather than being introduced and caught in just one.

    Batched, not all-at-once (added 2026-09-27 after a real 641,080-token round that produced zero
    tests - see _MAX_COVERAGE_FINDINGS_PER_ROUND above): only the _MAX_COVERAGE_FINDINGS_PER_ROUND
    gaps with the most real branching/error-handling logic are returned per call, highest-risk first.
    The rest are simply not returned this round, not tracked anywhere - since this function re-scans
    the real filesystem from scratch every call, a module that gets a real test between rounds drops
    off on its own next call, and the next-highest-priority gap naturally surfaces in its place.

    A project with more coverage gaps than fit in one round's cap does NOT need a separate invocation
    per batch: reviewer_node.py's MAX_ROUNDS check is itself per-batch for a fail driven purely by
    unresolved coverage debt (see its pure_coverage_fail logic) - each new batch this function starts
    returning gets its own fresh 3-round budget within the SAME invocation, keyed off the actual set
    of file paths currently surfaced. A genuine correctness bug still uses the original flat per-run
    cap, unchanged - only a coverage-only fail gets this per-batch treatment.
    """
    code_dir = Path(code_dir)
    if not epics_built or not code_dir.exists():
        return []

    source_files = []
    test_text_parts = []
    for path in sorted(code_dir.rglob("*.py")):
        if any(part in _TEST_SOURCE_EXCLUDE for part in path.parts):
            continue
        rel = path.relative_to(code_dir)
        is_test_file = (
            "test" in {p.lower() for p in rel.parts[:-1]}
            or path.name.startswith("test_")
            or path.name == "conftest.py"
        )
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if is_test_file:
            test_text_parts.append(text)
            continue
        source_files.append((path, text))

    all_test_text = "\n".join(test_text_parts)

    gaps = []  # (complexity_score, rel_path, module_stem) - real gaps, not yet capped
    for path, text in source_files:
        if not _DEFINITION_RE.search(text):
            continue  # nothing real to test - a docstring-only or pure re-export module
        module_stem = path.stem
        same_named_test_exists = any(
            True for _ in code_dir.rglob(f"test_{module_stem}.py")
        )
        referenced_in_tests = module_stem in all_test_text
        if same_named_test_exists or referenced_in_tests:
            continue
        rel = str(path.relative_to(code_dir))
        score = len(_COMPLEXITY_MARKER_RE.findall(text))
        gaps.append((score, rel, module_stem))

    if not gaps:
        return []

    # Highest real branching/decision logic first (most likely to hide a real bug if untested) -
    # rel path as the tiebreak purely for deterministic ordering, not a meaningful priority signal.
    gaps.sort(key=lambda g: (-g[0], g[1]))
    total_gap_count = len(gaps)
    surfaced = gaps[:_MAX_COVERAGE_FINDINGS_PER_ROUND]
    deferred_count = total_gap_count - len(surfaced)

    findings = []
    for _score, rel, module_stem in surfaced:
        deferred_note = (
            f" ({deferred_count} additional module(s) with existing coverage gaps are not shown this "
            f"round - fix these first, then re-run; they will surface on a later round once these "
            f"are addressed.)"
            if deferred_count else ""
        )
        findings.append({
            # Same shared shape check_stage_stub_regressions' findings use ("file", "confidence",
            # "message") so both check types route through reviewer_node.py's ONE generic
            # enforcement path (verify_mechanical_finding_engagement, the retry/force-injection logic) without
            # either needing special-casing there. "high" here too: real-world validation against
            # creative-team's own actual code found zero false positives (a module genuinely
            # referenced by a test, even indirectly via a fixture import, is correctly excluded
            # above) - a stage should not pass while shipping genuinely untested real logic.
            "file": rel,
            "confidence": "high",
            "message": (
                f"{rel}: contains real function/class definitions, but no test_{module_stem}.py "
                f"exists and '{module_stem}' is not referenced in any test file's content - no "
                f"apparent test coverage for this module, despite epic(s) reported built as of this "
                f"round: {', '.join(epics_built)}.{deferred_note}"
            ),
        })
    return findings
