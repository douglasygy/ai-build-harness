from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from ..external_findings import finding_line, resolve_finding
from ..ambiguity_answers import drop_dismissed, load_answers, rulings_block
from ..build_journal import REVIEW_ROUND_UNIT, record_journal_entry
from ..build_stages import check_stage_stub_regressions, check_stage_test_coverage_gaps
from ..config_loader import max_rounds
from ..constants import MAX_INTERNAL_FIX_ATTEMPTS, MAX_REVIEWER_TOKENS_PER_ROUND
from ..interface_contracts import check_interface_contracts
from ..json_extract import extract_json_object
from ..langfuse_setup import new_run_id, run_config
from ..model_fallback import attempt_fallback, resolve_llm_for_role
from ..retry import with_retries_nonempty
from ..provisional_fixed import clear_provisional
from ..review_waivers import format_waivers_for_prompt, load_waivers
from ..scale_roster import architect_decisions_path
from ..state import BuildState
from ..tech_quirks import record_tech_quirk_entry, with_tech_quirks
from ..token_guard import extract_tokens

SYSTEM_PROMPT = """You are an independent code reviewer. You are given a product spec and the finished code
for it - nothing else. You do not know how the code was produced. Your job is exactly two things: does the
code match what SPEC.md actually says, and does the code have real, demonstrable errors. Nothing else is
your job - never suggest optimizations, redesigns, refactors, or best-practice/style opinions. If something
isn't explicitly demanded by SPEC.md and isn't a genuine, demonstrable error, it is not a problem to report,
no matter how reasonable it might seem.

One specific, demonstrable defect to look for: code that reads a config key, column or field that the real
loader/schema never produces, whose tests pass only because they build a stub/fake config or schema that
happens to contain it. If a real loader or migration exists in the code you were given and the tests
substitute a hand-made stand-in, check the keys/columns the code reads against what the REAL one produces;
report a mismatch as a "correctness_bug" (must_fix when it makes the feature silently not work).

A second specific defect, related to the one above: a unit that CONSUMES data produced by another unit in
this codebase (a client, loader, collector, parser) but whose tests only feed it hand-built stand-in data
(DataFrames, dicts, fake config objects) shaped the way the consumer expects, never obtaining that data by
calling the REAL producer with only the network/SDK boundary faked. Such tests stay green while the seam is
broken (different index/column names, drop=True vs drop=False, different label values, fields the real
producer never fills). When a real producer exists in the code you were given, check what it actually
returns against what the consumer and its tests assume; report a mismatch as a "correctness_bug"
(must_fix when the real path would fail or silently misbehave), and report "consumer tested only against
hand-built producer output, no test uses the real producer" as a must_fix "correctness_bug" naming the
consumer file.

Report EVERY real defect you can find in this single review, not just the most important few. A finding you
hold back now only surfaces in a later round, and every extra round is a full rebuild and re-review at real
cost; you will not get a second look to "save it for next time". Severity already decides what blocks
delivery, so report a minor but real finding as "should_fix" or "nitpick" instead of leaving it out. Before
you answer, go through every file you were given once more for anything you skipped. Completeness applies
only to real, demonstrable problems that pass every check below - never invent or pad a finding to look
thorough.

Respond with ONLY a JSON object: {"verdict": "pass" or "fail", "problems": [...]}. "problems" is empty if
verdict is "pass". Each entry in "problems" is an OBJECT, not a plain string:

{"node": "...", "claim_type": "spec_requirement" or "correctness_bug" or "missing_file", "severity": "must_fix" or "should_fix" or "nitpick", "description": "...", "spec_quote": "...", "anchor": "...", "waiver_id": "..."}

- "node": the exact name of the single node, file, or function this applies to, copied character-for-character
  from the code you were given - never paraphrased, abbreviated, or invented. If a problem genuinely spans the
  whole artifact and cannot be attributed to one place, use "WORKFLOW" instead. This is not cosmetic: it is
  used to rebuild exactly the piece at fault instead of regenerating everything. EXCEPTION: for claim_type
  "missing_file", "node" is instead the path of the file that SHOULD exist but doesn't - it will not be found
  anywhere in the code you were given, and that's the entire point of the claim; do not use "correctness_bug"
  for this, it will be discarded as an invented name (see "anchor" below for how this is verified instead).
- "claim_type": "spec_requirement" when you are asserting the spec explicitly demands a specific name, value,
  string, or exact behavior the code doesn't have. "correctness_bug" when you are reporting a genuine defect
  (a crash, a logic error, data flowing from the wrong place, a broken condition) that you can point to
  directly in the code, independent of whatever wording the spec used. "missing_file" when a whole file that
  should exist - a test file for a real module, a file the spec names outright - is absent entirely, rather
  than defective; never use "correctness_bug" for a file that isn't there at all.
- "spec_quote": REQUIRED, and must be copied character-for-character from the SPEC text you were given, when
  claim_type is "spec_requirement" - this is checked mechanically against the actual spec text after you
  respond, and any claim without a real, verbatim quote from the SPEC text you were given is discarded before
  it ever reaches anyone. Leave it empty for "correctness_bug"/"missing_file" entries - neither needs spec
  wording, they need a real, checkable defect or a real, checkable reason.
- "anchor": REQUIRED when claim_type is "missing_file", otherwise omit it. The exact name of a file that DOES
  exist in the code you were given and is the concrete, real reason the missing file should exist (e.g. the
  real module "test_creative_director.py" would test) - copied character-for-character, checked the same way
  "node" is checked for the other claim types. A "missing_file" claim with no real anchor is discarded exactly
  like an invented node name - naming one real thing doesn't excuse inventing the other.
- "severity": REQUIRED on every problem. Only "must_fix" findings stop the build and send the builder back for
  another round - be honest and precise here, not maximalist; inflating severity to "be safe" costs real money
  and real time on a round that didn't need to happen.
  - "must_fix": a real, demonstrable defect against the SPEC, or a genuine security/data-loss/crash risk - the
    kind of thing that would hurt a real user or operator if it shipped as-is. This blocks delivery.
  - "should_fix": a genuine, verifiable defect or spec gap that does NOT block real usage today - an edge
    case, a robustness gap, a real but non-critical spec point. Worth fixing, but never worth blocking
    delivery over BY ITSELF - if there is also at least one "must_fix" this round, it's handed to the builder
    alongside it; if not, it's recorded and does not trigger a round on its own.
  - "nitpick": a real but genuinely minor or cosmetic observation - would improve the code but has no
    functional consequence. Never sent to the builder, never causes a round; only recorded so nothing you
    noticed is silently lost. If it wouldn't survive being read out loud to Douglas as "this must ship
    differently," it's a nitpick, not a must_fix.
- "waiver_id": Leave empty unless you are told below that a KNOWN LIMITATION exists for this project - if a
  finding you are about to report exactly matches one, DO NOT REPORT IT AT ALL (see the section below, when
  present, for the full instruction and the real accepted limitations for this project). This field exists
  only as a mechanical backstop if you report one anyway despite that instruction - it does not replace not
  reporting it in the first place.

Three things this mechanical check specifically exists to catch, because they happened for real, cost real
money, and must never happen again: (1) asserting "the spec requires node/field X" when X does not exist
anywhere in the spec text - grep your own claim against the spec text before writing it down, not just your
memory of what a similar spec might say; (2) naming a node or field in a complaint that does not actually
exist in the code you were given - a name you cannot find by searching the code text is a name you invented,
not a real problem; (3), the inverse of (2): a genuinely missing file reported as "correctness_bug" and
silently discarded as an invented name, because by definition a missing file's own path can never be found in
the code either - use "missing_file" with a real "anchor" for this instead, which is checked the opposite way
on purpose. All three failure modes are checked automatically and silently deleted from your review - so
inventing a name doesn't just fail to help, it wastes the round entirely. Only write down what you can
actually point to in the text you were given (or, for a missing file, what you can point to as the real reason
it should exist).

If you are given a "MECHANICAL STAGE-STUB CHECK" and/or "MECHANICAL TEST-COVERAGE CHECK" section,
every file named in either REQUIRES one of two responses from you, and silence is not a third
option: EITHER report it as a "correctness_bug" problem naming that file, OR add it to
"mechanical_check_dismissals" - a JSON object mapping that exact file path to a short, VERBATIM
quote copied character-for-character from that file's real, current content, proving the concern is
not real (genuinely implemented, not still a stub; or genuinely covered by a test you can point to).
A dismissal with no quote, or a quote that does not actually appear in that file, is mechanically
rejected exactly like a fabricated spec_quote above - this exists because it happened for real:
mechanical evidence reached a reviewer naming the exact file, and the reviewer said nothing and
passed anyway, three times in a row, burning over a million tokens with the stage's own headline
deliverable never built - and separately, a stated itemized testing requirement had zero measurable
effect across four straight rounds while hundreds of lines of untested real logic shipped.
"mechanical_check_dismissals" is only required when one of these sections is present; omit it
entirely otherwise.

If you are given a "BUILDER'S OWN ACCOUNT" section, it is an UNVERIFIED CLAIM, not evidence -
exactly as unverified as if the builder had said it to you directly, because that's exactly what
it is. The file list inside it is mechanically accurate (the framework recorded it, not the
builder), but the builder's stated reasoning for WHY it made those changes is not checked by
anyone before you see it. Use it only to understand intent faster - never let a plausible-sounding
explanation substitute for checking the actual code yourself, and never lower your bar for
evidence because an explanation sounded reasonable. If the code doesn't actually do what the
account claims, that gap is itself a real problem to report.

Optionally, include "tech_quirks": [...] in your response - a list of short, self-contained strings,
each one a genuinely reusable technical lesson about a LIBRARY, FRAMEWORK, or PLATFORM this code
depends on (never this project's own business logic, and never a restatement of a "problems" entry
above). Only include something you've directly confirmed by reading the actual code or its real
behavior - never a guess. Shape: "<Topic>: <the gotcha and what to do instead>". Every future build
on any project using this pipeline will see this, so leave it empty (or omit it) on the common round
where you have nothing new like this to report - most rounds do not.
"""

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _normalize_path_text(text: str) -> str:
    """_normalize, plus backslash->forward-slash - used ONLY for the "node" field's membership
    check below. code_text's own file headers are already forward-slash (_read_code uses
    .as_posix()), so this is defense in depth against the model writing a backslash path anyway
    (copying a traceback, or a Windows-flavored path from somewhere else in what it was given) -
    never trust a single fix to be the only thing standing between a real finding and being
    silently dropped as "invented"."""
    return _normalize(text.replace("\\", "/"))


_FILE_EXTENSIONS = {"py", "json", "md", "yaml", "yml", "txt", "bat", "toml", "cfg", "ini", "sh", "js", "ts"}


def _node_present(node: str, normalized_code: str) -> bool:
    """True if `node` is grounded in the code: as a literal path/name, or - for a dotted code
    reference like `Class.method` (which never appears literally, since code says `class Class` and
    `def method`) - when every dotted segment appears as a whole word. File names (`a.py`) are excluded
    from the split so an extension can't ground an invented file."""
    literal = _normalize_path_text(node)
    if literal in normalized_code:
        return True
    dotted = re.sub(r"\(.*\)\s*$", "", node.strip())
    parts = dotted.split(".")
    if len(parts) < 2 or parts[-1].lower() in _FILE_EXTENSIONS:
        return False
    if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p) for p in parts):
        return False
    return all(re.search(rf"\b{re.escape(p)}\b", normalized_code) for p in parts)


def _read_code(code_dir: Path) -> str:
    if not code_dir.exists():
        return "(no code found)"
    chunks = []
    for path in sorted(code_dir.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".md", ".txt", ".json", ".yaml", ".yml"}:
            # .as_posix() - not str() - 2026-09-28, Douglas's direct instruction, found via a real
            # reproduction: str(path.relative_to(code_dir)) uses backslashes on Windows, so every
            # header here read e.g. "creative_studio\storage_memory.py". The reviewer LLM naturally
            # writes forward-slash paths in its "node" field regardless of what it was shown, so a
            # plain substring check (verify_problems below) NEVER matched a real citation - every
            # genuine finding naming a real file was silently dropped as "invented", 100% of the
            # time, on Windows. This had presumably been broken since verify_problems was built;
            # today was just the first day a build survived long enough to reach it. Real incident:
            # a genuinely correct finding (two competing storage backend classes) got dropped this
            # way while a less important finding was kept instead.
            chunks.append(f"--- {path.relative_to(code_dir).as_posix()} ---\n{path.read_text(encoding='utf-8')}")
    return "\n\n".join(chunks) if chunks else "(no code found)"


def _parse_response(text: str) -> dict:
    return extract_json_object(text, context="reviewer response")


_VALID_SEVERITIES = {"must_fix", "should_fix", "nitpick"}


def _apply_waivers(problems: list, waivers: list) -> tuple[list, list]:
    """Mechanically drops any problem the reviewer itself tagged with a REAL waiver_id - never
    trusting the "don't report this" prompt instruction alone (same principle as everything else
    in this file: a model following an instruction reliably, every time, is never assumed).
    Returns (remaining, dropped_notes) - remaining is `problems` with any waived entries removed
    (for verify_problems to check exactly as before); dropped_notes is debug strings for stderr, so
    a drop is visible, never silent."""
    if not waivers or not problems:
        return list(problems or []), []
    valid_ids = {w.get("id") for w in waivers if w.get("id")}
    remaining, dropped = [], []
    for raw in problems:
        waiver_id = str(raw.get("waiver_id") or "").strip() if isinstance(raw, dict) else ""
        if waiver_id and waiver_id in valid_ids:
            dropped.append(f"dropped (tagged waiver_id {waiver_id!r}, an accepted known limitation): {raw.get('description', raw)!r}"[:250])
            continue
        remaining.append(raw)
    return remaining, dropped


def _flag_possible_unwaived_matches(kept_problems: list, waivers: list) -> list:
    """NOT a drop - a cheap, rough overlap check (does a waiver's own text share a distinctive word
    with a kept finding's own text) purely to print a DEBUG note when a kept finding might be the
    same known limitation the reviewer simply didn't tag. A mechanical text/substring match is too
    unreliable to auto-drop on (it could waive something that only superficially resembles the real
    waiver) - this only surfaces a possible miss for a human to notice, never acts on it."""
    if not waivers or not kept_problems:
        return []
    notes = []
    for waiver in waivers:
        waiver_words = {w for w in re.findall(r"[a-z]{5,}", str(waiver.get("text", "")).lower())}
        if not waiver_words:
            continue
        for problem in kept_problems:
            problem_words = set(re.findall(r"[a-z]{5,}", problem.lower()))
            if len(waiver_words & problem_words) >= 3:
                notes.append(
                    f"possible unwaived match: kept problem {problem[:150]!r} shares significant "
                    f"wording with waiver {waiver.get('id')!r} ({waiver.get('text', '')[:150]!r}) "
                    "but was not tagged with its waiver_id - not dropped, flagging only"
                )
    return notes


def verify_problems(problems: list, spec_text: str, code_text: str) -> tuple[list[str], list[str], list[str]]:
    """Mechanically grounds every claimed problem against the actual spec/code text - never trusts the
    reviewer's own compliance with the instructions above, because it has already demonstrated (a real
    incident, not a hypothetical) that it will confidently invent a "spec requires X" claim and a node
    name that exist nowhere, and that this is expensive: it drove a builder rebuild loop chasing
    unsatisfiable fabricated names straight into a full token-budget abort at ~6x a normal run's cost.

    claim_type "missing_file" (2026-09-29) is checked the OPPOSITE way from every other claim_type, on
    purpose: the ordinary invented-name check below requires `node` to appear in the code, but a
    missing-file claim's whole point is that `node` (the missing path) does NOT appear anywhere - the
    real incident this fixes is exactly that: two accurate "you're missing a test file" findings got
    silently dropped as "invented name," surviving only because an unrelated mechanical check happened
    to catch the same substance, which won't always be true. Grounded instead via `anchor`, a real,
    already-existing file the claim must name as its reason - checked the normal way (must actually
    appear in the code) - so a fabricated (node, anchor) pair still can't slip through on a made-up
    anchor, and a `node` that turns out to actually exist (not really missing after all) is rejected
    too.

    Severity (2026-09-29, Douglas's explicit direction, relayed via video-generator-78: "the reviewer
    ranking its finding as well, priority must-haves, good to have, nitpicks etc. And must haves must
    be delivered") is DIFFERENT in kind from every other check in this function - claim_type/node/
    spec_quote/anchor all have a real, independently-checkable ground truth (does this text exist),
    while severity is the reviewer's own subjective judgment, with no ground truth to verify it
    against - the same kind of unavoidable judgment call "verdict" itself has always been. What IS
    mechanically enforced: severity must be one of the three recognized values - anything missing or
    unrecognized defaults to "must_fix", the conservative choice (better to over-block a round than
    silently let an unclassified problem slip through unaddressed as if it were a nitpick).

    Returns (kept, rejected, kept_severities) - kept is the flattened "node: description" strings
    review_problems has always been (so nothing downstream that already treats it as plain text needs
    to change), positionally aligned with kept_severities (kept_severities[i] is kept[i]'s severity);
    rejected is a list of short debug strings explaining what got dropped and why, printed to stderr
    for visibility rather than silently vanishing.
    """
    normalized_spec = _normalize(spec_text)
    normalized_code = _normalize(code_text)
    kept: list[str] = []
    kept_severities: list[str] = []
    rejected: list[str] = []

    for raw in problems or []:
        if not isinstance(raw, dict):
            # Old-format plain string, or a malformed entry - can't verify a claim with no structure to
            # check, so it's dropped rather than trusted.
            rejected.append(f"dropped (not a structured problem object): {str(raw)[:200]}")
            continue

        node = str(raw.get("node") or "").strip()
        claim_type = str(raw.get("claim_type") or "").strip()
        description = str(raw.get("description") or "").strip()
        spec_quote = str(raw.get("spec_quote") or "").strip()
        anchor = str(raw.get("anchor") or "").strip()
        severity = str(raw.get("severity") or "").strip()
        if severity not in _VALID_SEVERITIES:
            severity = "must_fix"

        if not node or not description:
            rejected.append(f"dropped (missing node/description): {raw}")
            continue

        if claim_type == "missing_file":
            if _normalize_path_text(node) in normalized_code:
                rejected.append(
                    f"dropped ({node!r}: missing_file claim but this file is actually present in the "
                    f"code - not really missing): {description[:200]}"
                )
                continue
            if not anchor or _normalize_path_text(anchor) not in normalized_code:
                rejected.append(
                    f"dropped ({node!r}: missing_file claim has no real, existing anchor file to "
                    f"justify it - unverifiable): {description[:200]}"
                )
                continue
        elif node != "WORKFLOW" and not _node_present(node, normalized_code):
            rejected.append(f"dropped ({node!r} does not appear in the actual code - invented name): {description[:200]}")
            continue

        if claim_type == "spec_requirement":
            if not spec_quote or _normalize(spec_quote) not in normalized_spec:
                rejected.append(
                    f"dropped ({node}: claims a spec requirement but the quote does not appear in "
                    f"SPEC.md - fabricated): {description[:200]}"
                )
                continue

        kept.append(f"{node}: {description}")
        kept_severities.append(severity)

    return kept, rejected, kept_severities


def verify_mechanical_finding_engagement(mechanical_findings: list, kept_problems: list, dismissals: dict, code_dir: Path) -> list:
    """Mechanically checks that every mechanical finding (a stub-regression hit, a test-coverage
    gap, or any future check of the same shape - see pipeline/build_stages.py) got a real response,
    not silence.

    Real incident this exists to catch (2026-09-26): three consecutive live runs handed the
    reviewer a mechanical, file-naming finding exactly as designed, and the reviewer's response
    simply did not engage it at all - no problem reported, no dismissal, verdict "pass" - a stage's
    own headline deliverable was still a stub every single time, undetected, over a million tokens
    spent. Advisory evidence plus a prompt instruction to "please investigate" already failed this
    exact test live; this closes the gap the same way verify_problems() already does for a
    fabricated claim - by verifying the reviewer's OWN stated position, not trusting it. A second,
    separate finding type (test-coverage gaps) hit the identical non-engagement pattern immediately
    after being introduced, so this function was generalized to cover any mechanical finding sharing
    the same {"file", "confidence", "message"} shape, rather than staying stub-specific.

    A finding counts as engaged if EITHER: its file appears in kept_problems (the reviewer escalated
    it as a real problem - already independently verified by verify_problems() above, nothing more
    to check), OR dismissals[finding["file"]] is a real, non-empty string that actually appears,
    verbatim (whitespace-normalized), in that file's real, current content on disk - a dismissal
    with no quote, or a quote that doesn't really appear there, is exactly the fabrication
    verify_problems() already refuses to trust for a spec_quote, applied here to a file's own real
    content instead of SPEC.md.

    Returns the findings that were NOT engaged either way - callers decide what to do with them
    (retry, or mechanically force them into the problem list) based on each finding's own
    "confidence" field, since a "low" confidence finding deserves a real chance to be dismissed
    before being forced, while a "high" (near-zero false-positive rate) finding does not.
    """
    code_dir = Path(code_dir)
    dismissals = dismissals if isinstance(dismissals, dict) else {}
    unengaged = []
    for finding in mechanical_findings:
        file = finding["file"]
        if any(file in p for p in kept_problems):
            continue  # escalated as a real problem - genuinely engaged, already grounded

        quote = str(dismissals.get(file) or "").strip()
        if not quote:
            unengaged.append(finding)
            continue
        try:
            real_text = (code_dir / file).read_text(encoding="utf-8")
        except OSError:
            unengaged.append(finding)
            continue
        if _normalize(quote) not in _normalize(real_text):
            unengaged.append(finding)  # fabricated or wrong dismissal quote

    return unengaged


def reviewer_node(state: BuildState) -> BuildState:
    spec_text = Path(state["spec_path"]).read_text(encoding="utf-8")
    code_text = _read_code(Path(state["code_dir"]))

    project_dir = state.get("project_dir")
    platform = state.get("platform", "python")
    llm = resolve_llm_for_role("reviewer", project_dir=project_dir, platform=platform)
    run_id = new_run_id()
    config = run_config(run_id, tags=["reviewer"])

    user_prompt = f"SPEC:\n{spec_text}\n\nFINISHED CODE:\n{code_text}"

    if state.get("rulings"):
        user_prompt += "\n\n" + rulings_block(state["rulings"]) + (
            "\nDo not report a finding these rulings dismiss as a false positive or out of scope."
        )

    carried = state.get("carried_findings") or []
    if carried:
        user_prompt += (
            "\n\nPREVIOUSLY REPORTED FINDINGS (from the prior run of this project; the builder was "
            "asked to fix them first). Review the code fresh as usual, and additionally check each of "
            "these against the actual code: report any that is still present as a normal finding, and "
            "do not report the ones that are fixed:\n" + "\n".join(f"- {c}" for c in carried)
        )

    # Findings that stay open until the reviewer explicitly confirms them fixed - silence never
    # clears one. External: verified by a human/Agent 1, persisted across runs. Carried must-fix:
    # from the previous run's result.
    tracked = [
        {"id": e["id"], "line": e.get("line") or finding_line(e), "unit": e.get("unit") or "",
         "label": f"externally verified by {e.get('source', 'Douglas')}", "external": True}
        for e in (state.get("external_findings") or [])
    ] + [
        {"id": f"carried-{i}", "line": line, "unit": "", "label": "carried must-fix from the previous run", "external": False}
        for i, line in enumerate(state.get("carried_must_fix") or [], 1)
    ]
    tracked_lines = {t["line"] for t in tracked}
    for i, c in enumerate(state.get("claimed_fixed") or [], 1):
        if c["finding"] in tracked_lines:
            for t in tracked:
                if t["line"] == c["finding"]:
                    t["label"] += f'; the builder claims it is already fixed, citing: "{c["quote"]}"'
            continue
        tracked.append({
            "id": f"claimed-{i}", "line": c["finding"], "unit": "", "external": False, "severity": "should_fix",
            "label": f'the builder claims it is already fixed, citing: "{c["quote"]}"',
        })
    if tracked:
        user_prompt += (
            "\n\nEXTERNALLY VERIFIED / CARRIED MUST-FIX FINDINGS. These are real, already-verified defects "
            "(a human or another agent confirmed them). Do NOT pass while any still reproduces in the code. "
            "For each one: if it is still present, report it as a normal must_fix problem; if it is fixed, "
            "you MUST say so explicitly by adding it to a top-level JSON field "
            "\"finding_confirmations\": {\"<id>\": \"<short VERBATIM quote from the current code that proves "
            "the fix>\"}. Staying silent about one does NOT count as fixed - it will be re-opened as a "
            "must_fix automatically. Where the builder cites a quote, check it against the code yourself; "
            "do not accept it on trust:\n" + "\n".join(f"- [{t['id']}] {t['line']} ({t['label']})" for t in tracked)
        )

    # A partial build is a deliberate stop at a functional base, not an unfinished failure. Without
    # this the reviewer reported every not-yet-built slice as "missing nodes" and burned all three
    # rounds on it - in one case ending ambiguous because an IF node's true branch was unwired, when
    # that branch correctly leads into a slice that does not exist yet.
    if state.get("partial_build"):
        built = ", ".join(state.get("slices_built") or []) or "(unknown)"
        remaining = ", ".join(state.get("slices_remaining") or []) or "(none)"
        user_prompt += (
            f"\n\nIMPORTANT - THIS IS A DELIBERATE PARTIAL BUILD.\n"
            f"Only these functional slices have been built, and only they are in scope for this "
            f"review: {built}.\n"
            f"These slices are NOT built yet and are explicitly out of scope: {remaining}.\n"
            f"Do NOT report anything belonging to an unbuilt slice as missing, incomplete or "
            f"broken - it is scheduled, not lost. Connections that would lead into an unbuilt slice "
            f"are intentionally absent, and a branch with no target because its target lives in a "
            f"later slice is CORRECT, not an error. Review only whether the built slices above "
            f"correctly implement the parts of the spec they cover."
        )

    # Real incident (2026-09-29): a real, already-decided waiver (a symlink-escape edge case
    # Douglas explicitly declined to fix) kept coming back as a must_fix finding every round,
    # burning a whole run's budget re-litigating something already settled. See
    # pipeline/review_waivers.py's own module docstring for the full design and why matching is
    # the LLM's own job, not a mechanical text comparison.
    waivers = load_waivers(project_dir) if project_dir else []
    waivers_section = format_waivers_for_prompt(waivers)
    if waivers_section:
        user_prompt += "\n\n" + waivers_section

    if project_dir:
        architect_path = architect_decisions_path(project_dir)
        if architect_path.exists():
            try:
                architect_decisions = json.loads(architect_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                architect_decisions = {}
            contract_findings = check_interface_contracts(Path(state["code_dir"]), architect_decisions)
            if contract_findings:
                # A mechanical grep result, not a verified problem - it can only prove ABSENCE, never
                # prove a value is wired correctly wherever it does appear. Handed to the reviewer as
                # evidence to investigate against the real code, same as everything else it decides on;
                # anything it turns into an actual problem still goes through verify_problems() below
                # exactly like a claim the reviewer thought of on its own.
                bullets = "\n".join(f"- {f}" for f in contract_findings)
                user_prompt += (
                    f"\n\nMECHANICAL INTERFACE-CONTRACT CHECK (already run against the real code you "
                    f"were given, not your own claim to verify - investigate before deciding whether "
                    f"this is a real problem):\n{bullets}\n"
                    "If, after checking the actual code, this is a genuine mismatch, report it as a "
                    "correctness_bug naming the real file/node where the wrong value actually appears. "
                    "If the value is actually present under different real code you can point to (e.g. "
                    "computed rather than literal), do not report it."
                )

    # python_units_built (2026-09-28, replacing the retired python_epics_built): plain unit-name
    # list, same shape check_stage_stub_regressions/check_stage_test_coverage_gaps already expect -
    # PM-epic/checkpoint stage gating is retired for the python path, but the mechanical checks
    # themselves are untouched, just retargeted from epic names to unit names.
    epics_built = state.get("python_units_built")
    mechanical_findings: list = []
    stub_findings: list = []
    coverage_findings: list = []
    if epics_built:
        code_dir_for_checks = Path(state["code_dir"])
        stub_findings = check_stage_stub_regressions(code_dir_for_checks, epics_built)
        if stub_findings:
            # Same mechanical-evidence pattern as the interface-contract check above - real
            # incident this exists to catch (2026-09-26): a stage's headline epic was reported
            # built (reviewer verdict "pass") while its file was still, byte-for-byte, a stub
            # written back when that epic was deferred - never actually implemented.
            bullets = "\n".join(f"- {f['message']}" for f in stub_findings)
            user_prompt += (
                f"\n\nMECHANICAL STAGE-STUB CHECK (already run against the real code you were given, "
                f"not your own claim to verify):\n{bullets}\n"
                "This is strong evidence, not a certainty - a stub marker could coincidentally appear "
                "in a comment/string unrelated to the actual implementation state. Check the real file "
                "for that epic's actual implementation. Every file named above requires a response from "
                "you - see the required \"mechanical_check_dismissals\" field above if you are not "
                "reporting it as a correctness_bug. Silence is not a valid response to a file named here."
            )
        mechanical_findings.extend(stub_findings)

        coverage_findings = check_stage_test_coverage_gaps(code_dir_for_checks, epics_built)
        if coverage_findings:
            # Same enforcement path as the stub check - a real, separate incident hit the identical
            # non-engagement pattern immediately after this was introduced (2026-09-27): an explicit,
            # itemized, stage-binding testing requirement had zero measurable effect across 4 rounds
            # while hundreds of lines of untested real logic shipped and every round passed review.
            bullets = "\n".join(f"- {f['message']}" for f in coverage_findings)
            user_prompt += (
                f"\n\nMECHANICAL TEST-COVERAGE CHECK (already run against the real code you were "
                f"given, not your own claim to verify):\n{bullets}\n"
                "This is strong evidence, not a certainty - a module could genuinely be covered "
                "through an integration test this check can't detect. Check the real test files for "
                "that module's actual coverage. Every file named above requires a response from you - "
                "see the required \"mechanical_check_dismissals\" field above if you are not reporting "
                "it as a correctness_bug. Silence is not a valid response to a file named here."
            )
        mechanical_findings.extend(coverage_findings)

    # The safer version of "let the coder and reviewer talk" (2026-09-27, Douglas's own proposal,
    # built deliberately narrower than the literal ask): the file list is mechanical ground truth
    # (recorded by the builder loop itself, not the agent's own claim); the account text is the
    # agent's own free-text explanation, clearly labeled below as unverified - never the builder's
    # full reasoning/exploration transcript, which would let a persuasive explanation substitute
    # for the independent check this reviewer exists to do (see SYSTEM_PROMPT's own instruction on
    # this section, and AGENTS.md's original reviewer-independence design). Absent entirely for an
    # n8n build or any build that never reached "done" this way - no behavior change there.
    builder_files_touched = state.get("builder_files_touched")
    builder_account = state.get("builder_account")
    if builder_files_touched or builder_account:
        files_list = "\n".join(
            f"- {f.get('action')}: {f.get('path')}" for f in (builder_files_touched or [])
        ) or "(none recorded)"
        user_prompt += (
            f"\n\nBUILDER'S OWN ACCOUNT (see SYSTEM_PROMPT - the file list is mechanically "
            f"recorded, the explanation below it is the builder's own unverified claim):\n"
            f"Files touched this round:\n{files_list}\n\n"
            f"Builder's stated reasoning:\n{builder_account or '(none given)'}"
        )

    total_tokens = state.get("total_tokens", 0)
    data = None
    kept_problems: list[str] = []
    kept_severities: list[str] = []
    unengaged_findings: list = []
    code_dir_path = Path(state["code_dir"])
    # Computed once for this whole call (not once per retry attempt below) - see with_tech_quirks'
    # own docstring for why a stable prefix across every attempt matters even for a role that isn't
    # in a long tool-calling loop.
    system_prompt = with_tech_quirks(SYSTEM_PROMPT)
    for _attempt in range(1, MAX_INTERNAL_FIX_ATTEMPTS + 1):
        is_last_attempt = _attempt == MAX_INTERNAL_FIX_ATTEMPTS
        messages = [("system", system_prompt), ("user", user_prompt)]
        try:
            response = with_retries_nonempty(lambda: llm.invoke(messages, config=config))
        except Exception as exc:
            response = attempt_fallback(
                "reviewer", lambda fb_llm: fb_llm.invoke(messages, config=config),
                project_dir=project_dir, platform=platform, primary_error=exc,
            )
        total_tokens += extract_tokens(response)
        try:
            data = _parse_response(response.content)
        except ValueError as exc:
            # Same fragility class the builder hit: a model producing malformed JSON. Retry with
            # the parse error as feedback instead of crashing the whole run over one bad completion.
            data = None
            user_prompt += (
                f"\n\nYour previous response could not be parsed as valid JSON: {exc}\n"
                "Respond again with ONLY the JSON object specified, valid and correctly escaped."
            )
            continue

        for lesson in data.get("tech_quirks") or []:
            if isinstance(lesson, str) and lesson.strip():
                record_tech_quirk_entry(lesson, source="reviewer")

        unwaived_problems, waiver_drops = _apply_waivers(data.get("problems", []), waivers)
        for note in waiver_drops:
            print(f"DEBUG: reviewer claim {note}", file=sys.stderr)

        kept_problems, rejected, kept_severities = verify_problems(unwaived_problems, spec_text, code_text)
        for note in rejected:
            print(f"DEBUG: reviewer claim rejected - {note}", file=sys.stderr)
        for note in _flag_possible_unwaived_matches(kept_problems, waivers):
            print(f"DEBUG: {note}", file=sys.stderr)

        unengaged_findings = (
            verify_mechanical_finding_engagement(
                mechanical_findings, kept_problems, data.get("mechanical_check_dismissals"), code_dir_path
            )
            if mechanical_findings else []
        )
        # A "low" confidence finding (predates the exact stub marker, real but small false-positive
        # risk) gets one genuine retry chance to be engaged before it's forced in below. A "high"
        # finding (an exact stub marker, or a test-coverage gap - false positives are effectively
        # impossible for either) does not get a chance to be silently ignored a second time either -
        # it goes straight to forced.
        unengaged_low = [f for f in unengaged_findings if f["confidence"] != "high"]

        claims_need_retry = data.get("verdict") == "fail" and not kept_problems and rejected
        findings_need_retry = bool(unengaged_low) and not is_last_attempt

        if claims_need_retry or findings_need_retry:
            if claims_need_retry:
                # Every single claim was unverifiable - give it one chance to self-correct with that
                # fact stated plainly, rather than either trusting a "fail" with zero real evidence
                # (exactly the incident this whole check exists to prevent) or silently flipping it
                # to "pass" ourselves.
                user_prompt += (
                    f"\n\nNone of your {len(rejected)} claimed problem(s) could be verified - every "
                    "\"spec requires X\" claim had no matching quote in the actual SPEC text, and/or "
                    "every referenced node name does not exist in the actual code. If you have a "
                    "genuine, verifiable issue, state it again with a real quote or a real node name "
                    "this time. Otherwise respond with verdict \"pass\" and an empty problems list."
                )
            if findings_need_retry:
                # Real incident this retry exists to catch (2026-09-26/27): the reviewer's response
                # simply did not address a mechanical finding at all - no problem, no dismissal.
                unresolved = ", ".join(f["file"] for f in unengaged_low)
                user_prompt += (
                    f"\n\nYou did not engage {len(unengaged_low)} mechanical finding(s) at all: "
                    f"{unresolved}. Silence is not a valid response to a file named in a MECHANICAL "
                    "... CHECK section above - report it as a correctness_bug, or add it to "
                    "mechanical_check_dismissals with a real, verbatim quote from that exact file."
                )
            data = None
            continue

        break

    if data is None:
        # Ran out of attempts either on malformed JSON or on nothing but unverifiable claims. After
        # this many real chances to produce a grounded problem, "no verifiable problems found" is the
        # only defensible mechanical conclusion - not a "fail" resting on nothing. unengaged_findings
        # is deliberately NOT reset here - it still holds whatever the last successfully-parsed
        # attempt left unengaged, and is force-applied below regardless of how the claim-verification
        # path above concluded, so a mechanical finding is never silently dropped either way.
        data = {"verdict": "pass", "problems": []}
        kept_problems = []
        kept_severities = []

    # Snapshot BEFORE force-injection below adds anything, MUST_FIX ONLY (2026-09-29) - so a
    # should_fix/nitpick item alongside a coverage gap doesn't wrongly disqualify the per-batch
    # coverage-round-budget relaxation below, which cares whether a GENUINE BLOCKING problem exists
    # alongside the coverage debt, not whether a finding of any severity does.
    confirmations = data.get("finding_confirmations") if isinstance(data, dict) else None
    confirmations = confirmations if isinstance(confirmations, dict) else {}
    norm_code = " ".join(code_text.split())
    still_carried = list(state.get("carried_must_fix") or [])
    confirmed_ids = set()
    for t in tracked:
        quote = " ".join(str(confirmations.get(t["id"], "")).split())
        if quote and quote in norm_code:
            confirmed_ids.add(t["id"])
            if t["external"]:
                if project_dir:
                    resolve_finding(project_dir, t["id"], "reviewer", quote[:300])
            else:
                still_carried = [c for c in still_carried if c != t["line"]]
            print(f"DEBUG: reviewer confirmed {t['id']} fixed", file=sys.stderr)
            continue
        node = t["unit"].lower()
        if node and any(node in p.lower() for p in kept_problems):
            continue
        print(f"DEBUG: {t['id']} not confirmed fixed by the reviewer - re-opening as must_fix", file=sys.stderr)
        kept_problems.append(f"{t['line']} ({t['label']}; the reviewer did not confirm it fixed)")
        kept_severities.append(t.get("severity", "must_fix"))

    verified_must_fix_before_force = [p for p, s in zip(kept_problems, kept_severities) if s == "must_fix"]
    unengaged_stub = [f for f in unengaged_findings if f in stub_findings]
    unengaged_coverage = [f for f in unengaged_findings if f in coverage_findings]

    if unengaged_findings:
        # Mechanically force it in - never leave a properly-fed, properly-instructed, still-ignored
        # finding to simply vanish because the model chose not to engage it, no matter how many
        # chances it already had. This is the real backstop the advisory-only version of this check
        # (an earlier attempt at this same fix) did not have, and needed: mechanical evidence reached
        # the reviewer exactly as designed, three consecutive times, and it said nothing and passed
        # anyway - a prompt asking nicely was already proven insufficient before this was added.
        # Always must_fix (2026-09-29) - a mechanical finding the reviewer never even engaged with is
        # never a nitpick.
        for finding in unengaged_findings:
            print(f"DEBUG: forcing unengaged mechanical finding into problems - {finding['message']}", file=sys.stderr)
            kept_problems.append(
                f"{finding['file']}: reviewer did not engage this mechanical finding (no "
                f"correctness_bug reported, no verified dismissal) - {finding['message']}"
            )
            kept_severities.append("must_fix")

    # Severity-driven verdict (2026-09-29, Douglas's explicit direction, relayed via
    # video-generator-78 - "must haves must be delivered", should_fix/nitpick must not force a round
    # on their own). verdict is MECHANICALLY derived from must_fix presence, never trusted directly
    # from the model's own top-level "verdict" field - same "mechanical check over self-report"
    # principle as everything else in this function. A should_fix/nitpick-only round is a pass,
    # regardless of what the model itself said at the top level.
    ruled = load_answers(project_dir) if project_dir else []
    if ruled:
        keep = set(drop_dismissed(kept_problems, ruled))
        pairs = [(p, s) for p, s in zip(kept_problems, kept_severities) if p in keep]
        for dropped in set(kept_problems) - keep:
            print(f"DEBUG: dropping finding dismissed by a human ruling - {dropped[:200]}", file=sys.stderr)
        kept_problems = [p for p, _ in pairs]
        kept_severities = [s for _, s in pairs]
    must_fix = [p for p, s in zip(kept_problems, kept_severities) if s == "must_fix"]
    should_fix = [p for p, s in zip(kept_problems, kept_severities) if s == "should_fix"]
    nitpicks = [p for p, s in zip(kept_problems, kept_severities) if s == "nitpick"]

    round_num = state.get("review_round", 0) + 1
    verdict = "fail" if must_fix else "pass"

    # What actually gets fed back to the builder: must_fix always drives a round; should_fix rides
    # along ONLY when a round is already happening (it never triggers one on its own, per Douglas's
    # explicit direction). Nitpicks never reach the builder at all, only the run's own reported
    # output below (result["review_nitpicks"]), so nothing observed is silently lost even though
    # nothing loops on it either.
    review_problems = must_fix + should_fix if verdict == "fail" else []

    # Douglas's explicit direction (2026-09-27): MAX_ROUNDS should cap rounds spent on the CURRENT
    # batch of (at most 5) test-coverage findings, not the whole run's total coverage backlog - a
    # 22-gap project split into ~5 batches must not have its later batches inherit an already-spent
    # budget from earlier ones that has nothing to do with them. Scoped narrowly to fails that are
    # driven PURELY by unresolved coverage debt (no genuine reviewer-verified must_fix problem, no
    # unresolved stub-regression finding) - a real must_fix bug, whether alone or alongside coverage
    # debt, still uses the original flat per-run cap unchanged, since letting a genuine bug's round
    # budget silently balloon was never asked for and isn't safe to assume.
    rounds_cap = max_rounds(project_dir)
    reviewer_tokens_cap = MAX_REVIEWER_TOKENS_PER_ROUND * rounds_cap
    pure_coverage_fail = (
        verdict == "fail"
        and not verified_must_fix_before_force
        and not unengaged_stub
        and bool(unengaged_coverage)
    )
    if pure_coverage_fail:
        batch_key = ",".join(sorted(f["file"] for f in coverage_findings))
        prev_batch_key = state.get("coverage_batch_key")
        prev_batch_rounds = state.get("coverage_batch_rounds") or 0
        batch_rounds = prev_batch_rounds + 1 if batch_key == prev_batch_key else 1
        round_budget_exhausted = batch_rounds >= rounds_cap
    else:
        # Not a coverage-only fail (a pass, a genuine bug, or an unresolved stub finding) - no batch
        # to track, and no stale batch state should survive into whatever comes next.
        batch_key = None
        batch_rounds = 0
        round_budget_exhausted = round_num >= rounds_cap

    # Reviewer-only cumulative spend this run (see constants.py's MAX_REVIEWER_TOKENS_PER_RUN) - the
    # delta THIS round itself consumed (including any internal retries above) added onto whatever the
    # reviewer had already spent in earlier rounds. Deliberately never mixes in the builder's own
    # (legitimately large, unit-by-unit) spend, which total_tokens above still carries for reporting.
    reviewer_tokens_used = state.get("reviewer_tokens_used", 0) + (total_tokens - state.get("total_tokens", 0))

    if project_dir:
        clear_provisional(project_dir)
    result: BuildState = {
        **state,
        "review_round": round_num,
        "review_verdict": verdict,
        "review_problems": review_problems,
        # Full severity breakdown, always populated regardless of verdict (2026-09-29, Douglas's
        # explicit direction: "report all severities in the run output so nothing is hidden") - a
        # should_fix/nitpick never loops, but must still be visible to Douglas, not silently dropped
        # just because it didn't block anything.
        "review_must_fix": must_fix,
        "review_should_fix": should_fix,
        "review_nitpicks": nitpicks,
        "total_tokens": total_tokens,
        "reviewer_tokens_used": reviewer_tokens_used,
        "coverage_batch_key": batch_key,
        "coverage_batch_rounds": batch_rounds,
        "round_budget_exhausted": round_budget_exhausted,
        "carried_must_fix": still_carried,
        "claimed_fixed": [],
        "external_findings": [e for e in (state.get("external_findings") or []) if e["id"] not in confirmed_ids],
    }
    # A 1-round budget (Douglas: "i expect reviewer 1 round to go to builder not to me"): a failed
    # review is not the end - its findings go to the builder for one fix pass, then the run ends with
    # no second review. Only for a real must_fix fail, and never past the reviewer's token ceiling.
    result["final_fix_pass"] = False
    if rounds_cap == 1 and verdict == "fail" and round_budget_exhausted and must_fix:
        result["final_fix_pass"] = True
        result["round_budget_exhausted"] = False
    if reviewer_tokens_used >= reviewer_tokens_cap:
        result["final_fix_pass"] = False
        result["review_verdict"] = "fail"
        result["round_budget_exhausted"] = True
        result["review_problems"] = list(result["review_problems"] or []) + [
            f"Run stopped: reviewer's own cumulative spend this run exceeded its token budget "
            f"({reviewer_tokens_used} tokens across {round_num} round(s))."
        ]

    # Build-journal entry for this round (2026-09-28, Douglas's firm requirement - see
    # pipeline/build_journal.py) - mechanical, no new LLM call: a short summary of THIS round's own
    # verdict/kept-problems, reusing what was already computed above. Tagged REVIEW_ROUND_UNIT so
    # every later unit's own task sees it (a round's feedback isn't scoped to one unit). Uses
    # review_problems (what was actually asked of the builder this round), not every kept finding -
    # a nitpick that never reached the builder doesn't belong in "what this round's feedback was".
    if project_dir:
        problems_summary = "; ".join(review_problems[:3]) if review_problems else "no problems found"
        record_journal_entry(
            project_dir, REVIEW_ROUND_UNIT, "reviewer",
            f"Round {round_num}: verdict {verdict}. {problems_summary}",
        )
    return result
