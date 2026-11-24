"""preference_analyst: the EVALUATIVE half of the two-tier memory architecture (the "Gatekeeper") -
reads memory/decision_log.jsonl (Douglas's own explicit decisions), memory/build_log.jsonl
(build_scribe's objective per-run summaries, see pipeline/build_scribe.py), and
memory/agent1_action_log.jsonl (Agent 1's own self-reported actions in the four reserved
categories, see pipeline/agent1_action_log.py) and refreshes memory/decision-patterns.md - a
living, git-tracked synthesis of how Douglas actually decides things, what lessons each build run
leaves behind, and whether Agent 1 has overreached into a decision that was never actually his to
make. Meant to be fed as context into every gated role's prompts (concept-advocate/business-analyst,
PM, Architect) going forward - see AGENTS.md "Decision log & preference learning".

The overreach check is this pipeline's answer to a real incident: an earlier session picked a model
for a new role and used it without asking Douglas first, and nothing caught it automatically - he
caught it himself. This is the automated backstop for the next time that almost happens - it cannot
prevent an unlogged action, but it turns a logged one into something the Gatekeeper independently
checks against Douglas's actual decisions rather than trusting Agent 1's own account.

Runs automatically after every single build_and_review.py call (pipeline/build_and_review.py calls
synthesize_patterns()/write_patterns() directly, best-effort - a failure here never breaks a real
build's result). Also runnable standalone for a manual refresh:

    python -m pipeline.review_decisions

Overwrites memory/decision-patterns.md with a fresh synthesis of the FULL logs every time - the
logs themselves (not the doc) are the durable source of truth, so the doc is always safe to
regenerate from scratch rather than incrementally patched, and never silently drifts from what they
actually contain. To guard against a monolithic rewrite silently dropping a previously-identified
real pattern (the "context collapse" failure mode a self-modifying context loop is prone to), the
PREVIOUS synthesis is fed back in as additional input every time, with an explicit instruction not
to drop something from it without saying why.

Model scorecard: at genuine project completion (build_and_review.py passes
include_model_scorecard=True exactly when its own memory_review_required flag is set - see AGENTS.md
"Model scorecard"), the Gatekeeper also scores each model that actually participated, using
mechanically pre-counted failure data (never asking the model to do its own arithmetic) plus the
qualitative lesson content already in the logs. Runs at project-boundary cadence, not every call, to
keep this heavier judgment call from adding cost to the routine per-run synthesis.
"""
from __future__ import annotations

import json
import sys

from .agent1_action_log import load_actions
from .build_log import load_builds
from .decision_log import DECISION_LOG_PATH, load_decisions
from .json_extract import extract_json_object
from .langfuse_setup import new_run_id, run_config
from .model_fallback import attempt_fallback, resolve_llm_for_role
from .retry import with_retries_nonempty
from .token_guard import extract_tokens

PATTERNS_PATH = DECISION_LOG_PATH.parent / "decision-patterns.md"

SYSTEM_PROMPT = """You are the Gatekeeper: the evaluative half of a two-tier memory system for an AI
build pipeline directed by Douglas (a non-developer running an SME automation business).
You are given: DECISIONS (things Douglas explicitly decided, with his own stated reasons), BUILDS
(objective, non-evaluative summaries of what actually happened during build runs, written by a
separate descriptive-only role that never judges anything), AGENT1_ACTIONS (Agent 1's own
self-reported actions in four categories Douglas has reserved exclusively for himself: server/infra
setup, agent model selection, tool selection, and new folder/directory creation or access - Agent 1
may only suggest and recommend in these areas, never decide), and PREVIOUS_SYNTHESIS (the last
version of the document you are regenerating).

Your job:
1. Find real, honest patterns in how Douglas decides things, and extract concrete lessons from what
   the build runs reveal - so future proposals from this pipeline's roles can pre-empt his likely
   answer, and so recurring build problems get named instead of silently repeating.
2. AUDIT AGENT1_ACTIONS against DECISIONS. For each logged action, look for a plausible matching
   entry in DECISIONS (by topic and rough timing, not exact wording) that shows Douglas actually
   approved it. An action with no plausible matching decision is an overreach - flag it plainly.
   This audit exists because of a real incident: Agent 1 once picked and used a model for a new
   role without ever asking Douglas first, and nothing caught it automatically until he noticed
   himself. Be genuinely skeptical here - a vague or generic-sounding "description" is not itself
   evidence of approval, and Agent 1's own account of its actions should not be taken at face value
   any more than any other unverified claim.
3. Compare against PREVIOUS_SYNTHESIS. If a pattern, lesson, or flag that appeared there is not
   showing up in your new analysis, either carry it forward (if still supported) or explicitly note
   that it no longer holds and why. Never silently drop something a prior synthesis identified -
   a monolithic rewrite that quietly loses earlier findings is a known failure mode of exactly this
   kind of self-updating document, and this document has no other mechanism to catch it.

You are NOT deciding anything yourself, and you are NOT inventing rules never actually stated or
problems never actually seen - only naming patterns, lessons, and overreach flags concretely
supported by these logs. If the logs are too small or too mixed to support a real pattern yet, say
so plainly rather than overreaching from a couple of data points. Write in plain, direct language -
this document is read by both Douglas and by other AI roles in this pipeline as working context,
not as a report.

Respond with ONLY a JSON object:
{"patterns": [{"theme": "...", "pattern": "...", "evidence": ["...", ...], "confidence": "low" or
"medium" or "high"}], "build_lessons": [{"lesson": "...", "evidence": ["..."]}],
"overreach_flags": [{"action": "...", "category": "...", "concern": "..."}],
"carried_forward_note": "..." or null (mention here anything from PREVIOUS_SYNTHESIS you kept or
explicitly dropped and why - omit entirely if PREVIOUS_SYNTHESIS was empty),
"open_questions": ["... things the logs don't yet have enough data to answer ..."],
"note_if_logs_too_thin": "..." or null}
"""

MODEL_SCORECARD_INSTRUCTION = """
ADDITIONAL TASK - MODEL SCORECARD. Douglas's own words: "agent models are not set in stone. an
agent working well today could perform the worst tomorrow, and i have no control besides switching
them out." You are given MODEL_ROUND_FAILURES - a mechanical, pre-counted tally of failing rounds
per (role, model), computed in Python, not by you - never recompute or second-guess these counts,
only interpret them alongside the qualitative build_lessons and DECISIONS history.

For each model that actually participated (per BUILDS and MODEL_ROUND_FAILURES), give an honest
assessment: is its real-world performance in THIS pipeline still good, mixed, or concerning? Ground
every score in specific evidence from the logs - never a generic opinion about a model's reputation.
A model with zero or few failing rounds and no negative build_lessons is doing fine and needs no
comment beyond that. Do not manufacture concern where the evidence doesn't support it.

If a model's evidence is genuinely concerning (rising failure rate, recurring specific problems in
build_lessons), say so plainly and suggest a replacement ONLY if a concrete alternative was already
named somewhere in DECISIONS (e.g. an alternative considered and rejected when the role was first
set up, that might be worth revisiting) - cite that entry. If no such alternative exists in the
logs, say the evidence supports reconsidering this model but do NOT invent a specific replacement
model name that isn't grounded in something Douglas has actually seen - instead say Agent 1 should
research current alternatives with Douglas rather than act on a guess.

Add to your JSON response: "model_scorecard": [{"role": "...", "model": "...", "verdict": "good" or
"mixed" or "concerning", "evidence": ["..."], "suggested_replacement": "..." or null,
"replacement_basis": "..." or null (the DECISIONS entry that named this alternative, if any)}]
"""


def compute_build_trends(builds: list) -> dict:
    """Mechanical (no LLM) aggregate stats from build_log.jsonl run-scope entries. Deliberately not
    asked of the Gatekeeper's model - counting and averaging is arithmetic, not judgment, and
    arithmetic from an LLM is a reliability risk this function removes entirely. Returns {} when
    there isn't at least one run-scope entry with a recorded round count.
    """
    runs = [b for b in builds if b.get("scope", "run") == "run" and isinstance(b.get("rounds"), int)]
    if not runs:
        return {}
    runs_sorted = sorted(runs, key=lambda b: b.get("timestamp", ""))
    total = len(runs_sorted)
    passed = sum(1 for b in runs_sorted if b.get("status") == "pass")
    half = max(1, total // 2)
    earlier = runs_sorted[: total - half] if total > half else []
    recent = runs_sorted[-half:]

    def _avg_rounds(subset: list):
        vals = [b["rounds"] for b in subset if b.get("status") == "pass"]
        return round(sum(vals) / len(vals), 2) if vals else None

    return {
        "total_runs": total,
        "pass_rate": round(passed / total, 2),
        "avg_rounds_recent": _avg_rounds(recent),
        "avg_rounds_earlier": _avg_rounds(earlier) if earlier else None,
    }


def compute_model_round_failures(builds: list) -> dict:
    """Mechanical count of failing rounds per (role, model), from build_log.jsonl round-scope
    entries - pre-aggregated so the Gatekeeper judges trends from real counts instead of
    re-deriving arithmetic from a raw list itself.
    """
    counts: dict = {}
    for b in builds:
        if b.get("scope") != "round":
            continue
        key = f"{b.get('role', 'unknown')} ({b.get('model') or 'model unrecorded'})"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _render_markdown(
    analysis: dict, decision_count: int, build_count: int, action_count: int = 0, trends: dict | None = None,
) -> str:
    lines = [
        "# Decision patterns",
        "",
        "Living synthesis of `memory/decision_log.jsonl`, `memory/build_log.jsonl`, and"
        " `memory/agent1_action_log.jsonl`, regenerated by `pipeline/review_decisions.py` - never"
        " hand-edit this file, edit the logs and re-run instead.",
        "",
        f"Synthesized from {decision_count} logged decision(s), {build_count} logged build run(s),"
        f" and {action_count} logged Agent 1 action(s).",
        "",
    ]

    if trends:
        lines.append("## Build trends (mechanical, not model-judged)")
        lines.append("")
        lines.append(f"- {trends['total_runs']} completed run(s) with a recorded round count.")
        lines.append(f"- Overall pass rate: {trends['pass_rate'] * 100:.0f}%.")
        if trends.get("avg_rounds_earlier") is not None:
            lines.append(
                f"- Average rounds to pass: {trends['avg_rounds_recent']} recently vs "
                f"{trends['avg_rounds_earlier']} earlier."
            )
        elif trends.get("avg_rounds_recent") is not None:
            lines.append(f"- Average rounds to pass so far: {trends['avg_rounds_recent']}.")
        lines.append("")

    if analysis.get("overreach_flags"):
        lines.append("## ⚠ Overreach flags - Agent 1 actions with no matching Douglas decision")
        lines.append("")
        for flag in analysis["overreach_flags"]:
            lines.append(f"- **[{flag.get('category', 'unknown')}]** {flag.get('action', '')}")
            lines.append(f"  - {flag.get('concern', '')}")
        lines.append("")

    if analysis.get("model_scorecard"):
        lines.append("## Model scorecard")
        lines.append("")
        for entry in analysis["model_scorecard"]:
            lines.append(f"### {entry.get('role', 'unknown')} - `{entry.get('model', 'unknown')}` ({entry.get('verdict', 'unknown')})")
            for evidence in entry.get("evidence") or []:
                lines.append(f"- {evidence}")
            if entry.get("suggested_replacement"):
                lines.append(f"- **Consider:** {entry['suggested_replacement']}")
                if entry.get("replacement_basis"):
                    lines.append(f"  - Basis: {entry['replacement_basis']}")
            lines.append("")

    if analysis.get("note_if_logs_too_thin"):
        lines += ["> " + analysis["note_if_logs_too_thin"], ""]

    for pattern in analysis.get("patterns") or []:
        lines.append(f"## {pattern.get('theme', 'Untitled pattern')} (confidence: {pattern.get('confidence', 'unknown')})")
        lines.append("")
        lines.append(pattern.get("pattern", ""))
        lines.append("")
        for evidence in pattern.get("evidence") or []:
            lines.append(f"- {evidence}")
        lines.append("")

    if analysis.get("build_lessons"):
        lines.append("## Lessons from build runs")
        lines.append("")
        for lesson in analysis["build_lessons"]:
            lines.append(f"- {lesson.get('lesson', '')}")
            for evidence in lesson.get("evidence") or []:
                lines.append(f"  - {evidence}")
        lines.append("")

    if analysis.get("carried_forward_note"):
        lines += ["> Carried forward from the previous synthesis: " + analysis["carried_forward_note"], ""]

    if analysis.get("open_questions"):
        lines.append("## Open questions - not enough data yet")
        lines.append("")
        for question in analysis["open_questions"]:
            lines.append(f"- {question}")
        lines.append("")

    return "\n".join(lines)


def synthesize_patterns(project_dir=None, include_model_scorecard: bool = False) -> tuple[dict, int, int, int, int]:
    """Returns (analysis, decision_count, build_count, action_count, tokens_used). Raises on
    failure - callers decide how to degrade (build_and_review.py's call site treats this as
    best-effort).

    `include_model_scorecard`: only True at genuine project completion (see AGENTS.md "Model
    scorecard") - the routine per-call synthesis stays cheap; the heavier per-model judgment call
    only runs at the cadence Douglas actually asked for it.
    """
    decisions = load_decisions()
    builds = load_builds()
    actions = load_actions()
    previous_synthesis = PATTERNS_PATH.read_text(encoding="utf-8") if PATTERNS_PATH.exists() else ""

    system_prompt = SYSTEM_PROMPT
    payload = {"decisions": decisions, "builds": builds, "agent1_actions": actions}
    if include_model_scorecard:
        system_prompt += MODEL_SCORECARD_INSTRUCTION
        payload["model_round_failures"] = compute_model_round_failures(builds)
    # previous_synthesis always goes last, regardless of what else is in the payload - it's the one
    # thing that must NOT lose attention to a lost-in-middle effect, since its whole job is telling
    # the model not to silently drop something it found before.
    payload["previous_synthesis"] = previous_synthesis

    llm = resolve_llm_for_role("preference_analyst", project_dir=project_dir)
    run_id = new_run_id()
    config = run_config(run_id, tags=["preference_analyst", "decision_synthesis"])
    messages = [
        ("system", system_prompt),
        ("user", json.dumps(payload, indent=2)),
    ]
    try:
        response = with_retries_nonempty(lambda: llm.invoke(messages, config=config))
    except Exception as exc:
        response = attempt_fallback(
            "preference_analyst", lambda fb_llm: fb_llm.invoke(messages, config=config),
            project_dir=project_dir, primary_error=exc,
        )
    analysis = extract_json_object(response.content, context="preference_analyst response")
    tokens = extract_tokens(response)
    return analysis, len(decisions), len(builds), len(actions), tokens


def write_patterns(analysis: dict, decision_count: int, build_count: int, action_count: int = 0, builds: list | None = None) -> None:
    trends = compute_build_trends(builds) if builds is not None else None
    PATTERNS_PATH.write_text(
        _render_markdown(analysis, decision_count, build_count, action_count, trends), encoding="utf-8"
    )


def main() -> int:
    if not load_decisions() and not load_builds() and not load_actions():
        print(f"No decisions, builds, or actions logged yet - nothing to synthesize.", file=sys.stderr)
        return 0

    try:
        analysis, decision_count, build_count, action_count, tokens = synthesize_patterns(include_model_scorecard=True)
    except Exception as exc:  # OpenRouter outage, exhausted retries, malformed model output, etc.
        print(f"Could not synthesize decision patterns: {exc}", file=sys.stderr)
        return 1

    write_patterns(analysis, decision_count, build_count, action_count, builds=load_builds())
    print(
        f"Wrote {PATTERNS_PATH} from {decision_count} decision(s), {build_count} build(s), and "
        f"{action_count} action(s) ({tokens} tokens)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
