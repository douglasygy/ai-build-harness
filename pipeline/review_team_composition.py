"""Team-composition bounce between PM and Architect: a separate, dedicated gate (not bundled with
their other pre-build output, per Douglas's explicit choice) deciding whether THIS project
genuinely needs more than the standing roster - an additional build agent, a DevOps role, a second
independent reviewer, etc.

Meant to run after the concept is locked into SPEC.md (uses the approved spec as context) - team
shape is a downstream question of what's actually being built, not a concept-stage decision. It can
also run retrospectively against an already-built project's build summary (the shape a peer session
first tried this in practice) via --build-summary.

Neither model decides for Douglas: this produces a proposal only. Turning "additional_agents" into
individual approval prompts (so Douglas can approve a DevOps role but reject a second reviewer, or
vice versa) happens outside this script, the same way review_concept.py's decisions_needed become
real AskUserQuestion prompts. Approving nothing (an empty additional_agents list) is a fine, and the
DEFAULT expected, answer - most projects need nothing beyond the standing roster.

Any approved addition is scoped to THIS project alone: Claude Code (or whichever tool is driving
this) is responsible for writing it into the project's own models.yaml (never config/models.yaml)
and recording it via pipeline/team_roster.py's save_team_roster() - the global roster and every
other project stay untouched. See AGENTS.md "Team composition gate (PM + Architect)".

Usage:
    python -m pipeline.review_team_composition --spec path/to/SPEC.md \
        [--build-summary path/to/run_result.json] [--project-dir <dir>]

Prints ONE JSON object to stdout:
    {
      "summary": "...",
      "additional_agents": [
        {"role_name": "...", "reason": "...", "recommended_model": "...", "model_notes": "..."}
      ],
      "no_change_rationale": "..." or null
    }
or, on failure after retries: {"status": "error", "message": "..."}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .constants import MAX_TEAM_COMPOSITION_ROUNDS
from .json_extract import extract_json_object
from .langfuse_setup import new_run_id, run_config
from .model_fallback import attempt_fallback, resolve_llm_for_role
from .retry import with_retries_nonempty
from .token_guard import extract_tokens

PM_SYSTEM_PROMPT = """You are the PM for this project, evaluating TEAM COMPOSITION ONLY - not
product scope or epics/stories (that is a separate job). Read the approved spec (and build summary,
if given) and decide, from a coordination/workload angle: does this SPECIFIC project genuinely need
more than the standing roster (one build agent, one reviewer, the core intake/business-analyst
stages)? Concrete triggers worth naming: genuinely independent, parallelizable workflows or modules
that don't share state; a scale of work where a single build agent's serial pace would be a real
bottleneck Douglas would actually feel. Most projects need nothing extra - if you cannot point to a
concrete, project-specific reason, say so plainly. NEVER invent a need to pad the list; an empty
list is a fine, and the default expected, answer. Respond with ONLY a JSON object:
{"proposals": [{"role_name": "...", "reason": "..."}], "closing_note": "..."}
"""

ARCHITECT_SYSTEM_PROMPT = """You are the Architect for this project, evaluating TEAM COMPOSITION
ONLY - not system design itself (that is a separate job). Read the approved spec, the build summary
if given, and the PM's proposals. From a system-structure/risk angle: does this project have
genuinely independent components that could be built in parallel by a second build agent of a
DIFFERENT vendor (never the same vendor as the first build agent - this pipeline never lets a role
review or duplicate work using the same vendor it is meant to be independent of)? Does it need
infrastructure/deployment expertise beyond application logic (a DevOps role)? Does its risk profile
(money movement, PII, safety-critical behavior) warrant a second, independently-vendored reviewer?
For each addition you keep or add, name an EXACT OpenRouter model slug (e.g. "vendor/model-name")
suited to that role's actual job, with a one-line reason for that specific model choice - never a
placeholder or a generic "a capable model". Push back on any PM proposal that is not concretely
justified for THIS project. NEVER invent a need to pad the list; an empty list is a fine, and the
default expected, answer. Respond with ONLY a JSON object: {"agreed": [{"role_name": "...",
"reason": "...", "recommended_model": "...", "model_notes": "..."}], "rejected": [{"role_name":
"...", "why": "..."}], "closing_note": "..."}
"""

SYNTHESIS_SYSTEM_PROMPT = """You are the Architect, closing out this team-composition review with
the PM. Consolidate the whole exchange into a final handoff for Douglas: either a short list of
concretely-justified additional agents (each with an exact recommended OpenRouter model and why),
or a plain statement that the standing roster is sufficient for this project and why. Respond with
ONLY a JSON object: {"summary": "...", "additional_agents": [{"role_name": "...", "reason": "...",
"recommended_model": "...", "model_notes": "..."}], "no_change_rationale": "..." or null}
"""


def _call(role: str, system_prompt: str, user_prompt: str, run_id: str, tag: str, project_dir=None) -> tuple[dict, int]:
    llm = resolve_llm_for_role(role, project_dir=project_dir)
    config = run_config(run_id, tags=[role, tag])
    messages = [("system", system_prompt), ("user", user_prompt)]
    try:
        response = with_retries_nonempty(lambda: llm.invoke(messages, config=config))
    except Exception as exc:
        response = attempt_fallback(
            role, lambda fb_llm: fb_llm.invoke(messages, config=config), project_dir=project_dir, primary_error=exc
        )
    return extract_json_object(response.content, context=f"{role} response"), extract_tokens(response)


def _build_context_block(spec_text: str, build_summary_text: Optional[str]) -> str:
    block = f"APPROVED SPEC:\n{spec_text}"
    if build_summary_text:
        block = f"{block}\n\nBUILD SUMMARY (what has actually been built so far, if anything):\n{build_summary_text}"
    return block


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="Path to the approved SPEC.md")
    parser.add_argument(
        "--build-summary",
        help="Path to a run_result.json or similar build summary, for a retrospective review of an already-built project",
    )
    parser.add_argument(
        "--project-dir",
        help="Project folder, so this project's own models.yaml (if any) overrides the global model config",
    )
    args = parser.parse_args()
    project_dir = args.project_dir

    spec_text = Path(args.spec).read_text(encoding="utf-8")
    build_summary_text = Path(args.build_summary).read_text(encoding="utf-8") if args.build_summary else None
    run_id = new_run_id()

    transcript = _build_context_block(spec_text, build_summary_text)
    total_tokens = 0

    try:
        for round_num in range(1, MAX_TEAM_COMPOSITION_ROUNDS + 1):
            pm_out, tokens = _call(
                "pm",
                PM_SYSTEM_PROMPT,
                f"{transcript}\n\nRound {round_num}: propose any team-composition additions (or none).",
                run_id,
                "team_composition_pm",
                project_dir,
            )
            total_tokens += tokens
            transcript += f"\n\n[Round {round_num} - PM]: {json.dumps(pm_out)}"

            if not pm_out.get("proposals"):
                break

            architect_out, tokens = _call(
                "architect",
                ARCHITECT_SYSTEM_PROMPT,
                f"{transcript}\n\nRound {round_num}: respond to PM's proposals above.",
                run_id,
                "team_composition_architect",
                project_dir,
            )
            total_tokens += tokens
            transcript += f"\n\n[Round {round_num} - Architect]: {json.dumps(architect_out)}"

        final, tokens = _call(
            "architect",
            SYNTHESIS_SYSTEM_PROMPT,
            transcript,
            run_id,
            "team_composition_synthesis",
            project_dir,
        )
        total_tokens += tokens
    except Exception as exc:  # OpenRouter outage, exhausted retries, malformed model output, etc.
        print(json.dumps({"status": "error", "message": str(exc), "total_tokens": total_tokens}))
        return 1

    final["total_tokens"] = total_tokens
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
