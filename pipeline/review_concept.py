"""Concept-stage bounce between two AI perspectives before anything reaches the human.

concept_advocate (representing Agent 1 / Claude Code's drafting perspective) and
business_analyst (an independent, non-Anthropic second opinion) go back and forth for up
to MAX_CONCEPT_ROUNDS rounds - raising gaps/edge cases, resolving what's safe to resolve as
a coding/implementation detail, and refusing to resolve anything that's actually a
product-intent decision. Neither model ever decides what the product should do. At the end
they hand back ONE structured JSON object; Claude Code (the real Agent 1) reads it and turns
"decisions_needed" into actual decision prompts for Douglas (e.g. via AskUserQuestion),
folding "resolved_assumptions" into the spec.

Usage:
    python -m pipeline.review_concept --draft path/to/draft.md [--previous path/to/old_spec.md]

Prints ONE JSON object to stdout:
    {
      "summary": "...",
      "resolved_assumptions": ["...", ...],
      "decisions_needed": [
        {"question": "...", "options": ["...", ...], "why_it_matters": "..."},
        ...
      ]
    }
or, on failure after retries:
    {"status": "error", "message": "..."}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .config_loader import warn_if_business_analyst_is_anthropic, warn_if_concept_roles_share_vendor
from .constants import MAX_CONCEPT_ROUNDS
from .json_extract import extract_json_object
from .langfuse_setup import new_run_id, run_config
from .model_fallback import attempt_fallback, resolve_llm_for_role
from .retry import with_retries_nonempty
from .tech_quirks import record_tech_quirk_entry, with_tech_quirks
from .token_guard import extract_tokens

ANALYST_SYSTEM_PROMPT = """You are a senior business analyst reviewing a draft product spec together
with the engineer who drafted it ("Agent 1"). Read the spec and the conversation so far (if any).
Raise concrete gaps, unstated assumptions, edge cases, and requirement mismatches - be specific, not
generic. If Agent 1 already addressed a point well in a prior round, don't repeat it: either accept
it, or push back with a specific reason if the resolution is genuinely unsafe to assume without the
human's input. Respond with ONLY a JSON object:
{"concerns": [{"topic": "...", "detail": "..."}], "closing_note": "..."}
Set "concerns" to an empty list and use "closing_note" to say so if you have nothing further to add.
"""

ADVOCATE_SYSTEM_PROMPT = """You represent the drafting engineer's perspective on this product spec
("Agent 1"). You already wrote the draft based on the human's own words. A business analyst is
raising concerns. For each concern: if it is a coding/implementation detail you can reasonably
resolve with a sensible default, say how and why. If it genuinely changes what the product does, or
could reasonably go more than one way, do NOT decide it yourself - mark it as needing the human's
decision, phrased as a clear question with 2-4 concrete options (the human can always type something
else instead of picking one). You never make product-intent decisions - only the human does. Respond
with ONLY a JSON object:
{"responses": [{"topic": "...", "resolution": "..." or null, "needs_human": true or false,
"question_for_human": "..." or null, "options": ["..."] or null}]}
"""

SYNTHESIS_SYSTEM_PROMPT = """You represent the drafting engineer's perspective on this product spec
("Agent 1"). The exploration between you and the business analyst is over. Consolidate the whole
conversation into a final handoff for the human: what was explored, what you safely resolved
yourselves (brief rationale each - these are defaults the human can override but doesn't need to
actively decide), and what still genuinely needs the human's own decision, phrased as clear
questions with concrete options. Respond with ONLY a JSON object:
{"summary": "...", "resolved_assumptions": ["..."],
"decisions_needed": [{"question": "...", "options": ["..."], "why_it_matters": "..."}],
"tech_quirks": [...]}

"tech_quirks" is OPTIONAL and rare at this stage - a list of short, self-contained strings for a
genuinely reusable technical lesson about a LIBRARY, FRAMEWORK, or PLATFORM (never this project's own
business logic, and never a decision or assumption already captured above) that came up during this
exploration and that every future build on any project using this pipeline should see. Shape:
"<Topic>: <the gotcha and what to do instead>". Omit it (or leave it empty) on the ordinary case where
nothing like this came up - concept-stage exploration is about product intent, not technical platform
quirks, so this will usually be empty.
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


def _build_spec_block(draft_text: str, previous_text: Optional[str]) -> str:
    block = f"DRAFT SPEC:\n{draft_text}"
    if previous_text:
        block = f"PREVIOUS APPROVED SPEC (for context on what's changing):\n{previous_text}\n\n{block}"
    return block


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", required=True, help="Path to the draft spec text file")
    parser.add_argument("--previous", help="Path to the previously approved spec, if amending")
    parser.add_argument(
        "--project-dir",
        help="Project folder, so this project's own models.yaml (if any) overrides the global model config",
    )
    args = parser.parse_args()
    project_dir = args.project_dir

    warn_if_business_analyst_is_anthropic(project_dir)
    warn_if_concept_roles_share_vendor(project_dir)

    draft_text = Path(args.draft).read_text(encoding="utf-8")
    previous_text = Path(args.previous).read_text(encoding="utf-8") if args.previous else None
    run_id = new_run_id()

    transcript = _build_spec_block(draft_text, previous_text)
    total_tokens = 0
    # Computed once for this whole run, not once per round - see with_tech_quirks' own docstring.
    analyst_system_prompt = with_tech_quirks(ANALYST_SYSTEM_PROMPT)
    advocate_system_prompt = with_tech_quirks(ADVOCATE_SYSTEM_PROMPT)
    synthesis_system_prompt = with_tech_quirks(SYNTHESIS_SYSTEM_PROMPT)

    try:
        for round_num in range(1, MAX_CONCEPT_ROUNDS + 1):
            analyst_out, tokens = _call(
                "business_analyst",
                analyst_system_prompt,
                f"{transcript}\n\nRound {round_num}: raise your concerns (or say you have none).",
                run_id,
                "concept_bounce_analyst",
                project_dir,
            )
            total_tokens += tokens
            transcript += f"\n\n[Round {round_num} - business analyst]: {json.dumps(analyst_out)}"

            if not analyst_out.get("concerns"):
                break

            advocate_out, tokens = _call(
                "concept_advocate",
                advocate_system_prompt,
                f"{transcript}\n\nRound {round_num}: respond to each concern above.",
                run_id,
                "concept_bounce_advocate",
                project_dir,
            )
            total_tokens += tokens
            transcript += f"\n\n[Round {round_num} - Agent 1]: {json.dumps(advocate_out)}"

        final, tokens = _call(
            "concept_advocate",
            synthesis_system_prompt,
            transcript,
            run_id,
            "concept_bounce_synthesis",
            project_dir,
        )
        total_tokens += tokens
    except Exception as exc:  # OpenRouter outage, exhausted retries, malformed model output, etc.
        print(json.dumps({"status": "error", "message": str(exc), "total_tokens": total_tokens}))
        return 1

    for lesson in final.pop("tech_quirks", None) or []:
        if isinstance(lesson, str) and lesson.strip():
            record_tech_quirk_entry(lesson, source="concept (concept_advocate synthesis)")

    final["total_tokens"] = total_tokens
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
