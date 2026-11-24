"""PM's primary job: turn an approved spec into Epics and User Stories - scoping and sequencing
work Douglas has already approved at the concept stage, never deciding product intent itself. See
AGENTS.md "Scale-adaptive roles: PM and Architect" and "Wiring PM/Architect into the build".

Single-shot, not a bounce (unlike review_concept.py / review_team_composition.py) - this is PM's
own scoping judgment applied to an already-locked spec, not something that benefits from a second
adversarial perspective the way concept gaps or team-composition tradeoffs do.

Usage:
    python -m pipeline.review_pm --spec path/to/SPEC.md [--project-dir <dir>]

Prints ONE JSON object to stdout:
    {"summary": "...", "epics": [{"name": "...", "description": "...",
    "stories": [{"story": "...", "priority": "..."}], "non_functional_requirements": ["..."]}]}
or, on failure after retries: {"status": "error", "message": "..."}

Douglas approves this output the same way he approves everything else - nothing here writes it
anywhere. Once approved, Agent 1 (Claude Code) writes it verbatim to <project-dir>/.pm_epics.json,
which pipeline/nodes/builder_node.py then folds into every builder call as binding context - see
_augment_spec_with_scale_adaptive_context.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .json_extract import extract_json_object
from .langfuse_setup import new_run_id, run_config
from .model_fallback import attempt_fallback, resolve_llm_for_role
from .retry import with_retries_nonempty
from .token_guard import extract_tokens

SYSTEM_PROMPT = """You are the PM for this project. Read the approved spec and turn it into Epics
and User Stories - scoping and sequencing the work it already describes, with measurable
non-functional requirements where the spec implies them (performance, reliability, scale). You do
NOT decide what the product should do - that was already decided and locked into this spec. If the
spec is small enough to be one epic, say so plainly rather than inventing structure it doesn't need.
Respond with ONLY a JSON object: {"summary": "...", "epics": [{"name": "...", "description": "...",
"stories": [{"story": "...", "priority": "high" or "medium" or "low"}],
"non_functional_requirements": ["..."]}]}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="Path to the approved SPEC.md")
    parser.add_argument(
        "--project-dir",
        help="Project folder, so this project's own models.yaml (if any) overrides the global model config",
    )
    args = parser.parse_args()

    spec_text = Path(args.spec).read_text(encoding="utf-8")
    run_id = new_run_id()
    llm = resolve_llm_for_role("pm", project_dir=args.project_dir)
    config = run_config(run_id, tags=["pm", "epics_and_stories"])
    messages = [("system", SYSTEM_PROMPT), ("user", f"APPROVED SPEC:\n{spec_text}")]

    try:
        try:
            response = with_retries_nonempty(lambda: llm.invoke(messages, config=config))
        except Exception as exc:
            response = attempt_fallback(
                "pm", lambda fb_llm: fb_llm.invoke(messages, config=config),
                project_dir=args.project_dir, primary_error=exc,
            )
        result = extract_json_object(response.content, context="pm response")
        result["total_tokens"] = extract_tokens(response)
    except Exception as exc:  # OpenRouter outage, exhausted retries, malformed model output, etc.
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
