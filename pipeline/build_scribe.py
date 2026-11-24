"""build_scribe: the DESCRIPTIVE half of the two-tier memory architecture (Flash) - see AGENTS.md
"Decision log & preference learning". Its only job is turning one build_and_review.py run's result
into a clean, chronological, OBJECTIVE description: what happened, what changed, what blocked it.
It never judges whether any of it was the right call - that is preference_analyst's (the
Gatekeeper's) job, kept strictly separate so evaluation never gets contaminated by how the
description itself was phrased.

The input is run_result.json's content - the structured summary build_and_review.py already
produces (status, rounds, problems, the `built` inventory). This is not raw captured terminal
output (no such capture exists yet in this pipeline) - it is the best structured signal already
available without a separate stdout/stderr-capture project. If literal terminal-log capture is
wanted later, that is a separate, additional piece of plumbing, not something this module assumes.
"""
from __future__ import annotations

import json
from typing import Optional

from .json_extract import extract_json_object
from .langfuse_setup import new_run_id, run_config
from .model_fallback import attempt_fallback, resolve_llm_for_role
from .retry import with_retries_nonempty
from .token_guard import extract_tokens

SYSTEM_PROMPT = """You are a purely objective observer, summarizing one build run for a durable
record. Read the run result JSON. Describe ONLY what happened: what was built or attempted, what
changed, what the outcome was, and what (if anything) blocked it. NEVER evaluate whether any
decision was right or wrong, good or bad, or well-judged - that is strictly not your job, someone
else does that from your summary. If something is unclear from the data given, say plainly that it
is unclear rather than guessing or inferring intent. Respond with ONLY a JSON object:
{"what_happened": "...", "files_or_nodes_touched": ["..."], "outcome": "...", "blockers": ["..."]}
"""


def summarize_build_run(run_result: dict, project_dir: Optional[str] = None) -> tuple[dict, int]:
    """Returns (summary, tokens_used). Raises on failure - callers decide how to degrade (this
    pipeline's own call site treats it as best-effort and never lets it break a build's real
    result, since this is a side channel, not part of the build/review contract).
    """
    llm = resolve_llm_for_role("build_scribe", project_dir=project_dir)
    run_id = new_run_id()
    config = run_config(run_id, tags=["build_scribe"])
    messages = [("system", SYSTEM_PROMPT), ("user", json.dumps(run_result, indent=2))]
    try:
        response = with_retries_nonempty(lambda: llm.invoke(messages, config=config))
    except Exception as exc:
        response = attempt_fallback(
            "build_scribe", lambda fb_llm: fb_llm.invoke(messages, config=config),
            project_dir=project_dir, primary_error=exc,
        )
    return extract_json_object(response.content, context="build_scribe response"), extract_tokens(response)
