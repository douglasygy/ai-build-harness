"""Repo-native, cross-project build-activity log - the durable output of the Descriptive half of
the two-tier memory architecture (see pipeline/build_scribe.py and AGENTS.md "Decision log &
preference learning"). Append-only, git-tracked, mechanical (no LLM call happens in this module
itself - `summary` is build_scribe's already-produced output, this just records it). Distinct from
memory/decision_log.jsonl: that log is Douglas's own explicit decisions; this one is what a build
run actually did, objectively described.

Two scopes of entry:
- "run" (the default) - one per build_and_review.py call, summarizing its final result.
- "round" - one per FAILING round along the way (a reviewer round, an n8n repair round) - the
  friction worth learning from, logged only when something actually went wrong so a clean build
  never costs anything extra beyond its one "run" entry. See pipeline/graph.py and
  pipeline/nodes/builder_node.py's _log_repair_round for the call sites.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_LOG_PATH = REPO_ROOT / "memory" / "build_log.jsonl"


def record_build(
    project: str,
    status: str,
    summary: dict,
    scope: str = "run",
    round_num: Optional[int] = None,
    role: Optional[str] = None,
    model: Optional[str] = None,
    rounds: Optional[int] = None,
    total_tokens: Optional[int] = None,
) -> None:
    """`model`: the actual resolved model slug used for this entry's role at the time it ran (e.g.
    "qwen/qwen3-coder-next") - NOT just the role name, since the role stays constant while the
    model behind it can change over time (a project override, a CLI --builder-model flag, or the
    global config simply being updated later). This is what makes the model-scorecard synthesis
    (pipeline/review_decisions.py, run at project completion - see AGENTS.md "Model scorecard")
    possible at all: without it, a scorecard could only ever say "the reviewer role" performed some
    way, never which actual model did.

    `rounds`: for a run-scope entry only - the total review rounds that run took (from
    run_result.json's own "rounds" field), as a structured top-level number rather than buried
    inside build_scribe's free-form summary text. This is what makes the mechanical build-trend
    stats in pipeline/review_decisions.py's compute_build_trends() possible without asking an LLM
    to extract or average numbers from prose.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project": project,
        "scope": scope,
        "status": status,
        "summary": summary,
    }
    if round_num is not None:
        entry["round"] = round_num
    if role is not None:
        entry["role"] = role
    if model is not None:
        entry["model"] = model
    if rounds is not None:
        entry["rounds"] = rounds
    if total_tokens is not None:
        entry["total_tokens"] = total_tokens
    BUILD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_builds() -> list:
    if not BUILD_LOG_PATH.exists():
        return []
    builds = []
    with open(BUILD_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                builds.append(json.loads(line))
    return builds
