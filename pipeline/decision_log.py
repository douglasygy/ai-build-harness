"""Repo-native, cross-project decision log: the durable, tool-agnostic memory behind "learning
Douglas's decision-making" (see AGENTS.md "Decision log & preference learning"). Every gated
decision anywhere in this pipeline - concept approval, spec lock-in, a team-composition approval,
an ambiguity resolution, a model override, anything Douglas explicitly decided - gets appended here
as one line of JSON. Append-only, git-tracked, never rewritten in place: this is raw history, not a
summary (pipeline/review_decisions.py produces the summary, from this).

This is DELIBERATELY separate from Claude Code's own (or any other tool's) private memory system.
The whole reason it lives in the repo instead: it has to survive a change of AI tool, be visible to
the pipeline scripts themselves (not just to whichever assistant is in the chat), and be auditable
in git. Anything currently sitting only in a tool's private memory should be folded in here as the
authoritative copy going forward - see AGENTS.md for the standing rule.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DECISION_LOG_PATH = REPO_ROOT / "memory" / "decision_log.jsonl"


def record_decision(
    decision_point: str,
    choice: str,
    project: str = "",
    proposal_summary: str = "",
    options_presented: Optional[list] = None,
    rationale: str = "",
    proposed_by: Optional[str] = None,
    supersedes: Optional[str] = None,
) -> None:
    """Appends one decision record. `project` is a short name/label only (e.g. a project folder
    name, or "pipeline-design" for a decision about the framework itself) - never a full path,
    since this log is meant to stay readable and diffable in git.

    `proposed_by`: which agent's output this decision was actually deciding on, if any - e.g.
    "business_analyst (deepseek/deepseek-r1)" or "reviewer (qwen/qwen3-coder-next)". Douglas is
    always the one who decides (this log exists only for HIS decisions), but many decisions are
    Douglas ruling on a specific agent's proposal or verdict - naming which agent and which model
    produced it is what lets both a human reader and the Gatekeeper's model-scorecard synthesis
    (see pipeline/review_decisions.py) attribute outcomes to the right model. Omit for a decision
    with no single agent proposal behind it (e.g. a pure pipeline-design choice Douglas made
    directly with Agent 1).

    `supersedes`: the `decision_point` text (or a short unique fragment of it) of an earlier entry
    this one corrects or overrides, if any - e.g. the builder_n8n routing conflict resolution
    supersedes the entry that first set that routing. Lets a reader (or the Gatekeeper) find the
    relationship explicitly instead of inferring it from prose across the whole log.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project": project,
        "decision_point": decision_point,
        "proposal_summary": proposal_summary,
        "options_presented": options_presented or [],
        "choice": choice,
        "rationale": rationale,
    }
    if proposed_by is not None:
        entry["proposed_by"] = proposed_by
    if supersedes is not None:
        entry["supersedes"] = supersedes
    DECISION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DECISION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_decisions() -> list:
    if not DECISION_LOG_PATH.exists():
        return []
    decisions = []
    with open(DECISION_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                decisions.append(json.loads(line))
    return decisions
