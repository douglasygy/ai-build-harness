"""Records Agent 1's OWN actions in the four categories Douglas reserved exclusively for
himself - server/infra setup, agent model selection, tool selection, and new folder/directory
creation or access (see AGENTS.md "Hard rule - four decisions are always Douglas's"). This is the
direct response to a real incident: an earlier session picked a model for a new role and used it
without ever asking Douglas first, and nothing caught it automatically - he caught it himself.

Distinct from memory/decision_log.jsonl (Douglas's own decisions) - this is a log of what AGENT 1
DID, mechanical and self-reported at the moment of acting, so an independent audit (see
pipeline/review_decisions.py's overreach-flag synthesis) can check afterward whether each logged
action has a plausible matching Douglas decision nearby, rather than trusting Agent 1's own
say-so about whether it was authorized.

This module deliberately does NOT accept a self-graded "was this approved" flag - that would let
the same blind spot that caused the original incident grade its own homework. Recording an action
here is not a defense of it; the independent audit decides that from the two logs together.

CLI, since Agent 1 (a Claude Code conversation) calls this via the Bash/PowerShell tool, not via
direct Python import:

    python -m pipeline.agent1_action_log record <category> "<description>" [--project <name>]
    python -m pipeline.agent1_action_log list
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
ACTION_LOG_PATH = REPO_ROOT / "memory" / "agent1_action_log.jsonl"

RESERVED_CATEGORIES = (
    "server_infra",
    "model_selection",
    "tool_selection",
    "filesystem",
)


def record_action(category: str, description: str, project: Optional[str] = None) -> None:
    if category not in RESERVED_CATEGORIES:
        raise ValueError(f"Unknown category {category!r}; expected one of {RESERVED_CATEGORIES}")
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "description": description,
        "project": project,
    }
    ACTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_actions() -> list:
    if not ACTION_LOG_PATH.exists():
        return []
    lines = ACTION_LOG_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    record_p = sub.add_parser("record")
    record_p.add_argument("category", choices=RESERVED_CATEGORIES)
    record_p.add_argument("description")
    record_p.add_argument("--project", default=None)

    sub.add_parser("list")

    args = parser.parse_args()

    if args.cmd == "record":
        record_action(args.category, args.description, args.project)
        print(f"Logged [{args.category}] {args.description}")
        return 0

    if args.cmd == "list":
        actions = load_actions()
        if not actions:
            print("No actions logged yet.")
            return 0
        for a in actions:
            proj = f" ({a['project']})" if a.get("project") else ""
            print(f"{a['timestamp']} [{a['category']}]{proj} {a['description']}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
