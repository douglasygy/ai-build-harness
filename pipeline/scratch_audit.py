"""Reports stale scratch/checkpoint artifacts under projects/ - never deletes anything itself.

Filesystem-context's own guidance (see .claude/skills/context-engineering/filesystem-context)
flags unbounded scratch growth as a real gotcha: "agents create temp files without cleanup,
eventually consuming disk and making directory listings noisy." Build-agent already clears a
project's checkpoint on a genuine pass/fail-after-max-rounds completion (see
pipeline/checkpoint.py's clear()) - what's missing is visibility into checkpoints that never got
that far: an abandoned project, a run that stopped ambiguous and was never resumed, a throwaway
smoketest folder left behind.

Deliberately a REPORT, not an auto-delete: removing existing files is exactly the kind of
filesystem action AGENTS.md's "HARD RULE - four decisions are always Douglas's alone" reserves to
Douglas. This script only names candidates; deleting any of them is his call.

Usage:
    python -m pipeline.scratch_audit [--projects-dir projects] [--max-age-days 30]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .checkpoint import CHECKPOINT_FILENAME

DEFAULT_MAX_AGE_DAYS = 30


def find_stale_checkpoints(projects_dir, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> list:
    """Returns [{"project": ..., "checkpoint_path": ..., "age_days": ...}, ...] for every
    .build_checkpoint.json older than max_age_days - a real candidate for "this run was abandoned,
    not just mid-flight," since an active project gets its checkpoint touched every round.
    """
    projects_dir = Path(projects_dir)
    if not projects_dir.exists():
        return []
    now = time.time()
    stale = []
    for checkpoint_path in projects_dir.glob(f"*/{CHECKPOINT_FILENAME}"):
        age_days = (now - checkpoint_path.stat().st_mtime) / 86400
        if age_days >= max_age_days:
            stale.append({
                "project": checkpoint_path.parent.name,
                "checkpoint_path": str(checkpoint_path),
                "age_days": round(age_days, 1),
            })
    return sorted(stale, key=lambda s: -s["age_days"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects-dir", default="projects")
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    args = parser.parse_args()

    stale = find_stale_checkpoints(args.projects_dir, args.max_age_days)
    if not stale:
        print(f"No checkpoints older than {args.max_age_days} days under {args.projects_dir}/.")
        return 0

    print(f"{len(stale)} checkpoint(s) older than {args.max_age_days} days - candidates to review, not auto-deleted:")
    for entry in stale:
        print(f"  {entry['project']}: {entry['checkpoint_path']} ({entry['age_days']} days old)")
    print("\nA stale checkpoint usually means an interrupted or abandoned run. Delete it yourself "
          "(or ask Agent 1 to, explicitly) once you've confirmed the project is genuinely done with.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
