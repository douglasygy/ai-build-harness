"""Run-level checkpointing for build_and_review, for EVERY platform (python, n8n, anything added
later) - not tied to any one builder implementation.

Why this exists: a build can be killed mid-run by something outside its control (the OS reclaiming
memory being the case actually hit in practice). Without this, that meant re-running the whole
pipeline from scratch - re-planning, re-building and re-reviewing work that had already succeeded
and been paid for.

What it stores: the whole graph state (which round we're on, the reviewer's outstanding problems,
running token total, builder status) next to the project, keyed by a hash of the spec. If the spec
changed since the checkpoint was written, the checkpoint is ignored entirely rather than silently
resuming stale work against a spec that no longer matches.

This is deliberately separate from, and complementary to, the n8n builder's own per-node checkpoint:
- the per-node checkpoint resumes an interrupted build *within* a single builder call
- this one resumes an interrupted *run* between builder/reviewer rounds
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

CHECKPOINT_FILENAME = ".build_checkpoint.json"

# State keys worth persisting. Deliberately explicit rather than dumping the whole dict, so an
# unexpected non-serializable value added to the state later can't silently break checkpointing.
_PERSISTED_KEYS = (
    "project_dir",
    "spec_path",
    "code_dir",
    "platform",
    "n8n_workflow",
    "review_round",
    "review_verdict",
    "review_problems",
    "review_must_fix",
    "review_should_fix",
    "review_nitpicks",
    "builder_status",
    "ambiguity_question",
    "builder_model_override",
    "total_tokens",
    "reviewer_tokens_used",
    "python_until_unit_index",
    "python_units_built",
    "coverage_batch_key",
    "coverage_batch_rounds",
    "round_budget_exhausted",
    "final_fix_pass",
    # "live_smoke" is DELIBERATELY never persisted here - it's per-invocation consent to spend real
    # money (see AGENTS.md's live-smoke hard rule), and must never be silently inherited by a later
    # resumed run that never asked for it. Do not add it to this tuple.
)


def _stage_scope_key(state: dict) -> Optional[str]:
    """The python unit-decomposition equivalent of an n8n workflow slug (see
    pipeline/review_architect.py's build_units) - the scope tag that keeps a build stopped through
    unit 1 from ever having its checkpoint mistaken for one stopped through unit 2. None when this
    project's Architect decision declared no build_units (or there's no Architect at all) - every
    project without unit decomposition. Named "stage" for its RETURNED key shape (matches
    build_and_review.py's own python_unit_key f-string), not because staging still exists - PM-epic/
    checkpoint stage gating was retired for the python path 2026-09-28, Douglas's direct instruction.
    """
    index = state.get("python_until_unit_index")
    return f"unit{index}" if index is not None else None


def checkpoint_path(project_dir, workflow: Optional[str] = None, stage: Optional[str] = None) -> Path:
    """Scoped per n8n workflow (pipeline/n8n_workflow_target.py) and/or per python stage
    (pipeline/build_stages.py) so building one workflow/stage in a multi-workflow or multi-stage
    project can never resume/collide with a DIFFERENT one's checkpoint.

    Real incident this fixes (2026-09-26, reported live): a python project's stage 1 build
    completed (pass_partial) and its checkpoint was never cleared (see the clear() docstring below);
    a later `--until-stage 2` invocation found that same unscoped, still-"done" checkpoint,
    resumed straight at the reviewer with nothing to review, and reported a clean "pass" with
    `slices: null` - 238,423 tokens spent building nothing, on a run that looked identical to a real
    success. `workflow`/`stage` both falsy (every existing project, and any project with at most one
    n8n workflow and no PM epics) keeps the exact original filename - zero behavior change for the
    common case.
    """
    parts = [p for p in (workflow, stage) if p]
    filename = CHECKPOINT_FILENAME if not parts else f".build_checkpoint.{'.'.join(parts)}.json"
    return Path(project_dir) / filename


def spec_hash(spec_text: str) -> str:
    return hashlib.sha256(spec_text.encode("utf-8")).hexdigest()


def save_state(state: dict) -> None:
    project_dir = state.get("project_dir")
    spec_path = state.get("spec_path")
    if not project_dir or not spec_path:
        return
    try:
        spec_text = Path(spec_path).read_text(encoding="utf-8")
    except OSError:
        return
    payload = {
        "spec_hash": spec_hash(spec_text),
        "state": {k: state.get(k) for k in _PERSISTED_KEYS if k in state},
    }
    try:
        path = checkpoint_path(project_dir, workflow=state.get("n8n_workflow"), stage=_stage_scope_key(state))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except (OSError, TypeError):
        # Checkpointing is a convenience, never a reason to fail an otherwise-working build.
        pass


def load_state(project_dir, spec_text: str, workflow: Optional[str] = None, stage: Optional[str] = None) -> Optional[dict]:
    """Returns the saved state if a checkpoint exists and matches the current spec, else None."""
    path = checkpoint_path(project_dir, workflow=workflow, stage=stage)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("spec_hash") != spec_hash(spec_text):
        return None
    saved = payload.get("state")
    return saved if isinstance(saved, dict) else None


def clear(project_dir, workflow: Optional[str] = None, stage: Optional[str] = None) -> None:
    path = checkpoint_path(project_dir, workflow=workflow, stage=stage)
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def resume_stage_for(saved_state: dict) -> str:
    """Where a resumed run should re-enter the graph.

    'reviewer' when the builder had already finished and nothing was sent back to it yet - going
    back through the builder there would rebuild work that was already complete. 'builder' in every
    other case, including when the reviewer had already returned problems (re-entering the builder
    with those problems in hand is exactly the right next step anyway).
    """
    if saved_state.get("builder_status") == "done" and not saved_state.get("review_problems"):
        return "reviewer"
    return "builder"
