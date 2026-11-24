from __future__ import annotations

from typing import Literal, Optional, TypedDict


class BuildState(TypedDict, total=False):
    project_dir: str
    spec_path: str
    code_dir: str
    platform: str  # "python" (default) or "n8n" - selects builder prompt + self-test mechanism
    # Which of an Architect-approved system_shape's n8n workflows this build targets - see
    # pipeline/n8n_workflow_target.py. None (every project today) means the legacy flat
    # code/n8n/workflow.json path; a project whose Architect decision names more than one n8n
    # workflow requires this to disambiguate which one.
    n8n_workflow: Optional[str]
    review_round: int
    review_verdict: Optional[Literal["pass", "fail"]]
    # Only must_fix + (should_fix, when a round is already happening) - what's actually fed back to
    # the builder. See review_must_fix/review_should_fix/review_nitpicks below for the full,
    # always-populated severity breakdown (2026-09-29, Douglas's explicit direction: "the reviewer
    # ranking its finding as well, priority must-haves, good to have, nitpicks etc. And must haves
    # must be delivered" - only must_fix forces another round; nitpick never reaches the builder at
    # all, only the run's own reported output, so nothing observed is silently hidden).
    review_problems: list
    review_must_fix: list
    review_should_fix: list
    review_nitpicks: list
    # Findings a previous invocation ended with (run_result.json); shown to the reviewer as
    # "previously reported, verify fixed". Empty when none.
    carried_findings: list
    # Open externally verified findings (pipeline/external_findings.py): [{id, text, unit, source}].
    external_findings: list
    # Carried must-fix findings that must be explicitly confirmed fixed by the reviewer.
    carried_must_fix: list
    # Units whose tests never touch their dependencies' real producers (should_fix, heuristic).
    seam_warnings: list
    # [{id, finding, quote}] findings the builder claims were already fixed (ALREADY_FIXED); the
    # reviewer must confirm or re-open each.
    claimed_fixed: list
    # Per-invocation builder budget overrides (--agent-rounds / --read-first-round); never
    # checkpointed, like live_smoke.
    agent_rounds: Optional[int]
    read_first_round: bool
    builder_status: Literal["working", "ambiguous", "done"]
    ambiguity_question: Optional[str]
    builder_model_override: Optional[str]
    total_tokens: int
    # Reviewer-only cumulative spend this run (2026-09-29) - separate from total_tokens above on
    # purpose: total_tokens also carries the builder's own (legitimately large, unit-by-unit) spend,
    # which must never count against the reviewer's own budget. See constants.py's
    # MAX_REVIEWER_TOKENS_PER_RUN for the real incident this fixes. Reset to 0 only at the start of a
    # genuinely fresh run, same resume semantics as total_tokens itself.
    reviewer_tokens_used: int
    # Set only when resuming an interrupted run from a checkpoint - decides whether the graph
    # re-enters at the builder or skips straight to the reviewer. See pipeline/checkpoint.py.
    resume_stage: Optional[str]
    # Build only through this functional slice (name, or 1-based number) and stop, so a working
    # base can be reviewed before the rest is built. None/"all" builds everything.
    until_slice: Optional[str]
    # Python-platform unit decomposition (see pipeline/review_architect.py's build_units,
    # pipeline/nodes/builder_node.py's _build_python_units) - the python analog of until_slice,
    # replacing PM-epic/checkpoint stage gating entirely (2026-09-28, Douglas's direct instruction).
    # python_until_unit_index is the already-resolved 0-based target index to build through (None
    # when this project's Architect decision declared no build_units - unit decomposition simply
    # doesn't apply then, same as a project with no Architect at all).
    python_until_unit_index: Optional[int]
    # Cumulative unit names successfully built so far across every invocation of this same build
    # (persisted across checkpoint resumes) - what _build_python_units itself uses to skip an
    # already-built unit not named in this round's reviewer feedback, and what reviewer_node.py's
    # stub-regression check (pipeline/build_stages.py's check_stage_stub_regressions) verifies
    # against, in place of the retired python_epics_built.
    python_units_built: list
    # Douglas's rulings on earlier AMBIGUOUS questions (pipeline/ambiguity_answers.py).
    rulings: list
    # [{'test': nodeid, 'reason': str, 'tolerated_for': unit}] red tests that were already failing
    # when a unit started, tolerated at its acceptance; surfaced as should_fix findings.
    tolerated_failures: list
    # Reported back by the builder so a partial build is visible in run_result.json.
    slices_total: int
    slices_built: list
    slices_remaining: list
    partial_build: bool
    # Per-batch round budget for test-coverage debt (see reviewer_node.py's pure_coverage_fail
    # logic) - Douglas's explicit direction (2026-09-27): MAX_ROUNDS should cap rounds spent on the
    # CURRENT batch of (at most 5) coverage findings, not the whole run's total coverage backlog.
    # coverage_batch_key identifies which specific files make up the current batch (None when the
    # last round's fail wasn't purely coverage-debt-driven); coverage_batch_rounds resets to 1
    # whenever that key changes (a genuinely new batch, because prior modules got real coverage).
    coverage_batch_key: Optional[str]
    coverage_batch_rounds: int
    # The single authoritative "should the run actually stop" signal, computed by reviewer_node and
    # checked by graph.py's routing - NOT simply review_round >= MAX_ROUNDS, since a per-batch reset
    # can mean review_round is well past MAX_ROUNDS while the run should still continue.
    round_budget_exhausted: bool
    # A project whose round cap is exactly 1 (models.yaml max_rounds: 1) does not end on a failed
    # review: the single round's findings go to the builder for ONE fix pass, then the run ends with
    # no second review. Set by reviewer_node, honored by graph.py's routing.
    final_fix_pass: bool
    # Whether THIS invocation is authorized to run code/smoke_run.py in "live" mode (real calls to
    # real, paid vendors this project's own built code uses - see pipeline/nodes/builder_node.py's
    # AGENT_SYSTEM_PROMPT and AGENTS.md's live-smoke hard rule). Deliberately NOT in checkpoint.py's
    # _PERSISTED_KEYS: consent is per-invocation and must never be silently inherited by a later
    # resumed run that didn't ask for it again.
    live_smoke: bool
    # The agentic python builder's "builder account" (2026-09-27, Douglas's own suggestion,
    # deliberately built as the SAFER version of it - see reviewer_node.py's builder-account
    # section for why the reviewer's independent read of the actual code never gets replaced by
    # this). builder_files_touched is MECHANICAL ground truth - every real write_file/edit_file
    # call this round, recorded by the loop itself, never the agent's own claim about what it
    # changed. builder_account is the agent's own final free-text explanation of what it did and
    # why - a genuinely unverified claim, shown to the reviewer clearly labeled as exactly that.
    builder_files_touched: list
    builder_account: str
