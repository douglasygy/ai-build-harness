"""Runs the builder <-> reviewer loop once for a given project.

Usage:
    python -m pipeline.build_and_review --project-dir projects/acme-corp [--builder-model anthropic/claude-opus-5] [--platform n8n] [--fresh] [--fresh-n8n]

Writes and prints projects/<name>/run_result.json with one of:
    {"status": "pass", "rounds": N, "total_tokens": N}
    {"status": "failed_after_max_rounds", "rounds": 3, "problems": [...], "total_tokens": N}
    {"status": "ambiguous", "question": "...", "total_tokens": N}
    {"status": "error", "message": "..."}

If a previous run of this same spec was interrupted (e.g. the process was killed for memory), it
resumes from its checkpoint automatically rather than starting over. Pass --fresh to ignore any
checkpoint and start clean at the outer round level. That alone does NOT touch n8n's own per-node
artifacts (workflow.json, its checkpoint, conventions) - a node already on disk is deliberately
never rebuilt just because a new run started. Pass --fresh-n8n too for a genuine full wipe of those
as well - see wipe_n8n_artifacts in pipeline/nodes/builder_node.py for why this is a separate flag.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import checkpoint
from .build_log import load_builds, record_build
from .build_scribe import summarize_build_run
from .build_stages import next_unbuilt_stage_index
from .config_loader import ConfigError, effective_models, validate_builder_reviewer_vendors
from .graph import build_graph
from .n8n_workflow_target import n8n_subdir, resolve_workflow_slug
from .nodes.builder_node import wipe_n8n_artifacts, wipe_python_code
from .external_findings import dismiss_matching, finding_line, load_open_findings, record_finding, unmatched_needles
from .ambiguity_answers import drop_dismissed, format_rulings, load_answers, record_answer
from .build_journal import REVIEW_ROUND_UNIT, clear_journal, record_journal_entry
from .nodes.builder_node import _save_unit_baseline
from .review_architect import resolve_until_unit
from .review_decisions import synthesize_patterns, write_patterns
from .scale_roster import load_python_build_units, missing_prerequisites


def _build_summary(code_dir: Path, platform: str, workflow: str | None = None) -> dict:
    """A compact inventory of what was actually built, for Agent 1's concept-conformance check.

    The point is to make that check cheap and reliable: comparing a node/file inventory (and the
    conventions the build actually adopted) against SPEC.md is quick, whereas re-reading a 40KB
    generated workflow every time is not - and a check that's expensive is a check that gets skipped.
    """
    summary: dict = {"platform": platform}
    if platform == "n8n":
        n8n_dir = n8n_subdir(code_dir, workflow)
        workflow_path = n8n_dir / "workflow.json"
        if workflow:
            summary["workflow"] = workflow
        if workflow_path.exists():
            try:
                built_workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                summary["error"] = f"{workflow_path} could not be read as JSON"
                return summary
            nodes = [n for n in (built_workflow.get("nodes") or []) if isinstance(n, dict)]
            summary["workflow_name"] = built_workflow.get("name")
            summary["node_count"] = len(nodes)
            summary["nodes"] = [
                {"name": n.get("name"), "type": n.get("type")} for n in nodes
            ]
        conventions_path = n8n_dir / ".conventions.json"
        if conventions_path.exists():
            try:
                summary["conventions"] = json.loads(conventions_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    else:
        if code_dir.exists():
            summary["files"] = sorted(
                str(p.relative_to(code_dir))
                for p in code_dir.rglob("*")
                if p.is_file()
                and not p.name.startswith(".")
                # Build artifacts, not authored code - noise in a conformance check.
                and "__pycache__" not in p.parts
                and p.suffix != ".pyc"
            )
    return summary


def _is_project_complete(status: str, partial: bool) -> bool:
    """A real pass or a real stop after exhausting the round cap - never a partial-slice stop or an
    ambiguous question, neither of which means the project has actually concluded.
    """
    return status in ("pass", "failed_after_max_rounds") and not partial


def _question_already_ruled(question, answers) -> bool:
    norm = lambda s: " ".join(str(s or "").lower().split())  # noqa: E731
    q = norm(question)
    return bool(q) and any(norm(a.get("question")) == q for a in answers)


def _load_carried_findings(project_dir) -> dict:
    """Unresolved reviewer findings (must_fix, should_fix) the previous invocation ended with, from
    run_result.json - after a failed_after_max_rounds run, or a pass/pass_partial that still had
    must_fix/should_fix findings (nitpicks never carry). {} when there is nothing to carry."""
    path = Path(project_dir) / "run_result.json"
    try:
        prev = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    status = str(prev.get("status", "")) if isinstance(prev, dict) else ""
    failed = status.startswith("failed_after_max_rounds")
    if not failed and not status.startswith(("pass", "ambiguous")):
        return {}
    findings = prev.get("review_findings") or {}
    must = [str(x) for x in findings.get("must_fix") or []]
    should = [str(x) for x in findings.get("should_fix") or []]
    if failed and not must and not should:
        must = [str(x) for x in prev.get("problems") or []]
    if not must and not should:
        return {}
    return {"must_fix": must, "should_fix": should}


def _run_concluded(status: str) -> bool:
    """Whether THIS invocation's own scope (the whole project, or - for a staged/sliced build -
    just the stage/slice it targeted) has nothing left to resume, so its checkpoint should be
    cleared. Unlike _is_project_complete above, a "_partial" stop counts as concluded here: a
    deliberate stage/slice boundary is not a failure, and the code already on disk is the durable
    record of what that scope achieved - same principle already applied to n8n's own per-node
    checkpoint ("done - the workflow file is now the durable artifact"). Only "ambiguous" (a real
    question still waiting on Douglas) keeps its checkpoint, so re-running after answering resumes
    rather than restarting.

    Real incident this fixes (2026-09-26): the original check here was `status in ("pass",
    "failed_after_max_rounds")`, which never matches "pass_partial" - so a genuinely concluded
    stage's checkpoint was never cleared, stuck around matching the (unchanged) spec hash, and was
    later loaded by a DIFFERENT stage's invocation, skipping its builder entirely and reporting a
    clean pass having built nothing.
    """
    return status.startswith(("pass", "failed_after_max_rounds"))


def _estimate_run_cost(project_dir: Path, platform: str) -> None:
    """Best-effort, mechanical (no LLM call): prints a rough token estimate for the run about to
    start, from the historical average of past run-scope build_log.jsonl entries for this same
    project. Pure arithmetic on already-logged numbers - never a reason to delay or block a run, so
    any failure here is swallowed silently. Prints nothing (rather than a misleading "0 tokens"
    estimate) when there isn't at least one prior completed run for this project to average from.
    """
    try:
        prior = [
            b for b in load_builds()
            if b.get("project") == project_dir.name
            and b.get("scope", "run") == "run"
            and isinstance(b.get("total_tokens"), (int, float))
        ]
        if not prior:
            return
        avg_tokens = sum(b["total_tokens"] for b in prior) / len(prior)
        print(
            f"ESTIMATE: based on {len(prior)} prior run(s) of this project, expect roughly "
            f"{avg_tokens:,.0f} tokens for this run (actual cost may differ - see Langfuse after).",
            file=sys.stderr,
        )
    except Exception:  # noqa: BLE001 - a convenience printout must never block or break a run
        pass


def _update_memory(project_dir: Path, result: dict, models_used: dict | None = None) -> None:
    """Best-effort: runs the two-tier memory pipeline (build_scribe -> memory/build_log.jsonl ->
    preference_analyst -> memory/decision-patterns.md) after EVERY call, per Douglas's explicit
    choice - he was told this means real per-run model cost and chose it anyway, wanting the
    Gatekeeper woken on every finish/pause/blocker, not only at project boundaries. This is a side
    channel, never part of the build/review contract: any failure here (OpenRouter outage, a
    malformed model response) is logged and swallowed, never allowed to change `result` or this
    run's exit code.

    `models_used`: this run's resolved {role: {"model": ..., "source": ...}} (from
    config_loader.effective_models(), already computed once in main() for the startup printout) -
    threaded through so the run-level build_log entry names the actual model behind the builder
    role that ran, not just the role name. Reviewer's model is attributed per-round already (see
    pipeline/graph.py's _log_reviewer_round), since a single run can only have one reviewer model
    but the round is where reviewer failures actually get logged.

    Runs the model scorecard (see pipeline/review_decisions.py) only at genuine project completion
    (AGENTS.md "Model scorecard") - the routine per-call synthesis stays cheap; the heavier
    per-model judgment only runs at the cadence Douglas actually asked for it.
    """
    try:
        summary, _ = summarize_build_run(result, project_dir=str(project_dir))
        builder_model = None
        if models_used:
            for role, info in models_used.items():
                if role in ("builder", "builder_n8n"):
                    builder_model = info.get("model")
        record_build(
            project_dir.name, result.get("status", "unknown"), summary,
            model=builder_model, rounds=result.get("rounds"), total_tokens=result.get("total_tokens"),
        )
        # Reuses the same flag already computed in main() rather than re-deriving the same
        # project-completion logic a second way.
        include_scorecard = bool(result.get("memory_review_required"))
        analysis, decision_count, build_count, action_count, _ = synthesize_patterns(
            project_dir=str(project_dir), include_model_scorecard=include_scorecard,
        )
        write_patterns(analysis, decision_count, build_count, action_count, builds=load_builds())
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this must never break a real build
        print(f"WARNING: memory pipeline (build_scribe/preference_analyst) failed this run: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True, help="Path to the project folder, e.g. projects/acme-corp")
    parser.add_argument(
        "--builder-model",
        help="Override the configured builder model for this run, e.g. anthropic/claude-opus-5",
    )
    parser.add_argument(
        "--platform",
        choices=["python", "n8n"],
        default="python",
        help=(
            "Target platform for the builder. 'python' (default, unchanged behavior): Python files + "
            "pytest, via the 'builder' model role. 'n8n': an importable n8n workflow with all logic as "
            "Code-node JavaScript, built via the separate 'builder_n8n' model role, one small LLM call "
            "per node (plan -> per-node build+validate -> assemble -> structural validation/targeted "
            "repair) rather than one giant completion. Opt-in only - omitting this flag never changes "
            "behavior for non-n8n projects."
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any existing checkpoint for this project and start the run from scratch.",
    )
    parser.add_argument(
        "--fresh-n8n",
        action="store_true",
        help=(
            "n8n platform only. --fresh alone only clears the OUTER round-level checkpoint, not "
            "n8n's own per-node 'frozen once built' artifacts (workflow.json, its checkpoint, saved "
            "conventions, production-triggers sidecar) - a node already on disk is deliberately "
            "never rebuilt just because a new run started. Pass this too for a genuine full wipe, "
            "e.g. when a node was built wrong for a reason that's since been fixed elsewhere and "
            "needs to actually be regenerated, not silently reused."
        ),
    )
    parser.add_argument(
        "--rebuild-unit",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "python units only, repeatable: force the named build unit to be re-entered even though "
            "it looks 'already built' (interface + journal entry). Records an EMPTY other-tests "
            "baseline for it, so every failure currently red anywhere in the suite counts as new "
            "and must be fixed (or the unit will not complete) - use it when a finished unit broke "
            "its callers. Does not wipe any code."
        ),
    )
    parser.add_argument(
        "--fresh-python",
        action="store_true",
        help=(
            "python platform only, the analog of --fresh-n8n (2026-09-28, added after a real "
            "incident: a rebuild reported 'pass' having barely touched already-existing files, "
            "which looked like a genuine test but wasn't - the existing code was never actually "
            "cleared). --fresh alone only clears the OUTER round-level checkpoint, not real source "
            "files already on disk - a file already there is deliberately never rebuilt just "
            "because a new run started. Pass this too for a genuine clean-slate rebuild: wipes "
            "every file under code/ AND this project's build journal (a stale journal entry "
            "referencing code that no longer exists would mislead a later unit otherwise)."
        ),
    )
    parser.add_argument(
        "--until-slice",
        help=(
            "n8n only. Build only through this functional slice (its name, or a 1-based number) and "
            "stop, so a working functional base can be reviewed before the rest is built, instead of "
            "only being judgeable once every node exists. Omit to build everything."
        ),
    )
    parser.add_argument(
        "--workflow",
        help=(
            "n8n only, and only required when this project's approved Architect decision names more "
            "than one n8n workflow (see pipeline/n8n_workflow_target.py) - the n8n builder produces "
            "one workflow per invocation, so this says which one. Ignored (no effect) on any project "
            "with at most one n8n workflow - the ordinary case stays on the original flat "
            "code/n8n/workflow.json path unconditionally."
        ),
    )
    parser.add_argument(
        "--agent-rounds", type=int, default=None, metavar="N",
        help=(
            "python only. THIS INVOCATION ONLY (never persisted or checkpointed): max rounds per builder "
            "attempt (default: MAX_ROUNDS_PER_ATTEMPT). Recorded in the build log and journal."
        ),
    )
    parser.add_argument(
        "--read-first-round", action="store_true",
        help=(
            "python only. THIS INVOCATION ONLY: the builder's round 1 may be exploration only "
            "(read/list/search) without that counting as no progress or forcing a write; the progress "
            "requirement starts from round 2. Recorded in the build log and journal."
        ),
    )
    parser.add_argument(
        "--live-smoke",
        action="store_true",
        help=(
            "python only. code/smoke_run.py's mandatory dry-mode run (zero real cost - see "
            "pipeline/nodes/builder_node.py's AGENT_SYSTEM_PROMPT) happens automatically on every "
            "agentic build; this flag ADDITIONALLY authorizes THIS INVOCATION ONLY to also run it in "
            "'live' mode - real calls to whatever real, paid external vendors this project's own "
            "built code uses, at real cost. Never persisted, never inherited by a resumed run - see "
            "AGENTS.md's live-smoke hard rule. Douglas must give fresh, explicit consent for every "
            "single run this is passed on."
        ),
    )
    parser.add_argument(
        "--until-unit",
        help=(
            "python only, and only meaningful when this project's Architect decision declared "
            "build_units (see pipeline/review_architect.py) - the python-platform analog of "
            "--until-slice. Build only through this unit (its name, or a 1-based number) and stop; "
            "pass 'all' to build every remaining unit in one run. Omit to advance exactly one unit "
            "past whatever this project's last run reported built - same cadence-control shape as "
            "before (2026-09-28, Douglas's direct instruction: retargeted from PM-epic/checkpoint "
            "stages to Architect-decomposed units, which give the same 'don't fly blind through one "
            "long unsupervised run' property at finer granularity, since a properly-decomposed unit "
            "is already ordered, contracted, and sized for a genuine behavioral check). Has no "
            "effect on a project whose Architect decision declared no build_units - that stays a "
            "single, undecomposed run, same as a project with no Architect at all."
        ),
    )
    parser.add_argument(
        "--answer", action="append", default=[],
        help=(
            "Douglas's answer to the previous run's AMBIGUOUS question. Stored in "
            "<project>/.ambiguity_answers.jsonl and shown to the builder and reviewer as an "
            "authoritative ruling on this and every later run. Repeatable."
        ),
    )
    parser.add_argument(
        "--dismiss-finding", action="append", default=[],
        help=(
            "With --answer: a substring; carried findings containing it (case-insensitive) are "
            "dropped now and on later runs, e.g. for a finding Douglas ruled a false positive. Repeatable."
        ),
    )
    parser.add_argument(
        "--finding", action="append", default=[],
        help=(
            "Inject an externally verified finding (repeatable). Stored in .external_findings.jsonl, treated "
            "as must-fix on this and every later run until the reviewer confirms it fixed (with a verbatim "
            "code quote) or Douglas dismisses it. Start it with the unit/file/function name ('events.py: ...') "
            "or pass --finding-unit so it is matched to the right unit."
        ),
    )
    parser.add_argument("--finding-source", default="Douglas", help="Who verified the --finding(s), e.g. 'Agent 1 concept check'.")
    parser.add_argument("--finding-unit", default="", help="Unit/file/function name the --finding(s) apply to.")
    parser.add_argument(
        "--dismiss-external-finding", action="append", default=[],
        help="A substring; open external findings containing it are dismissed (Douglas's call). Repeatable.",
    )
    args = parser.parse_args()

    # .resolve() (2026-09-28, root-cause fix alongside the two subprocess cwd fixes in
    # builder_node.py/agent_tools.py): project_dir/code_dir used to stay whatever shape
    # --project-dir was typed in as, including relative. A relative code_dir handed to a subprocess
    # as ITS OWN cwd, or resolved by the project's own code (a config's storage_root defaulting to
    # "./data"), depends entirely on wherever build_and_review.py itself happened to be launched
    # from - real, evidenced incident: a smoke-test run wrote build-agent/data/acme-property/...
    # because the invoking shell's cwd was build-agent's own repo root. Resolving once here means
    # every downstream use (subprocess cwd, checkpoint paths, project_dir.name for the build log,
    # str(project_dir) passed into other functions) is unambiguous from this point on, rather than
    # depending on every future call site remembering to handle a possibly-relative path correctly.
    project_dir = Path(args.project_dir).resolve()
    spec_path = project_dir / "SPEC.md"
    code_dir = project_dir / "code"
    if not spec_path.exists():
        print(f"ERROR: {spec_path} does not exist. Write and approve the spec before building.")
        return 1
    code_dir.mkdir(parents=True, exist_ok=True)

    # Enforced, not advisory: if this project's roster (<project>/.roster.json) includes PM and/or
    # Architect, their approved decision file must already exist - see pipeline/roster.py and
    # AGENTS.md "Wiring PM/Architect into the build". A project with no roster file, or one with
    # neither role marked True, has nothing enforced here at all.
    roster_problems = missing_prerequisites(project_dir)
    if roster_problems:
        for problem in roster_problems:
            print(f"ERROR: {problem}")
        return 1

    # Real framework gap, closed 2026-09-26 (reported by a peer session, confirmed against this
    # code): review_architect.py's schema always allowed several SAME-PLATFORM n8n workflows, but
    # the n8n builder could only ever produce one, with no error anywhere - Architect could design
    # 3 n8n workflows and the builder would silently build one, overwriting whichever node happened
    # to write last. This is that check: refuse rather than guess which workflow was meant.
    n8n_workflow, workflow_problem = resolve_workflow_slug(project_dir, args.workflow)
    if workflow_problem:
        print(f"ERROR: {workflow_problem}")
        return 1

    # Python-platform unit decomposition (see pipeline/review_architect.py's build_units, and
    # pipeline/nodes/builder_node.py's _build_python_units), replacing PM-epic/checkpoint stage
    # gating entirely (2026-09-28, Douglas's direct instruction): a properly-decomposed unit
    # sequence (ordered, contracted, sized for a real behavioral check) already gives the "don't fly
    # blind through one long unsupervised run" property the old stage system existed for, at finer
    # granularity - Douglas controls cadence the same way, just retargeted at units. Has no effect
    # at all on a project whose Architect decision declared no build_units (including a project with
    # no Architect at all) - unchanged single-pass behavior.
    python_build_units = []
    python_unit_index = None
    if args.platform == "python":
        python_build_units = load_python_build_units(project_dir)
        if python_build_units:
            if args.until_unit:
                python_unit_index = resolve_until_unit(python_build_units, args.until_unit)
            else:
                python_unit_index = next_unbuilt_stage_index(project_dir, len(python_build_units))
            target = python_build_units[python_unit_index]
            print(
                f"UNIT: building through {python_unit_index + 1} of {len(python_build_units)} "
                f"({target.get('name', '')!r}). Pass --until-unit to target a different one, or "
                f"--until-unit all to build every remaining unit in this one run.",
                file=sys.stderr,
            )
    # Same scope tag pipeline/checkpoint.py's own save_state computes internally - kept in sync here
    # so a fresh/wipe/resume decided in THIS function never drifts from what a round actually saved
    # under. Real incident this fixes (2026-09-26, for the old stage system - the same reasoning
    # applies to units): with no unit scoping at all, a --until-unit 2 invocation could resume unit
    # 1's already-"done" checkpoint outright, re-enter at the reviewer, build nothing, and report a
    # clean "pass" for zero code.
    python_unit_key = f"unit{python_unit_index}" if python_unit_index is not None else None

    try:
        validate_builder_reviewer_vendors(
            builder_model=args.builder_model,
            platform=args.platform,
            project_dir=str(project_dir),
        )
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 1

    # Say out loud which models this run will actually use, and whether each came from the global
    # default or this project's own models.yaml - so a per-project override is never a silent surprise.
    models_used = effective_models(str(project_dir), platform=args.platform, builder_model=args.builder_model)
    for role, info in models_used.items():
        print(f"MODEL {role}: {info['model']} ({info['source']})", file=sys.stderr)

    _estimate_run_cost(project_dir, args.platform)

    spec_text = spec_path.read_text(encoding="utf-8")

    if args.fresh:
        checkpoint.clear(project_dir, workflow=n8n_workflow, stage=python_unit_key)

    if args.fresh_n8n:
        removed = wipe_n8n_artifacts(code_dir, workflow=n8n_workflow)
        if removed:
            print(f"WIPED n8n artifacts: {removed}", file=sys.stderr)
        else:
            print("WIPED n8n artifacts: nothing to remove.", file=sys.stderr)

    for _rebuild_name in args.rebuild_unit:
        _known = [u.get("name") for u in python_build_units]
        if _rebuild_name not in _known:
            raise SystemExit(f"--rebuild-unit {_rebuild_name!r}: no such unit. Known units: {_known}")
        _save_unit_baseline(project_dir, _rebuild_name, set())
        print(f"REBUILD: unit {_rebuild_name!r} will be re-entered with an empty other-tests baseline.", file=sys.stderr)

    if args.fresh_python:
        removed = wipe_python_code(code_dir)
        if removed:
            print(f"WIPED python code: {removed}", file=sys.stderr)
        else:
            print("WIPED python code: nothing to remove.", file=sys.stderr)
        journal_cleared = clear_journal(project_dir)
        print(f"WIPED build journal: {'removed' if journal_cleared else 'nothing to remove'}", file=sys.stderr)
        (Path(project_dir) / ".unit_test_baselines.json").unlink(missing_ok=True)

    initial_state = {
        "project_dir": str(project_dir),
        "spec_path": str(spec_path),
        "code_dir": str(code_dir),
        "review_round": 0,
        "review_verdict": None,
        "review_problems": [],
        "review_must_fix": [],
        "review_should_fix": [],
        "review_nitpicks": [],
        "builder_status": "working",
        "ambiguity_question": None,
        "builder_model_override": args.builder_model,
        "platform": args.platform,
        "n8n_workflow": n8n_workflow,
        "total_tokens": 0,
        "reviewer_tokens_used": 0,
        "resume_stage": None,
        "until_slice": args.until_slice,
        "python_until_unit_index": python_unit_index,
        "coverage_batch_key": None,
        "coverage_batch_rounds": 0,
        "round_budget_exhausted": False,
        "final_fix_pass": False,
        "live_smoke": args.live_smoke,
        "agent_rounds": args.agent_rounds,
        "read_first_round": args.read_first_round,
    }

    saved =None if args.fresh else checkpoint.load_state(project_dir, spec_text, workflow=n8n_workflow, stage=python_unit_key)
    if saved:
        resume_stage = checkpoint.resume_stage_for(saved)
        initial_state = {
            **initial_state,
            **saved,
            # Never inherit a stale terminal status from the checkpoint - this run decides afresh.
            "builder_status": "working",
            "ambiguity_question": None,
            "resume_stage": resume_stage,
            # Command-line flags always win over whatever the interrupted run was using.
            "platform": args.platform,
            "builder_model_override": args.builder_model,
            "n8n_workflow": n8n_workflow,
            "until_slice": args.until_slice,
            "python_until_unit_index": python_unit_index,
            # Consent is per-invocation, never inherited from a checkpoint (it also was never
            # persisted there in the first place - see checkpoint.py) - a resumed run only gets live
            # mode if --live-smoke is passed again, fresh, on this exact command.
            "live_smoke": args.live_smoke,
            "agent_rounds": args.agent_rounds,
            "read_first_round": args.read_first_round,
        }
        print(
            f"RESUMING interrupted run: round {saved.get('review_round', 0)}, "
            f"{saved.get('total_tokens', 0)} tokens already spent, re-entering at {resume_stage}.",
            file=sys.stderr,
        )

    if args.answer:
        try:
            prev_result = json.loads((project_dir / "run_result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prev_result = {}
        prev_question = prev_result.get("question") if str(prev_result.get("status", "")).startswith("ambiguous") else ""
        for answer_text in args.answer:
            record_answer(project_dir, prev_question or "", answer_text, args.dismiss_finding)
            record_journal_entry(
                project_dir, REVIEW_ROUND_UNIT, "reviewer",
                f"DOUGLAS RULING on an AMBIGUOUS question. Question: {prev_question or '(not recorded)'} "
                f"Answer: {answer_text}",
            )
            try:
                record_build(project_dir.name, "ruling", {"question": prev_question or "", "answer": answer_text}, scope="ruling")
            except Exception as exc:  # noqa: BLE001 - side channel, never blocks the build
                print(f"WARNING: could not log ruling to build log: {exc}", file=sys.stderr)
    answers = load_answers(project_dir)
    initial_state["rulings"] = format_rulings(answers)
    if answers:
        for key in ("review_problems", "review_must_fix", "review_should_fix", "review_nitpicks", "carried_findings"):
            if initial_state.get(key):
                initial_state[key] = drop_dismissed(initial_state[key], answers)

    for text in args.finding:
        entry = record_finding(project_dir, text, args.finding_source, args.finding_unit)
        record_journal_entry(
            project_dir, REVIEW_ROUND_UNIT, "reviewer",
            f"EXTERNAL FINDING {entry['id']} from {entry['source']}: {text}",
        )
    if args.dismiss_external_finding:
        _open_before = load_open_findings(project_dir)
        dismissed = dismiss_matching(project_dir, args.dismiss_external_finding)
        print(f"DISMISSED external finding(s): {dismissed}", file=sys.stderr)
        for _n in unmatched_needles(_open_before, args.dismiss_external_finding):
            print(f"WARNING: --dismiss-external-finding {_n!r} matched no open finding (id or text)", file=sys.stderr)
        _rr = project_dir / "run_result.json"
        try:
            _prev = json.loads(_rr.read_text(encoding="utf-8"))
            _still = load_open_findings(project_dir)
            if _still:
                _prev["open_external_findings"] = [{"id": e["id"], "source": e.get("source"), "text": e["text"]} for e in _still]
            else:
                _prev.pop("open_external_findings", None)
            _rr.write_text(json.dumps(_prev, indent=2), encoding="utf-8")
        except (OSError, ValueError):
            pass
    external = load_open_findings(project_dir)
    if external:
        ext_lines = [finding_line(e) for e in external]
        merged_must = ext_lines + [f for f in (initial_state.get("review_must_fix") or []) if f not in ext_lines]
        merged_problems = ext_lines + [f for f in (initial_state.get("review_problems") or []) if f not in ext_lines]
        initial_state = {
            **initial_state,
            "external_findings": [{**e, "line": l} for e, l in zip(external, ext_lines)],
            "review_must_fix": merged_must,
            "review_problems": merged_problems,
        }
        print(f"INJECTED {len(external)} open external finding(s) as must-fix.", file=sys.stderr)

    carried = {} if (saved or args.fresh or args.fresh_python) else _load_carried_findings(project_dir)
    if carried and answers:
        carried = {
            "must_fix": drop_dismissed(carried["must_fix"], answers),
            "should_fix": drop_dismissed(carried["should_fix"], answers),
        }
    if carried and (carried["must_fix"] or carried["should_fix"]):
        all_carried = carried["must_fix"] + carried["should_fix"]
        initial_state = {
            **initial_state,
            "carried_must_fix": carried["must_fix"],
            "review_problems": list(dict.fromkeys(initial_state.get("review_problems", []) + all_carried)),
            "review_must_fix": list(dict.fromkeys(initial_state.get("review_must_fix", []) + carried["must_fix"])),
            "review_should_fix": carried["should_fix"],
            "carried_findings": all_carried,
        }
        record_journal_entry(
            project_dir, REVIEW_ROUND_UNIT, "reviewer",
            f"Carried {len(carried['must_fix'])} must-fix and {len(carried['should_fix'])} should-fix "
            "unresolved finding(s) from the previous run into this run as round-0 builder feedback: "
            + " | ".join(all_carried)[:1500],
        )
        print(f"CARRIED {len(all_carried)} unresolved finding(s) from the previous run.", file=sys.stderr)

    if args.agent_rounds is not None or args.read_first_round:
        budget = {"agent_rounds": args.agent_rounds, "read_first_round": args.read_first_round}
        record_journal_entry(
            project_dir, REVIEW_ROUND_UNIT, "reviewer",
            f"BUILDER BUDGET OVERRIDE for this run only: rounds per attempt = "
            f"{args.agent_rounds if args.agent_rounds is not None else 'default'}, "
            f"read-first round 1 = {args.read_first_round}.",
        )
        try:
            record_build(project_dir.name, "builder_budget_override", budget, scope="override")
        except Exception as exc:  # noqa: BLE001 - side channel, never blocks the build
            print(f"WARNING: could not log builder budget override: {exc}", file=sys.stderr)

    graph = build_graph()
    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:  # OpenRouter outage, exhausted retries, malformed model output, etc.
        # Deliberately leave the checkpoint in place here - this is exactly the case where resuming
        # rather than restarting saves the most work.
        result = {"status": "error", "message": str(exc)}
        result_path = project_dir / "run_result.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        _update_memory(project_dir, result, models_used)
        print(json.dumps(result, indent=2))
        return 1

    if final_state.get("builder_status") == "ambiguous":
        result = {
            "status": "ambiguous",
            "question": final_state.get("ambiguity_question"),
            "total_tokens": final_state.get("total_tokens", 0),
        }
        if _question_already_ruled(result["question"], answers):
            result["already_ruled_on"] = True
            result["note"] = (
                "This question was already answered in .ambiguity_answers.jsonl; the builder asked "
                "again despite the ruling. Check the ruling's wording and that its --dismiss-finding "
                "substring matches the finding text."
            )
    elif final_state.get("final_fix_pass"):
        # 1-round budget: the reviewer's findings went to the builder for one fix pass. The builder
        # only reports "done" once its own verification (tests, smoke) passed - that is the whole
        # acceptance rule; there is deliberately no second review.
        result = {
            "status": "pass",
            "rounds": final_state.get("review_round"),
            "total_tokens": final_state.get("total_tokens", 0),
            "final_fix_pass": True,
            "accepted_without_rereview": True,
            "findings_sent_to_builder": final_state.get("review_must_fix") or [],
        }
    elif final_state.get("review_verdict") == "pass":
        result = {
            "status": "pass",
            "rounds": final_state.get("review_round"),
            "total_tokens": final_state.get("total_tokens", 0),
        }
    else:
        result = {
            "status": "failed_after_max_rounds",
            "rounds": final_state.get("review_round"),
            "problems": final_state.get("review_problems"),
            "total_tokens": final_state.get("total_tokens", 0),
        }

    # What was actually built, so Agent 1's concept-conformance check (see AGENTS.md) has a concrete
    # inventory to compare against SPEC.md rather than re-reading the whole artifact.
    result["built"] = _build_summary(code_dir, args.platform, workflow=n8n_workflow)
    result["concept_check_required"] = True
    # Transparency, same principle as AGENTS.md's "always name which agent/model acted" hard rule:
    # whether THIS run was authorized to spend real money on a live smoke run must be visible in the
    # result Agent 1 relays, not something Douglas has to infer or take on faith.
    result["live_smoke_authorized"] = args.live_smoke
    if args.agent_rounds is not None or args.read_first_round:
        result["builder_budget_override"] = {
            "agent_rounds": args.agent_rounds, "read_first_round": args.read_first_round,
        }

    # Full severity breakdown, always surfaced when the reviewer has actually run this invocation -
    # even on a "pass" (2026-09-29, Douglas's explicit direction: "report all severities in the run
    # output so nothing is hidden"). should_fix/nitpick never block delivery, but must still reach
    # Douglas, not vanish just because nothing looped on them.
    should_fix = list(final_state.get("review_should_fix") or [])
    for t in final_state.get("tolerated_failures") or []:
        should_fix.append(
            f"{t.get('owner') or t['test'].split('::', 1)[0]}: pre-existing failing test {t['test']} was "
            f"tolerated (already red before unit '{t.get('tolerated_for')}' started) and is still red - "
            f"fix the code or the test. {t.get('reason') or ''}".strip()
        )
    for w in final_state.get("seam_warnings") or []:
        if w not in should_fix:
            should_fix.append(w)
    if final_state.get("review_must_fix") or should_fix or final_state.get("review_nitpicks"):
        result["review_findings"] = {
            "must_fix": final_state.get("review_must_fix") or [],
            "should_fix": should_fix,
            "nitpick": final_state.get("review_nitpicks") or [],
        }

    open_external = load_open_findings(project_dir)
    if open_external:
        result["open_external_findings"] = [{"id": e["id"], "source": e.get("source"), "text": e["text"]} for e in open_external]

    # Standing reminder, same shape as concept_check_required above: a genuine project completion
    # (a real pass, or a real stop after exhausting the round cap - never a partial-slice stop or an
    # ambiguous question, which aren't the project concluding) means the mandatory memory review
    # (AGENTS.md "Mandatory memory review - after every project") isn't optional busywork - it's
    # what actually turns this run's friction into next time's fewer questions.
    if _is_project_complete(result["status"], bool(final_state.get("partial_build"))):
        result["memory_review_required"] = True

    if final_state.get("slices_total"):
        result["slices"] = {
            "total": final_state.get("slices_total"),
            "built": final_state.get("slices_built"),
            "remaining": final_state.get("slices_remaining"),
            "partial": bool(final_state.get("partial_build")),
        }
        if final_state.get("partial_build"):
            # A deliberate stop at a functional base, not a finished product - don't let a "pass"
            # here read as "the whole thing is built".
            result["status"] = f"{result['status']}_partial"

    # A run that reached a real verdict is finished - a later run targeting THIS SAME scope should
    # start clean rather than resuming this one. See _run_concluded's own docstring for the real
    # incident this guards against.
    if _run_concluded(result["status"]):
        checkpoint.clear(project_dir, workflow=n8n_workflow, stage=python_unit_key)

    result_path = project_dir / "run_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _update_memory(project_dir, result, models_used)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
