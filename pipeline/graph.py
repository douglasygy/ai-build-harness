from __future__ import annotations

import sys
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from . import checkpoint
from .build_log import record_build
from .build_scribe import summarize_build_run
from .config_loader import ConfigError, load_role
from .nodes.builder_node import builder_node
from .nodes.reviewer_node import reviewer_node
from .state import BuildState


def _checkpointed(node_fn):
    """Persist the run's state after a node finishes, so an interrupted run (e.g. the OS killing
    the process for memory) can resume instead of starting over. Platform-agnostic - applies to
    every builder implementation, not just n8n.
    """

    def wrapped(state: BuildState) -> BuildState:
        result = node_fn(state)
        checkpoint.save_state(result)
        return result

    wrapped.__name__ = getattr(node_fn, "__name__", "node")
    return wrapped


def _log_reviewer_round(state: BuildState, result: BuildState) -> None:
    """Best-effort: logs ONE failing reviewer round via the Descriptive Scribe - the friction worth
    learning from, not every round. A round that passes triggers no call, so a clean build (the
    common case) costs nothing extra beyond its final per-run summary. Never allowed to affect the
    actual builder<->reviewer loop - any failure here is logged and swallowed.
    """
    try:
        project_dir = state.get("project_dir")
        round_result = {
            "round": result.get("review_round"),
            "role": "reviewer",
            "status": result.get("review_verdict"),
            "problems": result.get("review_problems"),
        }
        summary, _ = summarize_build_run(round_result, project_dir=project_dir)
        project_name = Path(project_dir).name if project_dir else "unknown"
        try:
            model = load_role("reviewer", project_dir=project_dir).model
        except ConfigError:
            model = None
        record_build(
            project_name, "fail", summary, scope="round", round_num=result.get("review_round"),
            role="reviewer", model=model,
        )
    except Exception as exc:  # noqa: BLE001 - a logging side channel must never break a real build
        print(f"WARNING: round-level memory logging failed: {exc}", file=sys.stderr)


def _checkpointed_reviewer(node_fn):
    def wrapped(state: BuildState) -> BuildState:
        result = node_fn(state)
        checkpoint.save_state(result)
        if result.get("review_verdict") != "pass":
            _log_reviewer_round(state, result)
        return result

    wrapped.__name__ = getattr(node_fn, "__name__", "node")
    return wrapped


def route_entry(state: BuildState) -> str:
    """Normally a run starts at the builder. A resumed run whose builder had already finished (with
    nothing sent back to it) re-enters at the reviewer instead, so completed build work isn't redone.

    A resumed run whose round budget was already exhausted stops immediately rather than paying for
    another review round it isn't entitled to - otherwise an interrupted run could report more rounds
    than the cap allows (seen in practice: a resume reporting "rounds: 4" against a cap of 3).

    Checks round_budget_exhausted (computed and persisted by reviewer_node), NOT a raw
    review_round >= MAX_ROUNDS comparison - see route_after_reviewer for why those two are no
    longer the same thing.
    """
    if state.get("round_budget_exhausted"):
        return END
    return "reviewer" if state.get("resume_stage") == "reviewer" else "builder"


def route_after_builder(state: BuildState) -> str:
    if state.get("builder_status") == "ambiguous":
        return END
    if state.get("final_fix_pass"):
        return END
    return "reviewer"


def route_after_reviewer(state: BuildState) -> str:
    """Stops the run on a pass, or once round_budget_exhausted (set by reviewer_node) is true.

    Deliberately NOT a plain review_round >= MAX_ROUNDS check (2026-09-27, Douglas's explicit
    direction): when a round's fail verdict is driven purely by unresolved test-coverage debt (see
    reviewer_node.py's pure_coverage_fail), MAX_ROUNDS caps rounds spent on the CURRENT batch of at
    most 5 coverage findings, not the whole run's total backlog - review_round can legitimately climb
    well past MAX_ROUNDS while round_budget_exhausted stays false, as long as each successive batch
    is still making genuine progress. A genuine correctness-bug fail (or a fail mixed with one) still
    uses the original flat per-run cap - reviewer_node only relaxes this for coverage-only fails.
    """
    if state.get("review_verdict") == "pass":
        return END
    if state.get("round_budget_exhausted"):
        return END
    # final_fix_pass (1-round budget) falls through to the builder on purpose - see reviewer_node.
    return "builder"


def build_graph():
    graph = StateGraph(BuildState)
    graph.add_node("builder", _checkpointed(builder_node))
    graph.add_node("reviewer", _checkpointed_reviewer(reviewer_node))
    graph.add_conditional_edges(START, route_entry, {"builder": "builder", "reviewer": "reviewer", END: END})
    graph.add_conditional_edges("builder", route_after_builder, {"reviewer": "reviewer", END: END})
    graph.add_conditional_edges("reviewer", route_after_reviewer, {"builder": "builder", END: END})
    return graph.compile()
