"""Standing fallback model for when a role's normally-pinned model is genuinely unavailable, not
just having a transient blip (transient blips are handled by with_retries_nonempty in retry.py -
this module only engages once that has already exhausted every retry attempt).

Douglas's explicit direction (2026-09-26): if a role's pinned model is down, fall back to a single,
fixed, highest-capability model rather than leaving a project stuck retrying a broken model forever.
The fallback must be "seamless" for the call that triggered it (retried immediately against the
fallback, not surfaced as a failure) AND must stick for the rest of that project - every subsequent
call to that role in that project uses the fallback directly, without re-attempting the broken
primary each time - until Douglas explicitly clears it. This is model SELECTION (one of the four
decisions always reserved to Douglas - see AGENTS.md) happening automatically, which is exactly why
every fallback is loud (printed) and durably logged (record_decision), never silent.

FALLBACK_MODEL is pinned to an exact version, not a floating "-latest" alias, matching this repo's
existing convention (every other role in config/models.yaml is an exact pin) - a floating alias
would silently change behavior whenever the vendor updates what "latest" means, which is precisely
the kind of surprise this pipeline is built to avoid.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config_loader import builder_role_for, load_role
from .decision_log import record_decision
from .model_factory import get_chat_model
from .provider_pin import ask_provider_switch, looks_like_provider_failure, set_provider_override
from .retry import with_retries_nonempty

# Verified against OpenRouter's own listing 2026-09-26: anthropic/claude-fable-5, 1M context,
# 128K max output - Douglas's explicit pick, "highest" capability tier, for exactly this job.
FALLBACK_MODEL = "anthropic/claude-fable-5"

FALLBACK_STATE_FILENAME = ".model_fallback_state.json"


class FallbackBlockedError(Exception):
    """Raised instead of silently falling back, when doing so would break a hard-enforced rule:
    reviewer's vendor must differ from builder's (see config_loader.validate_builder_reviewer_
    vendors), or business_analyst's vendor must not collide with concept_advocate's (both would
    make FALLBACK_MODEL's own Anthropic vendor the collision - added 2026-09-29 after a real,
    silent double-occurrence live). Surfaces to Douglas rather than trading one broken guarantee
    for another automatically.
    """


def _state_path(project_dir) -> Path:
    return Path(project_dir) / FALLBACK_STATE_FILENAME


def load_fallback_state(project_dir) -> dict:
    path = _state_path(project_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def get_active_fallback(project_dir, role: str) -> Optional[str]:
    """The fallback model already in effect for this role in this project, if any - None means
    this role is still on its normally-pinned model."""
    if not project_dir:
        return None
    entry = load_fallback_state(project_dir).get(role)
    return entry.get("model") if entry else None


def _save_state(project_dir, state: dict) -> None:
    path = _state_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def clear_fallback(project_dir, role: Optional[str] = None) -> None:
    """Clears one role's fallback (or every role's, if role is None) - the only way this ever
    reverts; nothing in this module ever switches a project back to its primary model on its own.
    """
    state = load_fallback_state(project_dir)
    cleared = list(state.keys()) if role is None else ([role] if role in state else [])
    if role is None:
        state = {}
    else:
        state.pop(role, None)
    _save_state(project_dir, state)
    for r in cleared:
        record_decision(
            decision_point=f"Clear model fallback for role '{r}' in project {Path(project_dir).name}",
            choice="Reverted to the role's normally-pinned model in config/models.yaml.",
            project=Path(project_dir).name,
            proposed_by="Douglas",
        )


def _record_fallback(project_dir, role: str, primary_model: str, reason: str) -> None:
    state = load_fallback_state(project_dir)
    state[role] = {
        "model": FALLBACK_MODEL,
        "primary_model": primary_model,
        "since": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    _save_state(project_dir, state)
    banner = (
        f"MODEL FALLBACK ENGAGED: role '{role}' ({primary_model}) failed - {reason}. "
        f"This project now uses {FALLBACK_MODEL} for '{role}' on every call, not just this one, "
        f"until you clear it: python -m pipeline.model_fallback clear {project_dir} {role}"
    )
    print(banner, file=sys.stderr)
    record_decision(
        decision_point=f"Role '{role}' model fallback in project {Path(project_dir).name}",
        choice=f"Automatically switched '{role}' from {primary_model} to standing fallback {FALLBACK_MODEL} "
        f"for the rest of this project.",
        project=Path(project_dir).name,
        proposed_by=f"{role} ({primary_model})",
        rationale=reason,
    )


def _would_collide_with_builder_vendor(project_dir, platform: str) -> bool:
    """The one hard-enforced cross-vendor rule this fallback must never silently break: reviewer's
    vendor must differ from builder's (config_loader.validate_builder_reviewer_vendors). Falling
    reviewer back to Claude Fable is fine UNLESS this project's builder is also Anthropic already.
    """
    try:
        builder_vendor = load_role(builder_role_for(platform), project_dir=project_dir).vendor
    except Exception:
        return False
    return builder_vendor == FALLBACK_MODEL.split("/", 1)[0]


def resolve_llm_for_role(role: str, project_dir=None, model_override: Optional[str] = None, platform: str = "python"):
    """Drop-in replacement for model_factory.get_chat_model that checks for an already-active
    fallback first. If this role is already on fallback for this project, returns a client on the
    fallback model directly - the primary is not retried again on every call, per Douglas's "not
    just that output, that entire project" requirement.

    An explicit `model_override` (e.g. --builder-model on this specific run) still wins over an
    active fallback - it's a fresh, conscious choice Douglas is making right now, and this repo's
    existing convention (config_loader.load_role) already treats an explicit per-run override as
    the highest-precedence source of truth for a role's model.
    """
    if model_override:
        return get_chat_model(role, model_override=model_override, project_dir=project_dir)
    active = get_active_fallback(project_dir, role) if project_dir else None
    if active:
        return get_chat_model(role, model_override=active, project_dir=project_dir)
    return get_chat_model(role, model_override=model_override, project_dir=project_dir)


def attempt_fallback(role: str, invoke_fn, project_dir=None, platform: str = "python", primary_error: Exception = None):
    """Called when a role's primary call has already exhausted with_retries_nonempty. Tries the
    standing fallback model ONCE (itself retried the same way) so THIS call can still succeed even
    without a project to remember the choice in (e.g. preference_analyst's usual project-less
    invocation) - persisting the fallback for every later call in the same project only happens
    when project_dir is given, since that's the only case with anywhere durable to remember it.
    Raises FallbackBlockedError instead of a silent vendor collision for reviewer, or for
    business_analyst (2026-09-29, real incident: creative-team's concept bounce hit this exact
    silent collision TWICE live - business_analyst's primary failed with a real, unrelated error
    (context-window overflow), fell back to FALLBACK_MODEL (Anthropic) without a word, and the
    round quietly stopped being "concept_advocate's Claude vs. a genuinely independent second
    opinion" and became Claude arguing with itself, undetected. The existing
    warn_if_business_analyst_is_anthropic only checks the CONFIGURED primary model at startup - it
    has no way to catch a mid-run fallback silently changing the EFFECTIVE model later). Re-raises
    the original error, chained, if the fallback call fails too.

    `invoke_fn` takes the resolved fallback ChatOpenAI client and returns its `.invoke(...)` result
    - the caller supplies this so the exact same messages/config are reused, not re-derived here.
    """
    try:
        primary_model = load_role(role, project_dir=project_dir).model
    except Exception:
        primary_model = "unknown"

    pinned = load_role(role, project_dir=project_dir).provider_order if primary_model != "unknown" else None
    if pinned and primary_error is not None and looks_like_provider_failure(primary_error):
        chosen = ask_provider_switch(role, primary_model, pinned, project_dir, primary_error)
        if chosen:
            if project_dir:
                set_provider_override(project_dir, role, primary_model, chosen)
                llm = get_chat_model(role, project_dir=project_dir)
            else:
                llm = get_chat_model(role, project_dir=project_dir)
                llm.extra_body = {**(llm.extra_body or {}), "provider": {"order": [chosen], "allow_fallbacks": False}}
            response = with_retries_nonempty(lambda: invoke_fn(llm))
            record_decision(
                decision_point=f"Role '{role}' provider switch",
                choice=f"Douglas moved '{role}' ({primary_model}) from {', '.join(pinned)} to {chosen}; prompt cache restarted.",
                project=Path(project_dir).name if project_dir else "pipeline-design",
                proposed_by="Douglas",
                rationale=str(primary_error),
            )
            return response

    if role == "reviewer" and project_dir and _would_collide_with_builder_vendor(project_dir, platform):
        raise FallbackBlockedError(
            f"reviewer's primary model ({primary_model}) failed, but the standing fallback "
            f"({FALLBACK_MODEL}) is the same vendor as this project's builder - falling back "
            f"automatically would silently break the builder/reviewer vendor-independence "
            f"requirement. This needs your call, not an automatic one."
        ) from primary_error

    if role == "business_analyst" and FALLBACK_MODEL.split("/", 1)[0] == "anthropic":
        raise FallbackBlockedError(
            f"business_analyst's primary model ({primary_model}) failed, but the standing fallback "
            f"({FALLBACK_MODEL}) is Anthropic - concept_advocate is ALSO Anthropic, so falling "
            f"back automatically would silently turn the concept bounce's genuinely independent "
            f"second opinion into Claude arguing with itself. This needs your call, not an "
            f"automatic one."
        ) from primary_error

    fallback_llm = get_chat_model(role, model_override=FALLBACK_MODEL, project_dir=project_dir)
    try:
        response = with_retries_nonempty(lambda: invoke_fn(fallback_llm))
    except Exception as fallback_error:
        raise RuntimeError(
            f"Role '{role}' primary model ({primary_model}) failed: {primary_error}. "
            f"Fallback ({FALLBACK_MODEL}) also failed: {fallback_error}."
        ) from fallback_error

    if project_dir:
        _record_fallback(project_dir, role, primary_model, reason=str(primary_error))
    else:
        print(
            f"MODEL FALLBACK (one-off, no project to persist against): role '{role}' ({primary_model}) "
            f"failed - {primary_error}. Used {FALLBACK_MODEL} for this call only.",
            file=sys.stderr,
        )
        record_decision(
            decision_point=f"Role '{role}' model fallback (project-less call)",
            choice=f"Used standing fallback {FALLBACK_MODEL} for this one call after {primary_model} failed.",
            project="pipeline-design",
            proposed_by=f"{role} ({primary_model})",
            rationale=str(primary_error),
        )
    return response


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect or clear this project's model-fallback state")
    sub = parser.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", help="Show any active fallbacks for a project")
    list_p.add_argument("project_dir")

    clear_p = sub.add_parser("clear", help="Clear a fallback (or all of them) for a project")
    clear_p.add_argument("project_dir")
    clear_p.add_argument("role", nargs="?", default=None, help="Omit to clear every role's fallback")

    args = parser.parse_args()
    if args.command == "list":
        state = load_fallback_state(args.project_dir)
        if not state:
            print("No active fallbacks.")
        for role, entry in state.items():
            print(f"{role}: {entry['primary_model']} -> {entry['model']} (since {entry['since']}) - {entry['reason']}")
    elif args.command == "clear":
        clear_fallback(args.project_dir, args.role)
        print(f"Cleared fallback for {args.role or 'all roles'} in {args.project_dir}.")
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(_cli())
