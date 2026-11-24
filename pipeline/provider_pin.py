"""Default-host pinning and the provider-failure prompt (Douglas, 2026-09-30).

Every model is pinned to its vendor's own first-party OpenRouter host (config/provider_defaults.yaml)
so prompt caching stays on one host. If that host fails, the pipeline never silently moves to
another host (that ends the cache and changes behaviour): it asks Douglas. In a terminal it prompts
inline; when there is no terminal (Claude Code orchestrating) it stops with ProviderFailureError
carrying the alternatives and the exact command to choose one.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import yaml

DEFAULTS_PATH = Path(__file__).resolve().parent.parent / "config" / "provider_defaults.yaml"
STATE_FILENAME = ".provider_override_state.json"


class ProviderFailureError(Exception):
    """The pinned provider failed and Douglas has not chosen a different one yet."""


def default_provider_for(model: str) -> Optional[list]:
    try:
        hosts = (yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8")) or {}).get("default_hosts") or {}
    except (OSError, yaml.YAMLError):
        return None
    slug = hosts.get(model.split("/", 1)[0])
    return [slug] if slug else None


def _state_path(project_dir) -> Path:
    return Path(project_dir) / STATE_FILENAME


def _load_state(project_dir) -> dict:
    try:
        data = json.loads(_state_path(project_dir).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def get_provider_override(project_dir, role: str, model: str) -> Optional[list]:
    """Douglas's chosen provider for this role in this project - only valid for the model it was
    chosen for, so switching the role's model drops it automatically."""
    if not project_dir:
        return None
    entry = _load_state(project_dir).get(role)
    if entry and entry.get("model") == model and entry.get("provider"):
        return [entry["provider"]]
    return None


def set_provider_override(project_dir, role: str, model: str, provider: str) -> None:
    state = _load_state(project_dir)
    state[role] = {"model": model, "provider": provider}
    path = _state_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def clear_provider_override(project_dir, role: Optional[str] = None) -> None:
    state = {} if role is None else {k: v for k, v in _load_state(project_dir).items() if k != role}
    _state_path(project_dir).write_text(json.dumps(state, indent=2), encoding="utf-8")


def looks_like_provider_failure(error: Exception) -> bool:
    code = getattr(error, "status_code", None)
    if code in (404, 408, 429, 500, 502, 503, 504, 529):
        return True
    name = type(error).__name__.lower()
    if "timeout" in name or "connection" in name:
        return True
    msg = str(error).lower()
    return any(s in msg for s in ("provider", "no endpoints", "overloaded", "unavailable"))


def alternative_providers(model: str, current: list) -> list:
    """Other OpenRouter hosts currently listed for this model (public endpoint, best effort)."""
    try:
        with urllib.request.urlopen(f"https://openrouter.ai/api/v1/models/{model}/endpoints", timeout=10) as r:
            endpoints = json.loads(r.read().decode("utf-8"))["data"]["endpoints"]
    except Exception:
        return []
    skip = {c.split("/")[0] for c in current}
    seen: list = []
    for e in endpoints:
        tag = e.get("tag") or ""
        if tag and tag.split("/")[0] not in skip and tag not in seen:
            seen.append(tag)
    return seen


def ask_provider_switch(role: str, model: str, current: list, project_dir, error: Exception) -> Optional[str]:
    """Returns the provider Douglas chose, or None if he chose the standing fallback MODEL instead.
    Raises ProviderFailureError if he aborts or nobody can be asked (no terminal)."""
    alts = alternative_providers(model, current)
    cmd = f"python -m pipeline.provider_pin set <project_dir> {role} <provider>"
    header = (
        f"PROVIDER FAILED: role '{role}' ({model}) via {', '.join(current)}: {error}\n"
        f"Switching provider ENDS the prompt cache built up on {', '.join(current)} (first turns on the "
        f"new host are full price) and the new host may behave differently."
    )
    if not sys.stdin or not sys.stdin.isatty():
        raise ProviderFailureError(
            header + "\nAlternatives listed on OpenRouter: " + (", ".join(alts) or "(could not fetch)")
            + f"\nTo switch, run: {cmd}  (or retry later). Nothing was changed."
        ) from error
    print(header, file=sys.stderr)
    for i, a in enumerate(alts, 1):
        print(f"  {i}) switch to {a}", file=sys.stderr)
    print("  m) use the standing fallback MODEL instead\n  a) abort (keep the pin, stop the run)", file=sys.stderr)
    choice = input("Choice: ").strip().lower()
    if choice.isdigit() and 1 <= int(choice) <= len(alts):
        return alts[int(choice) - 1]
    if choice == "m":
        return None
    raise ProviderFailureError(header + "\nAborted by Douglas.") from error


def _cli() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Choose or clear a project's provider for a role")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("set")
    s.add_argument("project_dir")
    s.add_argument("role")
    s.add_argument("provider")
    c = sub.add_parser("clear")
    c.add_argument("project_dir")
    c.add_argument("role", nargs="?")
    ls = sub.add_parser("list")
    ls.add_argument("project_dir")
    a = p.parse_args()
    if a.command == "set":
        from .config_loader import load_role

        cfg = load_role(a.role, project_dir=a.project_dir)
        set_provider_override(a.project_dir, a.role, cfg.model, a.provider)
        print(f"{a.role} ({cfg.model}) in {a.project_dir} now routes to {a.provider}. Prompt cache restarts.")
    elif a.command == "clear":
        clear_provider_override(a.project_dir, a.role)
        print("Cleared; back to the default host.")
    else:
        print(json.dumps(_load_state(a.project_dir), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
