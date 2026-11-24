from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "models.yaml"
MODEL_LIMITS_PATH = REPO_ROOT / "config" / "model_limits.yaml"

# A project may carry its own models.yaml with the same shape as the global one. Anything it
# defines overrides the global default FOR THAT PROJECT ONLY - other projects and the global
# defaults are never touched. See docs in AGENTS.md ("Per-project model overrides").
PROJECT_CONFIG_FILENAME = "models.yaml"

# Tuning keys that are meaningful only for the specific model they were written for. If a project
# overrides `model` with a different vendor's model, these are NOT inherited from the global config
# (e.g. provider_order: ["anthropic"] must not silently follow a switch to a DeepSeek model), unless
# the project sets them explicitly itself.
_MODEL_SPECIFIC_KEYS = ("reasoning_max_tokens", "provider_order")


from .provider_pin import default_provider_for, get_provider_override  # noqa: E402


class ConfigError(Exception):
    pass


@dataclass
class RoleConfig:
    role: str
    model: str
    base_url: str
    temperature: float
    max_tokens: int
    reasoning_max_tokens: Optional[int] = None
    provider_order: Optional[list] = None
    source: str = "global"  # "global" or "project" - which config supplied the model id
    # "openrouter" (default) or "claude_code" (headless `claude -p` as the python builder). Only
    # ever read from a PROJECT's own models.yaml - never inherited globally - so it stays
    # project-scoped (trading-team trial, Douglas 2026-10-03).
    backend: str = "openrouter"

    @property
    def vendor(self) -> str:
        if self.backend == "claude_code":
            return "anthropic"
        return self.model.split("/", 1)[0]


def _load_raw() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def project_config_path(project_dir) -> Path:
    return Path(project_dir) / PROJECT_CONFIG_FILENAME


def _load_project_raw(project_dir: Optional[str]) -> dict:
    if not project_dir:
        return {}
    path = project_config_path(project_dir)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read project model config at {path}: {exc}") from exc


def _vendor_of(model: str) -> str:
    return model.split("/", 1)[0]


def _clamp_max_tokens(model: str, requested: int) -> int:
    """Lower `requested` to the model's known max output, warning when it changes."""
    try:
        with open(MODEL_LIMITS_PATH, "r", encoding="utf-8") as f:
            ceilings = (yaml.safe_load(f) or {}).get("max_output_tokens") or {}
    except (OSError, yaml.YAMLError):
        return requested
    ceiling = ceilings.get(model)
    if isinstance(ceiling, int) and requested > ceiling:
        print(
            f"WARNING: max_tokens {requested:,} exceeds {model}'s max output {ceiling:,}; using {ceiling:,}.",
            file=sys.stderr,
        )
        return ceiling
    return requested


def load_role(
    role: str,
    model_override: Optional[str] = None,
    project_dir: Optional[str] = None,
) -> RoleConfig:
    """Resolve a role's model config: global defaults, overlaid by this project's own models.yaml
    (if any), overlaid by an explicit per-run model_override (highest precedence).
    """
    raw = _load_raw()
    project_raw = _load_project_raw(project_dir)

    global_entry = dict((raw.get("roles") or {}).get(role) or {})
    project_entry = dict((project_raw.get("roles") or {}).get(role) or {})

    if not global_entry and not project_entry:
        raise ConfigError(
            f"No config for role '{role}' in {CONFIG_PATH}"
            + (f" or {project_config_path(project_dir)}" if project_dir else "")
        )

    entry = {**global_entry, **project_entry}

    # If the project (or a --builder-model flag) points this role at a different vendor's model,
    # don't let the global config's model-specific tuning silently ride along - it was written for
    # a different model and is usually wrong or actively harmful for the new one.
    effective_model = model_override or entry.get("model")
    if effective_model and global_entry.get("model"):
        if _vendor_of(effective_model) != _vendor_of(global_entry["model"]):
            for key in _MODEL_SPECIFIC_KEYS:
                if key not in project_entry:
                    entry.pop(key, None)

    model = model_override or entry["model"]
    if model_override:
        source = "run-override"
    elif "model" in project_entry:
        source = "project"
    else:
        source = "global"

    return RoleConfig(
        role=role,
        model=model,
        base_url=entry.get(
            "base_url",
            project_raw.get("openrouter_base_url")
            or raw.get("openrouter_base_url", "https://openrouter.ai/api/v1"),
        ),
        temperature=entry.get("temperature", 0.2),
        max_tokens=_clamp_max_tokens(model, entry.get("max_tokens", 4096)),
        # Optional: caps OpenRouter's "reasoning" token budget for models that support it. Without
        # this, a reasoning-capable model can spend its entire max_tokens on invisible reasoning
        # tokens and return empty visible content (finish_reason "length", 0 real output) - seen in
        # practice on the builder role. Unset by default so this never changes behavior for a role
        # that hasn't shown the problem.
        reasoning_max_tokens=entry.get("reasoning_max_tokens"),
        # Optional: pins OpenRouter to specific named providers for this model, skipping automatic
        # fallback to whatever else claims to serve it. Seen in practice: a large-context request
        # for a 1M-context model got routed to a "Cloudflare" provider reporting a 32768-token
        # limit and rejecting the request - not one of the model's legitimate providers. Unset by
        # default so this never changes behavior for a role that hasn't shown the problem.
        provider_order=(
            get_provider_override(project_dir, role, model)
            or entry.get("provider_order")
            or default_provider_for(model)
        ),
        source=source,
        backend=str(project_entry.get("backend") or "openrouter"),
    )


def builder_backend(project_dir: Optional[str], platform: str = "python") -> str:
    """'claude_code' only when this project's own models.yaml sets roles.builder.backend to it
    (python platform only); every other project uses the OpenRouter builder."""
    if platform == "n8n" or not project_dir:
        return "openrouter"
    backend = str(((_load_project_raw(project_dir).get("roles") or {}).get("builder") or {}).get("backend") or "openrouter")
    return "claude_code" if backend == "claude_code" else "openrouter"


def builder_role_for(platform: str) -> str:
    return "builder_n8n" if platform == "n8n" else "builder"


def max_run_tokens(project_dir: Optional[str] = None) -> int:
    """Per-run token ceiling. A project's own models.yaml may raise/lower it for that project only."""
    project_raw = _load_project_raw(project_dir)
    if "max_run_tokens" in project_raw:
        return int(project_raw["max_run_tokens"])
    return int(_load_raw().get("max_run_tokens", 200000))


def max_rounds(project_dir: Optional[str] = None) -> int:
    """Builder/reviewer round cap. A project's own models.yaml may set `max_rounds` for that project
    only; every other project keeps constants.MAX_ROUNDS."""
    from .constants import MAX_ROUNDS

    project_raw = _load_project_raw(project_dir)
    if "max_rounds" not in project_raw:
        return MAX_ROUNDS
    try:
        value = int(project_raw["max_rounds"])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"max_rounds in {project_config_path(project_dir)} must be an integer") from exc
    if value < 1:
        raise ConfigError(f"max_rounds in {project_config_path(project_dir)} must be >= 1, got {value}")
    return value


def effective_models(
    project_dir: Optional[str] = None,
    platform: str = "python",
    builder_model: Optional[str] = None,
) -> dict:
    """Every role's resolved model for this project, plus where each came from. Used to show
    Douglas exactly what a project will run on before a build starts.

    builder_model must be threaded through explicitly: a --builder-model override reaches the
    actual build call fine on its own (load_role takes it directly), but this function feeds only
    the diagnostic printout - without passing it here too, the printout silently kept showing the
    config default even while the real run correctly used the override underneath it, which is
    exactly the "never a silent surprise" promise this printout exists to keep.
    """
    builder_role = builder_role_for(platform)
    roles = ["concept_advocate", "business_analyst", builder_role, "reviewer"]
    out = {}
    for role in roles:
        override = builder_model if role == builder_role else None
        try:
            cfg = load_role(role, model_override=override, project_dir=project_dir)
            out[role] = {"model": cfg.model, "source": cfg.source}
        except ConfigError as exc:
            out[role] = {"model": None, "source": f"error: {exc}"}
    return out


def validate_builder_reviewer_vendors(
    builder_model: Optional[str] = None,
    platform: str = "python",
    project_dir: Optional[str] = None,
) -> None:
    builder_role = builder_role_for(platform)
    builder = load_role(builder_role, model_override=builder_model, project_dir=project_dir)
    reviewer = load_role("reviewer", project_dir=project_dir)
    if builder.vendor == reviewer.vendor:
        raise ConfigError(
            f"reviewer vendor ({reviewer.vendor}) must differ from {builder_role} vendor "
            f"({builder.vendor}). Edit config/models.yaml, this project's models.yaml, or pick a "
            f"different --builder-model."
        )


def warn_if_business_analyst_is_anthropic(project_dir: Optional[str] = None) -> None:
    ba = load_role("business_analyst", project_dir=project_dir)
    if ba.vendor == "anthropic":
        print(
            "WARNING: business_analyst role is set to an Anthropic model. Douglas's default "
            "preference is a non-Anthropic model here for independent review (concept intake is "
            "already Claude). Edit config/models.yaml if this is unintended.",
        )


def warn_if_concept_roles_share_vendor(project_dir: Optional[str] = None) -> None:
    advocate = load_role("concept_advocate", project_dir=project_dir)
    analyst = load_role("business_analyst", project_dir=project_dir)
    if advocate.vendor == analyst.vendor:
        print(
            "WARNING: concept_advocate and business_analyst are configured to the same vendor "
            f"({advocate.vendor}). The point of bouncing ideas between them is a genuinely "
            "independent second perspective - edit config/models.yaml if this is unintended.",
        )
