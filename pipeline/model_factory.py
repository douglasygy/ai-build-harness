from __future__ import annotations

import os
from typing import Optional

from langchain_openai import ChatOpenAI

from .config_loader import load_role

# Providers confirmed (via OpenRouter's own per-provider caching docs, checked directly 2026-09-28,
# not assumed) to require this explicit opt-in field before they cache anything at all. This is
# deliberately NOT "anthropic/ only" anymore - Douglas's own direct instruction: caching must not
# silently stop working just because a role's model gets switched one day. Most providers this
# pipeline could plausibly use (DeepSeek, xAI/Grok, Moonshot, Groq, Z.AI) cache AUTOMATICALLY with
# zero request changes, as long as the prompt's static content forms a stable, byte-identical prefix
# - for those, sending this field would do nothing useful, so they're deliberately left out rather
# than sent a no-op. Anthropic and Qwen are the two confirmed exceptions that need it explicitly -
# adding qwen/ here is a real, direct fix on its own: this pipeline's reviewer role
# (qwen/qwen3-coder-next) was getting NO caching benefit at all before this, silently, because the
# old check only ever matched anthropic/. Keyed on MODEL PREFIX (checked wherever any role's model is
# resolved), not on a specific role - if a role's model changes to another provider in this set, or
# out of it, this keeps doing the right thing with no further code change. If Douglas moves a role to
# a provider NOT in this set and OpenRouter's docs later show it also needs an explicit opt-in
# (unlike the automatic providers above), add its prefix here - check OpenRouter's prompt-caching
# docs first rather than assuming either way.
_EXPLICIT_CACHE_CONTROL_PREFIXES = ("anthropic/", "qwen/")


def get_chat_model(
    role: str,
    model_override: Optional[str] = None,
    project_dir: Optional[str] = None,
) -> ChatOpenAI:
    cfg = load_role(role, model_override=model_override, project_dir=project_dir)
    api_key = os.environ.get("OPENROUTER_API_KEY", "not-needed-for-local-endpoint")
    extra_body = {}
    if cfg.reasoning_max_tokens:
        # OpenRouter's unified reasoning param. Without an explicit cap, a reasoning-capable model
        # can consume the entire max_tokens budget on invisible reasoning and return empty visible
        # content. max_tokens must stay comfortably above this value so real output has room left.
        extra_body["reasoning"] = {"max_tokens": cfg.reasoning_max_tokens}
    if cfg.provider_order:
        # Pins routing to named providers and disables automatic fallback to an unlisted one -
        # see config_loader.py for why this matters in practice.
        extra_body["provider"] = {"order": cfg.provider_order, "allow_fallbacks": False}
    if cfg.model.startswith(_EXPLICIT_CACHE_CONTROL_PREFIXES):
        # See _EXPLICIT_CACHE_CONTROL_PREFIXES above for which providers land here and why. A single
        # root-level cache_control field, not per-message breakpoints - for Anthropic specifically,
        # OpenRouter's own docs confirm the cached region "automatically advances as conversations
        # grow," which is exactly the agentic builder's own shape (_build_python_agentic resends the
        # WHOLE accumulated conversation every turn - a real transcript showed turn 14 alone costing
        # ~46,000 tokens, almost all of it content already sent on every earlier turn). Cache reads
        # are priced at a fraction of normal input cost on every provider in the explicit set; cache
        # writes carry a modest premium the first time, paid back many times over across a multi-turn
        # attempt or a run of many separate same-role calls sharing a static prefix (e.g. the n8n
        # per-node builder calling the same system prompt on every node in a workflow).
        extra_body["cache_control"] = {"type": "ephemeral"}
    return ChatOpenAI(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=api_key,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        extra_body=extra_body or None,
    )
