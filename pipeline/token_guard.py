from __future__ import annotations

from typing import Optional

from .config_loader import max_run_tokens


def extract_tokens(response) -> int:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return 0
    return usage.get("total_tokens", 0) or 0


def over_budget(total_tokens: int, project_dir: Optional[str] = None) -> bool:
    return total_tokens >= max_run_tokens(project_dir)


def extract_usage(response) -> dict:
    """Per-call token breakdown: input, cached input (billed at a fraction), output. Zeros when the
    provider reports nothing."""
    usage = getattr(response, "usage_metadata", None) or {}
    details = usage.get("input_token_details") or {}
    return {
        "input": usage.get("input_tokens", 0) or 0,
        "cached": details.get("cache_read", 0) or 0,
        "output": usage.get("output_tokens", 0) or 0,
    }
