"""Mechanical (no LLM call) check that a project's actual built code uses the interface contract(s)
the Architect pinned in .architect_decisions.json's "interface_contracts" field
(pipeline/review_architect.py). Real incident this exists to catch: pinning an intended contract
does not guarantee two independently-built pieces (an n8n shell and a Python service, say) actually
honor it - an n8n shell and its LangGraph service each independently invented a different
response_type vocabulary despite the contract having been documented as intent, and every real reply
silently vanished at runtime.

Deliberately narrow: this can only reliably detect ABSENCE (none of the pinned values appear
anywhere in the code, a strong signal something invented a different vocabulary instead), never
confirm correctness with certainty (a value could appear as a literal string yet still be wired
wrong in some other way). Feeds into pipeline/nodes/reviewer_node.py as a grounded hint, not a
separate gate - the reviewer's own already-hardened, mechanically-verified problem-reporting
mechanism decides whether and how to act on it, the same way it decides about anything else.
"""
from __future__ import annotations

from pathlib import Path

_CODE_EXTENSIONS = {".py", ".json", ".js", ".md", ".yaml", ".yml"}


def check_interface_contracts(code_dir: Path, architect_decisions: dict) -> list[str]:
    """Returns plain-English findings - never raises, never blocks a build. Empty when there are no
    contracts to check, or when every pinned discriminator value appears somewhere in the code.
    """
    contracts = (architect_decisions or {}).get("interface_contracts") or []
    if not contracts or not code_dir.exists():
        return []

    code_text_parts = []
    for path in code_dir.rglob("*"):
        if path.is_file() and path.suffix in _CODE_EXTENSIONS:
            try:
                code_text_parts.append(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
    code_text = "\n".join(code_text_parts)

    findings = []
    for contract in contracts:
        if not isinstance(contract, dict):
            continue
        between = contract.get("between") or "the pinned interface contract"
        field = contract.get("discriminator_field")
        values = [v for v in (contract.get("discriminator_values") or []) if v]
        missing = [v for v in values if str(v) not in code_text]
        if missing:
            findings.append(
                f"Interface contract between {between} pins {field!r} to include {missing!r}, but "
                f"none of these exact values appear anywhere in the built code - one or both sides "
                f"may be using a different vocabulary than what was pinned."
            )
    return findings
