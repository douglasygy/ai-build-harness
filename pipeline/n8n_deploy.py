"""Deploy step for the self-hosted path: turns a verified n8n workflow (real, webhook-typed
triggers built and proven against Douglas's dev instance - see pipeline/n8n_verify.py) into its
production form.

Mechanical only, same principle as trigger/adapter construction at build time: nothing here is
decided by an LLM. The ONLY things that change from the verified workflow are, per trigger recorded
in the project's `.production_triggers.json` sidecar (written by
pipeline/nodes/builder_node.py's save_production_triggers):
  1. that trigger node's `type`/`parameters`, swapped from a verify-mode webhook to its real
     production shape (e.g. n8n-nodes-base.telegramTrigger).
  2. its adapter's reshaping code, swapped to match (see production_adapter_js below).

Everything else - every other node, every connection, every name - is untouched, so the artifact
that ships is recognizably the same thing that was verified, not a fresh reconstruction of it.

This module never talks to a live n8n instance. It only ever writes files to
<project>/code/n8n/deploy/. Getting workflow.json onto a real target instance (a client's own
self-hosted n8n, or Douglas's own beyond the dev/verify instance) is a separate, deliberate action
outside this pipeline's automatic reach - consistent with "nothing ever deploys automatically."
"""

from __future__ import annotations

import copy
import uuid

from .n8n_verify import VERIFY_ADAPTER_JS, adapter_name_for

# Per-production-type adapter overrides. VERIFY_ADAPTER_JS ("item.json.body ?? item.json") already
# degrades correctly for a real trigger's un-wrapped output - a native trigger has no `.body`
# envelope, so the fallback already produces the right shape. This map exists only for a production
# type whose real output needs more than a body-unwrap; empty until one is actually found.
_PRODUCTION_ADAPTER_OVERRIDES: dict = {}


def production_adapter_js(production_type: str) -> str:
    return _PRODUCTION_ADAPTER_OVERRIDES.get(production_type, VERIFY_ADAPTER_JS)


def _fresh_production_webhook_path(trigger_name: str) -> str:
    """A production n8n-nodes-base.webhook trigger gets a freshly random path, deliberately NOT the
    deterministic verify_webhook_path - that route has already been exercised repeatedly against
    the dev instance and is not a fresh, unguessable production route.
    """
    return f"n8n-{uuid.uuid4().hex[:24]}"


def build_production_workflow(workflow: dict, triggers: list, project_name: str = "") -> dict:
    """Returns a deep copy of `workflow` with each declared trigger's node swapped to its recorded
    production type/parameters, and its adapter's code swapped to match. A trigger with no entry in
    `triggers` (or an empty list) leaves the workflow completely unchanged - there is nothing to
    swap without a recorded production shape to swap it to.
    """
    produced = copy.deepcopy(workflow)
    triggers_by_name = {t.get("name"): t for t in (triggers or []) if t.get("name")}
    if not triggers_by_name:
        return produced

    # Derived from adapter_name_for itself, never a hardcoded string prefix - a real incident
    # (see adapter_name_for's docstring) already came from a hardcoded "Normalize: " prefix here
    # silently going stale the moment that reserved name format changed. Looking it up through the
    # function it's defined in means this can't happen again the same way.
    adapter_to_trigger = {adapter_name_for(name): name for name in triggers_by_name}

    for node in produced.get("nodes") or []:
        name = node.get("name")
        if name in triggers_by_name:
            trigger = triggers_by_name[name]
            production_type = trigger.get("production_type") or node.get("type")
            params = dict(trigger.get("production_parameters") or {})
            if production_type == "n8n-nodes-base.webhook" and "path" not in params:
                params["path"] = _fresh_production_webhook_path(name)
            node["type"] = production_type
            node["parameters"] = params
        elif name in adapter_to_trigger:
            production_type = triggers_by_name[adapter_to_trigger[name]].get("production_type", "")
            node["parameters"] = {"jsCode": production_adapter_js(production_type)}

    return produced


# Credential types a production trigger/action node needs, so the bootstrap manifest can name them
# without an LLM guessing - deliberately small and only grown when a real node type needs it, same
# "don't fake it generically" stance as default_test_cases in n8n_verify.py.
_CREDENTIAL_HINTS = {
    "n8n-nodes-base.telegramTrigger": "telegramApi",
    "n8n-nodes-base.telegram": "telegramApi",
}


def bootstrap_manifest(workflow_name: str, triggers: list, produced_workflow: dict) -> dict:
    """A plain-language checklist for whoever installs `produced_workflow` on a real n8n instance.
    Credentials are never embedded or exported from this pipeline (same rule as everywhere else
    here) - this file is how that gap gets communicated up front instead of silently discovered the
    first time the workflow is activated and n8n refuses for a missing credential.
    """
    trigger_steps = [
        {
            "trigger": trigger.get("name"),
            "production_type": trigger.get("production_type", ""),
            "production_parameters": trigger.get("production_parameters") or {},
        }
        for trigger in (triggers or [])
        if trigger.get("name")
    ]

    credentials_required = sorted({
        _CREDENTIAL_HINTS[node.get("type")]
        for node in (produced_workflow.get("nodes") or [])
        if node.get("type") in _CREDENTIAL_HINTS
    })

    return {
        "workflow_name": workflow_name,
        "triggers": trigger_steps,
        "credentials_required": credentials_required,
        "instructions": [
            "Import workflow.json into the target n8n instance (Workflows -> Import from File).",
            "For each entry in credentials_required, create that credential type in the target "
            "instance and attach it to the node(s) that need it - credentials never travel in this "
            "file, by design.",
            "Confirm each trigger's parameters above match the real external service for this "
            "deployment target (bot token owner, channel, cron schedule, etc.).",
            "Activate the workflow only after every required credential is attached - n8n refuses "
            "to activate a workflow with a missing credential, the same check live verification "
            "already exercised against a placeholder credential during the build.",
        ],
    }
