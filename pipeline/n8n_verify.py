"""Live verification for n8n builds - the n8n equivalent of running pytest.

The builder pushes a verification-only copy of the workflow to the DEV instance, fires it through an
injected webhook, reads the real execution, and hands genuine per-node failures back. That replaces
"assume it's correct", which is how confidently-fabricated workflows got as far as the reviewer.

Two deliberate design decisions worth understanding:

1. The verification copy has its ORIGINAL TRIGGERS REMOVED and a single injected webhook wired in
   their place. Verification must be deterministic and on-demand - an active copy still carrying
   cron triggers would fire on its own schedule and perform real actions (sending Telegram messages,
   writing data tables) at 7am whether or not anyone asked it to.

2. The injected webhook is TAGGED, so it can be removed again with certainty. A webhook node in an
   active workflow is a live, publicly reachable, unauthenticated URL - fine on Douglas's own dev
   instance, not something to hand a client. Deployment MUST strip it: see strip_verification_nodes
   and the hard rule in AGENTS.md.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from .n8n_client import VERIFY_PREFIX, N8nClient, N8nError, extract_failures

# Marks every node this pipeline injected for verification. Deployment strips anything carrying it.
VERIFY_TAG = "__build_agent_verification__"
VERIFY_TRIGGER_NAME = "Build Agent Verification Trigger"
VERIFY_NORMALIZER_NAME = "Build Agent Verification Payload"

# A webhook node emits {headers, params, query, body}, but the trigger it stands in for emits the
# event itself at the root. Without reshaping, downstream nodes look for e.g. message.text, find
# nothing, and take the wrong branch - verification would then report confident false failures on a
# perfectly good workflow. Proved live: a /help command was classified as a non-text message.
_NORMALIZER_JS = "return $input.all().map(item => ({ json: item.json.body ?? item.json }));"

_TRIGGER_HINTS = ("trigger", "cron", "webhook", "schedule", "interval")

# --- Shared with pipeline/nodes/builder_node.py's mechanical trigger construction ---------------
# Lives here, not there, so builder_node.py can import from this module without a circular import
# (this module needs nothing from builder_node.py; builder_node.py needs verify_native_workflow).

VERIFY_ADAPTER_JS = "return $input.all().map(item => ({ json: item.json.body ?? item.json }));"


def adapter_name_for(trigger_name: str) -> str:
    """Reserved, deliberately distinctive-looking name - see the real incident this format fixes
    in _N8N_PLAN_SYSTEM_PROMPT's TRIGGER section (pipeline/nodes/builder_node.py). The previous
    format ("Normalize: {trigger_name}") was a completely natural name for a plan author to
    independently choose for its OWN semantic normalization node - and when it did, the mechanical
    adapter silently won the name collision (nodes_by_name.setdefault), discarding the plan's real
    field-flattening logic with no error of any kind. The generic envelope-unwrap adapter kept
    running under that name instead, and downstream nodes referencing the flattened fields it was
    supposed to produce got emptystring/undefined - which is a completely different failure
    signature (a live Telegram API rejection days later) from anything resembling "duplicate name".
    """
    return f"[Trigger Adapter] {trigger_name}"


def verify_webhook_path(project_name: str, trigger_name: str) -> str:
    """Deterministic per-trigger webhook path, so re-running the same project fires the same route
    instead of registering a new one every time.
    """
    return f"build-agent-verify-{uuid.uuid5(uuid.NAMESPACE_URL, f'build-agent/n8n-verify-webhook-path:{project_name}:{trigger_name}').hex[:16]}"


def default_test_cases(production_type: str) -> list:
    """A small default fixture set for a trigger, inferred from what it becomes in production.
    Deliberately modest - a real regression suite (which node/case matters for THIS project) is an
    authoring job, meant for the Architect role once it exists, not something to fake with generic
    guesses. This exists so verification finds more than nothing in the meantime, and so
    load_test_cases has a fallback when a project hasn't supplied its own set yet.
    """
    if "telegram" in (production_type or "").lower():
        return [
            {"name": "command", "payload": {"message": {"text": "/start", "chat": {"id": 1}, "from": {"id": 1}}}},
            {"name": "plain text", "payload": {"message": {"text": "hello", "chat": {"id": 1}, "from": {"id": 1}}}},
        ]
    return [{"name": "default", "payload": {}}]


def load_test_cases(project_dir, trigger: dict) -> list:
    """A project may supply its own fixture set at <project_dir>/n8n_test_cases.json, keyed by
    trigger name: {"Trigger Name": [{"name": "...", "payload": {...}}, ...]}. That file always wins
    when present and non-empty for this trigger; otherwise falls back to the generic default. This
    is the seam the future Architect role authors into - verification doesn't need it to exist to
    already be useful, just gets sharper once it does.
    """
    name = (trigger or {}).get("name")
    if project_dir:
        path = Path(project_dir) / "n8n_test_cases.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                cases = data.get(name) if isinstance(data, dict) else None
                if isinstance(cases, list) and cases:
                    return cases
            except (OSError, json.JSONDecodeError):
                pass
    return default_test_cases((trigger or {}).get("production_type", ""))


def is_trigger_node(node: dict) -> bool:
    node_type = str((node or {}).get("type", "")).lower()
    return any(hint in node_type for hint in _TRIGGER_HINTS)


def is_verification_node(node: dict) -> bool:
    return bool(
        (node or {}).get("notes") == VERIFY_TAG
        or (node or {}).get("name") in (VERIFY_TRIGGER_NAME, VERIFY_NORMALIZER_NAME)
    )


def strip_verification_nodes(workflow: dict):
    """Remove every pipeline-injected verification node and any connection touching it.

    This is what deployment calls. It is deterministic because the pipeline tagged the nodes itself
    rather than trying to recognise them by shape after the fact.
    """
    nodes = [n for n in (workflow.get("nodes") or []) if not is_verification_node(n)]
    removed = [n.get("name") for n in (workflow.get("nodes") or []) if is_verification_node(n)]
    kept_names = {n.get("name") for n in nodes}

    connections = {}
    for source, outputs in (workflow.get("connections") or {}).items():
        if source not in kept_names:
            continue
        rebuilt = []
        for branch in (outputs or {}).get("main") or []:
            rebuilt.append([t for t in (branch or []) if (t or {}).get("node") in kept_names])
        if any(rebuilt):
            connections[source] = {"main": rebuilt}
    return {**workflow, "nodes": nodes, "connections": connections}, removed


def build_verification_copy(workflow: dict, webhook_path: str):
    """A copy of the workflow driven by one injected webhook instead of its real triggers."""
    original_nodes = workflow.get("nodes") or []
    original_connections = workflow.get("connections") or {}

    trigger_names = {n.get("name") for n in original_nodes if is_trigger_node(n)}

    # The stand-in takes the PRIMARY trigger's own name. Downstream nodes routinely reference the
    # trigger by name - $('Receive Telegram Message') - and simply deleting it breaks every such
    # reference with "Referenced node doesn't exist". Proved live: the flow died at Prepare LLM
    # Request for exactly that reason. Impersonating the trigger keeps those references working.
    # Known limitation: with several triggers, only the primary one's name is preserved.
    # Chosen by how often other nodes REFERENCE the trigger by name, not by how many connections it
    # has. Fan-out picked the wrong one in practice: a seed-data manual trigger had more outgoing
    # edges than the Telegram trigger, so the stand-in wore the wrong name and $('Receive Telegram
    # Message') still failed. What matters is which name the expressions actually depend on.
    serialized = json.dumps(workflow)
    primary_trigger = None
    best_score = (-1, -1)
    for name in trigger_names:
        references = serialized.count(f"$('{name}')") + serialized.count(f'$("{name}")')
        fanout = sum(len(branch or []) for branch in (original_connections.get(name) or {}).get("main") or [])
        if (references, fanout) > best_score:
            primary_trigger, best_score = name, (references, fanout)
    normalizer_name = primary_trigger or VERIFY_NORMALIZER_NAME

    # Whatever the real triggers fed into becomes what the injected webhook feeds into, so the
    # workflow's actual logic still runs end to end.
    entry_targets = []
    for source, outputs in original_connections.items():
        if source not in trigger_names:
            continue
        for branch in (outputs or {}).get("main") or []:
            for target in branch or []:
                name = (target or {}).get("node")
                if name and name not in trigger_names and name not in entry_targets:
                    entry_targets.append(name)

    kept = [n for n in original_nodes if n.get("name") not in trigger_names]
    kept_names = {n.get("name") for n in kept}

    trigger_node = {
        "id": str(uuid.uuid4()),
        "name": VERIFY_TRIGGER_NAME,
        "type": "n8n-nodes-base.webhook",
        "typeVersion": 2,
        "position": [0, 0],
        "notes": VERIFY_TAG,  # the marker deployment strips on
        "parameters": {"path": webhook_path, "httpMethod": "POST", "responseMode": "onReceived"},
    }

    connections = {}
    for source, outputs in original_connections.items():
        if source in trigger_names or source not in kept_names:
            continue
        rebuilt = []
        for branch in (outputs or {}).get("main") or []:
            rebuilt.append([t for t in (branch or []) if (t or {}).get("node") in kept_names])
        if any(rebuilt):
            connections[source] = {"main": rebuilt}

    # webhook -> normalizer -> whatever the real triggers fed into, so downstream nodes receive the
    # test payload in the same shape the original trigger would have produced.
    normalizer_node = {
        "id": str(uuid.uuid4()),
        "name": normalizer_name,
        "type": "n8n-nodes-base.code",
        "typeVersion": 2,
        "position": [200, 0],
        "notes": VERIFY_TAG,
        "parameters": {"jsCode": _NORMALIZER_JS},
    }

    if entry_targets:
        connections[VERIFY_TRIGGER_NAME] = {
            "main": [[{"node": normalizer_name, "type": "main", "index": 0}]]
        }
        connections[normalizer_name] = {
            "main": [[{"node": name, "type": "main", "index": 0} for name in entry_targets if name in kept_names]]
        }

    return (
        {**workflow, "nodes": [trigger_node, normalizer_node] + kept, "connections": connections},
        entry_targets,
    )


def verification_workflow_name(project_name: str) -> str:
    return f"{VERIFY_PREFIX} {project_name}"


def _fire_one_case(client: N8nClient, workflow_id, path: str, payload: dict, settle_seconds: float) -> list:
    """Fires one webhook call, finds the execution IT produced (not just the most recent one - a
    prior case in the same run could otherwise be mistaken for this one), and returns its problems.
    """
    executions_before = {e.get("id") for e in client.list_executions(workflow_id, limit=10)}
    fire_results = []
    found = None
    for attempt in range(2):
        time.sleep(3.0 if attempt == 0 else 5.0)
        fire_results.append(_fire_webhook(client.base_url, path, payload))
        time.sleep(settle_seconds)
        for execution in client.list_executions(workflow_id, limit=10):
            if execution.get("id") not in executions_before:
                found = execution
                break
        if found:
            break
    if not found:
        return [
            f"deployed and triggered but no execution was recorded. Webhook responses: "
            f"{fire_results}. A 404 means the route was not registered; anything else means the "
            f"request arrived but produced no run."
        ]
    return extract_failures(client.get_execution(found.get("id")))


# Node type -> expected credential type, for repairing a malformed/un-wrapped credentials shape
# (see resolve_credential_references). Deliberately small and grown only when a real node type
# needs it - same "don't fake it generically" stance as default_test_cases above and
# n8n_deploy.py's own _CREDENTIAL_HINTS (kept separate to avoid a circular import between the two
# modules - n8n_deploy.py already imports from here).
_CREDENTIAL_TYPE_HINTS = {
    "n8n-nodes-base.telegram": "telegramApi",
    "n8n-nodes-base.telegramTrigger": "telegramApi",
}


def resolve_credential_references(nodes: list, client: N8nClient) -> list:
    """Ensures every node that needs a credential (per _CREDENTIAL_TYPE_HINTS) actually references
    the real one that exists on the target n8n instance - adding the reference outright if the node
    has none at all, repairing it if it's malformed, and replacing the id/name if it's simply wrong
    - instead of ever trusting what an isolated per-node build call invented or omitted.

    Real incidents this fixes, all seen in practice across separate build attempts of the same
    project: four different Telegram nodes with four different, entirely wrong credential
    references (a bare string, two distinct made-up placeholder ids); a node given an un-wrapped
    {id, name} object instead of the real type-keyed shape; and - separately, after both of those
    were fixed - four Telegram nodes built with no "credentials" field at all. Each is a different
    isolated per-node call independently failing to know something only the target instance
    actually knows (whether a credential exists, and under what id) - fixing one shape of the
    problem does not fix the others, which is why this function now handles all three explicitly
    rather than assuming a node's own build output is a reasonable starting point to merely correct.

    Mutates `nodes` in place (each node's own `credentials` dict) and returns a list of
    node-attributed problem strings for any credential type referenced but not found anywhere on
    the instance - a clear, attributable message ("create this credential first") instead of
    letting n8n's own generic activation error be the only signal of what to actually do.

    If the instance's API key doesn't permit reading credentials at all (a real, documented
    restriction - some n8n versions/API scopes reject it), this is a no-op: nodes are left exactly
    as built, and whatever error n8n's own activation attempt produces surfaces same as before -
    never worse than the pre-existing behavior.
    """
    try:
        available = client.list_credentials()
    except N8nError:
        return []

    by_type: dict = {}
    for cred in available:
        by_type.setdefault(cred.get("type"), []).append(cred)

    problems = []
    for node in nodes:
        creds = node.get("credentials")
        node_type = str(node.get("type", ""))
        expected_type = _CREDENTIAL_TYPE_HINTS.get(node_type)

        if not isinstance(creds, dict) or not creds:
            # Missing entirely - a real incident: a node of a type that definitely needs a
            # credential (n8n-nodes-base.telegram) was built with no "credentials" field at all,
            # four separate times, since a per-node build call has no way to know a real
            # credential exists until this resolution step runs. If we know what type this node
            # type needs, add it outright rather than leaving it to fail activation with a bare
            # "missing required credential" and no indication resolution ever ran.
            if not expected_type:
                continue
            creds = {}
            node["credentials"] = creds
        elif "id" in creds or "name" in creds:
            # Defensive shape fix: a node is sometimes built with an un-wrapped credential object
            # directly under "credentials" (e.g. {"id": ..., "name": ...}) instead of the real,
            # type-keyed n8n shape ({"telegramApi": {"id": ..., "name": ...}}). Left as-is, the
            # loop below would treat "id"/"name" themselves as credential TYPES to look up - a
            # real incident, reported as "no credential of type 'id' exists".
            if not expected_type:
                continue
            creds = {expected_type: creds}
            node["credentials"] = creds

        cred_types_needed = set(creds.keys()) | ({expected_type} if expected_type else set())
        for cred_type in cred_types_needed:
            matches = by_type.get(cred_type) or []
            if not matches:
                problems.append(
                    f"{node.get('name', '?')}: no credential of type '{cred_type}' exists on this "
                    f"n8n instance yet - create one before this can verify."
                )
                continue
            existing = creds.get(cred_type)
            existing_name = existing.get("name") if isinstance(existing, dict) else None
            chosen = next((m for m in matches if m.get("name") == existing_name), None) or matches[0]
            creds[cred_type] = {"id": str(chosen.get("id")), "name": chosen.get("name")}
    return problems


def verify_native_workflow(
    workflow: dict,
    project_name: str,
    triggers: list,
    client: Optional[N8nClient] = None,
    settle_seconds: float = 6.0,
    project_dir=None,
) -> tuple:
    """Verify a workflow that already has real, named webhook triggers built into it from the start
    (see pipeline/nodes/builder_node.py's mechanical trigger construction) - the successor to
    verify_live/build_verification_copy above.

    No retrofit is needed or performed here: the workflow is pushed exactly as built, under a
    verify-prefixed name for safety, and each declared trigger's own webhook is fired directly. This
    exists because build_verification_copy's retrofit-a-finished-workflow approach reproducibly
    failed ("no execution recorded", confirmed via the instance's own execution log, not just a
    missing-result artifact of our query) once applied to a real, complex, hand-built workflow -
    while a workflow that was webhook-native from construction worked cleanly every time it was
    tried. Building it right the first time turned out simpler than fixing the retrofit.

    Fires a small SET of fixture cases per trigger (load_test_cases: the project's own
    n8n_test_cases.json if present, else a generic default), not just one payload - a single happy-
    path payload only ever proves the happy path works. Each case's problems are prefixed with the
    trigger and case name, so a failure is traceable to exactly which scenario broke.

    Returns (passed, problems) - problems are node-attributed strings in the same shape reviewer
    feedback takes, so the builder's existing targeted-repair path handles them unchanged.
    """
    if not triggers:
        return True, []

    client = client or N8nClient()
    name = verification_workflow_name(project_name)

    # Fix credential references against what's real on this instance BEFORE pushing - see
    # resolve_credential_references. Mutates workflow["nodes"] in place; the caller is responsible
    # for re-persisting workflow.json afterward so the resolved ids survive into the artifact.
    credential_problems = resolve_credential_references(workflow.get("nodes") or [], client)
    if credential_problems:
        return False, credential_problems

    saved = client.upsert_verification_workflow(name, workflow.get("nodes") or [], workflow.get("connections") or {})
    workflow_id = saved.get("id")

    try:
        client.activate_workflow(workflow_id)
    except N8nError as exc:
        return False, [f"WORKFLOW: could not activate the verification copy to run it: {exc}"]

    all_problems = []
    try:
        for trigger in triggers:
            trigger_name = (trigger or {}).get("name")
            if not trigger_name:
                continue
            node = next((n for n in (workflow.get("nodes") or []) if n.get("name") == trigger_name), None)
            path = ((node or {}).get("parameters") or {}).get("path")
            if not path:
                all_problems.append(f"{trigger_name}: could not find this trigger's webhook path in the built workflow.")
                continue

            for case in load_test_cases(project_dir, trigger):
                case_name = case.get("name", "case")
                problems = _fire_one_case(client, workflow_id, path, case.get("payload") or {}, settle_seconds)
                all_problems.extend(f"{trigger_name} [{case_name}]: {p}" for p in problems)
    finally:
        try:
            client.deactivate_workflow(workflow_id)
        except N8nError:
            pass

    return (not all_problems), all_problems


def verify_live(
    workflow: dict,
    project_name: str,
    test_payload: Optional[dict] = None,
    client: Optional[N8nClient] = None,
    settle_seconds: float = 6.0,
):
    """Push, fire, and read back. Returns (passed, problems).

    Problems are node-attributed strings in the same shape the reviewer produces, so the builder's
    existing targeted-repair path can act on them without special handling.
    """
    client = client or N8nClient()
    name = verification_workflow_name(project_name)
    webhook_path = f"build-agent-verify-{uuid.uuid5(uuid.NAMESPACE_URL, name).hex[:16]}"

    copy, entry_targets = build_verification_copy(workflow, webhook_path)
    if not entry_targets:
        return False, [
            "WORKFLOW: no trigger node feeds into anything, so there is no entry point to verify. "
            "Every workflow needs at least one trigger wired to the rest of the flow."
        ]

    saved = client.upsert_verification_workflow(name, copy["nodes"], copy["connections"])
    workflow_id = saved.get("id")

    try:
        client.activate_workflow(workflow_id)
    except N8nError as exc:
        return False, [f"WORKFLOW: could not activate the verification copy to run it: {exc}"]

    try:
        # n8n registers the webhook route a moment AFTER activation returns, so firing immediately
        # can hit a path that isn't live yet and silently produce no execution at all.
        executions = []
        fire_results = []
        for attempt in range(2):
            time.sleep(3.0 if attempt == 0 else 5.0)
            fire_results.append(_fire_webhook(client.base_url, webhook_path, test_payload or {}))
            time.sleep(settle_seconds)  # executions are recorded asynchronously
            executions = client.list_executions(workflow_id, limit=1)
            if executions:
                break
        if not executions:
            return False, [
                f"WORKFLOW: deployed and triggered but no execution was recorded. Webhook responses: "
                f"{fire_results}. A 404 means the route was not registered; anything else means the "
                f"request arrived but produced no run."
            ]
        detail = client.get_execution(executions[0].get("id"))
        problems = extract_failures(detail)
        return (not problems), problems
    finally:
        # Always leave the instance quiet, even if verification blew up midway.
        try:
            client.deactivate_workflow(workflow_id)
        except N8nError:
            pass


def _fire_webhook(base_url: str, path: str, payload: dict) -> str:
    """Returns a short description of what the webhook call did.

    The status is reported rather than swallowed: a 404 here means the route isn't registered, which
    looks identical to "the workflow ran but produced nothing" if the error is discarded. Losing
    that distinction turns a one-line diagnosis into guesswork.
    """
    url = f"{base_url.rstrip('/')}/webhook/{path}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        # Non-2xx doesn't necessarily mean it didn't run - the execution log is the real source of
        # truth - but the code itself is the fastest signal when nothing runs at all.
        return f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:120]}"
    except Exception as exc:  # noqa: BLE001
        raise N8nError(f"could not reach the verification webhook at {url}: {type(exc).__name__}: {exc}") from exc
