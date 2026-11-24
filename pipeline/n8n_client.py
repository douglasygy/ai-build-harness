"""Minimal n8n REST API client - the piece that lets the builder actually check its work.

Why this exists: the Python builder has pytest telling it the truth about what it wrote. The n8n
builder had nothing, so it produced confident fabrications and the reviewer, equally blind,
confirmed them. This client is the n8n equivalent of running the tests: push the workflow to a real
instance, let it execute, and read what actually happened per node.

Verified against a live instance: GET /workflows, GET /executions and
GET /executions/{id}?includeData=true all work, and the execution detail includes
data.resultData.runData with per-node results plus lastNodeExecuted - which is exactly the
ground truth needed to hand real failures back to the builder.

SAFETY RULES BAKED IN HERE:
- Only ever touches a workflow whose name carries the VERIFY_PREFIX. Existing workflows - including
  anything Douglas built by hand and is actively using - are never updated or deleted.
- Never activates a workflow. Verification runs are manual executions only.
- Points at N8N_API_URL from .env, which is documented as a DEV instance. Client/production hosts
  are deploy targets, never verification targets.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Every workflow this pipeline creates is named with this prefix, so it is always distinguishable
# from anything a human made and can never be confused with a live one.
VERIFY_PREFIX = "[build-agent verify]"


class N8nError(Exception):
    pass


class N8nClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 60):
        self.base_url = (base_url or os.environ.get("N8N_API_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("N8N_API_KEY") or ""
        self.timeout = timeout
        if not self.base_url or not self.api_key:
            raise N8nError(
                "N8N_API_URL and N8N_API_KEY must be set (see .env.example). Without them the n8n "
                "builder has no way to check whether what it built actually works."
            )

    def _request(self, method: str, path: str, payload: Optional[dict] = None):
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "X-N8N-API-KEY": self.api_key,
                "Accept": "application/json",
                **({"Content-Type": "application/json"} if body else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                text = response.read().decode("utf-8")
                return json.loads(text) if text else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise N8nError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc
        except Exception as exc:  # noqa: BLE001 - network/DNS/timeout all surface the same way here
            raise N8nError(f"{method} {path} failed: {type(exc).__name__}: {exc}") from exc

    # --- reads -------------------------------------------------------------------------------

    def list_workflows(self, limit: int = 100) -> list:
        return (self._request("GET", f"/api/v1/workflows?limit={limit}") or {}).get("data", [])

    def find_workflow_by_name(self, name: str) -> Optional[dict]:
        return next((w for w in self.list_workflows() if w.get("name") == name), None)

    def list_credentials(self, limit: int = 250) -> list:
        """Credential id/name/type for everything on this instance - never the decrypted secret
        data, which n8n's API never returns. Used to resolve a node's credential reference against
        what actually exists, rather than trusting an id an isolated per-node build call invented
        (it has no way to know a real credential's id in advance - see n8n_verify.py's
        _resolve_credential_references). Some instances/API-key scopes reject this read entirely
        (n8n-mcp's own tool documents the same restriction) - callers should treat an N8nError here
        as "can't resolve, leave nodes as-is" rather than a fatal problem.
        """
        return (self._request("GET", f"/api/v1/credentials?limit={limit}") or {}).get("data", [])

    def list_executions(self, workflow_id: str, limit: int = 5) -> list:
        path = f"/api/v1/executions?limit={limit}&workflowId={workflow_id}"
        return (self._request("GET", path) or {}).get("data", [])

    def get_execution(self, execution_id) -> dict:
        return self._request("GET", f"/api/v1/executions/{execution_id}?includeData=true") or {}

    # --- writes (verification workflows only) --------------------------------------------------

    def _guard_name(self, name: str) -> None:
        if not name.startswith(VERIFY_PREFIX):
            raise N8nError(
                f"Refusing to write workflow {name!r}: this client only ever creates or updates "
                f"workflows named with the {VERIFY_PREFIX!r} prefix, so it can never overwrite a "
                f"workflow you built or are using."
            )

    def upsert_verification_workflow(self, name: str, nodes: list, connections: dict) -> dict:
        """Create or update a verification-only copy of a workflow. Never activates it."""
        self._guard_name(name)
        payload = {"name": name, "nodes": nodes, "connections": connections, "settings": {}}
        existing = self.find_workflow_by_name(name)
        if existing:
            return self._request("PUT", f"/api/v1/workflows/{existing['id']}", payload)
        return self._request("POST", "/api/v1/workflows", payload)

    def activate_workflow(self, workflow_id) -> dict:
        """Activate a verification workflow so its webhook URL goes live. Guarded by name, so this
        can never activate a workflow the pipeline didn't create.
        """
        self._guard_id_is_verification(workflow_id)
        return self._request("POST", f"/api/v1/workflows/{workflow_id}/activate")

    def deactivate_workflow(self, workflow_id) -> dict:
        self._guard_id_is_verification(workflow_id)
        return self._request("POST", f"/api/v1/workflows/{workflow_id}/deactivate")

    def _guard_id_is_verification(self, workflow_id) -> None:
        match = next((w for w in self.list_workflows() if str(w.get("id")) == str(workflow_id)), None)
        if not match:
            raise N8nError(f"No workflow with id {workflow_id!r} on this instance.")
        self._guard_name(match.get("name") or "")

    def delete_verification_workflow(self, name: str) -> None:
        self._guard_name(name)
        existing = self.find_workflow_by_name(name)
        if existing:
            self._request("DELETE", f"/api/v1/workflows/{existing['id']}")


def extract_failures(execution: dict) -> list:
    """Turn an execution into concrete, node-attributed problems the builder can act on.

    Attribution matters for the same reason it does in reviewer feedback: a problem naming the node
    it belongs to can be repaired by rebuilding that node, while an unattributed one forces a
    full replan.
    """
    problems = []
    result_data = ((execution.get("data") or {}).get("resultData") or {})

    top_error = result_data.get("error")
    if top_error:
        node_name = ((top_error.get("node") or {}).get("name")) if isinstance(top_error, dict) else None
        message = top_error.get("message") if isinstance(top_error, dict) else str(top_error)
        problems.append(f"{node_name or 'WORKFLOW'}: {message}")

    for node_name, runs in (result_data.get("runData") or {}).items():
        for run in runs or []:
            error = (run or {}).get("error")
            if error:
                message = error.get("message") if isinstance(error, dict) else str(error)
                problems.append(f"{node_name}: {message}")

    status = execution.get("status")
    if status and status not in ("success", "running", "waiting") and not problems:
        # Failed with nothing attributable - still report it rather than calling the run clean.
        problems.append(f"WORKFLOW: execution finished with status {status!r} and no node-level error detail.")
    return problems


def executed_node_names(execution: dict) -> list:
    result_data = ((execution.get("data") or {}).get("resultData") or {})
    return list((result_data.get("runData") or {}).keys())
