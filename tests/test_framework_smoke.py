import json
import os
import subprocess
import sys
import sysconfig

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import checkpoint  # noqa: E402
from pipeline.config_loader import (  # noqa: E402
    ConfigError,
    effective_models,
    load_role,
    max_run_tokens,
    validate_builder_reviewer_vendors,
)
from pipeline.model_factory import get_chat_model  # noqa: E402
from pipeline import graph as graph_module  # noqa: E402
from pipeline.graph import build_graph  # noqa: E402
from pipeline.n8n_client import (  # noqa: E402
    N8nClient,
    N8nError,
    VERIFY_PREFIX,
    executed_node_names,
    extract_failures,
)
from pipeline.n8n_verify import (  # noqa: E402
    VERIFY_TAG,
    VERIFY_TRIGGER_NAME,
    adapter_name_for,
    build_verification_copy,
    default_test_cases,
    is_verification_node,
    load_test_cases,
    resolve_credential_references,
    strip_verification_nodes,
    verify_native_workflow,
    verify_webhook_path,
)
from pipeline import agent1_action_log as agent1_action_log_module  # noqa: E402
from pipeline.agent1_action_log import load_actions, record_action  # noqa: E402
from pipeline import build_log as build_log_module  # noqa: E402
from pipeline import decision_log as decision_log_module  # noqa: E402
from pipeline import tech_quirks as tech_quirks_module  # noqa: E402
from pipeline import build_journal as build_journal_module  # noqa: E402
from pipeline import review_waivers as review_waivers_module  # noqa: E402
from pipeline import n8n_verify as n8n_verify_module  # noqa: E402
from pipeline.build_log import load_builds, record_build  # noqa: E402
from pipeline.decision_log import load_decisions, record_decision  # noqa: E402
from pipeline.n8n_deploy import bootstrap_manifest, build_production_workflow  # noqa: E402
from pipeline.review_decisions import (  # noqa: E402
    _render_markdown,
    compute_build_trends,
    compute_model_round_failures,
)
from pipeline.scale_roster import missing_prerequisites, save_roster  # noqa: E402
from pipeline.scratch_audit import find_stale_checkpoints  # noqa: E402
from pipeline.team_roster import load_team_roster, save_team_roster  # noqa: E402
from pipeline import build_and_review as build_and_review_module  # noqa: E402
from pipeline.build_and_review import _estimate_run_cost, _is_project_complete, _run_concluded  # noqa: E402
from pipeline.nodes.reviewer_node import verify_problems, verify_mechanical_finding_engagement  # noqa: E402
from pipeline.nodes import reviewer_node as reviewer_node_module  # noqa: E402
from pipeline.interface_contracts import check_interface_contracts  # noqa: E402
from pipeline.review_architect import (  # noqa: E402
    only_build_units_changed,
    reconsideration_engaged,
    resolve_until_unit,
    build_limits_sheet,
    validate_build_units,
)
from pipeline.retry import EmptyResponseError, with_retries_nonempty, with_retries_tool_call  # noqa: E402
from pipeline.model_fallback import (  # noqa: E402
    FALLBACK_MODEL,
    FallbackBlockedError,
    attempt_fallback,
    clear_fallback,
    get_active_fallback,
    resolve_llm_for_role,
)
from pipeline.n8n_workflow_target import (  # noqa: E402
    n8n_subdir,
    n8n_workflows,
    resolve_workflow_slug,
    slugify,
)
from pipeline.deploy_n8n import _deploy_one  # noqa: E402
from pipeline.build_stages import (  # noqa: E402
    check_stage_stub_regressions,
    check_stage_test_coverage_gaps,
    load_pm_epics,
    next_unbuilt_stage_index,
    stub_marker,
)
from pipeline.constants import MAX_AGENT_TURNS, MAX_ROUNDS  # noqa: E402
from pipeline.agent_tools import make_agent_tools  # noqa: E402
from pipeline import agent_tools as agent_tools_module  # noqa: E402
from pipeline.nodes import builder_node as builder_node_module  # noqa: E402
from pipeline.nodes.builder_node import (  # noqa: E402
    _augment_spec_with_scale_adaptive_context,
    _assemble_n8n_workflow,
    _conventions_path,
    _production_triggers_path,
    _format_conventions,
    _is_stub_js,
    _node_available,
    _autofix_js_casing,
    _expected_output_count,
    _find_unavailable_api,
    _validate_n8n_versioning,
    _validate_n8n_workflow,
    _parse_n8n_plan,
    build_trigger_and_adapter_nodes,
    insert_adapters_into_connections,
    save_production_triggers,
    _repair_node_in_place,
    _resolve_until_slice,
    _slices_from_plan,
    _validate_node_type,
    _PLAN_NODE_RE,
    _prune_dead_branches,
    wipe_n8n_artifacts,
    wipe_python_code,
)


def test_config_loads():
    role = load_role("builder")
    assert role.model


def test_vendor_validation_passes_with_default_config():
    validate_builder_reviewer_vendors()


def test_vendor_validation_rejects_same_vendor():
    reviewer_model = load_role("reviewer").model
    try:
        validate_builder_reviewer_vendors(builder_model=reviewer_model)
    except ConfigError:
        pass
    else:
        raise AssertionError("Expected ConfigError when builder and reviewer share a vendor")


def test_graph_builds():
    graph = build_graph()
    assert graph is not None


def test_get_chat_model_enables_prompt_caching_for_an_anthropic_model():
    # 2026-09-27, Douglas's own question about "compounding turn fetch" cost in the agentic
    # builder's growing tool-calling conversations - confirmed against OpenRouter's own docs
    # (automatic caching via a single root-level cache_control field) before wiring this in.
    model = get_chat_model("builder", model_override="anthropic/claude-opus-5")
    assert model.extra_body.get("cache_control") == {"type": "ephemeral"}


def test_get_chat_model_enables_prompt_caching_for_a_qwen_model():
    # 2026-09-28, Douglas's direct instruction: caching must not be a Claude-only harness feature.
    # Qwen is a REAL, confirmed exception alongside Anthropic (OpenRouter's own docs: Qwen's caching
    # also requires this explicit opt-in, unlike most other providers) - and this pipeline's actual
    # reviewer role (qwen/qwen3-coder-next) was silently getting zero caching benefit before this
    # fix, since the old check only ever matched anthropic/.
    model = get_chat_model("reviewer", model_override="qwen/qwen3-coder-next")
    assert model.extra_body.get("cache_control") == {"type": "ephemeral"}


def test_get_chat_model_skips_cache_control_for_an_automatic_caching_provider():
    # DeepSeek (builder_n8n's real model) caches AUTOMATICALLY per OpenRouter's own docs - zero
    # request changes needed, so sending it this field would be a meaningless no-op rather than a
    # real fix. Confirms the explicit-prefix set stays a deliberate allowlist, not "send it to
    # everyone and hope OpenRouter sorts it out."
    model = get_chat_model("builder_n8n", model_override="deepseek/deepseek-v4-pro")
    assert (model.extra_body or {}).get("cache_control") is None


def _write_project_config(tmp_path, data):
    (tmp_path / "models.yaml").write_text(json.dumps(data), encoding="utf-8")  # YAML is a superset of JSON


def test_project_config_overrides_model_for_that_project_only(tmp_path):
    global_model = load_role("reviewer").model
    _write_project_config(tmp_path, {"roles": {"reviewer": {"model": "someone-else/custom-reviewer"}}})

    scoped = load_role("reviewer", project_dir=str(tmp_path))
    assert scoped.model == "someone-else/custom-reviewer"
    assert scoped.source == "project"
    # The global default, and therefore every other project, is untouched.
    assert load_role("reviewer").model == global_model


def test_project_override_drops_model_specific_tuning_from_a_different_vendor(tmp_path):
    # The global builder pins provider_order/reasoning settings meant for its own vendor's model.
    # A project pointing the role at another vendor must not silently inherit them.
    _write_project_config(tmp_path, {"roles": {"builder": {"model": "someone-else/some-model"}}})
    scoped = load_role("builder", project_dir=str(tmp_path))
    assert scoped.provider_order is None
    assert scoped.reasoning_max_tokens is None


def test_project_can_override_max_run_tokens(tmp_path):
    _write_project_config(tmp_path, {"max_run_tokens": 12345})
    assert max_run_tokens(str(tmp_path)) == 12345
    assert max_run_tokens() != 12345


def test_effective_models_reports_every_role():
    report = effective_models(platform="n8n")
    assert set(report) == {"concept_advocate", "business_analyst", "builder_n8n", "reviewer"}
    assert all(info["model"] for info in report.values())


def test_effective_models_reflects_a_builder_model_override():
    # The diagnostic printout silently kept showing the config default even while the real build
    # correctly used a --builder-model override underneath it - the printout exists specifically
    # to make an override visible, so this is the exact case it must get right.
    report = effective_models(platform="python", builder_model="someone-else/override-model")
    assert report["builder"]["model"] == "someone-else/override-model"
    assert report["builder"]["source"] == "run-override"
    # The override must be scoped to the builder role only - it must never leak into others.
    assert report["reviewer"]["model"] != "someone-else/override-model"


def test_checkpoint_round_trips_and_respects_spec_changes(tmp_path):
    spec = tmp_path / "SPEC.md"
    spec.write_text("original spec", encoding="utf-8")
    state = {
        "project_dir": str(tmp_path),
        "spec_path": str(spec),
        "review_round": 2,
        "review_problems": ["NodeA: broken"],
        "builder_status": "done",
        "total_tokens": 999,
    }
    checkpoint.save_state(state)

    resumed = checkpoint.load_state(tmp_path, "original spec")
    assert resumed["review_round"] == 2
    assert resumed["total_tokens"] == 999

    # An edited spec must invalidate the checkpoint rather than resume stale work against it.
    assert checkpoint.load_state(tmp_path, "a different spec") is None

    checkpoint.clear(tmp_path)
    assert checkpoint.load_state(tmp_path, "original spec") is None


def test_resume_stage_skips_builder_only_when_build_was_complete():
    assert checkpoint.resume_stage_for({"builder_status": "done", "review_problems": []}) == "reviewer"
    # Outstanding reviewer problems mean the builder genuinely has work to do again.
    assert checkpoint.resume_stage_for({"builder_status": "done", "review_problems": ["x"]}) == "builder"
    assert checkpoint.resume_stage_for({"builder_status": "working"}) == "builder"


def test_assembler_puts_branches_on_separate_outputs():
    # An IF node's true/false paths must land on outputs 0 and 1. Putting both on output 0 makes
    # n8n run them together, silently turning a decision into an unconditional fan-out.
    workflow = _assemble_n8n_workflow(
        "t",
        {},
        [
            {"from": "Check", "to": "WhenTrue", "from_output": 0},
            {"from": "Check", "to": "WhenFalse", "from_output": 1},
        ],
    )
    outputs = workflow["connections"]["Check"]["main"]
    assert [t["node"] for t in outputs[0]] == ["WhenTrue"]
    assert [t["node"] for t in outputs[1]] == ["WhenFalse"]


def test_assembler_still_accepts_legacy_pair_connections():
    workflow = _assemble_n8n_workflow("t", {}, [["A", "B"]])
    assert workflow["connections"]["A"]["main"][0][0]["node"] == "B"


def _code_node(name, node_id="f47ac10b-58cc-4372-a567-0e02b2c3d479"):
    return {"name": name, "type": "n8n-nodes-base.code", "id": node_id,
            "parameters": {"jsCode": "const a = 1;\nreturn [{json:{a}}];"}}


def test_assembly_replaces_duplicate_node_ids_and_keeps_them_stable():
    # Nodes are generated in isolation, so the model can't pick a globally-unique id - in one real
    # build it emitted the same placeholder UUID for 11 nodes, which n8n rejects outright.
    nodes = {n: _code_node(n) for n in ["A", "B", "C"]}
    first = _assemble_n8n_workflow("Flow", nodes, [])
    ids = [n["id"] for n in first["nodes"]]
    assert len(set(ids)) == 3

    # Derived, not random: rebuilding the same workflow yields the same ids rather than churning.
    again = _assemble_n8n_workflow("Flow", {k: _code_node(k) for k in ["A", "B", "C"]}, [])
    assert [n["id"] for n in again["nodes"]] == ids


def test_expected_output_count_reads_rules_and_gives_up_when_unsure():
    assert _expected_output_count({"type": "n8n-nodes-base.if"}) == 2
    switch = {"type": "n8n-nodes-base.switch",
              "parameters": {"rules": {"values": [{"a": 1}, {"b": 2}]}, "fallbackOutput": 1}}
    assert _expected_output_count(switch) == 3  # two rules plus the fallback
    # An unfamiliar parameter shape must return None, not a guess that becomes a false problem.
    assert _expected_output_count({"type": "n8n-nodes-base.switch", "parameters": {"weird": 1}}) is None
    assert _expected_output_count({"type": "n8n-nodes-base.code"}) is None


def test_validator_flags_branches_wired_beyond_the_rules_that_exist(tmp_path):
    workflow = {
        "name": "t",
        "nodes": [
            {"name": "RouteCommand", "type": "n8n-nodes-base.switch", "id": "1",
             "parameters": {"rules": {"values": [{"a": 1}]}, "fallbackOutput": 1}},
            {"name": "X", "type": "n8n-nodes-base.noOp", "id": "2", "parameters": {}},
            {"name": "Y", "type": "n8n-nodes-base.noOp", "id": "3", "parameters": {}},
            {"name": "Z", "type": "n8n-nodes-base.noOp", "id": "4", "parameters": {}},
        ],
        "connections": {"RouteCommand": {"main": [
            [{"node": "X", "type": "main", "index": 0}],
            [{"node": "Y", "type": "main", "index": 0}],
            [{"node": "Z", "type": "main", "index": 0}],
        ]}},
    }
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(workflow), encoding="utf-8")
    ok, output = _validate_n8n_workflow(path)
    assert not ok and "can never fire" in output


def test_validator_flags_duplicate_node_ids(tmp_path):
    workflow = {
        "name": "t",
        "nodes": [_code_node("A", "same-id"), _code_node("B", "same-id")],
        "connections": {},
    }
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(workflow), encoding="utf-8")
    ok, output = _validate_n8n_workflow(path)
    assert not ok and "unique" in output


def _workflow_with_cron_and_webhook():
    return {
        "name": "wf",
        "nodes": [
            {"name": "Cron 7am", "type": "n8n-nodes-base.cron", "id": "1", "parameters": {}},
            {"name": "Telegram In", "type": "n8n-nodes-base.telegramTrigger", "id": "2", "parameters": {}},
            {"name": "Do Work", "type": "n8n-nodes-base.code", "id": "3", "parameters": {"jsCode": "return items;"}},
            {"name": "Send", "type": "n8n-nodes-base.telegram", "id": "4", "parameters": {}},
        ],
        "connections": {
            "Cron 7am": {"main": [[{"node": "Do Work", "type": "main", "index": 0}]]},
            "Telegram In": {"main": [[{"node": "Do Work", "type": "main", "index": 0}]]},
            "Do Work": {"main": [[{"node": "Send", "type": "main", "index": 0}]]},
        },
    }


def test_verification_copy_replaces_real_triggers_with_one_webhook():
    # An activated copy still carrying cron triggers would fire on its own schedule and perform real
    # actions - sending messages, writing data tables - without anyone asking it to.
    copy, entry_targets = build_verification_copy(_workflow_with_cron_and_webhook(), "verify-path")
    names = [n["name"] for n in copy["nodes"]]
    assert VERIFY_TRIGGER_NAME in names
    # The real triggers no longer FIRE - but the primary one's name survives as the stand-in, so
    # downstream $('Cron 7am') style references still resolve (see below).
    assert entry_targets == ["Do Work"]
    stand_in = copy["connections"][VERIFY_TRIGGER_NAME]["main"][0][0]["node"]
    assert copy["connections"][stand_in]["main"][0][0]["node"] == "Do Work"
    assert copy["connections"]["Do Work"]["main"][0][0]["node"] == "Send"
    # Whatever the stand-in is called, it must be a tagged code node, not a real trigger.
    stand_in_node = next(n for n in copy["nodes"] if n["name"] == stand_in)
    assert stand_in_node["type"] == "n8n-nodes-base.code"
    assert is_verification_node(stand_in_node)


def test_stand_in_impersonates_the_primary_trigger_so_references_survive():
    # Downstream nodes reference the trigger by name ($('Receive Telegram Message')). Deleting it
    # outright broke every such reference with "Referenced node doesn't exist" - proved live.
    workflow = {
        "nodes": [
            {"name": "Receive Telegram Message", "type": "n8n-nodes-base.telegramTrigger", "id": "1", "parameters": {}},
            {"name": "Rare Cron", "type": "n8n-nodes-base.cron", "id": "2", "parameters": {}},
            {"name": "A", "type": "n8n-nodes-base.code", "id": "3", "parameters": {"jsCode": "return items;"}},
            {"name": "B", "type": "n8n-nodes-base.code", "id": "4", "parameters": {"jsCode": "return items;"}},
        ],
        "connections": {
            # The Telegram trigger fans out to two nodes, the cron to one - so Telegram is primary.
            "Receive Telegram Message": {"main": [[{"node": "A", "type": "main", "index": 0},
                                                   {"node": "B", "type": "main", "index": 0}]]},
            "Rare Cron": {"main": [[{"node": "A", "type": "main", "index": 0}]]},
        },
    }
    copy, _ = build_verification_copy(workflow, "p")
    names = [n["name"] for n in copy["nodes"]]
    assert "Receive Telegram Message" in names  # the name survives, as a tagged stand-in
    assert next(n for n in copy["nodes"] if n["name"] == "Receive Telegram Message")["type"] == "n8n-nodes-base.code"


def test_stripping_removes_injected_nodes_and_their_connections():
    copy, _ = build_verification_copy(_workflow_with_cron_and_webhook(), "verify-path")
    clean, removed = strip_verification_nodes(copy)
    # Both injected nodes go: the webhook AND the trigger stand-in. Identification is by tag, so a
    # stand-in wearing the original trigger's name is still removed.
    assert VERIFY_TRIGGER_NAME in removed and len(removed) == 2
    assert all(not is_verification_node(n) for n in clean["nodes"])
    # No dangling connection may reference the removed node - that would break the import it was
    # stripped to protect.
    assert VERIFY_TRIGGER_NAME not in clean["connections"]
    for outputs in clean["connections"].values():
        for branch in outputs["main"]:
            assert all(t["node"] != VERIFY_TRIGGER_NAME for t in branch)


def test_injected_node_is_identifiable_by_tag_not_by_shape():
    copy, _ = build_verification_copy(_workflow_with_cron_and_webhook(), "verify-path")
    injected = next(n for n in copy["nodes"] if n["name"] == VERIFY_TRIGGER_NAME)
    assert injected["notes"] == VERIFY_TAG
    # A webhook the USER legitimately wants must never be mistaken for an injected one.
    assert not is_verification_node({"name": "Client Webhook", "type": "n8n-nodes-base.webhook"})


def test_execution_failures_are_extracted_and_node_attributed():
    execution = {
        "status": "error",
        "data": {"resultData": {
            "lastNodeExecuted": "Send Reminder",
            "error": {"message": "Cannot read properties of undefined", "node": {"name": "Send Reminder"}},
            "runData": {
                "Load Routes": [{"error": {"message": "Data table 'Route Library' not found"}}],
                "Format Message": [{}],
            },
        }},
    }
    problems = extract_failures(execution)
    # Attribution matters for the same reason it does in reviewer feedback: a named node can be
    # rebuilt on its own, an unattributed problem forces a full replan.
    assert any(p.startswith("Send Reminder:") for p in problems)
    assert any(p.startswith("Load Routes:") for p in problems)
    assert executed_node_names(execution) == ["Load Routes", "Format Message"]


def test_clean_execution_reports_no_failures():
    execution = {"status": "success", "data": {"resultData": {"runData": {"A": [{}]}}}}
    assert extract_failures(execution) == []


def test_failed_execution_without_node_detail_is_still_reported():
    # A failure we can't attribute must never be silently treated as a clean run.
    problems = extract_failures({"status": "crashed", "data": {"resultData": {"runData": {}}}})
    assert problems and problems[0].startswith("WORKFLOW:")


def test_client_refuses_to_write_workflows_it_did_not_create():
    client = N8nClient(base_url="https://example.invalid", api_key="x")
    for name in ["Running Bot v2", "anything else"]:
        try:
            client.upsert_verification_workflow(name, [], {})
        except N8nError as exc:
            assert "Refusing to write" in str(exc)
        else:
            raise AssertionError(f"guard must refuse {name!r} - it could overwrite a live workflow")
    # A properly-prefixed name passes the guard (and then fails on the unreachable host, as expected).
    try:
        client.upsert_verification_workflow(f"{VERIFY_PREFIX} demo", [], {})
    except N8nError as exc:
        assert "Refusing to write" not in str(exc)


def test_fabricated_n8n_apis_are_rejected():
    # Every one of these was confidently invented by a model and fails only at runtime.
    for js in [
        "const rows = $getWorkflowDataTable('Routes').getAll();\nreturn rows;",
        "const r = await $httpNode.request({url: 'https://x'});\nreturn [{json: r}];",
        "const k = $credentials.googleMaps.apiKey;\nreturn [{json: {k}}];",
        "await $telegram.sendMessage(1, 'hi');\nreturn [{json: {}}];",
        "$cancelScheduledExecution('abc');\nreturn [{json: {}}];",
        "const { DateTime } = require('luxon');\nreturn [{json: {}}];",
    ]:
        assert _find_unavailable_api(js), f"should have been rejected: {js[:40]}"


def test_real_n8n_code_apis_are_accepted():
    real = (
        "const staticData = $getWorkflowStaticData('global');\n"
        "const prev = $('Fetch Weather').item.json;\n"
        "const res = await $helpers.httpRequest({ url: 'https://api.example.com' });\n"
        "const now = DateTime.now().setZone('Asia/Singapore').toISO();\n"
        "return $input.all().map(i => ({ json: { ...i.json, now, res, prev } }));"
    )
    assert _find_unavailable_api(real) is None
    assert _is_stub_js(real) is None


def test_switch_and_if_version_syntax_pairing():
    # The worst bug found in live testing: v3-style rules on a typeVersion 1 Switch pass structural
    # validation, save and render fine, then match nothing at runtime with no error.
    filter_rules = {"rules": {"values": [{"conditions": {"options": {}, "conditions": [{"leftValue": "a"}],
                                                        "combinator": "and"}}]}}
    assert _validate_n8n_versioning({"type": "n8n-nodes-base.switch", "typeVersion": 1,
                                     "parameters": filter_rules})
    assert _validate_n8n_versioning({"type": "n8n-nodes-base.switch", "typeVersion": 3,
                                     "parameters": filter_rules}) is None
    assert _validate_n8n_versioning({"type": "n8n-nodes-base.if", "typeVersion": 1,
                                     "parameters": {"conditions": {"options": {}, "conditions": [{"leftValue": "a"}],
                                                                   "combinator": "and"}}})
    # A numeric fallbackOutput creates no real fallback port.
    assert _validate_n8n_versioning({"type": "n8n-nodes-base.switch", "typeVersion": 3,
                                     "parameters": {"fallbackOutput": 1}})
    assert _validate_n8n_versioning({"type": "n8n-nodes-base.switch", "typeVersion": 3,
                                     "parameters": {"fallbackOutput": "extra"}}) is None
    # Non-branching nodes are never touched by this check.
    assert _validate_n8n_versioning({"type": "n8n-nodes-base.code", "typeVersion": 2,
                                     "parameters": {"jsCode": "x"}}) is None


def test_node_type_typo_is_rejected_but_unknown_types_pass():
    assert _validate_node_type("n8n-nodes-base.codee")  # near-miss of a real type
    assert _validate_node_type("n8n-nodes-base.code") is None
    # Not in the known list at all - allowed, since that list is for typo detection, not allowlisting.
    assert _validate_node_type("n8n-nodes-custom.brandNewThing") is None


def test_autofix_corrects_the_casing_mistake_that_blocked_a_real_build():
    # deepseek-v4-pro produced '.toisostring' on six consecutive attempts for one node across two
    # runs, ignoring the retry feedback each time. The correction is fully determined, so it is
    # applied mechanically rather than spending another LLM round-trip on it.
    fixed, fixes = _autofix_js_casing("const ts = new Date().toisostring();")
    assert "toISOString" in fixed
    assert fixes == ["toisostring -> toISOString"]
    assert _is_stub_js(fixed + "\nreturn [{json:{ts}}];") is None


def test_autofix_preserves_member_access_and_n8n_property_paths():
    # A naive replace of '.parseint(' with 'parseInt(' would drop the dot and corrupt the code,
    # and rewriting item.json.* would break the single most common pattern in n8n Code nodes.
    fixed, _ = _autofix_js_casing("const n = value.parseint(x);")
    assert "value.parseInt(x)" in fixed

    untouched = "const v = item.json.stringify; const w = items[0].json.parse;"
    assert _autofix_js_casing(untouched)[0] == untouched


def test_autofix_leaves_correct_code_alone():
    good = "const j = JSON.stringify(item.json.data);\nconst m = Math.floor(2.5);"
    assert _autofix_js_casing(good) == (good, [])


def test_node_repair_only_touches_code_nodes():
    node = {"name": "N", "type": "n8n-nodes-base.code",
            "parameters": {"jsCode": "const t = d.toisostring();"}}
    assert _repair_node_in_place(node) == ["toisostring -> toISOString"]
    assert "toISOString" in node["parameters"]["jsCode"]

    non_code = {"name": "C", "type": "n8n-nodes-base.cron", "parameters": {"jsCode": "x.tostring()"}}
    assert _repair_node_in_place(non_code) == []


def test_js_casing_mistakes_are_caught():
    problem = _is_stub_js("const id = Math.random().tostring(36);\nreturn [{json:{id}}];")
    assert problem and "toString" in problem


def test_valid_n8n_code_node_js_is_accepted():
    # n8n runs Code-node bodies inside a function, so a top-level return is correct there.
    assert _is_stub_js("const items = $input.all();\nconst out = items.map(i => i.json);\nreturn out;") is None


def test_broken_js_is_rejected_when_node_is_available():
    if not _node_available():
        return  # syntax checking degrades to heuristics without a Node binary
    problem = _is_stub_js("const x = 1;\n});\\return items;")
    assert problem and "not valid JavaScript" in problem


DELIMITED_PLAN = '''<<<WORKFLOW: Test Flow>>>

<<<CONVENTIONS>>>
timezone: Asia/Singapore
- Score is an integer 0-100, never 1-10.
- Payloads look like {"chat_id": 123} using the "Bot" credential.
<<<END_CONVENTIONS>>>

<<<NODE: Cron: 7am Check-in>>>
type: n8n-nodes-base.cron
slice: checkin
purpose: Fire at 07:00. Emits {"trigger": "morning"}.
Second line of prose with a backslash path C:\\Users\\dougl.
<<<END_NODE>>>

<<<NODE: Weather Is Good?>>>
type: n8n-nodes-base.if
slice: checkin
purpose: Branch on weather.
<<<END_NODE>>>

<<<NODE: Seed Library>>>
type: n8n-nodes-base.code
slice: setup
purpose: Load seed data.
<<<END_NODE>>>

<<<CONNECTIONS>>>
Cron: 7am Check-in -> Weather Is Good? [0]
Weather Is Good? -> Seed Library [0]
Weather Is Good? -> Cron: 7am Check-in [1]
<<<END_CONNECTIONS>>>
'''


TRIGGER_PLAN = '''<<<WORKFLOW: Trigger Test>>>

<<<TRIGGER: Receive Telegram Message>>>
production_type: n8n-nodes-base.telegramTrigger
production_parameters: {"updates": ["message"]}
<<<END_TRIGGER>>>

<<<TRIGGER: Daily Cron>>>
production_type: n8n-nodes-base.cron
production_parameters: {}
<<<END_TRIGGER>>>

<<<NODE: Handle Message>>>
type: n8n-nodes-base.code
slice: main
purpose: handle it
<<<END_NODE>>>

<<<CONNECTIONS>>>
Receive Telegram Message -> Handle Message [0]
Daily Cron -> Handle Message [0]
<<<END_CONNECTIONS>>>
'''


def test_plan_parses_multiple_triggers_with_production_metadata():
    plan = _parse_n8n_plan(TRIGGER_PLAN)
    assert len(plan["triggers"]) == 2
    telegram = next(t for t in plan["triggers"] if t["name"] == "Receive Telegram Message")
    assert telegram["production_type"] == "n8n-nodes-base.telegramTrigger"
    assert telegram["production_parameters"] == {"updates": ["message"]}
    cron = next(t for t in plan["triggers"] if t["name"] == "Daily Cron")
    assert cron["production_type"] == "n8n-nodes-base.cron"
    # Triggers must never leak into the regular node list - they're built mechanically, not by the LLM.
    assert [n["name"] for n in plan["nodes"]] == ["Handle Message"]


def test_trigger_and_adapter_nodes_are_built_mechanically_and_webhook_typed():
    triggers = [{"name": "Receive Telegram Message", "production_type": "n8n-nodes-base.telegramTrigger"}]
    nodes = build_trigger_and_adapter_nodes(triggers, "TestFlow")
    assert set(nodes) == {"Receive Telegram Message", adapter_name_for("Receive Telegram Message")}
    # The trigger keeps its real declared name and is a webhook in verify mode - never anything the
    # LLM invented, so there is nothing to impersonate and nothing whose name can drift.
    assert nodes["Receive Telegram Message"]["type"] == "n8n-nodes-base.webhook"
    assert nodes["Receive Telegram Message"]["parameters"]["path"] == verify_webhook_path("TestFlow", "Receive Telegram Message")
    assert nodes[adapter_name_for("Receive Telegram Message")]["type"] == "n8n-nodes-base.code"


def test_webhook_path_is_stable_across_rebuilds():
    a = verify_webhook_path("proj", "Trigger")
    b = verify_webhook_path("proj", "Trigger")
    assert a == b  # re-running the same project must fire the same route, not register a new one


def test_default_test_cases_shapes_telegram_and_falls_back_generic():
    telegram_cases = default_test_cases("n8n-nodes-base.telegramTrigger")
    assert len(telegram_cases) >= 2  # a command case and a plain-text case, at minimum
    names = [c["name"] for c in telegram_cases]
    assert len(names) == len(set(names))  # each case must be independently traceable in a failure
    for case in telegram_cases:
        assert "message" in case["payload"] and "text" in case["payload"]["message"]
    generic = default_test_cases("n8n-nodes-base.cron")
    assert generic == [{"name": "default", "payload": {}}]


def test_load_test_cases_prefers_project_override(tmp_path):
    trigger = {"name": "Receive Telegram Message", "production_type": "n8n-nodes-base.telegramTrigger"}
    (tmp_path / "n8n_test_cases.json").write_text(
        json.dumps({"Receive Telegram Message": [{"name": "custom", "payload": {"foo": "bar"}}]}),
        encoding="utf-8",
    )
    assert load_test_cases(str(tmp_path), trigger) == [{"name": "custom", "payload": {"foo": "bar"}}]


def test_load_test_cases_falls_back_when_no_project_override_exists(tmp_path):
    trigger = {"name": "Receive Telegram Message", "production_type": "n8n-nodes-base.telegramTrigger"}
    assert load_test_cases(str(tmp_path), trigger) == default_test_cases("n8n-nodes-base.telegramTrigger")
    assert load_test_cases(None, trigger) == default_test_cases("n8n-nodes-base.telegramTrigger")


def test_insert_adapters_reroutes_trigger_output_through_the_adapter():
    triggers = [{"name": "Receive Telegram Message"}]
    connections = [{"from": "Receive Telegram Message", "to": "Handle Message", "from_output": 0}]
    adjusted = insert_adapters_into_connections(connections, triggers)
    adapter = adapter_name_for("Receive Telegram Message")
    # Trigger -> adapter is added, and the trigger's original edge now leaves from the adapter -
    # the plan's author never has to know the adapter exists.
    assert any(c["from"] == "Receive Telegram Message" and c["to"] == adapter for c in adjusted)
    assert any(c["from"] == adapter and c["to"] == "Handle Message" for c in adjusted)
    assert not any(c["from"] == "Receive Telegram Message" and c["to"] == "Handle Message" for c in adjusted)


def test_insert_adapters_is_a_no_op_without_triggers():
    connections = [{"from": "A", "to": "B", "from_output": 0}]
    assert insert_adapters_into_connections(connections, []) == connections


def test_production_triggers_sidecar_round_trips(tmp_path):
    triggers = [{"name": "T", "production_type": "n8n-nodes-base.cron", "production_parameters": {}}]
    save_production_triggers(tmp_path, triggers)
    saved = json.loads((tmp_path / "n8n" / ".production_triggers.json").read_text(encoding="utf-8"))
    assert saved == triggers


def test_concurrent_node_building_merges_tokens_and_builds_everything(monkeypatch):
    def fake_build(llm, config, spec, all_names, total_tokens, conventions):
        total_tokens[0] += 100  # simulates real token spend inside the (mocked) LLM call
        return {"name": spec["name"], "type": "n8n-nodes-base.code"}

    monkeypatch.setattr(builder_node_module, "_build_n8n_node", fake_build)

    specs = [{"name": f"N{i}"} for i in range(6)]  # 6 specs, concurrency 4 -> exercises 2 batches
    total_tokens = [0]
    built = {}
    error = builder_node_module._build_nodes_concurrently(
        None, None, specs, [s["name"] for s in specs], {}, total_tokens, None,
        lambda name, node: built.__setitem__(name, node),
    )
    assert error is None
    assert set(built) == {s["name"] for s in specs}
    # Each concurrent call gets its own token cell merged back on the main thread - if two threads
    # instead raced on the same shared counter, tokens would go missing under concurrency.
    assert total_tokens[0] == 600


def test_concurrent_node_building_stops_and_names_the_failure(monkeypatch):
    def fake_build(llm, config, spec, all_names, total_tokens, conventions):
        if spec["name"] == "Bad":
            raise ValueError("boom")
        total_tokens[0] += 10
        return {"name": spec["name"]}

    monkeypatch.setattr(builder_node_module, "_build_n8n_node", fake_build)
    specs = [{"name": "Good1"}, {"name": "Bad"}, {"name": "Good2"}]
    error = builder_node_module._build_nodes_concurrently(
        None, None, specs, [], {}, [0], None, lambda name, node: None,
    )
    assert error and "Bad" in error


def test_verify_native_workflow_fires_every_case_and_prefixes_problems(monkeypatch):
    monkeypatch.setattr(n8n_verify_module.time, "sleep", lambda s: None)
    fired = []
    monkeypatch.setattr(n8n_verify_module, "_fire_webhook", lambda base_url, path, payload: fired.append(payload) or "200 OK")

    # command -> before/after, plain text -> before/after: 4 list_executions calls, in that order.
    executions_script = iter([
        [],
        [{"id": "run-1"}],
        [{"id": "run-1"}],
        [{"id": "run-1"}, {"id": "run-2"}],
    ])
    execution_details = {
        "run-1": {},  # clean run - no problems
        "run-2": {"data": {"resultData": {"error": {"node": {"name": "Handle"}, "message": "boom"}}}},
    }

    class FakeClient:
        base_url = "https://fake.invalid"

        def upsert_verification_workflow(self, name, nodes, connections):
            return {"id": "wf1"}

        def activate_workflow(self, workflow_id):
            pass

        def deactivate_workflow(self, workflow_id):
            pass

        def list_executions(self, workflow_id, limit=10):
            return next(executions_script)

        def get_execution(self, execution_id):
            return execution_details[execution_id]

        def list_credentials(self, limit=250):
            return []

    workflow = {
        "nodes": [{"name": "Receive Telegram Message", "parameters": {"path": "abc"}}],
        "connections": {},
    }
    triggers = [{"name": "Receive Telegram Message", "production_type": "n8n-nodes-base.telegramTrigger"}]

    passed, problems = verify_native_workflow(workflow, "Proj", triggers, client=FakeClient(), settle_seconds=0)

    assert passed is False
    # A single happy-path payload only ever proves the happy path - both fixture cases must actually fire.
    assert len(fired) == 2
    # Only the failing case's problem survives, attributed to both the trigger and which case broke.
    assert problems == ["Receive Telegram Message [plain text]: Handle: boom"]


def test_verify_native_workflow_passes_when_all_cases_clean(monkeypatch):
    monkeypatch.setattr(n8n_verify_module.time, "sleep", lambda s: None)
    monkeypatch.setattr(n8n_verify_module, "_fire_webhook", lambda *a, **k: "200 OK")

    executions_script = iter([[], [{"id": "run-1"}]])  # one generic case -> one before/after pair
    deactivated = []

    class FakeClient:
        base_url = "https://fake.invalid"

        def upsert_verification_workflow(self, name, nodes, connections):
            return {"id": "wf1"}

        def activate_workflow(self, workflow_id):
            pass

        def deactivate_workflow(self, workflow_id):
            deactivated.append(workflow_id)

        def list_executions(self, workflow_id, limit=10):
            return next(executions_script)

        def get_execution(self, execution_id):
            return {}

        def list_credentials(self, limit=250):
            return []

    workflow = {"nodes": [{"name": "T", "parameters": {"path": "abc"}}], "connections": {}}
    triggers = [{"name": "T", "production_type": "n8n-nodes-base.cron"}]

    passed, problems = verify_native_workflow(workflow, "Proj", triggers, client=FakeClient(), settle_seconds=0)

    assert passed is True
    assert problems == []
    assert deactivated == ["wf1"]  # the verification copy must always be deactivated afterward


def test_decision_log_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_log_module, "DECISION_LOG_PATH", tmp_path / "decision_log.jsonl")
    record_decision(
        "test decision point", "chose X", project="demo",
        proposal_summary="summary", options_presented=["X", "Y"], rationale="because",
    )
    record_decision("second point", "chose Y", project="demo")

    decisions = load_decisions()
    assert len(decisions) == 2
    assert decisions[0]["decision_point"] == "test decision point"
    assert decisions[0]["choice"] == "chose X"
    assert decisions[0]["options_presented"] == ["X", "Y"]
    assert decisions[1]["rationale"] == ""  # defaults are always present, never omitted


def test_decision_log_load_is_empty_when_no_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_log_module, "DECISION_LOG_PATH", tmp_path / "nope.jsonl")
    assert load_decisions() == []


def test_load_tech_quirks_is_empty_when_no_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(tech_quirks_module, "TECH_QUIRKS_PATH", tmp_path / "tech-quirks.md")
    assert tech_quirks_module.load_tech_quirks() == ""


def test_record_tech_quirk_entry_creates_the_file_with_a_header_and_the_entry(tmp_path, monkeypatch):
    path = tmp_path / "tech-quirks.md"
    monkeypatch.setattr(tech_quirks_module, "TECH_QUIRKS_PATH", path)
    tech_quirks_module.record_tech_quirk_entry("LangGraph Runtime: has no .config, use context instead.", source="reviewer")

    content = path.read_text(encoding="utf-8")
    assert "# Tech Quirks" in content
    assert "LangGraph Runtime: has no .config, use context instead." in content
    assert "(reviewer, " in content  # attribution is always present, never a bare unattributed claim


def test_record_tech_quirk_entry_appends_on_a_second_call(tmp_path, monkeypatch):
    path = tmp_path / "tech-quirks.md"
    monkeypatch.setattr(tech_quirks_module, "TECH_QUIRKS_PATH", path)
    tech_quirks_module.record_tech_quirk_entry("First lesson.", source="architect")
    tech_quirks_module.record_tech_quirk_entry("Second lesson.", source="builder (n8n)")

    content = path.read_text(encoding="utf-8")
    assert "First lesson." in content
    assert "Second lesson." in content
    assert content.count("# Tech Quirks") == 1  # the header is written once, never duplicated


def test_record_tech_quirk_entry_is_a_no_op_for_an_empty_lesson(tmp_path, monkeypatch):
    path = tmp_path / "tech-quirks.md"
    monkeypatch.setattr(tech_quirks_module, "TECH_QUIRKS_PATH", path)
    tech_quirks_module.record_tech_quirk_entry("   ", source="reviewer")
    assert not path.exists()


def test_with_tech_quirks_returns_the_prompt_unchanged_when_nothing_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(tech_quirks_module, "TECH_QUIRKS_PATH", tmp_path / "tech-quirks.md")
    assert tech_quirks_module.with_tech_quirks("You are a reviewer.") == "You are a reviewer."


def test_with_tech_quirks_prepends_recorded_lessons_and_frames_them_as_unverified(tmp_path, monkeypatch):
    path = tmp_path / "tech-quirks.md"
    monkeypatch.setattr(tech_quirks_module, "TECH_QUIRKS_PATH", path)
    tech_quirks_module.record_tech_quirk_entry("n8n Slack node: emails must be URL-encoded.", source="builder (n8n)")

    result = tech_quirks_module.with_tech_quirks("You are a reviewer.")
    assert result.endswith("You are a reviewer.")  # the role's own prompt text is preserved verbatim at the end
    assert "n8n Slack node: emails must be URL-encoded." in result
    assert "UNVERIFIED CLAIM" in result  # never presented as automatic ground truth


def test_make_tech_quirk_tool_records_a_lesson_attributed_to_its_bound_source(tmp_path, monkeypatch):
    path = tmp_path / "tech-quirks.md"
    monkeypatch.setattr(tech_quirks_module, "TECH_QUIRKS_PATH", path)
    tool = tech_quirks_module.make_tech_quirk_tool(source="builder (python, agentic)")

    result = tool.invoke({"lesson": "LangGraph Runtime: has no .config, use context instead."})
    assert "Recorded" in result
    content = path.read_text(encoding="utf-8")
    assert "LangGraph Runtime: has no .config, use context instead." in content
    assert "(builder (python, agentic), " in content


def test_make_tech_quirk_tool_rejects_an_empty_lesson(tmp_path, monkeypatch):
    path = tmp_path / "tech-quirks.md"
    monkeypatch.setattr(tech_quirks_module, "TECH_QUIRKS_PATH", path)
    tool = tech_quirks_module.make_tech_quirk_tool(source="reviewer")

    result = tool.invoke({"lesson": "   "})
    assert "ERROR" in result
    assert not path.exists()


def test_build_journal_record_and_load_all(tmp_path):
    build_journal_module.record_journal_entry(tmp_path, "storage", "builder", "Built InMemoryStorage with save()/load().")
    build_journal_module.record_journal_entry(tmp_path, "validation", "builder", "Built validators using storage's InMemoryStorage.")
    entries = build_journal_module.load_all_entries(tmp_path)
    assert len(entries) == 2
    assert entries[0]["unit"] == "storage"
    assert entries[1]["unit"] == "validation"


def test_build_journal_record_is_a_no_op_for_an_empty_summary(tmp_path):
    build_journal_module.record_journal_entry(tmp_path, "storage", "builder", "   ")
    assert build_journal_module.load_all_entries(tmp_path) == []
    assert not (tmp_path / ".build_journal.jsonl").exists()


def test_build_journal_record_truncates_an_overly_long_summary(tmp_path):
    build_journal_module.record_journal_entry(tmp_path, "storage", "builder", "x" * 5000)
    entries = build_journal_module.load_all_entries(tmp_path)
    assert len(entries) == 1
    assert len(entries[0]["summary"]) < 1000
    assert entries[0]["summary"].endswith("(truncated)")


def test_build_journal_load_relevant_entries_is_scoped_not_the_whole_log(tmp_path):
    # The bounded-by-design guarantee: a unit only sees ITS OWN history plus whatever it explicitly
    # depends on - never the whole build's accumulated log, which would grow forever as the build
    # progresses (the same compounding-resend problem context-pruning already fixed, one level up).
    build_journal_module.record_journal_entry(tmp_path, "storage", "builder", "Built storage.")
    build_journal_module.record_journal_entry(tmp_path, "validation", "builder", "Built validation.")
    build_journal_module.record_journal_entry(tmp_path, "vendors", "builder", "Built vendors - unrelated to storage or validation.")

    relevant = build_journal_module.load_relevant_entries(tmp_path, "validation", depends_on=["storage"])
    units_seen = {e["unit"] for e in relevant}
    assert units_seen == {"storage", "validation"}
    assert "vendors" not in units_seen


def test_build_journal_load_relevant_entries_always_includes_review_round_entries(tmp_path):
    build_journal_module.record_journal_entry(tmp_path, "storage", "builder", "Built storage.")
    build_journal_module.record_journal_entry(
        tmp_path, build_journal_module.REVIEW_ROUND_UNIT, "reviewer", "Round 1: verdict fail, 2 problems kept.",
    )
    relevant = build_journal_module.load_relevant_entries(tmp_path, "vendors", depends_on=[])
    units_seen = {e["unit"] for e in relevant}
    assert build_journal_module.REVIEW_ROUND_UNIT in units_seen
    assert "storage" not in units_seen  # not a dependency of "vendors" and not a review round


def test_build_journal_format_for_prompt_is_empty_string_when_no_entries():
    assert build_journal_module.format_journal_for_prompt([]) == ""


def test_add_waiver_generates_a_slug_id_and_records_a_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_log_module, "DECISION_LOG_PATH", tmp_path / "decision_log.jsonl")
    entry = review_waivers_module.add_waiver(
        tmp_path, text="Symlink escape in disk_storage.py bypasses the namespace check",
        reason="Requires a link created inside Douglas's own output folder - Phase 1 never does that.",
    )
    assert entry["id"].startswith("symlink-escape-in-disk-storage")
    assert len(entry["id"]) <= 40
    assert entry["waived_by"] == "Douglas"
    loaded = review_waivers_module.load_waivers(tmp_path)
    assert loaded == [entry]
    decisions = decision_log_module.load_decisions()
    assert any("Symlink escape" in d.get("decision_point", "") for d in decisions)


def test_add_waiver_refuses_a_duplicate_explicit_id(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_log_module, "DECISION_LOG_PATH", tmp_path / "decision_log.jsonl")
    review_waivers_module.add_waiver(tmp_path, text="First finding", reason="r1", id="dup-id")
    with pytest.raises(ValueError):
        review_waivers_module.add_waiver(tmp_path, text="Second finding", reason="r2", id="dup-id")


def test_load_waivers_is_empty_when_no_file_exists(tmp_path):
    assert review_waivers_module.load_waivers(tmp_path) == []


def test_format_waivers_for_prompt_is_empty_string_when_no_waivers():
    assert review_waivers_module.format_waivers_for_prompt([]) == ""


def test_format_waivers_for_prompt_includes_id_text_and_reason():
    text = review_waivers_module.format_waivers_for_prompt([
        {"id": "symlink-escape", "text": "Symlink escape bypasses namespace check", "reason": "Douglas declined for now"},
    ])
    assert "symlink-escape" in text
    assert "Symlink escape bypasses namespace check" in text
    assert "Douglas declined for now" in text
    assert "do not report" in text.lower()


def test_build_journal_format_for_prompt_labels_entries_as_unverified():
    entries = [{"unit": "storage", "role": "builder", "summary": "Built InMemoryStorage."}]
    text = build_journal_module.format_journal_for_prompt(entries)
    assert "Built InMemoryStorage." in text
    assert "UNVERIFIED" in text or "unverified" in text.lower()


def test_clear_journal_removes_an_existing_journal_and_reports_true(tmp_path):
    build_journal_module.record_journal_entry(tmp_path, "storage", "builder", "Built storage.")
    assert build_journal_module.clear_journal(tmp_path) is True
    assert not (tmp_path / ".build_journal.jsonl").exists()
    assert build_journal_module.load_all_entries(tmp_path) == []


def test_clear_journal_is_false_when_nothing_to_remove(tmp_path):
    assert build_journal_module.clear_journal(tmp_path) is False


def test_team_roster_round_trips(tmp_path):
    approved = [{
        "role_name": "devops",
        "reason": "needs infra automation this project specifically requires",
        "recommended_model": "x/y",
        "approved_on": "2026-09-07",
    }]
    save_team_roster(tmp_path, approved)
    assert load_team_roster(tmp_path) == approved


def test_team_roster_is_empty_without_a_saved_file(tmp_path):
    assert load_team_roster(tmp_path) == []


def test_render_decision_patterns_markdown_includes_patterns_lessons_and_open_questions():
    analysis = {
        "patterns": [{
            "theme": "Hosting",
            "pattern": "Prefers self-hosted over cloud unless a client specifically needs otherwise.",
            "evidence": ["decided self-hosted n8n over Cloud for the dev instance"],
            "confidence": "high",
        }],
        "build_lessons": [{
            "lesson": "Telegram builds repeatedly fail on a missing credential, not a code bug.",
            "evidence": ["run on 2026-09-05 stopped after 3 rounds on the same credential error"],
        }],
        "open_questions": ["Not enough data yet on multi-tenant credential isolation preference"],
        "note_if_logs_too_thin": None,
    }
    markdown = _render_markdown(analysis, decision_count=3, build_count=2, action_count=1)
    assert "Hosting" in markdown
    assert "Prefers self-hosted" in markdown
    assert "decided self-hosted n8n" in markdown
    assert "Telegram builds repeatedly fail" in markdown
    assert "stopped after 3 rounds" in markdown
    assert "Not enough data yet" in markdown
    assert "3 logged decision" in markdown
    assert "2 logged build run" in markdown
    assert "1 logged Agent 1 action" in markdown


def test_render_decision_patterns_markdown_includes_overreach_flags():
    analysis = {
        "overreach_flags": [{
            "action": "Selected MiniMax M3 for the preference_analyst role and used it live",
            "category": "model_selection",
            "concern": "No matching decision_log entry approving this model choice before use.",
        }],
    }
    markdown = _render_markdown(analysis, decision_count=0, build_count=0, action_count=1)
    assert "Overreach flags" in markdown
    assert "model_selection" in markdown
    assert "Selected MiniMax M3" in markdown
    assert "No matching decision_log entry" in markdown


def test_render_decision_patterns_markdown_omits_overreach_section_when_none_flagged():
    markdown = _render_markdown({"patterns": []}, decision_count=1, build_count=0, action_count=0)
    assert "Overreach flags" not in markdown


def test_agent1_action_log_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(agent1_action_log_module, "ACTION_LOG_PATH", tmp_path / "agent1_action_log.jsonl")
    record_action("model_selection", "Picked Gemini 3.1 Pro for preference_analyst", project="pipeline-design")

    actions = load_actions()
    assert len(actions) == 1
    assert actions[0]["category"] == "model_selection"
    assert actions[0]["description"] == "Picked Gemini 3.1 Pro for preference_analyst"
    assert actions[0]["project"] == "pipeline-design"


def test_agent1_action_log_rejects_unknown_category(tmp_path, monkeypatch):
    monkeypatch.setattr(agent1_action_log_module, "ACTION_LOG_PATH", tmp_path / "agent1_action_log.jsonl")
    with pytest.raises(ValueError):
        record_action("scope_creep", "something not in the reserved list")


def test_agent1_action_log_is_empty_when_no_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(agent1_action_log_module, "ACTION_LOG_PATH", tmp_path / "nope.jsonl")
    assert load_actions() == []


def test_record_decision_round_trips_proposed_by_and_supersedes(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_log_module, "DECISION_LOG_PATH", tmp_path / "decision_log.jsonl")
    record_decision(
        "Which model for builder_n8n?", "Keep DeepSeek V4 Pro", project="pipeline-design",
        proposed_by="business_analyst (deepseek/deepseek-r1)", supersedes="An earlier routing decision",
    )
    decisions = load_decisions()
    assert len(decisions) == 1
    assert decisions[0]["proposed_by"] == "business_analyst (deepseek/deepseek-r1)"
    assert decisions[0]["supersedes"] == "An earlier routing decision"


def test_record_decision_omits_proposed_by_and_supersedes_when_not_given(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_log_module, "DECISION_LOG_PATH", tmp_path / "decision_log.jsonl")
    record_decision("A plain decision", "Choice", project="pipeline-design")
    decisions = load_decisions()
    assert "proposed_by" not in decisions[0]
    assert "supersedes" not in decisions[0]


def test_record_build_round_trips_model_rounds_and_total_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(build_log_module, "BUILD_LOG_PATH", tmp_path / "build_log.jsonl")
    record_build("demo", "pass", {"what_happened": "x"}, model="qwen/qwen3-coder-next", rounds=2, total_tokens=5000)
    builds = load_builds()
    assert builds[0]["model"] == "qwen/qwen3-coder-next"
    assert builds[0]["rounds"] == 2
    assert builds[0]["total_tokens"] == 5000


def test_augment_spec_adds_precedence_rule_only_when_something_was_added(tmp_path):
    assert "PRECEDENCE" not in _augment_spec_with_scale_adaptive_context("SPEC text", str(tmp_path))

    (tmp_path / ".architect_decisions.json").write_text('{"system_shape": "two workflows"}', encoding="utf-8")
    augmented = _augment_spec_with_scale_adaptive_context("SPEC text", str(tmp_path))
    assert "PRECEDENCE" in augmented
    assert "ARCHITECT DECISIONS take precedence" in augmented


def test_compute_build_trends_is_empty_without_recorded_rounds():
    assert compute_build_trends([{"scope": "run", "status": "pass"}]) == {}


def test_compute_build_trends_computes_pass_rate_and_avg_rounds():
    builds = [
        {"scope": "run", "status": "pass", "rounds": 1, "timestamp": "2026-09-01T00:00:00Z"},
        {"scope": "run", "status": "pass", "rounds": 3, "timestamp": "2026-09-02T00:00:00Z"},
        {"scope": "run", "status": "failed_after_max_rounds", "rounds": 3, "timestamp": "2026-09-03T00:00:00Z"},
        {"scope": "round", "status": "fail", "role": "reviewer"},  # round-scope entries are ignored
    ]
    trends = compute_build_trends(builds)
    assert trends["total_runs"] == 3
    assert trends["pass_rate"] == round(2 / 3, 2)


def test_compute_model_round_failures_counts_by_role_and_model():
    builds = [
        {"scope": "round", "role": "reviewer", "model": "qwen/qwen3-coder-next"},
        {"scope": "round", "role": "reviewer", "model": "qwen/qwen3-coder-next"},
        {"scope": "round", "role": "builder_n8n", "model": "deepseek/deepseek-v4-pro"},
        {"scope": "run", "role": "reviewer", "model": "qwen/qwen3-coder-next"},  # run-scope ignored
    ]
    counts = compute_model_round_failures(builds)
    assert counts == {
        "reviewer (qwen/qwen3-coder-next)": 2,
        "builder_n8n (deepseek/deepseek-v4-pro)": 1,
    }


def test_render_decision_patterns_markdown_includes_model_scorecard():
    analysis = {
        "model_scorecard": [{
            "role": "reviewer", "model": "qwen/qwen3-coder-next", "verdict": "concerning",
            "evidence": ["Failed 4 of the last 5 rounds on genuine misses, not false positives"],
            "suggested_replacement": "Reconsider the alternative from the original role setup",
            "replacement_basis": "decision_log entry naming an alternative considered when reviewer was chosen",
        }],
    }
    markdown = _render_markdown(analysis, decision_count=0, build_count=0)
    assert "Model scorecard" in markdown
    assert "qwen/qwen3-coder-next" in markdown
    assert "concerning" in markdown
    assert "Failed 4 of the last 5 rounds" in markdown
    assert "Reconsider the alternative" in markdown


def test_render_decision_patterns_markdown_includes_build_trends_section():
    trends = {"total_runs": 5, "pass_rate": 0.8, "avg_rounds_recent": 1.5, "avg_rounds_earlier": 2.2}
    markdown = _render_markdown({}, decision_count=0, build_count=5, trends=trends)
    assert "Build trends" in markdown
    assert "5 completed run" in markdown
    assert "80%" in markdown
    assert "1.5" in markdown and "2.2" in markdown


def test_render_decision_patterns_markdown_carries_forward_note():
    analysis = {"carried_forward_note": "Kept the Hosting pattern, still supported by recent decisions."}
    markdown = _render_markdown(analysis, decision_count=1, build_count=0)
    assert "Carried forward from the previous synthesis" in markdown
    assert "Kept the Hosting pattern" in markdown


def test_estimate_run_cost_prints_nothing_without_prior_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(build_and_review_module, "load_builds", lambda: [])
    _estimate_run_cost(tmp_path, "n8n")
    assert capsys.readouterr().err == ""


def test_estimate_run_cost_prints_average_from_prior_runs(tmp_path, monkeypatch, capsys):
    project = tmp_path / "demo"
    monkeypatch.setattr(build_and_review_module, "load_builds", lambda: [
        {"project": "demo", "scope": "run", "total_tokens": 4000},
        {"project": "demo", "scope": "run", "total_tokens": 6000},
        {"project": "other-project", "scope": "run", "total_tokens": 999999},
    ])
    _estimate_run_cost(project, "n8n")
    err = capsys.readouterr().err
    assert "ESTIMATE" in err
    assert "2 prior run" in err
    assert "5,000" in err  # average of 4000 and 6000, other-project excluded


def test_find_stale_checkpoints_is_empty_when_none_exist(tmp_path):
    assert find_stale_checkpoints(tmp_path / "nope") == []


def test_verify_problems_keeps_a_grounded_spec_requirement_claim():
    spec = "The workflow must include a node named 'Authorization IF' that gates access."
    code = "--- workflow.json ---\n{\"name\": \"Authorization IF\", \"type\": \"if\"}"
    problems = [{
        "node": "Authorization IF", "claim_type": "spec_requirement",
        "description": "missing the required gating logic",
        "spec_quote": "a node named 'Authorization IF' that gates access",
    }]
    kept, rejected, severities = verify_problems(problems, spec, code)
    assert kept == ["Authorization IF: missing the required gating logic"]
    assert rejected == []
    # No "severity" field given - must mechanically default to the conservative "must_fix".
    assert severities == ["must_fix"]


def test_verify_problems_keeps_a_correctness_bug_with_no_quote_needed():
    spec = "Send a message when a route is submitted."
    code = "--- workflow.json ---\n{\"name\": \"Send Score Message\", \"type\": \"code\"}"
    problems = [{
        "node": "Send Score Message", "claim_type": "correctness_bug",
        "description": "reads the score from the wrong upstream node", "spec_quote": "",
    }]
    kept, rejected, severities = verify_problems(problems, spec, code)
    assert kept == ["Send Score Message: reads the score from the wrong upstream node"]
    assert rejected == []


def test_verify_problems_resolves_dotted_class_method_names_but_not_invented_or_file_names():
    code = "--- openrouter_client.py ---\nclass OpenRouterClient:\n    def call(self, model):\n        pass\n"
    def _p(node):
        return [{"node": node, "claim_type": "correctness_bug", "description": "bug"}]
    for node in ("OpenRouterClient.call", "OpenRouterClient.call()", "OpenRouterClient.call(self, model)"):
        kept, rejected, _ = verify_problems(_p(node), "", code)
        assert len(kept) == 1 and rejected == [], node
    for node in ("OpenRouterClient.missing_method", "Ghost.call", "ghost.py"):
        kept, rejected, _ = verify_problems(_p(node), "", code)
        assert kept == [] and len(rejected) == 1, node


def test_read_code_uses_forward_slash_headers_not_the_platform_separator(tmp_path):
    # 2026-09-28, Douglas's direct instruction, verified via a real reproduction: on Windows,
    # str(path.relative_to(code_dir)) uses backslashes, so every file header the reviewer was shown
    # read e.g. "creative_studio\storage_memory.py". The reviewer LLM naturally writes forward-slash
    # paths in its "node" field regardless of what it was shown, so a plain substring check never
    # matched a real citation - every genuine finding naming a real file was silently dropped as
    # "invented", 100% of the time, on Windows.
    (tmp_path / "creative_studio").mkdir()
    (tmp_path / "creative_studio" / "storage_memory.py").write_text("class InMemoryStorage:\n    pass\n", encoding="utf-8")
    code_text = reviewer_node_module._read_code(tmp_path)
    assert "--- creative_studio/storage_memory.py ---" in code_text
    assert "\\" not in code_text.split("---")[1]  # the header itself, not incidentally inside file content


def test_verify_problems_keeps_a_finding_naming_a_file_in_a_subdirectory():
    # End-to-end version of the fix above: a real nested-path citation must survive verify_problems
    # exactly as _read_code (forward-slash headers) and a normal reviewer (forward-slash citations)
    # actually produce it - this is the shape that was silently broken.
    spec = "Storage must use a single, consistently-named backend class."
    code = "--- creative_studio/storage_memory.py ---\nclass InMemoryStorage:\n    pass\n"
    problems = [{
        "node": "creative_studio/storage_memory.py", "claim_type": "correctness_bug",
        "description": "defines InMemoryStorage but conftest.py expects InMemoryStorageBackend", "spec_quote": "",
    }]
    kept, rejected, _severities = verify_problems(problems, spec, code)
    assert kept == ["creative_studio/storage_memory.py: defines InMemoryStorage but conftest.py expects InMemoryStorageBackend"]
    assert rejected == []


def test_verify_problems_tolerates_a_backslash_node_against_forward_slash_code():
    # Defense in depth, independent of the _read_code fix above: even if a model writes a
    # backslash-style path in "node" (copying a traceback, or a Windows-flavored path from
    # somewhere else in what it was given), the membership check must still recognize it.
    code = "--- creative_studio/storage_memory.py ---\nclass InMemoryStorage:\n    pass\n"
    problems = [{
        "node": "creative_studio\\storage_memory.py", "claim_type": "correctness_bug",
        "description": "wrong class name", "spec_quote": "",
    }]
    kept, rejected, _severities = verify_problems(problems, "", code)
    assert rejected == []
    assert len(kept) == 1


def test_verify_problems_rejects_fabricated_spec_requirement_the_real_incident():
    # Recreates the actual reported incident, verbatim in shape: the reviewer (qwen/qwen3-coder-next,
    # seo-blog-bot project) claimed "spec requires node named 'Authorization IF'" when SPEC.md never
    # named any literal node at all - confirmed by direct grep, zero matches, twice, with different
    # invented names each time. This mechanical check exists specifically because of this incident.
    spec = "The bot checks whether the requester is authorized before proceeding."
    code = "--- workflow.json ---\n{\"name\": \"Check Authorization\", \"type\": \"if\"}"
    problems = [{
        "node": "Check Authorization", "claim_type": "spec_requirement",
        "description": "wrong name - spec requires \"Authorization IF\" but code has \"Check Authorization\"",
        "spec_quote": "Authorization IF",
    }]
    kept, rejected, _severities = verify_problems(problems, spec, code)
    assert kept == []
    assert len(rejected) == 1
    assert "fabricated" in rejected[0]


def test_verify_problems_rejects_invented_node_name_not_in_code():
    spec = "Send a text message when required information is missing."
    code = "--- workflow.json ---\n{\"name\": \"Prompt For Info\", \"type\": \"code\"}"
    problems = [{
        "node": "Send Text Message", "claim_type": "spec_requirement",
        "description": "missing: spec requires this node for missing-info prompts",
        "spec_quote": "Send a text message when required information is missing",
    }]
    kept, rejected, _severities = verify_problems(problems, spec, code)
    assert kept == []
    assert len(rejected) == 1
    assert "invented name" in rejected[0]


def test_verify_problems_keeps_a_genuine_missing_file_claim_with_a_real_anchor():
    # The real creative-team incident (2026-09-29): "tests/test_creative_director.py does not exist,
    # despite creative_director.py being a core module" - a correctness_bug claim naming the missing
    # path would be rejected as "invented name" (the path can never appear in code that lacks it).
    # missing_file with a real anchor must survive instead.
    spec = "Every core module must have a corresponding test file."
    code = "--- creative_director.py ---\nclass CreativeDirector:\n    pass\n"
    problems = [{
        "node": "tests/test_creative_director.py", "claim_type": "missing_file",
        "description": "no test file exists for creative_director.py, a core module",
        "anchor": "creative_director.py",
    }]
    kept, rejected, _severities = verify_problems(problems, spec, code)
    assert kept == ["tests/test_creative_director.py: no test file exists for creative_director.py, a core module"]
    assert rejected == []


def test_verify_problems_rejects_missing_file_claim_with_no_real_anchor():
    spec = "spec"
    code = "--- creative_director.py ---\nclass CreativeDirector:\n    pass\n"
    problems = [{
        "node": "tests/test_creative_director.py", "claim_type": "missing_file",
        "description": "invented pairing", "anchor": "nonexistent_module.py",
    }]
    kept, rejected, _severities = verify_problems(problems, spec, code)
    assert kept == []
    assert "no real, existing anchor" in rejected[0]


def test_verify_problems_rejects_missing_file_claim_when_the_file_actually_exists():
    spec = "spec"
    code = "--- tests/test_creative_director.py ---\ndef test_it(): pass\n\n--- creative_director.py ---\nclass CreativeDirector:\n    pass\n"
    problems = [{
        "node": "tests/test_creative_director.py", "claim_type": "missing_file",
        "description": "wrongly claimed missing", "anchor": "creative_director.py",
    }]
    kept, rejected, _severities = verify_problems(problems, spec, code)
    assert kept == []
    assert "not really missing" in rejected[0]


def test_verify_problems_rejects_malformed_non_dict_entries():
    kept, rejected, _severities = verify_problems(["a plain string, not the new structured format"], "spec", "code")
    assert kept == []
    assert "not a structured problem object" in rejected[0]


def test_verify_problems_tolerant_of_whitespace_differences_in_quote():
    spec = "The workflow   must\ninclude a retry step."
    code = "--- workflow.json ---\n{\"name\": \"Retry Step\"}"
    problems = [{
        "node": "Retry Step", "claim_type": "spec_requirement",
        "description": "missing", "spec_quote": "must include a retry step",
    }]
    kept, rejected, _severities = verify_problems(problems, spec, code)
    assert kept == ["Retry Step: missing"]
    assert rejected == []


def test_verify_problems_allows_workflow_level_attribution_without_existing_in_code():
    spec = "Every branch must be wired."
    problems = [{
        "node": "WORKFLOW", "claim_type": "correctness_bug",
        "description": "an unrelated branch has no target", "spec_quote": "",
    }]
    kept, rejected, _severities = verify_problems(problems, spec, "--- workflow.json ---\n{}")
    assert kept == ["WORKFLOW: an unrelated branch has no target"]
    assert rejected == []


def _make_stub_finding(file="teams/text_team.py", epic="Text Team", confidence="high"):
    return {"file": file, "epic": epic, "confidence": confidence, "message": f"{file}: stub for {epic!r}"}


def test_verify_mechanical_finding_engagement_the_real_incident_silence_is_unengaged(tmp_path):
    # Recreates the actual reported incident (2026-09-26, three consecutive live runs): the
    # reviewer's response contained neither a correctness_bug for the file nor any dismissal at all.
    (tmp_path / "text_team.py").write_text("def not_implemented(): pass\n", encoding="utf-8")
    finding = _make_stub_finding(file="text_team.py")
    unengaged = verify_mechanical_finding_engagement([finding], kept_problems=[], dismissals={}, code_dir=tmp_path)
    assert unengaged == [finding]


def test_verify_mechanical_finding_engagement_engaged_via_escalation(tmp_path):
    (tmp_path / "text_team.py").write_text("def not_implemented(): pass\n", encoding="utf-8")
    finding = _make_stub_finding(file="text_team.py")
    kept_problems = ["text_team.py: this is genuinely still a stub, confirmed"]
    unengaged = verify_mechanical_finding_engagement([finding], kept_problems, dismissals={}, code_dir=tmp_path)
    assert unengaged == []


def test_verify_mechanical_finding_engagement_engaged_via_real_verbatim_quote(tmp_path):
    (tmp_path / "text_team.py").write_text(
        "def build_subgraph():\n    return real_gatekeeper_wiring()\n", encoding="utf-8",
    )
    finding = _make_stub_finding(file="text_team.py")
    dismissals = {"text_team.py": "return real_gatekeeper_wiring()"}
    unengaged = verify_mechanical_finding_engagement([finding], kept_problems=[], dismissals=dismissals, code_dir=tmp_path)
    assert unengaged == []


def test_verify_mechanical_finding_engagement_rejects_a_fabricated_quote_not_actually_in_the_file(tmp_path):
    # Same fabrication class verify_problems() already rejects for a spec_quote, applied here to a
    # file's own real content instead of SPEC.md.
    (tmp_path / "text_team.py").write_text("def not_implemented(): pass\n", encoding="utf-8")
    finding = _make_stub_finding(file="text_team.py")
    dismissals = {"text_team.py": "this quote does not appear anywhere in the real file"}
    unengaged = verify_mechanical_finding_engagement([finding], kept_problems=[], dismissals=dismissals, code_dir=tmp_path)
    assert unengaged == [finding]


def test_verify_mechanical_finding_engagement_rejects_an_empty_dismissal_quote(tmp_path):
    (tmp_path / "text_team.py").write_text("def not_implemented(): pass\n", encoding="utf-8")
    finding = _make_stub_finding(file="text_team.py")
    unengaged = verify_mechanical_finding_engagement([finding], [], dismissals={"text_team.py": "   "}, code_dir=tmp_path)
    assert unengaged == [finding]


def test_verify_mechanical_finding_engagement_is_whitespace_tolerant(tmp_path):
    (tmp_path / "text_team.py").write_text(
        "def build_subgraph():\n    return   real_gatekeeper_wiring(  )\n", encoding="utf-8",
    )
    finding = _make_stub_finding(file="text_team.py")
    # The dismissal quote has different (but semantically identical) whitespace from the real file.
    dismissals = {"text_team.py": "return real_gatekeeper_wiring( )"}
    unengaged = verify_mechanical_finding_engagement([finding], kept_problems=[], dismissals=dismissals, code_dir=tmp_path)
    assert unengaged == []


def test_verify_mechanical_finding_engagement_handles_missing_file_safely(tmp_path):
    finding = _make_stub_finding(file="does_not_exist.py")
    dismissals = {"does_not_exist.py": "anything"}
    unengaged = verify_mechanical_finding_engagement([finding], [], dismissals=dismissals, code_dir=tmp_path)
    assert unengaged == [finding]


def test_verify_mechanical_finding_engagement_partitions_mixed_engaged_and_unengaged(tmp_path):
    (tmp_path / "a.py").write_text("real_implementation_a()\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def not_implemented(): pass\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("real_implementation_c()\n", encoding="utf-8")
    finding_a = _make_stub_finding(file="a.py", epic="Epic A")
    finding_b = _make_stub_finding(file="b.py", epic="Epic B")
    finding_c = _make_stub_finding(file="c.py", epic="Epic C")

    unengaged = verify_mechanical_finding_engagement(
        [finding_a, finding_b, finding_c],
        kept_problems=["c.py: still a stub"],  # C escalated
        dismissals={"a.py": "real_implementation_a()"},  # A properly dismissed
        code_dir=tmp_path,
    )
    assert unengaged == [finding_b]  # only B was neither escalated nor dismissed


def test_reviewer_node_forces_a_fail_when_it_silently_ignores_a_stub_finding(tmp_path, monkeypatch):
    # End-to-end recreation of the actual reported incident through the REAL reviewer_node(), not
    # just the isolated helper: the reviewer's response ignores a fed stub-check finding completely
    # (no correctness_bug, no mechanical_check_dismissals field at all) - exactly what happened live,
    # three consecutive times, 1M+ tokens, before this mechanical force-injection existed.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild the Text Team subgraph.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "text_team.py").write_text(
        f'"""{stub_marker("Text Team")}"""\n', encoding="utf-8",
    )

    class _FakeLLM:
        def invoke(self, messages, config=None):
            # Ignores the fed evidence entirely - a bare pass, matching the real reported response.
            return _FakeResponse('{"verdict": "pass", "problems": []}')

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _FakeLLM())

    state = {
        "project_dir": str(tmp_path),
        "spec_path": str(spec_path),
        "code_dir": str(code_dir),
        "platform": "python",
        "python_units_built": ["Text Team"],
        "review_round": 0,
        "total_tokens": 0,
    }
    result = reviewer_node_module.reviewer_node(state)

    assert result["review_verdict"] == "fail"
    assert any("text_team.py" in p for p in result["review_problems"])
    assert any("did not engage" in p for p in result["review_problems"])


def test_reviewer_node_accepts_a_genuine_verified_dismissal(tmp_path, monkeypatch):
    # The mirror-image case: the reviewer DOES engage, with a real quote from the real file - the
    # forcing mechanism must not fire when the evidence was genuinely, verifiably addressed.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild the Text Team subgraph.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "text_team.py").write_text(
        f'"""{stub_marker("Text Team")}"""\ndef build_subgraph():\n    return real_wiring()\n',
        encoding="utf-8",
    )

    class _FakeLLM:
        def invoke(self, messages, config=None):
            return _FakeResponse(json.dumps({
                "verdict": "pass", "problems": [],
                "mechanical_check_dismissals": {"text_team.py": "return real_wiring()"},
            }))

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _FakeLLM())

    state = {
        "project_dir": str(tmp_path),
        "spec_path": str(spec_path),
        "code_dir": str(code_dir),
        "platform": "python",
        "python_units_built": ["Text Team"],
        "review_round": 0,
        "total_tokens": 0,
    }
    result = reviewer_node_module.reviewer_node(state)

    assert result["review_verdict"] == "pass"
    assert result["review_problems"] == []


def test_resolve_credential_references_fills_in_a_missing_credentials_field():
    # Real incident: four Telegram nodes built with no "credentials" field at all.
    nodes = [{"name": "Send Message", "type": "n8n-nodes-base.telegram", "parameters": {}}]

    class FakeClient:
        def list_credentials(self):
            return [{"id": "42", "name": "My Telegram Bot", "type": "telegramApi"}]

    problems = resolve_credential_references(nodes, FakeClient())
    assert problems == []
    assert nodes[0]["credentials"] == {"telegramApi": {"id": "42", "name": "My Telegram Bot"}}


def test_resolve_credential_references_repairs_unwrapped_shape():
    # Real incident: a node given {"id": ..., "name": ...} directly instead of the type-keyed shape.
    nodes = [{
        "name": "Send Message", "type": "n8n-nodes-base.telegram",
        "credentials": {"id": "invented-id", "name": "wrong"},
    }]

    class FakeClient:
        def list_credentials(self):
            return [{"id": "42", "name": "My Telegram Bot", "type": "telegramApi"}]

    problems = resolve_credential_references(nodes, FakeClient())
    assert problems == []
    assert nodes[0]["credentials"] == {"telegramApi": {"id": "42", "name": "My Telegram Bot"}}


def test_resolve_credential_references_replaces_a_wrong_id():
    nodes = [{
        "name": "Send Message", "type": "n8n-nodes-base.telegram",
        "credentials": {"telegramApi": {"id": "1", "name": "made-up"}},
    }]

    class FakeClient:
        def list_credentials(self):
            return [{"id": "42", "name": "My Telegram Bot", "type": "telegramApi"}]

    problems = resolve_credential_references(nodes, FakeClient())
    assert problems == []
    assert nodes[0]["credentials"]["telegramApi"]["id"] == "42"


def test_resolve_credential_references_reports_a_genuinely_missing_credential():
    nodes = [{"name": "Send Message", "type": "n8n-nodes-base.telegram", "parameters": {}}]

    class FakeClient:
        def list_credentials(self):
            return []  # nothing of type telegramApi exists on this instance yet

    problems = resolve_credential_references(nodes, FakeClient())
    assert len(problems) == 1
    assert "Send Message" in problems[0] and "telegramApi" in problems[0]


def test_resolve_credential_references_is_a_noop_when_the_api_key_cant_list_credentials():
    nodes = [{"name": "Send Message", "type": "n8n-nodes-base.telegram", "parameters": {}}]

    class FakeClient:
        def list_credentials(self):
            raise N8nError("this API key scope cannot list credentials")

    problems = resolve_credential_references(nodes, FakeClient())
    assert problems == []
    assert "credentials" not in nodes[0]  # left exactly as built, never worse than before


def test_wipe_n8n_artifacts_removes_existing_files_and_reports_nothing_missing(tmp_path):
    n8n_dir = tmp_path / "n8n"
    n8n_dir.mkdir()
    (n8n_dir / "workflow.json").write_text("{}", encoding="utf-8")

    removed = wipe_n8n_artifacts(tmp_path)
    assert any("workflow.json" in r for r in removed)
    assert not (n8n_dir / "workflow.json").exists()


def test_wipe_n8n_artifacts_is_empty_when_nothing_to_remove(tmp_path):
    assert wipe_n8n_artifacts(tmp_path) == []


def test_wipe_python_code_removes_files_and_subdirectories(tmp_path):
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")
    sub = tmp_path / "graph"
    sub.mkdir()
    (sub / "nodes.py").write_text("x = 1", encoding="utf-8")

    removed = wipe_python_code(tmp_path)
    assert any("main.py" in r for r in removed)
    assert any("graph" in r for r in removed)
    assert list(tmp_path.iterdir()) == []


def test_wipe_python_code_is_empty_when_code_dir_does_not_exist(tmp_path):
    missing = tmp_path / "code"
    assert wipe_python_code(missing) == []


def test_prune_dead_branches_removes_only_the_dead_output_indices():
    problems_text = "Switch node 'Route Response' defines 3 output(s) but has connections on output(s) [3]"
    connections_plan = [
        {"from": "Route Response", "to": "A", "from_output": 0},
        {"from": "Route Response", "to": "B", "from_output": 3},
        {"from": "Other Node", "to": "C", "from_output": 3},  # different node, same index - must survive
    ]
    pruned, changed = _prune_dead_branches(problems_text, connections_plan)
    assert changed is True
    assert {"from": "Route Response", "to": "B", "from_output": 3} not in pruned
    assert {"from": "Route Response", "to": "A", "from_output": 0} in pruned
    assert {"from": "Other Node", "to": "C", "from_output": 3} in pruned


def test_prune_dead_branches_reports_no_change_when_nothing_matches():
    pruned, changed = _prune_dead_branches("no problems here", [{"from": "A", "to": "B", "from_output": 0}])
    assert changed is False
    assert pruned == [{"from": "A", "to": "B", "from_output": 0}]


def test_plan_node_regex_tolerates_a_miscounted_opening_delimiter():
    text = "<<<NODE: My Node>>\ntype: n8n-nodes-base.code\n<<<END_NODE>>>"
    match = _PLAN_NODE_RE.search(text)
    assert match is not None
    assert match.group("name") == "My Node"


def test_find_stale_checkpoints_flags_old_ones_not_recent_ones(tmp_path):
    import os
    import time as time_module

    projects_dir = tmp_path / "projects"
    old_project = projects_dir / "abandoned"
    fresh_project = projects_dir / "active"
    old_project.mkdir(parents=True)
    fresh_project.mkdir(parents=True)

    old_checkpoint = old_project / ".build_checkpoint.json"
    fresh_checkpoint = fresh_project / ".build_checkpoint.json"
    old_checkpoint.write_text("{}", encoding="utf-8")
    fresh_checkpoint.write_text("{}", encoding="utf-8")

    forty_days_ago = time_module.time() - 40 * 86400
    os.utime(old_checkpoint, (forty_days_ago, forty_days_ago))

    stale = find_stale_checkpoints(projects_dir, max_age_days=30)
    assert len(stale) == 1
    assert stale[0]["project"] == "abandoned"
    assert stale[0]["age_days"] >= 30


def test_build_log_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(build_log_module, "BUILD_LOG_PATH", tmp_path / "build_log.jsonl")
    record_build("demo", "pass", {"what_happened": "built 3 nodes", "outcome": "pass", "blockers": []})
    record_build("demo", "ambiguous", {"what_happened": "stopped", "outcome": "ambiguous", "blockers": ["credential"]})

    builds = load_builds()
    assert len(builds) == 2
    assert builds[0]["status"] == "pass"
    assert builds[0]["summary"]["what_happened"] == "built 3 nodes"
    assert builds[1]["summary"]["blockers"] == ["credential"]


def test_build_log_load_is_empty_when_no_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(build_log_module, "BUILD_LOG_PATH", tmp_path / "nope.jsonl")
    assert load_builds() == []


def test_build_log_round_scoped_entry_carries_round_and_role(tmp_path, monkeypatch):
    monkeypatch.setattr(build_log_module, "BUILD_LOG_PATH", tmp_path / "build_log.jsonl")
    record_build("demo", "fail", {"what_happened": "round 2 failed"}, scope="round", round_num=2, role="reviewer")

    builds = load_builds()
    assert len(builds) == 1
    assert builds[0]["scope"] == "round"
    assert builds[0]["round"] == 2
    assert builds[0]["role"] == "reviewer"


def test_checkpointed_reviewer_logs_only_on_failing_rounds(monkeypatch):
    logged = []
    monkeypatch.setattr(graph_module, "summarize_build_run", lambda round_result, project_dir=None: ({"what_happened": "x"}, 0))
    monkeypatch.setattr(graph_module, "record_build", lambda *a, **k: logged.append((a, k)))
    monkeypatch.setattr(graph_module, "load_role", lambda role, project_dir=None: type("Cfg", (), {"model": "fake/reviewer-model"})())

    def fake_pass(state):
        return {**state, "review_round": 1, "review_verdict": "pass", "review_problems": []}

    def fake_fail(state):
        return {**state, "review_round": 2, "review_verdict": "fail", "review_problems": ["X: broke"]}

    state = {"project_dir": None, "spec_path": None}
    graph_module._checkpointed_reviewer(fake_pass)(state)
    assert logged == []  # a clean round costs nothing extra beyond the final per-run summary

    graph_module._checkpointed_reviewer(fake_fail)(state)
    assert len(logged) == 1
    (project_name, status, _summary), kwargs = logged[0]
    assert (project_name, status) == ("unknown", "fail")
    assert kwargs == {"scope": "round", "round_num": 2, "role": "reviewer", "model": "fake/reviewer-model"}


def test_checkpointed_reviewer_swallows_logging_errors(monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(graph_module, "summarize_build_run", boom)

    def fake_fail(state):
        return {**state, "review_round": 1, "review_verdict": "fail", "review_problems": ["X: broke"]}

    result = graph_module._checkpointed_reviewer(fake_fail)({"project_dir": None, "spec_path": None})
    assert result["review_verdict"] == "fail"  # the real loop is unaffected by the logging failure
    assert "WARNING" in capsys.readouterr().err


def test_log_repair_round_logs_with_round_and_role(tmp_path, monkeypatch):
    logged = []
    monkeypatch.setattr(builder_node_module, "summarize_build_run", lambda round_result, project_dir=None: ({"what_happened": "x"}, 0))
    monkeypatch.setattr(builder_node_module, "record_build", lambda *a, **k: logged.append((a, k)))

    fake_llm = type("Llm", (), {"model_name": "fake/builder-model"})()
    builder_node_module._log_repair_round(str(tmp_path), tmp_path.name, 2, "Node X: broke", llm=fake_llm)

    assert len(logged) == 1
    (project_name, status, _summary), kwargs = logged[0]
    assert (project_name, status) == (tmp_path.name, "fail")
    assert kwargs == {"scope": "round", "round_num": 2, "role": "builder_n8n", "model": "fake/builder-model"}


def test_log_repair_round_swallows_errors(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(builder_node_module, "summarize_build_run", boom)
    builder_node_module._log_repair_round(str(tmp_path), tmp_path.name, 1, "problem")  # must not raise
    assert "WARNING" in capsys.readouterr().err


def test_missing_prerequisites_is_empty_with_no_roster_file(tmp_path):
    assert missing_prerequisites(tmp_path) == []


def test_missing_prerequisites_is_empty_when_neither_role_marked(tmp_path):
    save_roster(tmp_path, {"pm": False, "architect": False})
    assert missing_prerequisites(tmp_path) == []


def test_missing_prerequisites_flags_pm_without_approved_epics(tmp_path):
    save_roster(tmp_path, {"pm": True})
    problems = missing_prerequisites(tmp_path)
    assert len(problems) == 1
    assert "PM" in problems[0] and "review_pm" in problems[0]


def test_missing_prerequisites_flags_architect_without_approved_decisions(tmp_path):
    save_roster(tmp_path, {"architect": True})
    problems = missing_prerequisites(tmp_path)
    assert len(problems) == 1
    assert "Architect" in problems[0] and "review_architect" in problems[0]


def test_missing_prerequisites_is_satisfied_once_the_decision_files_exist(tmp_path):
    save_roster(tmp_path, {"pm": True, "architect": True})
    (tmp_path / ".pm_epics.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".architect_decisions.json").write_text("{}", encoding="utf-8")
    assert missing_prerequisites(tmp_path) == []


def test_run_pytest_sets_cwd_to_code_dir(tmp_path):
    # Real, evidenced incident (2026-09-28): _run_pytest never set cwd, so a relative path the
    # PROJECT's own code resolves (e.g. a config's storage_root defaulting to "./data") resolved
    # against wherever build_and_review.py itself was launched from, not the project being built -
    # real smoke-test output landed in build-agent's own repo (build-agent/data/acme-property/...).
    # This test would fail under the old code: without cwd set, the relative path below resolves
    # wherever this test SUITE happens to run from, not code_dir.
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "test_cwd.py").write_text(
        "from pathlib import Path\n\n"
        "def test_relative_path_resolves_inside_code_dir():\n"
        "    Path('marker.txt').write_text('here', encoding='utf-8')\n"
        "    assert Path('marker.txt').resolve().parent == Path(__file__).resolve().parent\n",
        encoding="utf-8",
    )
    passed, output = builder_node_module._run_pytest(code_dir)
    assert passed, output
    assert (code_dir / "marker.txt").exists()


def test_run_pytest_works_correctly_even_with_a_relative_code_dir(tmp_path):
    # code_dir is not guaranteed absolute (build_and_review.py's project_dir used to skip
    # .resolve() before this same fix) - a naive fix that set cwd=str(code_dir) while ALSO passing
    # str(code_dir) as pytest's own target argument would double-resolve a relative code_dir
    # against the NEW cwd (e.g. "proj/code" -> cwd "proj/code" + target "proj/code" -> a
    # nonexistent nested "proj/code/proj/code"). The real fix targets "." instead.
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "test_x.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")

    relative_code_dir = Path(os.path.relpath(code_dir, start=Path.cwd()))
    passed, output = builder_node_module._run_pytest(relative_code_dir)
    assert passed, output


def test_check_package_still_imports_works_correctly_even_with_a_relative_code_dir(tmp_path):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "storage.py").write_text("class InMemoryStorage:\n    pass\n", encoding="utf-8")
    (code_dir / "test_storage.py").write_text(
        "from storage import InMemoryStorage\n\ndef test_it():\n    assert InMemoryStorage()\n",
        encoding="utf-8",
    )
    relative_code_dir = Path(os.path.relpath(code_dir, start=Path.cwd()))
    assert agent_tools_module._check_package_still_imports(relative_code_dir) == ""


def test_run_smoke_fails_when_smoke_run_is_missing(tmp_path):
    passed, output = builder_node_module._run_smoke(tmp_path)
    assert passed is False
    assert "missing" in output


def test_run_smoke_fails_when_the_script_crashes(tmp_path):
    (tmp_path / "smoke_run.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    passed, output = builder_node_module._run_smoke(tmp_path)
    assert passed is False
    assert "crashed" in output
    assert "boom" in output


def test_run_smoke_fails_with_no_output_at_all(tmp_path):
    (tmp_path / "smoke_run.py").write_text("pass\n", encoding="utf-8")
    passed, output = builder_node_module._run_smoke(tmp_path)
    assert passed is False
    assert "no output" in output


def test_run_smoke_fails_when_the_last_line_is_not_json(tmp_path):
    (tmp_path / "smoke_run.py").write_text("print('not json')\n", encoding="utf-8")
    passed, output = builder_node_module._run_smoke(tmp_path)
    assert passed is False
    assert "not the required JSON" in output


def test_run_smoke_fails_when_ran_is_false(tmp_path):
    # A caught exception silently reported as success - exactly what the contract forbids.
    (tmp_path / "smoke_run.py").write_text(
        "import json\nprint(json.dumps({'ran': False, 'output_summary': 'x'}))\n", encoding="utf-8",
    )
    passed, output = builder_node_module._run_smoke(tmp_path)
    assert passed is False
    assert "looks like a stub" in output


def test_run_smoke_fails_when_output_summary_is_empty(tmp_path):
    (tmp_path / "smoke_run.py").write_text(
        "import json\nprint(json.dumps({'ran': True, 'output_summary': '   '}))\n", encoding="utf-8",
    )
    passed, output = builder_node_module._run_smoke(tmp_path)
    assert passed is False


def test_run_smoke_passes_with_a_genuine_summary(tmp_path):
    (tmp_path / "smoke_run.py").write_text(
        "import json\nprint(json.dumps({'ran': True, 'output_summary': 'produced 3 real items'}))\n",
        encoding="utf-8",
    )
    passed, output = builder_node_module._run_smoke(tmp_path)
    assert passed is True
    assert output == ""


def test_run_smoke_actually_threads_the_mode_through_the_environment(tmp_path):
    # Proves SMOKE_MODE really reaches the subprocess, not just that the function signature accepts
    # a mode argument - the script only reports a real run when the mode matches what it expects.
    (tmp_path / "smoke_run.py").write_text(
        "import json, os\n"
        "mode = os.environ.get('SMOKE_MODE')\n"
        "print(json.dumps({'ran': mode == 'live', 'output_summary': f'mode was {mode}'}))\n",
        encoding="utf-8",
    )
    dry_passed, _ = builder_node_module._run_smoke(tmp_path, mode="dry")
    live_passed, _ = builder_node_module._run_smoke(tmp_path, mode="live")
    assert dry_passed is False
    assert live_passed is True


def test_run_smoke_reports_a_timeout_cleanly(tmp_path, monkeypatch):
    (tmp_path / "smoke_run.py").write_text("pass\n", encoding="utf-8")

    def fake_run(*a, **k):
        raise builder_node_module.subprocess.TimeoutExpired(cmd="smoke_run.py", timeout=120)

    monkeypatch.setattr(builder_node_module.subprocess, "run", fake_run)
    passed, output = builder_node_module._run_smoke(tmp_path)
    assert passed is False
    assert "timed out" in output


def _get_tool(code_dir, name):
    tools = {t.name: t for t in make_agent_tools(code_dir)}
    return tools[name]


def test_agent_tool_list_files_lists_real_entries(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y = 2\n", encoding="utf-8")
    result = _get_tool(tmp_path, "list_files").invoke({"path": "."})
    assert "a.py" in result
    assert "sub/b.py" in result


def test_agent_tool_list_files_rejects_path_traversal(tmp_path):
    result = _get_tool(tmp_path, "list_files").invoke({"path": "../../outside"})
    assert "ERROR" in result
    assert "outside the allowed directories" in result


def test_agent_tool_list_files_reports_a_missing_path(tmp_path):
    result = _get_tool(tmp_path, "list_files").invoke({"path": "nope"})
    assert "ERROR" in result
    assert "does not exist" in result


def test_agent_tool_search_files_finds_a_real_match_with_file_and_line(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\ndef intake_node(state, runtime):\n    pass\n", encoding="utf-8")
    result = _get_tool(tmp_path, "search_files").invoke({"pattern": "def intake_node"})
    assert "a.py:2:" in result
    assert "def intake_node" in result


def test_agent_tool_search_files_searches_across_multiple_files(tmp_path):
    (tmp_path / "a.py").write_text("call_it()\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("call_it()\n", encoding="utf-8")
    result = _get_tool(tmp_path, "search_files").invoke({"pattern": "call_it"})
    assert "a.py:1:" in result
    assert "sub/b.py:1:" in result


def test_agent_tool_search_files_reports_no_matches_cleanly(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    result = _get_tool(tmp_path, "search_files").invoke({"pattern": "nonexistent_symbol"})
    assert "No matches" in result


def test_agent_tool_search_files_falls_back_to_literal_text_for_an_invalid_regex(tmp_path):
    (tmp_path / "a.py").write_text("cfg = Config(x=1)\n", encoding="utf-8")
    result = _get_tool(tmp_path, "search_files").invoke({"pattern": "Config("})
    assert "ERROR" not in result
    assert "a.py:1" in result


def test_agent_tool_search_files_rejects_path_traversal(tmp_path):
    result = _get_tool(tmp_path, "search_files").invoke({"pattern": "x", "path": "../../outside"})
    assert "ERROR" in result
    assert "outside the allowed directories" in result


def test_agent_tool_search_files_skips_unreadable_binary_files(tmp_path):
    (tmp_path / "data.bin").write_bytes(b"\xff\xfe\x00\x01binary")
    (tmp_path / "a.py").write_text("needle_value = 1\n", encoding="utf-8")
    result = _get_tool(tmp_path, "search_files").invoke({"pattern": "needle_value"})
    assert "a.py:1:" in result


def test_agent_tool_read_file_returns_content_with_line_numbers(tmp_path):
    (tmp_path / "a.py").write_text("first\nsecond\n", encoding="utf-8")
    result = _get_tool(tmp_path, "read_file").invoke({"path": "a.py"})
    assert result == "1\tfirst\n2\tsecond"


def test_agent_tool_read_file_rejects_path_traversal(tmp_path):
    result = _get_tool(tmp_path, "read_file").invoke({"path": "../../../etc/passwd"})
    assert "ERROR" in result
    assert "outside the allowed directories" in result


def test_agent_tool_read_file_reports_a_missing_file(tmp_path):
    result = _get_tool(tmp_path, "read_file").invoke({"path": "nope.py"})
    assert "ERROR" in result
    assert "does not exist" in result


def test_agent_tool_read_file_rejects_reading_a_directory(tmp_path):
    (tmp_path / "sub").mkdir()
    result = _get_tool(tmp_path, "read_file").invoke({"path": "sub"})
    assert "ERROR" in result
    assert "directory" in result


def test_agent_tool_write_file_creates_a_new_file(tmp_path):
    result = _get_tool(tmp_path, "write_file").invoke({"path": "new.py", "content": "z = 3\n"})
    assert "new file" in result
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "z = 3\n"


def test_agent_tool_write_file_overwrites_an_empty_existing_file(tmp_path):
    (tmp_path / "a.py").write_text("   \n", encoding="utf-8")  # whitespace-only - not "real content"
    result = _get_tool(tmp_path, "write_file").invoke({"path": "a.py", "content": "new = 1\n"})
    assert "overwrote existing" in result
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "new = 1\n"


def test_agent_tool_write_file_rejects_overwriting_a_file_with_real_content(tmp_path):
    # 2026-09-28, Douglas's direct instruction: a real transcript showed an internal fix-loop
    # attempt doing a full write_file rewrite of a file that already worked, silently inventing a
    # different class name than what another file already depended on. Mechanical, not advisory -
    # edit_file remains the escape hatch, even for a deliberate full rewrite.
    (tmp_path / "a.py").write_text("class InMemoryStorage:\n    pass\n", encoding="utf-8")
    result = _get_tool(tmp_path, "write_file").invoke({"path": "a.py", "content": "class InMemoryStorageBackend:\n    pass\n"})
    assert "ERROR" in result
    assert "edit_file" in result
    # Rejected BEFORE touching the file - the real, working content must survive untouched.
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "class InMemoryStorage:\n    pass\n"


def test_agent_tool_write_file_still_creates_a_genuinely_new_file(tmp_path):
    (tmp_path / "a.py").write_text("class InMemoryStorage:\n    pass\n", encoding="utf-8")
    result = _get_tool(tmp_path, "write_file").invoke({"path": "b.py", "content": "class Other:\n    pass\n"})
    assert "new file" in result
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "class Other:\n    pass\n"


def test_agent_tool_write_file_rejects_path_traversal(tmp_path):
    result = _get_tool(tmp_path, "write_file").invoke({"path": "../escape.py", "content": "x\n"})
    assert "ERROR" in result
    assert not (tmp_path.parent / "escape.py").exists()


def test_agent_tool_write_file_flags_a_syntax_error_immediately(tmp_path):
    # Mechanical, ast.parse-based - no LLM call, surfaced in the tool's OWN return value so the model
    # sees the problem on the very next turn, instead of only discovering it many turns later via
    # pytest/smoke at the very end of a whole attempt.
    result = _get_tool(tmp_path, "write_file").invoke({"path": "bad.py", "content": "def f(:\n    pass\n"})
    assert "Wrote" in result  # the file still gets written - the model needs to see and fix it, not lose the attempt
    assert agent_tools_module._SYNTAX_ERROR_MARKER in result
    assert (tmp_path / "bad.py").exists()


def test_agent_tool_write_file_rejects_a_bare_placeholder_word(tmp_path):
    # Real incident (2026-09-29, creative-team): the builder created
    # creative_studio/creative_director_v2.py containing only the 12 characters "placeholder" - the
    # existing overwrite/empty guards don't catch this (syntactically valid, non-empty, a genuinely
    # new file), so it landed on disk unchecked.
    result = _get_tool(tmp_path, "write_file").invoke({"path": "creative_director_v2.py", "content": "placeholder"})
    assert "ERROR" in result
    assert not (tmp_path / "creative_director_v2.py").exists()


def test_agent_tool_write_file_allows_a_docstring_only_module(tmp_path):
    # Must not over-reject: a minimal but real __init__.py-style docstring-only module is a string
    # Constant, not a bare Name - a real, if trivial, legitimate pattern.
    result = _get_tool(tmp_path, "write_file").invoke({"path": "__init__.py", "content": '"""Package marker."""\n'})
    assert "Wrote" in result
    assert (tmp_path / "__init__.py").exists()


def test_agent_tool_write_file_does_not_flag_valid_python(tmp_path):
    result = _get_tool(tmp_path, "write_file").invoke({"path": "good.py", "content": "def f():\n    return 1\n"})
    assert agent_tools_module._SYNTAX_ERROR_MARKER not in result


def test_agent_tool_write_file_does_not_syntax_check_non_python_files(tmp_path):
    result = _get_tool(tmp_path, "write_file").invoke({"path": "data.json", "content": "{not valid json or python"})
    assert agent_tools_module._SYNTAX_ERROR_MARKER not in result


def test_agent_tool_read_file_can_read_installed_library_source(tmp_path):
    # Real fix for a real incident (2026-09-27): the agent had no way to read WHY an installed
    # library (LangGraph) behaved unexpectedly, because read access was scoped to the project only.
    # pytest is guaranteed installed in this venv - the whole suite depends on it importing.
    site_packages = Path(sysconfig.get_paths()["purelib"])
    assert (site_packages / "pytest" / "__init__.py").exists()
    result = _get_tool(tmp_path, "read_file").invoke({"path": "site-packages/pytest/__init__.py"})
    assert "ERROR" not in result
    assert result.startswith("1\t")


def test_agent_tool_list_files_can_list_an_installed_library(tmp_path):
    result = _get_tool(tmp_path, "list_files").invoke({"path": "site-packages/pytest"})
    assert "ERROR" not in result
    assert "__init__.py" in result


def test_agent_tool_search_files_can_search_an_installed_library(tmp_path):
    result = _get_tool(tmp_path, "search_files").invoke({"pattern": "__version__", "path": "site-packages/pytest"})
    assert "ERROR" not in result
    assert "site-packages/pytest" in result


def test_agent_tool_read_file_library_root_still_rejects_traversal(tmp_path):
    result = _get_tool(tmp_path, "read_file").invoke({"path": "site-packages/../../../../etc/passwd"})
    assert "ERROR" in result


def test_agent_tool_read_file_reports_missing_file_inside_library_root(tmp_path):
    result = _get_tool(tmp_path, "read_file").invoke({"path": "site-packages/pytest/definitely_not_a_real_file.py"})
    assert "ERROR" in result
    assert "does not exist" in result


def test_agent_tool_write_file_never_touches_the_real_installed_library(tmp_path):
    # write_file/edit_file/run_command never learn about the site-packages/ prefix at all - a path
    # using it must land INSIDE the project as an ordinary, literally-named subfolder, and the real
    # installed library must be completely untouched.
    site_packages = Path(sysconfig.get_paths()["purelib"])
    real_content_before = (site_packages / "pytest" / "__init__.py").read_text(encoding="utf-8")

    result = _get_tool(tmp_path, "write_file").invoke({
        "path": "site-packages/pytest/__init__.py", "content": "x = 1\n",
    })
    assert "Wrote" in result
    assert (tmp_path / "site-packages" / "pytest" / "__init__.py").read_text(encoding="utf-8") == "x = 1\n"

    real_content_after = (site_packages / "pytest" / "__init__.py").read_text(encoding="utf-8")
    assert real_content_after == real_content_before


def test_agent_tool_edit_file_replaces_a_unique_match(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    result = _get_tool(tmp_path, "edit_file").invoke({
        "path": "a.py", "old_string": "return 1", "new_string": "return 2",
    })
    assert result == "Edited 'a.py'."
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"


def test_agent_tool_edit_file_rejects_a_missing_file(tmp_path):
    result = _get_tool(tmp_path, "edit_file").invoke({"path": "nope.py", "old_string": "x", "new_string": "y"})
    assert "ERROR" in result
    assert "does not exist" in result


def test_agent_tool_edit_file_rejects_zero_matches(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    result = _get_tool(tmp_path, "edit_file").invoke({
        "path": "a.py", "old_string": "return 99", "new_string": "return 2",
    })
    assert "ERROR" in result
    assert "not found" in result
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "def f():\n    return 1\n"


def test_agent_tool_edit_file_rejects_a_non_unique_match(tmp_path):
    (tmp_path / "a.py").write_text("return 1\nreturn 1\n", encoding="utf-8")
    result = _get_tool(tmp_path, "edit_file").invoke({
        "path": "a.py", "old_string": "return 1", "new_string": "return 2",
    })
    assert "ERROR" in result
    assert "appears 2 times" in result
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "return 1\nreturn 1\n"


def test_agent_tool_edit_file_rejects_path_traversal(tmp_path):
    result = _get_tool(tmp_path, "edit_file").invoke({
        "path": "../../outside.py", "old_string": "x", "new_string": "y",
    })
    assert "ERROR" in result
    assert "outside the project directory" in result


def test_agent_tool_edit_file_flags_a_syntax_error_immediately(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    result = _get_tool(tmp_path, "edit_file").invoke({
        "path": "a.py", "old_string": "return 1", "new_string": "return 1(",
    })
    assert "Edited" in result
    assert agent_tools_module._SYNTAX_ERROR_MARKER in result


def test_check_package_still_imports_is_clean_when_nothing_is_collected_yet(tmp_path):
    # Exit code 5 ("no tests collected") must NOT be treated as an import failure - it just means
    # nothing references anything yet, which is the ordinary early state of a fresh build.
    assert agent_tools_module._check_package_still_imports(tmp_path) == ""


def test_check_package_still_imports_is_clean_when_everything_resolves(tmp_path):
    (tmp_path / "storage.py").write_text("class InMemoryStorage:\n    pass\n", encoding="utf-8")
    (tmp_path / "test_storage.py").write_text(
        "from storage import InMemoryStorage\n\ndef test_it():\n    assert InMemoryStorage()\n",
        encoding="utf-8",
    )
    assert agent_tools_module._check_package_still_imports(tmp_path) == ""


def test_check_package_still_imports_catches_a_real_cross_file_break(tmp_path):
    # The exact real incident this exists for: a name a DIFFERENT file already depends on getting
    # renamed away, discovered only via a full pytest run at the very end of an attempt before this.
    (tmp_path / "storage.py").write_text("class InMemoryStorageBackend:\n    pass\n", encoding="utf-8")
    (tmp_path / "conftest.py").write_text("from storage import InMemoryStorage\n", encoding="utf-8")
    result = agent_tools_module._check_package_still_imports(tmp_path)
    assert "IMPORT CHECK FAILED" in result
    assert "InMemoryStorage" in result


def test_agent_tool_write_file_catches_a_cross_file_import_break_on_a_new_file(tmp_path):
    (tmp_path / "conftest.py").write_text("from storage import InMemoryStorage\n", encoding="utf-8")
    result = _get_tool(tmp_path, "write_file").invoke({
        "path": "storage.py", "content": "class InMemoryStorageBackend:\n    pass\n",
    })
    assert "Wrote" in result
    assert "IMPORT CHECK FAILED" in result


def test_agent_tool_edit_file_catches_a_cross_file_import_break(tmp_path):
    (tmp_path / "storage.py").write_text("class InMemoryStorage:\n    pass\n", encoding="utf-8")
    (tmp_path / "conftest.py").write_text("from storage import InMemoryStorage\n", encoding="utf-8")
    result = _get_tool(tmp_path, "edit_file").invoke({
        "path": "storage.py", "old_string": "InMemoryStorage", "new_string": "InMemoryStorageBackend",
    })
    assert "Edited" in result
    assert "IMPORT CHECK FAILED" in result


def test_agent_tool_write_file_skips_import_check_for_non_python_files(tmp_path):
    (tmp_path / "conftest.py").write_text("from storage import InMemoryStorage\n", encoding="utf-8")
    result = _get_tool(tmp_path, "write_file").invoke({"path": "data.json", "content": '{"a": 1}'})
    assert "IMPORT CHECK" not in result


def test_agent_tool_run_command_executes_an_allowed_python_command(tmp_path):
    result = _get_tool(tmp_path, "run_command").invoke({"command": "python -c \"print('hello')\""})
    assert "exit code 0" in result
    assert "hello" in result


def test_agent_tool_run_command_rejects_a_disallowed_command(tmp_path):
    result = _get_tool(tmp_path, "run_command").invoke({"command": "pip install requests"})
    assert "ERROR" in result
    assert "not allowed" in result


def test_agent_tool_run_command_rejects_git(tmp_path):
    result = _get_tool(tmp_path, "run_command").invoke({"command": "git status"})
    assert "ERROR" in result
    assert "not allowed" in result


def test_agent_tool_run_command_reports_a_nonzero_exit(tmp_path):
    result = _get_tool(tmp_path, "run_command").invoke({"command": "python -c \"import sys; sys.exit(1)\""})
    assert "exit code 1" in result


def test_agent_tool_run_command_runs_pytest_against_real_files(tmp_path):
    (tmp_path / "test_x.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    result = _get_tool(tmp_path, "run_command").invoke({"command": "pytest -q"})
    assert "exit code 0" in result


class _FakeToolResponse:
    """Stands in for a real AIMessage from a tool-bound LLM: tool_calls (a list of
    {"name", "args", "id"} dicts, empty/None to signal "done") plus content (used only when there
    are no tool_calls - a plain summary, or an "AMBIGUOUS: ..." sentinel). `tokens` defaults to 0
    (matching every existing caller, none of which care about token accounting) - set it to give
    extract_tokens() a real, non-zero usage_metadata for a test that specifically exercises the
    token-budget mechanism."""

    def __init__(self, content="", tool_calls=None, tokens=0):
        self.content = content
        self.tool_calls = tool_calls or []
        self.response_metadata = {}
        self.usage_metadata = {"total_tokens": tokens}


def _tool_call(name: str, call_id: str, **args) -> dict:
    return {"name": name, "args": args, "id": call_id}


class _ScriptedAgentLLM:
    """A fake tool-bound LLM: bind_tools is a no-op returning self (the real tool objects aren't
    needed by a scripted fake), invoke returns the next response in the script, repeating the last
    one if the script runs out - so a test can drive an exact multi-turn conversation."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, config=None):
        idx = min(self._calls, len(self._responses) - 1)
        self._calls += 1
        return self._responses[idx]


def _agentic_state(tmp_path, spec_path, code_dir, **extra):
    return {
        "project_dir": str(tmp_path),
        "spec_path": str(spec_path),
        "code_dir": str(code_dir),
        "platform": "python",
        "review_problems": [],
        "total_tokens": 0,
        **extra,
    }


_REAL_PYTEST_TWO_FAILURES_OUTPUT = """FF                                                                       [100%]
================================== FAILURES ===================================
__________________________________ test_one ___________________________________

    def test_one():
>       assert 1 == 2, "one isn't two"
E       AssertionError: one isn't two
E       assert 1 == 2

test_sample.py:2: AssertionError
__________________________________ test_two ___________________________________

    def test_two():
>       raise ImportError("cannot import name 'InMemoryStorage' from 'storage'")
E       ImportError: cannot import name 'InMemoryStorage' from 'storage'

test_sample.py:5: ImportError
=========================== short test summary info ===========================
FAILED test_sample.py::test_one
FAILED test_sample.py::test_two
2 failed in 0.20s
"""  # captured verbatim from a real pytest run (2026-09-28), not hand-written - see the ground truth check


_ALL_PASS_OUTPUT = """..                                                                       [100%]
2 passed in 0.10s
"""

_CONTENT_WRITER_ONLY_FAILS_OUTPUT = """.F                                                                       [100%]
================================== FAILURES ===================================
______________________________ test_content_writer_thing ______________________

    def test_content_writer_thing():
>       assert False

E       AssertionError

tests/test_content_writer.py:2: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_content_writer.py::test_content_writer_thing
1 failed, 1 passed in 0.20s
"""

_BOTH_STORAGE_AND_CONTENT_WRITER_FAIL_OUTPUT = """FF                                                                       [100%]
================================== FAILURES ===================================
__________________________________ test_storage_thing __________________________

    def test_storage_thing():
>       assert False

E       AssertionError

tests/test_storage.py:2: AssertionError
______________________________ test_content_writer_thing ______________________

    def test_content_writer_thing():
>       assert False

E       AssertionError

tests/test_content_writer.py:2: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_storage.py::test_storage_thing
FAILED tests/test_content_writer.py::test_content_writer_thing
2 failed in 0.20s
"""


def _content_writer_unit():
    return {
        "name": "content_writer", "purpose": "content writer", "files": ["content_writer.py"],
        "interface": {"exports": [{
            "name": "content_writer_node", "kind": "function", "signature": "...", "description": "...",
        }]},
        "depends_on": [], "verification_hint": "a real behavioral check",
    }


def test_build_python_units_tolerates_a_pre_existing_unrelated_test_failure(tmp_path, monkeypatch):
    # The real creative-team incident (2026-09-29): "storage" could never complete while
    # content_writer's own separate, already-known bug kept the WHOLE test suite red (_run_pytest
    # always runs the whole suite, never scoped to one unit) - so storage never advanced, and
    # content_writer never got its own dedicated turn again. The model's only lever to pass ITS OWN
    # gate was to keep reaching into content_writer.py itself - confirmed live, ~1M tokens spent with
    # zero progress.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "storage.py").write_text("class Storage:\n    pass\n", encoding="utf-8")
    (code_dir / "content_writer.py").write_text("def content_writer_node():\n    pass\n", encoding="utf-8")

    build_units = [_real_build_unit("storage"), _content_writer_unit()]

    monkeypatch.setattr(builder_node_module, "_finding_change_problems", lambda code_dir, baseline: [])
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (False, _CONTENT_WRITER_ONLY_FAILS_OUTPUT))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))
    llm = _ScriptedAgentLLM([_FakeToolResponse(content="Done.")])

    state = _agentic_state(
        tmp_path, spec_path, code_dir,
        python_units_built=["content_writer"],  # already built, NOT this round's target
        review_problems=["storage.py: minor cleanup needed"],  # triggers storage's own rebuild
    )
    result = builder_node_module._build_python_units(state, llm, "config", "BASE SPEC", code_dir, build_units, None)

    assert result["builder_status"] == "done"
    assert "storage" in result["python_units_built"]
    tolerated = result["tolerated_failures"]
    assert tolerated and all(t["owner"] == "content_writer" for t in tolerated)


def test_build_python_units_still_requires_the_current_units_own_tests_to_pass(tmp_path, monkeypatch):
    # The tolerance above must never let a unit skip fixing its OWN genuine bug just because
    # something unrelated was already broken too - the fix relaxes cross-unit noise, not this
    # unit's own verification.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "storage.py").write_text("class Storage:\n    pass\n", encoding="utf-8")
    (code_dir / "content_writer.py").write_text("def content_writer_node():\n    pass\n", encoding="utf-8")

    build_units = [_real_build_unit("storage"), _content_writer_unit()]

    # Every call - baseline and every attempt - reports BOTH storage's own test and content_writer's
    # test failing: storage's own bug is never actually fixed.
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (False, _BOTH_STORAGE_AND_CONTENT_WRITER_FAIL_OUTPUT))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))
    llm = _ScriptedAgentLLM([_FakeToolResponse(content="I believe this is correct.")])

    state = _agentic_state(
        tmp_path, spec_path, code_dir,
        python_units_built=["content_writer"],
        review_problems=["storage.py: minor cleanup needed"],
    )
    result = builder_node_module._build_python_units(state, llm, "config", "BASE SPEC", code_dir, build_units, None)

    assert result["builder_status"] == "ambiguous"
    assert "storage" not in result["python_units_built"]


def test_build_python_units_still_catches_a_new_regression_in_a_different_unit(tmp_path, monkeypatch):
    # The tolerance above must never mask a GENUINELY NEW failure this attempt introduced elsewhere -
    # only a failure that was ALREADY there before this attempt started is tolerated.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "storage.py").write_text("class Storage:\n    pass\n", encoding="utf-8")
    (code_dir / "content_writer.py").write_text("def content_writer_node():\n    pass\n", encoding="utf-8")

    build_units = [_real_build_unit("storage"), _content_writer_unit()]

    calls = []

    def _fake_run_pytest(code_dir):
        calls.append(1)
        if len(calls) == 1:
            return True, _ALL_PASS_OUTPUT  # baseline: content_writer was genuinely fine before
        return False, _CONTENT_WRITER_ONLY_FAILS_OUTPUT  # after storage's attempt: now it isn't

    monkeypatch.setattr(builder_node_module, "_run_pytest", _fake_run_pytest)
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))
    llm = _ScriptedAgentLLM([_FakeToolResponse(content="Done.")])

    state = _agentic_state(
        tmp_path, spec_path, code_dir,
        python_units_built=["content_writer"],
        review_problems=["storage.py: minor cleanup needed"],
    )
    result = builder_node_module._build_python_units(state, llm, "config", "BASE SPEC", code_dir, build_units, None)

    assert result["builder_status"] == "ambiguous"


def test_test_belongs_to_unit_matches_via_the_test_prefix_convention():
    assert builder_node_module._test_belongs_to_unit("tests/test_storage.py::test_thing", ["storage.py"]) is True
    assert builder_node_module._test_belongs_to_unit("tests/test_storage.py::test_thing", ["content_writer.py"]) is False


def test_parse_pytest_failures_extracts_test_and_reason_from_real_output():
    findings = builder_node_module._parse_pytest_failures(_REAL_PYTEST_TWO_FAILURES_OUTPUT)
    assert findings == [
        {"test": "test_sample.py::test_one", "reason": "AssertionError: one isn't two"},
        {"test": "test_sample.py::test_two", "reason": "ImportError: cannot import name 'InMemoryStorage' from 'storage'"},
    ]


def test_parse_pytest_failures_is_empty_for_output_with_no_failed_lines():
    # A collection error, or any pytest output shape without the short-summary "FAILED" lines this
    # parser depends on - must return [] cleanly, never raise, so the caller's fallback to the full
    # raw output is what the model actually sees.
    assert builder_node_module._parse_pytest_failures("ERROR collecting test_sample.py\n") == []
    assert builder_node_module._parse_pytest_failures("") == []


def test_builder_node_agentic_feeds_structured_pytest_findings_to_the_next_attempt(tmp_path, monkeypatch):
    # 2026-09-28, Douglas's direct instruction: loop 1 (this internal retry loop) must hand the next
    # attempt STRUCTURED findings, the same discipline loop 2 (builder<->reviewer) already has via
    # "REVIEWER PROBLEMS TO FIX" - not just a giant blob of raw pytest text.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    llm = _MessageCapturingAgentLLM([
        _FakeToolResponse(content="Done."),  # attempt 1: reports done immediately, verification then runs for real and fails
        _FakeToolResponse(content="Done."),  # attempt 2: same, so we can inspect its OWN opening task_prompt
    ])
    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(
        builder_node_module, "_run_pytest",
        lambda code_dir: (False, _REAL_PYTEST_TWO_FAILURES_OUTPUT),
    )

    builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert len(llm.invoke_message_contents) >= 2
    attempt_2_task_prompt = llm.invoke_message_contents[1][1]  # [system, human] - the human task_prompt
    assert "VERIFICATION FAILURES TO FIX" in attempt_2_task_prompt
    assert "test_sample.py::test_one" in attempt_2_task_prompt
    assert "AssertionError: one isn't two" in attempt_2_task_prompt
    assert "prefer edit_file" in attempt_2_task_prompt
    # The full raw output must still be there too - the structured list is an addition, not a
    # replacement, since the mechanical extraction is best-effort by design.
    assert "short test summary info" in attempt_2_task_prompt


def test_builder_node_agentic_writes_files_via_real_tools_and_reports_done(tmp_path, monkeypatch):
    # Multi-turn: write the module, write the mandatory smoke test, then signal done (no more tool
    # calls) - proves the dispatch loop actually executes sequential tool calls across turns and
    # writes real files, not just that the function returns something.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    llm = _ScriptedAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call(
            "write_file", "1", path="module_a.py", content="def do_thing():\n    return 1\n",
        )]),
        _FakeToolResponse(tool_calls=[_tool_call(
            "write_file", "2", path="smoke_run.py",
            content="import json\nprint(json.dumps({'ran': True, 'output_summary': 'module_a ran fine'}))\n",
        )]),
        _FakeToolResponse(content="Done - wrote module_a.py and smoke_run.py, both verified."),
    ])
    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "done"
    assert (code_dir / "module_a.py").read_text(encoding="utf-8") == "def do_thing():\n    return 1\n"


def test_builder_node_agentic_can_record_a_tech_quirk_via_its_own_tool(tmp_path, monkeypatch):
    # 2026-09-28, Douglas's direct instruction: any role should be able to write a genuinely
    # reusable technical lesson to the shared, cross-project ledger - the builder's own version of
    # this is a real tool (record_tech_quirk), not a structured-output field, since it's the one
    # role with an actual tool-calling loop.
    quirks_path = tmp_path / "tech-quirks.md"
    monkeypatch.setattr(tech_quirks_module, "TECH_QUIRKS_PATH", quirks_path)

    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    llm = _ScriptedAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call(
            "record_tech_quirk", "1",
            lesson="LangGraph Runtime: has no .config attribute, use the injected context instead.",
        )]),
        _FakeToolResponse(tool_calls=[_tool_call(
            "write_file", "2", path="module_a.py", content="def do_thing():\n    return 1\n",
        )]),
        _FakeToolResponse(content="Done."),
    ])
    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "done"

    content = quirks_path.read_text(encoding="utf-8")
    assert "LangGraph Runtime: has no .config attribute, use the injected context instead." in content
    assert "(builder (python, agentic), " in content
    # Recording a quirk is neither exploration nor a substantive code write - it must not, by
    # itself, satisfy anything the escalation mechanism is tracking.
    touched_paths = [f["path"] for f in result["builder_files_touched"]]
    assert touched_paths == ["module_a.py"]  # only the real write is mechanically recorded here


def test_builder_node_agentic_prepends_existing_tech_quirks_to_the_system_prompt(tmp_path, monkeypatch):
    quirks_path = tmp_path / "tech-quirks.md"
    monkeypatch.setattr(tech_quirks_module, "TECH_QUIRKS_PATH", quirks_path)
    tech_quirks_module.record_tech_quirk_entry("n8n Slack node: emails must be URL-encoded.", source="builder (n8n)")

    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    seen_system_messages = []

    class _CapturingLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages, config=None):
            seen_system_messages.append(messages[0].content)
            return _FakeToolResponse(content="Done.")

    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: _CapturingLLM())
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert "n8n Slack node: emails must be URL-encoded." in seen_system_messages[0]
    assert "UNVERIFIED CLAIM" in seen_system_messages[0]


def test_extract_and_record_tech_quirk_strips_the_line_and_records_it(tmp_path, monkeypatch):
    # The n8n per-node builder is a strict single-JSON-object contract, not a tool-calling loop, so
    # its write mechanism is a single optional leading line rather than a tool - see
    # _N8N_NODE_SYSTEM_PROMPT's own instruction.
    quirks_path = tmp_path / "tech-quirks.md"
    monkeypatch.setattr(tech_quirks_module, "TECH_QUIRKS_PATH", quirks_path)

    raw = 'TECH_QUIRK: n8n Slack node requires URL-encoded emails.\n{"name": "Node A", "type": "n8n-nodes-base.set"}'
    remaining = builder_node_module._extract_and_record_tech_quirk(raw)

    assert remaining.strip() == '{"name": "Node A", "type": "n8n-nodes-base.set"}'
    content = quirks_path.read_text(encoding="utf-8")
    assert "n8n Slack node requires URL-encoded emails." in content
    assert "(builder (n8n), " in content


def test_extract_and_record_tech_quirk_leaves_ordinary_responses_untouched(tmp_path, monkeypatch):
    quirks_path = tmp_path / "tech-quirks.md"
    monkeypatch.setattr(tech_quirks_module, "TECH_QUIRKS_PATH", quirks_path)

    raw = '{"name": "Node A", "type": "n8n-nodes-base.set"}'
    remaining = builder_node_module._extract_and_record_tech_quirk(raw)

    assert remaining == raw  # completely unchanged - the common case, nothing to report
    assert not quirks_path.exists()


def test_builder_node_agentic_escalates_immediately_with_no_nudge_phase(tmp_path, monkeypatch):
    # Redesigned 2026-09-28 (Douglas's own reaction to a real transcript: "one of my 3 max rounds
    # could end up 1 full round purely read-only... it should just know it's supposed to write
    # code" - correctly identified as a real design flaw, not expected model behavior). The
    # original version of this gave a polite, ignorable text nudge before doing anything mechanical
    # - a real transcript proved that doesn't work (ignored twice in a row). There is no longer a
    # nudge-first phase at all: crossing the exploration-call threshold escalates immediately.
    #
    # Also fixed in the same pass: the threshold now counts individual exploration CALLS, not
    # turns - the same transcript had NINE list_files calls in a single turn, which a turn-based
    # counter would have counted as just "1". This test deliberately batches many calls into one
    # turn (like the real transcript did) to prove the call-count, not the turn-count, is what
    # trips the threshold.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    n = builder_node_module._MAX_EXPLORATION_CALLS_BEFORE_FORCED_WRITE
    responses = [
        # All n exploration calls batched into a SINGLE turn - a turn-based counter would never
        # trip on this; a call-based counter must.
        _FakeToolResponse(tool_calls=[_tool_call("list_files", str(i), path=".") for i in range(n)]),
        _FakeToolResponse(tool_calls=[_tool_call(
            "write_file", "w", path="smoke_run.py",
            content="import json\nprint(json.dumps({'ran': True, 'output_summary': 'ok'}))\n",
        )]),
        _FakeToolResponse(content="Done."),
    ]
    captured_message_lists = []

    class _RecordingLLM:
        def __init__(self, responses):
            self._responses = list(responses)
            self._calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages, config=None):
            captured_message_lists.append(list(messages))
            idx = min(self._calls, len(self._responses) - 1)
            self._calls += 1
            return self._responses[idx]

    llm = _RecordingLLM(responses)
    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "done"

    # The very next call (2nd) already has the escalation message - no earlier "please write
    # something" nudge call happened first.
    assert len(captured_message_lists) == 3
    escalation_call_messages = captured_message_lists[1]
    assert any(
        "exploration calls" in str(getattr(m, "content", "")) and "Only write_file and edit_file are available" in str(getattr(m, "content", ""))
        for m in escalation_call_messages
    )


def test_builder_node_agentic_fails_when_smoke_run_is_never_written(tmp_path, monkeypatch):
    # The smoke test is now MANDATORY (no opt-in marker) - an agent that writes real code but never
    # writes smoke_run.py must not be allowed to report done.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    llm = _ScriptedAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call(
            "write_file", "1", path="module_a.py", content="def do_thing():\n    return 1\n",
        )]),
        _FakeToolResponse(content="Done."),
    ])
    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "ambiguous"
    assert "SMOKE-TEST FAILURE" in result["ambiguity_question"]
    assert "missing" in result["ambiguity_question"]


def test_builder_node_agentic_respects_the_ambiguous_signal(tmp_path, monkeypatch):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    llm = _ScriptedAgentLLM([
        _FakeToolResponse(content="AMBIGUOUS: should the export be CSV or JSON?"),
    ])
    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "ambiguous"
    assert result["ambiguity_question"] == "should the export be CSV or JSON?"


def test_builder_node_agentic_never_runs_live_smoke_without_explicit_consent(tmp_path, monkeypatch):
    # The consent hard rule, proven at the code level: dry mode always runs, live mode must never
    # fire unless state["live_smoke"] is explicitly True - the common case (key absent entirely)
    # must behave exactly like an explicit False.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    llm = _ScriptedAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call(
            "write_file", "1", path="smoke_run.py",
            content="import json\nprint(json.dumps({'ran': True, 'output_summary': 'ok'}))\n",
        )]),
        _FakeToolResponse(content="Done."),
    ])
    modes_seen = []
    real_run_smoke = builder_node_module._run_smoke

    def spying_run_smoke(code_dir, mode="dry"):
        modes_seen.append(mode)
        return real_run_smoke(code_dir, mode=mode)

    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", spying_run_smoke)

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "done"
    assert modes_seen == ["dry"]


def test_builder_node_agentic_runs_live_smoke_when_explicitly_authorized(tmp_path, monkeypatch):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    llm = _ScriptedAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call(
            "write_file", "1", path="smoke_run.py",
            content="import json\nprint(json.dumps({'ran': True, 'output_summary': 'ok'}))\n",
        )]),
        _FakeToolResponse(content="Done."),
    ])
    modes_seen = []
    real_run_smoke = builder_node_module._run_smoke

    def spying_run_smoke(code_dir, mode="dry"):
        modes_seen.append(mode)
        return real_run_smoke(code_dir, mode=mode)

    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", spying_run_smoke)

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir, live_smoke=True))
    assert result["builder_status"] == "done"
    assert modes_seen == ["dry", "live"]


def test_builder_node_agentic_per_attempt_token_cap_ends_only_that_attempt(tmp_path, monkeypatch):
    # 2026-09-28, Douglas's exact numbers: a fresh attempt gets its OWN 500K budget, not inherited
    # exhaustion from a previous attempt - "imagine a human coded once, redo but wiped his brain
    # entirely." Two big calls cross MAX_TOKENS_PER_ATTEMPT well before MAX_AGENT_TURNS (10) would -
    # proves the cap ends the ATTEMPT (verification still runs on whatever's on disk), not the whole
    # invocation, and that attempt 2 gets a genuinely fresh budget rather than inheriting the first.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    per_call = builder_node_module.MAX_TOKENS_PER_ATTEMPT // 2 + 1
    llm = _ScriptedAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call("list_files", "1", path=".")], tokens=per_call),
        _FakeToolResponse(tool_calls=[_tool_call("list_files", "2", path=".")], tokens=per_call),  # crosses the cap here - attempt 1 ends
        _FakeToolResponse(tool_calls=[_tool_call(  # attempt 2: fresh attempt_tokens=0
            "write_file", "w", path="module_a.py", content="def do_thing():\n    return 1\n",
        )], tokens=100),
        _FakeToolResponse(content="Done.", tokens=100),
    ])
    pytest_calls = []

    def fake_run_pytest(code_dir):
        pytest_calls.append(1)
        return (len(pytest_calls) >= 2, "still broken")  # fails attempt 1, passes from attempt 2 on

    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", fake_run_pytest)
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "done"
    assert len(pytest_calls) == 2  # attempt 1 (failed) + attempt 2 (passed) - not a 3rd attempt
    assert llm._calls == 4  # attempt 1 used exactly 2 turns (the cap), attempt 2 used exactly 2
    assert result["total_tokens"] == per_call * 2 + 100 + 100  # lifetime cumulative still accurate


def test_builder_node_agentic_per_invocation_token_cap_ends_the_whole_invocation(tmp_path, monkeypatch):
    # The per-invocation cap (1.5M = 500K x 3) is the backstop for individual attempts that each
    # legitimately overshoot their own per-attempt cap in one large turn rather than landing neatly
    # on it - loop 2's own MAX_ROUNDS is the real lifetime backstop now, not a token count.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    over_attempt_cap = builder_node_module.MAX_TOKENS_PER_ATTEMPT + 1

    class _NeverStopsLLM:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages, config=None):
            self.calls += 1
            return _FakeToolResponse(tool_calls=[_tool_call("list_files", "x", path=".")], tokens=over_attempt_cap)

    llm = _NeverStopsLLM()
    pytest_calls = []

    def fake_run_pytest(code_dir):
        # A DIFFERENT failure each attempt - real evidence today showed each attempt's own
        # exploration differs even when it doesn't fix anything, so distinct failures are the
        # realistic case. Deliberately avoids also tripping the separate "two identical failures"
        # dedup rule, which would otherwise stop the run after attempt 2 for an unrelated reason
        # and never let this test reach the invocation cap it's actually trying to exercise -
        # confirmed this distinction matters by hitting that exact interaction while writing this.
        pytest_calls.append(1)
        return False, f"failure variant {len(pytest_calls)}"

    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", fake_run_pytest)

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "ambiguous"
    assert str(builder_node_module.MAX_TOKENS_PER_INVOCATION) in result["ambiguity_question"]
    assert "invocation" in result["ambiguity_question"].lower()
    # Stopped on attempt 3's FIRST call, not by naturally exhausting 3 whole attempts each via
    # their own per-attempt cap (which would have taken exactly as many calls, but returned a
    # DIFFERENT, per-attempt-worded message instead - the assertions above prove which one fired).
    assert llm.calls == 3


def test_builder_node_agentic_exhausts_the_turn_and_attempt_budget(tmp_path, monkeypatch):
    # A fake that never stops calling tools - proves both caps actually bound the loop rather than
    # spinning forever, and that the failure is reported rather than silently swallowed. A DIFFERENT
    # pytest failure message each attempt (mocked, deterministic) so the "two identical failures
    # stop early" rule (covered by its own dedicated test below) never fires here - this test is
    # specifically about the outer attempt budget itself still terminating correctly.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    class _NeverStopsLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages, config=None):
            return _FakeToolResponse(tool_calls=[_tool_call("list_files", "x", path=".")])

    pytest_calls = []

    def _fake_run_pytest(code_dir):
        pytest_calls.append(1)
        return False, f"failure variant {len(pytest_calls)}"  # different every call - never a dedup match

    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: _NeverStopsLLM())
    monkeypatch.setattr(builder_node_module, "_run_pytest", _fake_run_pytest)

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "ambiguous"
    assert "could not get verification passing" in result["ambiguity_question"]
    assert "Ran out of turns" in result["ambiguity_question"]


def test_builder_node_agentic_stops_early_on_two_identical_ran_out_of_turns_failures(tmp_path, monkeypatch):
    # 2026-09-28 fix: independent verification now runs even on a ran-out-of-turns attempt, so two
    # such attempts producing the SAME real verification failure are correctly recognized as
    # identical and stop the run after attempt 2 - previously this whole path skipped verification
    # entirely (an immediate `continue`), so this dedup rule silently never applied to it at all,
    # and a run could burn all 3 attempts even when every single one hit the identical failure.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    class _NeverStopsLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages, config=None):
            return _FakeToolResponse(tool_calls=[_tool_call("list_files", "x", path=".")])

    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: _NeverStopsLLM())
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (False, "the exact same failure every time"))

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "ambiguous"
    assert "identical verification failure" in result["ambiguity_question"]
    assert "stopping rather than repeating it a third time" in result["ambiguity_question"]


def test_builder_node_agentic_verifies_and_reports_done_even_after_running_out_of_turns(tmp_path, monkeypatch):
    # The actual real-world fix (2026-09-28, verified against a real creative-team transcript): an
    # attempt that ran out of turns mid-tool-call must still get independent verification on
    # whatever real work landed on disk, rather than having it discarded unchecked. Mechanical truth
    # wins over an incomplete self-report - if what's already there happens to pass, that's done.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    class _WritesThenNeverStopsLLM:
        def __init__(self):
            self._calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages, config=None):
            self._calls += 1
            if self._calls == 1:
                return _FakeToolResponse(tool_calls=[_tool_call(
                    "write_file", "1", path="module_a.py", content="def do_thing():\n    return 1\n",
                )])
            # Keeps calling tools forever after that - never explicitly reports done, so this
            # attempt can only end via running out of turns, with real work already on disk.
            return _FakeToolResponse(tool_calls=[_tool_call("list_files", f"x{self._calls}", path=".")])

    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: _WritesThenNeverStopsLLM())
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "done"
    assert result["builder_files_touched"] == [{"path": "module_a.py", "action": "write_file"}]


def test_final_fix_pass_is_not_accepted_when_the_builder_only_ran_out_of_turns(tmp_path, monkeypatch):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    class _WritesThenNeverStopsLLM:
        def __init__(self):
            self._calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages, config=None):
            self._calls += 1
            if self._calls == 1:
                return _FakeToolResponse(tool_calls=[_tool_call(
                    "write_file", f"w{self._calls}", path="module_a.py", content="x = 1\n",
                )])
            return _FakeToolResponse(tool_calls=[_tool_call("list_files", f"x{self._calls}", path=".")])

    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: _WritesThenNeverStopsLLM())
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    state = {**_agentic_state(tmp_path, spec_path, code_dir), "final_fix_pass": True,
             "review_problems": ["module_a.py: a real must-fix bug"]}
    result = builder_node_module.builder_node(state)
    assert result["builder_status"] == "ambiguous"


def test_builder_node_agentic_does_not_count_zero_progress_exhaustion_as_done(tmp_path, monkeypatch):
    # The inverse real incident (2026-09-29): a unit ran out of turns in EVERY attempt, writing
    # NOTHING - its own file byte-identical to what was already there before the attempt started -
    # yet the run reported it "done" because pytest/smoke happened to pass on that untouched,
    # pre-existing code (a stale leftover, e.g. from before an Architect re-split changed this
    # unit's own contract). A pass that predates this attempt's own work is not evidence of anything
    # THIS attempt did - unlike the test above (real progress, just unreported), this attempt made
    # NO progress at all and must not be credited for a pass that was already true beforehand.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "context.py").write_text("class CDContext:\n    pass\n", encoding="utf-8")

    class _NeverWritesLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages, config=None):
            return _FakeToolResponse(tool_calls=[_tool_call("list_files", "x", path=".")])

    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: _NeverWritesLLM())
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))

    assert result["builder_status"] == "ambiguous"
    assert "without writing or editing anything" in result["ambiguity_question"].lower()
    # The pre-existing file must genuinely be untouched - confirms this really was zero progress.
    assert (code_dir / "context.py").read_text(encoding="utf-8") == "class CDContext:\n    pass\n"


def test_builder_node_agentic_stops_early_on_two_identical_verification_failures(tmp_path, monkeypatch):
    # Douglas's explicit direction: two consecutive attempts with the IDENTICAL failure must stop
    # immediately rather than spend a 3rd attempt repeating something already proven not to change.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    done_response = _FakeToolResponse(content="Done.")
    # Never writes smoke_run.py, every attempt - the exact same verification failure every time.
    llm = _ScriptedAgentLLM([done_response])
    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "ambiguous"
    assert "identical verification failure" in result["ambiguity_question"]
    # Stopped after the 2nd attempt, not the full 3 - only 2 LLM calls happened.
    assert llm._calls == 2


class _ToolTrackingAgentLLM:
    """Like _ScriptedAgentLLM, but bind_tools() itself is observable - records the sorted tool
    names each call was actually bound with, so a test can prove escalation really does swap to a
    smaller, restricted tool binding (a mechanical guarantee - the API is never even offered the
    other tools), not just that the surrounding bookkeeping ran."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._calls = 0
        self.invoke_tool_names: list = []

    def bind_tools(self, tools):
        bound_names = sorted(t.name for t in tools)

        class _Bound:
            def invoke(_self, messages, config=None):
                self.invoke_tool_names.append(bound_names)
                idx = min(self._calls, len(self._responses) - 1)
                self._calls += 1
                return self._responses[idx]

        return _Bound()


def test_builder_node_agentic_escalates_to_a_restricted_toolset_immediately(tmp_path, monkeypatch):
    # Redesigned 2026-09-28 (Douglas's own reaction to a real transcript - see the "no nudge phase"
    # test above for the full incident): crossing the exploration-call threshold escalates on the
    # VERY NEXT call. There is no longer a first "nudge, but still allow exploring" occurrence
    # before the toolset is actually restricted - a real transcript showed that phase gets ignored.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    # n calls split across 2 turns (like the real transcript, which batched many calls per turn),
    # not n turns of 1 call each - MAX_AGENT_TURNS is also 10, and one-per-turn would exhaust the
    # whole attempt's turn budget just reaching the threshold, leaving no turn left to observe the
    # escalation land within this same attempt.
    n = builder_node_module._MAX_EXPLORATION_CALLS_BEFORE_FORCED_WRITE
    half = n // 2
    responses = (
        [
            _FakeToolResponse(tool_calls=[_tool_call("list_files", f"e{i}", path=".") for i in range(half)]),
            _FakeToolResponse(tool_calls=[_tool_call("list_files", f"e{i}", path=".") for i in range(half, n)]),
        ]
        + [_FakeToolResponse(tool_calls=[_tool_call(
            "write_file", "w", path="smoke_run.py",
            content="import json\nprint(json.dumps({'ran': True, 'output_summary': 'ok'}))\n",
        )])]
        + [_FakeToolResponse(content="Done.")]
    )
    llm = _ToolTrackingAgentLLM(responses)
    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "done"

    full_toolset = sorted(["list_files", "search_files", "read_file", "write_file", "edit_file", "run_command", "record_tech_quirk"])
    restricted_toolset = sorted(["write_file", "edit_file", "record_tech_quirk"])
    # Both exploration turns (2 turns, n calls total) used the full toolset.
    assert llm.invoke_tool_names[0] == full_toolset
    assert llm.invoke_tool_names[1] == full_toolset
    # The VERY NEXT call - immediately after crossing the threshold, no nudge-only call in between -
    # was already mechanically restricted to write/edit only.
    assert llm.invoke_tool_names[2] == restricted_toolset
    # Once it wrote something, the restriction lifted for the final "done" call.
    assert llm.invoke_tool_names[3] == full_toolset


def test_builder_node_agentic_escalation_leaves_real_turns_even_one_call_per_turn(tmp_path, monkeypatch):
    # The real, pathological worst case: the model calls exactly ONE exploration tool per turn, no
    # batching at all. _MAX_EXPLORATION_CALLS_BEFORE_FORCED_WRITE is deliberately kept below
    # MAX_AGENT_TURNS specifically so this case still has real turns left after escalation, rather
    # than crossing the threshold on the attempt's very last turn and wasting it via "ran out of
    # turns" instead of ever actually forcing a write.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    n = builder_node_module._MAX_EXPLORATION_CALLS_BEFORE_FORCED_WRITE
    assert n < builder_node_module.MAX_AGENT_TURNS, "the cap must leave real headroom, not equal the turn budget"

    responses = (
        [_FakeToolResponse(tool_calls=[_tool_call("list_files", f"e{i}", path=".")]) for i in range(n)]
        + [_FakeToolResponse(tool_calls=[_tool_call(
            "write_file", "w", path="smoke_run.py",
            content="import json\nprint(json.dumps({'ran': True, 'output_summary': 'ok'}))\n",
        )])]
        + [_FakeToolResponse(content="Done.")]
    )
    llm = _ScriptedAgentLLM(responses)
    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    # Must actually succeed within THIS attempt - not fall through to "ran out of turns" and need a
    # second, fresh attempt just because escalation had nowhere left to land.
    assert result["builder_status"] == "done"


def test_is_substantive_write_rejects_a_scratch_named_file():
    # Real incident this exists for (2026-09-28, Douglas reviewing a transcript himself): a 98-byte
    # throwaway `_probe.py` satisfied a bare "wrote something" check and gamed escalation.
    assert not builder_node_module._is_substantive_write("_probe.py", "x = 1\n", "Wrote (new file) '_probe.py' - 6 bytes.")
    assert not builder_node_module._is_substantive_write("scratch.py", "x = 1\n", "Wrote 'scratch.py' - 6 bytes.")


def test_is_substantive_write_rejects_a_syntax_error():
    result_with_marker = f"Wrote (new file) 'bad.py' - 10 bytes.\n\nWARNING - {agent_tools_module._SYNTAX_ERROR_MARKER}: nope"
    assert not builder_node_module._is_substantive_write("bad.py", "def f(:\n", result_with_marker)


def test_is_substantive_write_rejects_empty_content():
    assert not builder_node_module._is_substantive_write("real_name.py", "   \n\n", "Wrote 'real_name.py' - 3 bytes.")


def test_is_substantive_write_accepts_a_real_flat_script_with_no_def_or_class():
    # An earlier version of this check required a def/class in every .py write - a real test caught
    # it rejecting exactly this shape (a legitimate driver script with no function needed) as if it
    # were fake filler. Content and byte-count don't reliably distinguish real-but-small from
    # throwaway (the real _probe.py incident was also under 100 bytes) - the filename is the signal.
    content = "import json\nprint(json.dumps({'ran': True, 'output_summary': 'ok'}))\n"
    assert builder_node_module._is_substantive_write("smoke_run.py", content, "Wrote 'smoke_run.py' - 70 bytes.")


def test_is_substantive_write_accepts_real_non_python_content():
    assert builder_node_module._is_substantive_write("workflow.json", '{"nodes": []}', "Wrote 'workflow.json' - 13 bytes.")


def test_builder_node_agentic_a_scratch_named_write_does_not_lift_escalation(tmp_path, monkeypatch):
    # The actual end-to-end proof, not just the unit-level check above: once escalated, writing an
    # obviously-scratch-named file must NOT lift the restriction or reset the exploration counter -
    # the API-level restriction must stay in place for the very next call too.
    monkeypatch.setattr(builder_node_module, "MAX_AGENT_TURNS", 30)  # needs more turns than the cap
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    n = builder_node_module._MAX_EXPLORATION_CALLS_BEFORE_FORCED_WRITE
    responses = (
        [_FakeToolResponse(tool_calls=[_tool_call("list_files", f"e{i}", path=".")]) for i in range(n)]
        + [_FakeToolResponse(tool_calls=[_tool_call("write_file", "probe", path="_probe.py", content="x = 1\n")])]
        + [_FakeToolResponse(tool_calls=[_tool_call(
            "write_file", "real", path="smoke_run.py",
            content="import json\nprint(json.dumps({'ran': True, 'output_summary': 'ok'}))\n",
        )])]
        + [_FakeToolResponse(content="Done.")]
    )
    llm = _ToolTrackingAgentLLM(responses)
    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "done"

    restricted_toolset = sorted(["write_file", "edit_file", "record_tech_quirk"])
    full_toolset = sorted(["list_files", "search_files", "read_file", "write_file", "edit_file", "run_command", "record_tech_quirk"])
    # The call right after the scratch write is STILL restricted - the scratch write did not lift it.
    assert llm.invoke_tool_names[-2] == restricted_toolset
    # Only the real write lifts it, for the final "done" call.
    assert llm.invoke_tool_names[-1] == full_toolset
    # touched_files still mechanically records the scratch write - it just doesn't count as progress.
    touched_paths = [f["path"] for f in result["builder_files_touched"]]
    assert "_probe.py" in touched_paths
    assert (code_dir / "_probe.py").exists()


class _MessageCapturingAgentLLM:
    """Like _ScriptedAgentLLM, but also records each call's messages list content (a chat-completions
    API is stateless - the WHOLE list is resent every turn) - so a test can verify pruning actually
    shrank an earlier turn's content by the time a later turn is called, not just that the run
    completes."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._calls = 0
        self.invoke_message_contents: list = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, config=None):
        self.invoke_message_contents.append([getattr(m, "content", None) for m in messages])
        idx = min(self._calls, len(self._responses) - 1)
        self._calls += 1
        return self._responses[idx]


def test_prune_stale_exploration_results_replaces_content_in_place():
    from langchain_core.messages import ToolMessage
    keep = ToolMessage(content="This turn's own result - not passed in, stays untouched by caller.", tool_call_id="keep")
    stale_a = ToolMessage(content="a big file's worth of content from three turns ago", tool_call_id="a")
    stale_b = ToolMessage(content="another stale result", tool_call_id="b")

    pruned = builder_node_module._prune_stale_exploration_results({"a": stale_a, "b": stale_b})

    assert pruned == 2
    assert stale_a.content == builder_node_module._PRUNED_EXPLORATION_PLACEHOLDER
    assert stale_b.content == builder_node_module._PRUNED_EXPLORATION_PLACEHOLDER
    assert keep.content == "This turn's own result - not passed in, stays untouched by caller."


def test_prune_stale_exploration_results_does_not_double_count_an_already_pruned_message():
    from langchain_core.messages import ToolMessage
    already_pruned = ToolMessage(content=builder_node_module._PRUNED_EXPLORATION_PLACEHOLDER, tool_call_id="a")
    assert builder_node_module._prune_stale_exploration_results({"a": already_pruned}) == 0


def test_builder_node_agentic_prunes_an_earlier_turns_exploration_result_after_real_progress(tmp_path, monkeypatch):
    # The actual fix for the real "compounding turn fetch" problem (2026-09-28, Douglas's direct
    # instruction, wanted fixed now rather than deferred): once real progress (a substantive write)
    # happens, an EARLIER turn's exploration result must stop being resent in full on every
    # subsequent turn - replaced with a short placeholder instead, never deleted (would break
    # tool_call_id pairing).
    monkeypatch.setattr(builder_node_module, "MAX_AGENT_TURNS", 30)  # needs more turns than the cap
    monkeypatch.setattr(builder_node_module, "PRUNE_STALE_EXPLORATION_MID_ATTEMPT", True)
    monkeypatch.setattr(builder_node_module, "_MAX_EXPLORATION_CALLS_BEFORE_FORCED_WRITE", 7)
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("x = 1\n", encoding="utf-8")

    llm = _MessageCapturingAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call("read_file", "1", path="a.py")]),  # turn 1: exploration
        _FakeToolResponse(tool_calls=[_tool_call(  # turn 2: substantive write - should prune turn 1's read
            "write_file", "2", path="module_a.py", content="def do_thing():\n    return 1\n",
        )]),
        _FakeToolResponse(tool_calls=[_tool_call("list_files", "3", path=".")]),  # turn 3: exploration again
        _FakeToolResponse(content="Done."),  # turn 4
    ])
    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "done"
    assert len(llm.invoke_message_contents) == 4

    # Turn 2's call (right after turn 1's read_file) still saw turn 1's real content - nothing had
    # been pruned yet at that point (pruning only happens once a SUBSEQUENT substantive write lands).
    assert any("x = 1" in str(c) for c in llm.invoke_message_contents[1])

    # By turn 3 (called AFTER turn 2's substantive write), turn 1's read_file result must have been
    # replaced - the real file content is gone from what's being resent, but the placeholder is there.
    turn_3_contents = llm.invoke_message_contents[2]
    assert not any("x = 1" in str(c) for c in turn_3_contents)
    assert any(c == builder_node_module._PRUNED_EXPLORATION_PLACEHOLDER for c in turn_3_contents)

    # Turn 3's OWN exploration result (the list_files call) is NOT pruned yet by the final call -
    # nothing substantive happened after it within this attempt to trigger that.
    final_contents = llm.invoke_message_contents[3]
    assert any(
        c is not None and c != builder_node_module._PRUNED_EXPLORATION_PLACEHOLDER and "module_a.py" in str(c)
        for c in final_contents
    )


def test_builder_node_agentic_prunes_on_escalation_even_with_no_substantive_write_ever(tmp_path, monkeypatch):
    # The real failure shape a real creative-team rebuild demonstrated (2026-09-28): 3 full attempts,
    # 30 turns, 798,485 tokens, NOTHING substantive ever written - so a prune trigger tied only to
    # "a substantive write happened" (see the test above) would never have fired even once across
    # the whole run, and tokens would have compounded completely unchecked. Escalating is itself
    # proof this stretch was unproductive, so it must ALSO trigger a prune, independent of whether a
    # write ever comes.
    monkeypatch.setattr(builder_node_module, "PRUNE_STALE_EXPLORATION_MID_ATTEMPT", True)
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("x = 1\n", encoding="utf-8")

    n = builder_node_module._MAX_EXPLORATION_CALLS_BEFORE_FORCED_WRITE
    responses = (
        [_FakeToolResponse(tool_calls=[_tool_call("read_file", f"e{i}", path="a.py" if i == 0 else ".")])
         for i in range(1)]  # one real read_file call, whose content should get pruned once escalation fires
        + [_FakeToolResponse(tool_calls=[_tool_call("list_files", f"e{i}", path=str(i))]) for i in range(1, n)]
        + [_FakeToolResponse(tool_calls=[_tool_call(
            "write_file", "w", path="smoke_run.py",
            content="import json\nprint(json.dumps({'ran': True, 'output_summary': 'ok'}))\n",
        )])]
        + [_FakeToolResponse(content="Done.")]
    )
    llm = _MessageCapturingAgentLLM(responses)
    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "done"

    # By the FINAL call, the very first read_file's real content must be gone - pruned at the moment
    # escalation fired, even though no substantive write had happened yet at that point.
    final_contents = llm.invoke_message_contents[-1]
    assert not any("x = 1" in str(c) for c in final_contents)
    assert any(c == builder_node_module._PRUNED_EXPLORATION_PLACEHOLDER for c in final_contents)


def test_builder_node_agentic_short_circuits_a_duplicate_read(tmp_path, monkeypatch):
    # Real transcript: the exact same site-packages/langgraph/graph/state.py was read twice,
    # 46,000+ tokens apart, with nothing having changed in between. The second read of an
    # unchanged file must never reach the real tool again.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("x = 1\n", encoding="utf-8")

    llm = _ScriptedAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call("read_file", "1", path="a.py")]),
        _FakeToolResponse(tool_calls=[_tool_call("read_file", "2", path="a.py")]),  # same path again
        _FakeToolResponse(tool_calls=[_tool_call(
            "write_file", "w", path="smoke_run.py",
            content="import json\nprint(json.dumps({'ran': True, 'output_summary': 'ok'}))\n",
        )]),
        _FakeToolResponse(content="Done."),
    ])
    real_dispatch = builder_node_module._dispatch_tool_call
    dispatched_names = []

    def spying_dispatch(tools_by_name, tool_call):
        dispatched_names.append(tool_call["name"])
        return real_dispatch(tools_by_name, tool_call)

    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_dispatch_tool_call", spying_dispatch)

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "done"
    # The second read_file call was short-circuited before ever reaching the real dispatch.
    assert dispatched_names.count("read_file") == 1


def test_exploration_call_signature_treats_equivalent_root_paths_as_the_same_call():
    # Real transcript evidence (2026-09-28): list_files('.'), list_files(''), list_files('/'), then
    # list_files('.') again - four cosmetically different spellings of "the project root", none of
    # them caught by any guard because none of them were byte-identical strings.
    sig = builder_node_module._exploration_call_signature
    assert sig("list_files", {"path": "."}) == sig("list_files", {"path": ""})
    assert sig("list_files", {"path": "."}) == sig("list_files", {"path": "/"})
    assert sig("list_files", {"path": "src"}) != sig("list_files", {"path": "."})
    assert sig("read_file", {"path": "."}) is None  # read_file keeps its own separate, simpler guard


def test_builder_node_agentic_short_circuits_a_duplicate_list_files_with_an_equivalent_path(tmp_path, monkeypatch):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    llm = _ScriptedAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call("list_files", "1", path=".")]),
        _FakeToolResponse(tool_calls=[_tool_call("list_files", "2", path="")]),  # equivalent, not identical
        _FakeToolResponse(tool_calls=[_tool_call("list_files", "3", path="/")]),  # also equivalent
        _FakeToolResponse(tool_calls=[_tool_call(
            "write_file", "w", path="smoke_run.py",
            content="import json\nprint(json.dumps({'ran': True, 'output_summary': 'ok'}))\n",
        )]),
        _FakeToolResponse(content="Done."),
    ])
    real_dispatch = builder_node_module._dispatch_tool_call
    dispatched_names = []

    def spying_dispatch(tools_by_name, tool_call):
        dispatched_names.append(tool_call["name"])
        return real_dispatch(tools_by_name, tool_call)

    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_dispatch_tool_call", spying_dispatch)

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "done"
    # Only the FIRST of the three equivalent calls ever reached the real dispatch.
    assert dispatched_names.count("list_files") == 1


def test_builder_node_agentic_does_not_confuse_a_different_search_pattern_at_the_same_path(tmp_path, monkeypatch):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("def foo():\n    pass\ndef bar():\n    pass\n", encoding="utf-8")

    llm = _ScriptedAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call("search_files", "1", pattern="foo", path=".")]),
        _FakeToolResponse(tool_calls=[_tool_call("search_files", "2", pattern="bar", path=".")]),  # different pattern, same path
        _FakeToolResponse(tool_calls=[_tool_call(
            "write_file", "w", path="smoke_run.py",
            content="import json\nprint(json.dumps({'ran': True, 'output_summary': 'ok'}))\n",
        )]),
        _FakeToolResponse(content="Done."),
    ])
    real_dispatch = builder_node_module._dispatch_tool_call
    dispatched_names = []

    def spying_dispatch(tools_by_name, tool_call):
        dispatched_names.append(tool_call["name"])
        return real_dispatch(tools_by_name, tool_call)

    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_dispatch_tool_call", spying_dispatch)

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "done"
    # A genuinely different pattern is NOT a duplicate, even at the same path - both must dispatch.
    assert dispatched_names.count("search_files") == 2


def test_builder_node_agentic_allows_rereading_after_a_write_clears_the_cache(tmp_path, monkeypatch):
    # The duplicate-read guard must not become a permanent block - once any file is written or
    # edited, a genuinely-changed file (or any other) must be freely re-readable again.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("x = 1\n", encoding="utf-8")

    llm = _ScriptedAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call("read_file", "1", path="a.py")]),
        _FakeToolResponse(tool_calls=[_tool_call(
            "write_file", "w", path="smoke_run.py",
            content="import json\nprint(json.dumps({'ran': True, 'output_summary': 'ok'}))\n",
        )]),
        _FakeToolResponse(tool_calls=[_tool_call("read_file", "2", path="a.py")]),  # re-read after a write elsewhere
        _FakeToolResponse(content="Done."),
    ])
    real_dispatch = builder_node_module._dispatch_tool_call
    dispatched_names = []

    def spying_dispatch(tools_by_name, tool_call):
        dispatched_names.append(tool_call["name"])
        return real_dispatch(tools_by_name, tool_call)

    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_dispatch_tool_call", spying_dispatch)

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "done"
    # Both read_file calls reached the real dispatch - the write in between cleared the cache.
    assert dispatched_names.count("read_file") == 2


def test_describe_tool_call_shows_the_real_command_for_run_command():
    # Real bug this fixes (2026-09-27): the original logging only ever printed a "path" key, so
    # every run_command call showed up as the useless run_command('') - its real argument is
    # named "command".
    result = builder_node_module._describe_tool_call({"name": "run_command", "args": {"command": "pytest -q"}})
    assert result == "'pytest -q'"


def test_describe_tool_call_shows_pattern_and_path_for_search_files():
    result = builder_node_module._describe_tool_call({"name": "search_files", "args": {"pattern": "foo", "path": "bar"}})
    assert "pattern='foo'" in result
    assert "path='bar'" in result


def test_describe_tool_call_shows_path_and_length_for_write_file():
    result = builder_node_module._describe_tool_call({"name": "write_file", "args": {"path": "a.py", "content": "xyz"}})
    assert "'a.py'" in result
    assert "3 chars" in result


def test_describe_tool_call_falls_back_to_path_for_other_tools():
    result = builder_node_module._describe_tool_call({"name": "list_files", "args": {"path": "sub"}})
    assert result == "'sub'"


def test_dispatch_tool_call_reports_an_unknown_tool_name(tmp_path):
    tools_by_name = {t.name: t for t in make_agent_tools(tmp_path)}
    result = builder_node_module._dispatch_tool_call(tools_by_name, {"name": "delete_everything", "args": {}})
    assert "unknown tool" in result


def test_dispatch_tool_call_catches_an_unexpected_tool_exception(tmp_path):
    class _BoomTool:
        name = "boom"

        def invoke(self, args):
            raise RuntimeError("kaboom")

    result = builder_node_module._dispatch_tool_call({"boom": _BoomTool()}, {"name": "boom", "args": {}})
    assert "unexpected error" in result
    assert "kaboom" in result


def test_with_retries_tool_call_accepts_empty_content_when_tool_calls_present():
    # The actual bug this exists to prevent: with_retries_nonempty would treat a legitimate,
    # empty-content tool-calling response as a failure and retry it needlessly.
    response = _FakeToolResponse(content="", tool_calls=[_tool_call("read_file", "1", path="a.py")])
    assert with_retries_tool_call(lambda: response) is response


def test_with_retries_tool_call_still_retries_a_genuinely_empty_response():
    calls = []

    def fn():
        calls.append(1)
        return _FakeToolResponse(content="", tool_calls=[])

    with pytest.raises(RuntimeError):
        with_retries_tool_call(fn, max_attempts=2, base_delay=0)
    assert len(calls) == 2


def test_augment_spec_is_unchanged_without_any_approved_decisions(tmp_path):
    spec = "SPEC: build a thing."
    assert _augment_spec_with_scale_adaptive_context(spec, str(tmp_path)) == spec


def test_augment_spec_folds_in_approved_architect_and_pm_decisions(tmp_path):
    (tmp_path / ".architect_decisions.json").write_text('{"system_shape": "two workflows"}', encoding="utf-8")
    (tmp_path / ".pm_epics.json").write_text('{"epics": ["onboarding"]}', encoding="utf-8")
    augmented = _augment_spec_with_scale_adaptive_context("SPEC: build a thing.", str(tmp_path))
    assert "SPEC: build a thing." in augmented
    assert "ARCHITECT DECISIONS" in augmented and "two workflows" in augmented
    assert "PM EPICS & STORIES" in augmented and "onboarding" in augmented


def test_augment_spec_with_no_project_dir_is_a_no_op():
    assert _augment_spec_with_scale_adaptive_context("SPEC text", None) == "SPEC text"


def test_augment_spec_strips_build_units_when_requested(tmp_path):
    # Real incident (2026-09-29): the whole Architect decision - including EVERY unit's own
    # build_units entry, not just the one this call actually needs - got resent every turn of
    # every attempt (it lives in the first message, never pruned the way exploration results are).
    # ~55-62K tokens per turn on a real project, tripping the 500K per-attempt cap in 9 turns for a
    # change of a few lines. build_units is redundant with what _format_unit_task already hands
    # forward precisely (this unit's own contract, its dependencies' contracts) - only IT should be
    # stripped, never data_placement/credential_architecture/interface_contracts/tech_quirks, which
    # are genuinely cross-cutting and not unit-specific.
    (tmp_path / ".architect_decisions.json").write_text(json.dumps({
        "summary": "A creative studio.",
        "system_shape": {
            "workflow_count": 1,
            "workflows": [{
                "name": "creative-studio", "platform": "python",
                "build_units": [{"name": "storage", "purpose": "..."}, {"name": "config", "purpose": "..."}],
            }],
        },
        "data_placement": "Local disk under output_root.",
        "credential_architecture": "Single OpenRouter key via env var.",
    }), encoding="utf-8")

    augmented = _augment_spec_with_scale_adaptive_context("SPEC text", str(tmp_path), strip_build_units=True)

    # The actual per-unit DATA is gone - the word "build_units" itself may still legitimately
    # appear once, in the explanatory replacement note (checked separately below).
    assert '"name": "storage"' not in augmented
    assert '"name": "config"' not in augmented
    # Everything else - genuinely cross-cutting, not unit-specific - must survive.
    assert "Local disk under output_root" in augmented
    assert "Single OpenRouter key via env var" in augmented
    assert "YOUR UNIT THIS ROUND" in augmented  # the replacement note pointing at where it actually lives now


def test_augment_spec_keeps_build_units_by_default(tmp_path):
    # No regression for n8n or a non-decomposed python project - the default must be unchanged.
    (tmp_path / ".architect_decisions.json").write_text(json.dumps({
        "system_shape": {"workflows": [{"name": "x", "build_units": [{"name": "storage"}]}]},
    }), encoding="utf-8")
    augmented = _augment_spec_with_scale_adaptive_context("SPEC text", str(tmp_path))
    assert '"name": "storage"' in augmented


def test_is_project_complete_true_for_real_completion_states():
    assert _is_project_complete("pass", partial=False) is True
    assert _is_project_complete("failed_after_max_rounds", partial=False) is True


def test_is_project_complete_false_for_partial_or_ambiguous():
    assert _is_project_complete("pass", partial=True) is False
    assert _is_project_complete("failed_after_max_rounds", partial=True) is False
    assert _is_project_complete("ambiguous", partial=False) is False


def _verified_telegram_workflow():
    triggers = [{
        "name": "Receive Telegram Message",
        "production_type": "n8n-nodes-base.telegramTrigger",
        "production_parameters": {"updates": ["message"]},
    }]
    nodes = build_trigger_and_adapter_nodes(triggers, "TestFlow")
    workflow = {
        "name": "TestFlow",
        "nodes": [nodes["Receive Telegram Message"], nodes["[Trigger Adapter] Receive Telegram Message"], {"name": "Handle", "type": "n8n-nodes-base.code"}],
        "connections": {},
    }
    return workflow, triggers


def test_build_production_workflow_swaps_only_the_trigger_and_its_adapter():
    workflow, triggers = _verified_telegram_workflow()
    produced = build_production_workflow(workflow, triggers, "TestFlow")

    trigger_node = next(n for n in produced["nodes"] if n["name"] == "Receive Telegram Message")
    assert trigger_node["type"] == "n8n-nodes-base.telegramTrigger"
    assert trigger_node["parameters"] == {"updates": ["message"]}

    adapter_node = next(n for n in produced["nodes"] if n["name"] == "[Trigger Adapter] Receive Telegram Message")
    assert adapter_node["parameters"]["jsCode"] == n8n_verify_module.VERIFY_ADAPTER_JS  # no override for telegram (yet)

    # Everything else - the untouched downstream node, the original object - is left alone.
    other_node = next(n for n in produced["nodes"] if n["name"] == "Handle")
    assert other_node == {"name": "Handle", "type": "n8n-nodes-base.code"}
    original_trigger_node = next(n for n in workflow["nodes"] if n["name"] == "Receive Telegram Message")
    assert original_trigger_node["type"] == "n8n-nodes-base.webhook"  # the source workflow is never mutated


def test_build_production_workflow_gives_a_webhook_trigger_a_fresh_path():
    triggers = [{"name": "Hook", "production_type": "n8n-nodes-base.webhook", "production_parameters": {}}]
    nodes = build_trigger_and_adapter_nodes(triggers, "TestFlow")
    workflow = {"nodes": [nodes["Hook"], nodes["[Trigger Adapter] Hook"]], "connections": {}}

    produced = build_production_workflow(workflow, triggers, "TestFlow")
    trigger_node = next(n for n in produced["nodes"] if n["name"] == "Hook")
    # The production path must never be the deterministic, already-exercised verify-mode path.
    assert trigger_node["parameters"]["path"] != nodes["Hook"]["parameters"]["path"]


def test_build_production_workflow_is_a_no_op_without_recorded_triggers():
    workflow, _ = _verified_telegram_workflow()
    produced = build_production_workflow(workflow, [], "TestFlow")
    assert produced == workflow


def test_bootstrap_manifest_names_required_credentials_and_trigger_shape():
    workflow, triggers = _verified_telegram_workflow()
    produced = build_production_workflow(workflow, triggers, "TestFlow")
    manifest = bootstrap_manifest("TestFlow", triggers, produced)

    assert manifest["credentials_required"] == ["telegramApi"]
    assert manifest["triggers"] == [{
        "trigger": "Receive Telegram Message",
        "production_type": "n8n-nodes-base.telegramTrigger",
        "production_parameters": {"updates": ["message"]},
    }]
    assert manifest["instructions"]  # a human installing this workflow always gets a checklist


def test_delimited_plan_parses_content_that_breaks_json():
    # Unescaped quotes, braces and backslashes in prose are exactly what broke the JSON plan
    # format (one bad quote 29,000 chars in cost the whole plan and all its retries).
    plan = _parse_n8n_plan(DELIMITED_PLAN)
    assert plan["workflow_name"] == "Test Flow"
    assert plan["conventions"]["timezone"] == "Asia/Singapore"
    assert any("chat_id" in rule for rule in plan["conventions"]["shared_rules"])
    # Node names containing a colon must survive, and multi-line purposes must be kept whole.
    assert [n["name"] for n in plan["nodes"]] == ["Cron: 7am Check-in", "Weather Is Good?", "Seed Library"]
    assert "Second line of prose" in plan["nodes"][0]["purpose"]


def test_delimited_plan_keeps_branch_output_indices():
    plan = _parse_n8n_plan(DELIMITED_PLAN)
    branches = [c for c in plan["connections"] if c["from"] == "Weather Is Good?"]
    assert sorted(c["from_output"] for c in branches) == [0, 1]


def test_delimited_plan_slices_cover_every_node():
    plan = _parse_n8n_plan(DELIMITED_PLAN)
    assert [s["name"] for s in plan["slices"]] == ["checkin", "setup"]
    assert sorted(n for s in plan["slices"] for n in s["nodes"]) == sorted(n["name"] for n in plan["nodes"])


def test_plan_parser_handles_legacy_json_ambiguous_and_garbage():
    legacy = json.dumps({"workflow_name": "Old", "nodes": [{"name": "A", "type": "t", "purpose": "p"}]})
    assert _parse_n8n_plan(legacy)["workflow_name"] == "Old"
    assert _parse_n8n_plan("AMBIGUOUS: which timezone?")["ambiguous_question"] == "which timezone?"
    try:
        _parse_n8n_plan("prose with no blocks at all")
    except ValueError:
        pass
    else:
        raise AssertionError("Unparseable plan must raise, not silently produce an empty plan")


def _plan_with_slices():
    return {
        "nodes": [{"name": n, "type": "n8n-nodes-base.code", "purpose": "p"} for n in ["A", "B", "C", "D"]],
        "slices": [
            {"name": "checkin", "purpose": "daily check-in", "nodes": ["A", "B"]},
            {"name": "routing", "purpose": "route selection", "nodes": ["C"]},
        ],
    }


def test_slices_cover_every_node_even_when_plan_forgets_one():
    plan = _plan_with_slices()
    slices = _slices_from_plan(plan, plan["nodes"])
    assert [s["name"] for s in slices] == ["checkin", "routing", "unassigned"]
    # "D" was in no slice - it must still get built, not silently dropped.
    assert slices[-1]["nodes"] == ["D"]
    assert sorted(n for s in slices for n in s["nodes"]) == ["A", "B", "C", "D"]


def test_slices_fall_back_to_one_slice_when_plan_has_none():
    nodes = [{"name": "A", "type": "n8n-nodes-base.code", "purpose": "p"}]
    slices = _slices_from_plan({}, nodes)
    assert len(slices) == 1 and slices[0]["nodes"] == ["A"]


def test_until_slice_resolves_by_name_number_and_default():
    plan = _plan_with_slices()
    slices = _slices_from_plan(plan, plan["nodes"])
    assert _resolve_until_slice(slices, "checkin") == 0
    assert _resolve_until_slice(slices, "ROUTING") == 1  # case-insensitive
    assert _resolve_until_slice(slices, "1") == 0  # 1-based number
    assert _resolve_until_slice(slices, None) == len(slices) - 1  # default: build everything
    assert _resolve_until_slice(slices, "nonexistent") == len(slices) - 1  # never silently build nothing


def test_conventions_block_renders_rules_and_is_empty_when_absent():
    rendered = _format_conventions({"timezone": "Asia/Singapore", "shared_rules": ["Score is 1-10"]})
    assert "Asia/Singapore" in rendered
    assert "Score is 1-10" in rendered
    assert _format_conventions({}) == ""


def test_check_interface_contracts_flags_missing_discriminator_value(tmp_path):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "shell.py").write_text('response_type = "prompt"\n', encoding="utf-8")
    decisions = {
        "interface_contracts": [
            {
                "between": "n8n shell and LangGraph service",
                "discriminator_field": "response_type",
                "discriminator_values": ["text", "file"],
            }
        ]
    }
    findings = check_interface_contracts(code_dir, decisions)
    assert len(findings) == 1
    assert "response_type" in findings[0]
    assert "['text', 'file']" in findings[0]


def test_check_interface_contracts_passes_when_all_values_present(tmp_path):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "shell.py").write_text('response_type = "text"\n', encoding="utf-8")
    (code_dir / "service.py").write_text('RESPONSE_KIND = "file"\n', encoding="utf-8")
    decisions = {
        "interface_contracts": [
            {
                "between": "n8n shell and LangGraph service",
                "discriminator_field": "response_type",
                "discriminator_values": ["text", "file"],
            }
        ]
    }
    assert check_interface_contracts(code_dir, decisions) == []


def test_check_interface_contracts_noop_with_no_contracts_or_missing_dir(tmp_path):
    assert check_interface_contracts(tmp_path, {}) == []
    assert check_interface_contracts(tmp_path / "nonexistent", {"interface_contracts": [{"discriminator_values": ["x"]}]}) == []


def test_reconsideration_engaged_rejects_verbatim_repeat_the_real_incident():
    # This recreates the actual reported failure: asked to reconsider, the Architect's re-run
    # produced the byte-for-byte original decision, never even mentioning the new information (n8n).
    previous = {"summary": "Standalone Python/LangGraph, no n8n involvement.", "system_shape": {"workflow_count": 1}}
    new = {"summary": "Standalone Python/LangGraph, no n8n involvement.", "system_shape": {"workflow_count": 1}}
    engaged, problem = reconsideration_engaged(previous, new, "Douglas's n8n instance is already on an always-on VPS")
    assert engaged is False
    assert "identical" in problem


def test_reconsideration_engaged_rejects_answer_never_mentioning_new_terms():
    previous = {"summary": "Standalone Python/LangGraph recommended."}
    new = {"summary": "Standalone Python/LangGraph remains the best choice for state management reasons."}
    engaged, problem = reconsideration_engaged(previous, new, "Douglas's n8n instance is already on an always-on VPS")
    assert engaged is False
    assert "never mentions" in problem


def test_reconsideration_engaged_accepts_a_response_that_addresses_the_new_reason():
    previous = {"summary": "Standalone Python/LangGraph recommended."}
    new = {"summary": "Hybrid: thin n8n shell on the existing always-on VPS, LangGraph service owns state."}
    engaged, _ = reconsideration_engaged(previous, new, "Douglas's n8n instance is already on an always-on VPS")
    assert engaged is True


def _decision_with_workflow(build_units=None, data_placement="S3 bucket per client", summary="Initial decision."):
    return {
        "summary": summary,
        "system_shape": {
            "workflow_count": 1,
            "workflows": [{"name": "Main", "scope": "the whole product", "platform": "python", "build_units": build_units or []}],
        },
        "data_placement": data_placement,
        "credential_architecture": "per-client isolation",
        "interface_contracts": [],
        "regression_strategy": "unit tests per module",
        "total_tokens": 12345,
    }


def test_only_build_units_changed_accepts_when_only_build_units_and_summary_differ():
    # The actual approved upgrade path: add build_units, touch nothing else.
    previous = _decision_with_workflow(build_units=[], summary="Original.")
    new = _decision_with_workflow(
        build_units=[{"name": "storage", "purpose": "...", "interface": {"exports": []}, "depends_on": [], "verification_hint": "..."}],
        summary="Added build_units per the upgrade reason - nothing else changed.",
    )
    new["total_tokens"] = 999  # always different, always expected
    in_scope, _ = only_build_units_changed(previous, new)
    assert in_scope is True


def test_only_build_units_changed_rejects_when_something_else_also_changed():
    # The real risk this exists to catch: the model also quietly revises an already-approved field
    # it wasn't asked to touch.
    previous = _decision_with_workflow(build_units=[], data_placement="S3 bucket per client")
    new = _decision_with_workflow(
        build_units=[{"name": "storage", "purpose": "...", "interface": {"exports": []}, "depends_on": [], "verification_hint": "..."}],
        data_placement="Postgres, multi-tenant",  # NOT asked to change this
    )
    in_scope, explanation = only_build_units_changed(previous, new)
    assert in_scope is False
    assert "ONLY add build_units" in explanation


def test_only_build_units_changed_handles_a_previous_decision_with_no_build_units_key_at_all():
    # The exact real-world shape this was built for: an existing project's decision file predates
    # the build_units field entirely (no key, not even an empty list) - confirmed via a real
    # project's actual .architect_decisions.json, not assumed.
    previous = {
        "summary": "Initial decision.",
        "system_shape": {"workflow_count": 1, "workflows": [{"name": "Main", "scope": "the whole product", "platform": "python"}]},
        "data_placement": "S3 bucket per client", "credential_architecture": "per-client isolation",
        "interface_contracts": [], "regression_strategy": "unit tests per module", "total_tokens": 12345,
    }
    new = _decision_with_workflow(
        build_units=[{"name": "storage", "purpose": "...", "interface": {"exports": []}, "depends_on": [], "verification_hint": "..."}],
        summary="Added build_units - nothing else changed.",
    )
    in_scope, _ = only_build_units_changed(previous, new)
    assert in_scope is True


def test_only_build_units_changed_does_not_mutate_its_inputs():
    previous = _decision_with_workflow(build_units=[])
    previous_copy = json.loads(json.dumps(previous))
    new = _decision_with_workflow(build_units=[{"name": "x", "purpose": "", "interface": {"exports": []}, "depends_on": [], "verification_hint": ""}])
    only_build_units_changed(previous, new)
    assert previous == previous_copy


def _real_unit(name, depends_on=None):
    return {
        "name": name,
        "purpose": f"{name} module",
        "files": [f"{name}.py"],
        "effort_points": 1,
        "interface": {"exports": [{"name": name.title(), "kind": "class", "signature": f"class {name.title()}: ...", "description": "..."}]},
        "depends_on": depends_on or [],
        "verification_hint": f"a real call into {name} produces the expected real output",
    }


def test_validate_build_units_rejects_too_many_files_exports_or_effort():
    unit = _real_unit("big")
    unit["files"] = [f"f{i}.py" for i in range(4)]
    assert any("files" in p for p in validate_build_units([unit]))

    unit = _real_unit("wide")
    unit["interface"]["exports"] = [
        {"name": f"F{i}", "kind": "function", "signature": "f()", "description": "."} for i in range(6)
    ]
    assert any("exports" in p for p in validate_build_units([unit]))

    for bad in (0, 4, None, "2", True):
        unit = _real_unit("effort")
        unit["effort_points"] = bad
        assert any("effort_points" in p for p in validate_build_units([unit])), bad


def test_build_limits_sheet_reports_live_role_caps_and_sprint_capacity():
    sheet = build_limits_sheet(None)
    assert "builder:" in sheet and "max output" in sheet
    assert "64,000" in sheet
    assert "sprint capacity" in sheet


def test_validate_build_units_accepts_a_well_formed_decomposition():
    units = [_real_unit("storage"), _real_unit("validation", depends_on=["storage"])]
    assert validate_build_units(units) == []


def test_validate_build_units_is_a_no_op_for_an_empty_or_missing_list():
    # Optional - a project/workflow that doesn't need this is not an error.
    assert validate_build_units([]) == []
    assert validate_build_units(None) == []


def test_validate_build_units_rejects_a_unit_with_no_exports():
    unit = _real_unit("storage")
    unit["interface"] = {"exports": []}
    problems = validate_build_units([unit])
    assert len(problems) == 1
    assert "storage" in problems[0] and "exports" in problems[0]


def test_validate_build_units_rejects_a_unit_whose_exports_are_all_constants():
    unit = _real_unit("config")
    unit["interface"] = {"exports": [{"name": "DEFAULT_TIMEOUT", "kind": "constant", "signature": "30", "description": "..."}]}
    problems = validate_build_units([unit])
    assert len(problems) == 1
    assert "constants" in problems[0]


def test_validate_build_units_rejects_a_missing_verification_hint():
    unit = _real_unit("storage")
    unit["verification_hint"] = ""
    problems = validate_build_units([unit])
    assert len(problems) == 1
    assert "verification_hint" in problems[0]


def test_validate_build_units_rejects_a_forward_reference():
    # "storage" listed first but claiming to depend on "validation", which comes AFTER it -
    # mechanically rejected regardless of whether "validation" exists anywhere in the list.
    units = [_real_unit("storage", depends_on=["validation"]), _real_unit("validation")]
    problems = validate_build_units(units)
    assert len(problems) == 1
    assert "storage" in problems[0] and "validation" in problems[0]


def test_validate_build_units_rejects_a_self_reference():
    units = [_real_unit("storage", depends_on=["storage"])]
    problems = validate_build_units(units)
    assert len(problems) == 1


def test_validate_build_units_rejects_a_duplicate_name():
    units = [_real_unit("storage"), _real_unit("storage")]
    problems = validate_build_units(units)
    assert any("duplicates" in p for p in problems)


def test_resolve_until_unit_none_or_all_builds_everything():
    units = [_real_unit("A"), _real_unit("B"), _real_unit("C")]
    assert resolve_until_unit(units, None) == 2
    assert resolve_until_unit(units, "all") == 2


def test_resolve_until_unit_matches_by_name_or_number():
    units = [_real_unit("A"), _real_unit("B"), _real_unit("C")]
    assert resolve_until_unit(units, "B") == 1
    assert resolve_until_unit(units, "2") == 1


def test_resolve_until_unit_falls_back_to_everything_when_unmatched():
    units = [_real_unit("A"), _real_unit("B")]
    assert resolve_until_unit(units, "nonexistent") == 1


class _FakeResponse:
    def __init__(self, content, response_metadata=None):
        self.content = content
        self.response_metadata = response_metadata or {}


def test_with_retries_nonempty_retries_past_a_transient_empty_response():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) == 1:
            return _FakeResponse("")
        return _FakeResponse("real content")

    result = with_retries_nonempty(fn, max_attempts=3, base_delay=0)
    assert result.content == "real content"
    assert len(calls) == 2


def test_with_retries_nonempty_raises_after_exhausting_all_attempts_on_empty():
    # This recreates the actual reported incident: HTTP 200, empty content, every attempt.
    with pytest.raises(RuntimeError, match="empty/whitespace content"):
        with_retries_nonempty(lambda: _FakeResponse("   "), max_attempts=2, base_delay=0)


def test_get_active_fallback_is_none_by_default(tmp_path):
    assert get_active_fallback(tmp_path, "reviewer") is None


def test_attempt_fallback_succeeds_and_persists_for_the_rest_of_the_project(tmp_path, monkeypatch):
    # A fallback is durably logged via record_decision - must never write to the real repo-wide
    # decision log from a test, only to a throwaway path.
    monkeypatch.setattr(decision_log_module, "DECISION_LOG_PATH", tmp_path / "decision_log.jsonl")
    # "pm" has no hard vendor constraint (unlike reviewer or, since 2026-09-29, business_analyst),
    # so this exercises the plain success path.
    response = attempt_fallback(
        "pm",
        lambda fb_llm: _FakeResponse("fallback worked"),
        project_dir=tmp_path,
        primary_error=RuntimeError("primary model is down"),
    )
    assert response.content == "fallback worked"
    assert get_active_fallback(tmp_path, "pm") == FALLBACK_MODEL

    # "Not just that output, that entire project": a later resolve for the same role must pick up
    # the fallback directly, without needing another failure first.
    llm = resolve_llm_for_role("pm", project_dir=tmp_path)
    assert getattr(llm, "model_name", None) == FALLBACK_MODEL

    clear_fallback(tmp_path, "pm")
    assert get_active_fallback(tmp_path, "pm") is None


def test_attempt_fallback_without_project_dir_is_one_off_and_persists_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_log_module, "DECISION_LOG_PATH", tmp_path / "decision_log.jsonl")
    response = attempt_fallback(
        "preference_analyst",
        lambda fb_llm: _FakeResponse("one-off fallback"),
        project_dir=None,
        primary_error=RuntimeError("gemini is down"),
    )
    assert response.content == "one-off fallback"


def test_attempt_fallback_raises_combined_error_when_fallback_also_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_log_module, "DECISION_LOG_PATH", tmp_path / "decision_log.jsonl")

    def always_fails(fb_llm):
        raise RuntimeError("claude fable is also down")

    with pytest.raises(RuntimeError, match="also failed"):
        attempt_fallback(
            "pm", always_fails, project_dir=tmp_path, primary_error=RuntimeError("primary is down")
        )
    # A failed fallback must not be recorded as if it succeeded.
    assert get_active_fallback(tmp_path, "pm") is None


def test_attempt_fallback_blocks_reviewer_when_it_would_collide_with_builder_vendor(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_log_module, "DECISION_LOG_PATH", tmp_path / "decision_log.jsonl")
    # A project whose builder is an Anthropic model (project-local override; the global default is
    # OpenAI now) is the same vendor as FALLBACK_MODEL - falling reviewer back to Claude Fable would
    # silently put it on the same vendor as the builder, breaking vendor independence.
    (tmp_path / "models.yaml").write_text(
        "roles:\n  builder:\n    model: \"anthropic/claude-sonnet-5\"\n    temperature: 1\n    max_tokens: 4096\n",
        encoding="utf-8",
    )
    with pytest.raises(FallbackBlockedError, match="vendor-independence"):
        attempt_fallback(
            "reviewer", lambda fb_llm: _FakeResponse("should not get here"),
            project_dir=tmp_path, platform="python", primary_error=RuntimeError("qwen is down"),
        )
    assert get_active_fallback(tmp_path, "reviewer") is None


def test_attempt_fallback_allows_reviewer_when_builder_is_a_different_vendor(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_log_module, "DECISION_LOG_PATH", tmp_path / "decision_log.jsonl")
    (tmp_path / "models.yaml").write_text(
        "roles:\n  builder:\n    model: \"deepseek/deepseek-v4-pro\"\n    temperature: 0.1\n    max_tokens: 4096\n",
        encoding="utf-8",
    )
    response = attempt_fallback(
        "reviewer", lambda fb_llm: _FakeResponse("fallback ok, no collision"),
        project_dir=tmp_path, platform="python", primary_error=RuntimeError("qwen is down"),
    )
    assert response.content == "fallback ok, no collision"
    assert get_active_fallback(tmp_path, "reviewer") == FALLBACK_MODEL


def test_attempt_fallback_blocks_business_analyst_since_the_fallback_is_anthropic(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_log_module, "DECISION_LOG_PATH", tmp_path / "decision_log.jsonl")
    # Real incident (2026-09-29, creative-team's concept bounce): this happened live, twice, and
    # silently - business_analyst's primary hit a real, unrelated error (context-window overflow),
    # fell back to FALLBACK_MODEL (Anthropic) without a word, and the round quietly stopped being a
    # genuinely independent second opinion and became Claude arguing with itself. Unconditional,
    # unlike reviewer's check - FALLBACK_MODEL is always Anthropic today, and concept_advocate is
    # always Anthropic too, so this is not a sometimes-collision like builder/reviewer's.
    with pytest.raises(FallbackBlockedError, match="Claude arguing with itself"):
        attempt_fallback(
            "business_analyst", lambda fb_llm: _FakeResponse("should not get here"),
            project_dir=tmp_path, platform="python", primary_error=RuntimeError("context window exceeded"),
        )
    assert get_active_fallback(tmp_path, "business_analyst") is None


def test_resolve_llm_for_role_prefers_explicit_override_over_active_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_log_module, "DECISION_LOG_PATH", tmp_path / "decision_log.jsonl")
    attempt_fallback(
        "builder", lambda fb_llm: _FakeResponse("fallback engaged"),
        project_dir=tmp_path, primary_error=RuntimeError("sonnet is down"),
    )
    llm = resolve_llm_for_role("builder", project_dir=tmp_path, model_override="anthropic/claude-opus-5")
    assert getattr(llm, "model_name", None) == "anthropic/claude-opus-5"


def _write_architect_decision(project_dir, workflows):
    (project_dir / ".architect_decisions.json").write_text(
        json.dumps({"system_shape": {"workflow_count": len(workflows), "workflows": workflows}}),
        encoding="utf-8",
    )


def test_slugify_handles_spaces_punctuation_and_case():
    assert slugify("Creative Director") == "creative-director"
    assert slugify("  Text/Team!! ") == "text-team"
    assert slugify("already-a-slug") == "already-a-slug"


def test_slugify_falls_back_to_workflow_for_empty_name():
    assert slugify("") == "workflow"
    assert slugify("   ") == "workflow"


def test_n8n_workflows_is_empty_with_no_architect_decision(tmp_path):
    assert n8n_workflows(tmp_path) == []


def test_n8n_workflows_filters_to_n8n_platform_only(tmp_path):
    _write_architect_decision(tmp_path, [
        {"name": "Telegram shell", "scope": "trigger+delivery", "platform": "n8n"},
        {"name": "Core service", "scope": "orchestration", "platform": "python"},
    ])
    workflows = n8n_workflows(tmp_path)
    assert len(workflows) == 1
    assert workflows[0]["name"] == "Telegram shell"


def test_resolve_workflow_slug_is_none_none_with_zero_or_one_n8n_workflow(tmp_path):
    assert resolve_workflow_slug(tmp_path, None) == (None, None)
    assert resolve_workflow_slug(tmp_path, "anything") == (None, None)

    _write_architect_decision(tmp_path, [{"name": "Only One", "scope": "s", "platform": "n8n"}])
    assert resolve_workflow_slug(tmp_path, None) == (None, None)


def test_resolve_workflow_slug_errors_when_ambiguous_and_no_workflow_given(tmp_path):
    # Recreates the actual reported gap: 3 same-platform n8n workflows, no way to tell them apart.
    _write_architect_decision(tmp_path, [
        {"name": "Creative Director", "scope": "s1", "platform": "n8n"},
        {"name": "Text Team", "scope": "s2", "platform": "n8n"},
        {"name": "Image Team", "scope": "s3", "platform": "n8n"},
    ])
    slug, problem = resolve_workflow_slug(tmp_path, None)
    assert slug is None
    assert "Creative Director" in problem and "Text Team" in problem and "Image Team" in problem
    assert "--workflow" in problem


def test_resolve_workflow_slug_matches_by_name_or_slug_case_insensitively(tmp_path):
    _write_architect_decision(tmp_path, [
        {"name": "Creative Director", "scope": "s1", "platform": "n8n"},
        {"name": "Text Team", "scope": "s2", "platform": "n8n"},
    ])
    slug, problem = resolve_workflow_slug(tmp_path, "Creative Director")
    assert (slug, problem) == ("creative-director", None)

    slug, problem = resolve_workflow_slug(tmp_path, "text-team")
    assert (slug, problem) == ("text-team", None)


def test_resolve_workflow_slug_errors_on_unknown_workflow_name(tmp_path):
    _write_architect_decision(tmp_path, [
        {"name": "Creative Director", "scope": "s1", "platform": "n8n"},
        {"name": "Text Team", "scope": "s2", "platform": "n8n"},
    ])
    slug, problem = resolve_workflow_slug(tmp_path, "Video Team")
    assert slug is None
    assert "Video Team" in problem and "Creative Director" in problem


def test_n8n_subdir_legacy_vs_scoped(tmp_path):
    assert n8n_subdir(tmp_path, None) == tmp_path / "n8n"
    assert n8n_subdir(tmp_path, "creative-director") == tmp_path / "n8n" / "creative-director"


def test_checkpoint_path_is_unchanged_filename_when_no_workflow(tmp_path):
    assert checkpoint.checkpoint_path(tmp_path) == tmp_path / ".build_checkpoint.json"
    assert checkpoint.checkpoint_path(tmp_path, workflow=None) == tmp_path / ".build_checkpoint.json"


def test_checkpoint_path_is_scoped_per_workflow(tmp_path):
    assert checkpoint.checkpoint_path(tmp_path, workflow="creative-director") == (
        tmp_path / ".build_checkpoint.creative-director.json"
    )


def test_save_load_state_keeps_two_workflows_checkpoints_independent(tmp_path):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("shared spec for both workflows", encoding="utf-8")

    state_a = {
        "project_dir": str(tmp_path), "spec_path": str(spec_path), "n8n_workflow": "creative-director",
        "review_round": 1, "total_tokens": 100,
    }
    state_b = {
        "project_dir": str(tmp_path), "spec_path": str(spec_path), "n8n_workflow": "text-team",
        "review_round": 3, "total_tokens": 999,
    }
    checkpoint.save_state(state_a)
    checkpoint.save_state(state_b)

    loaded_a = checkpoint.load_state(tmp_path, spec_path.read_text(encoding="utf-8"), workflow="creative-director")
    loaded_b = checkpoint.load_state(tmp_path, spec_path.read_text(encoding="utf-8"), workflow="text-team")
    assert loaded_a["review_round"] == 1 and loaded_a["total_tokens"] == 100
    assert loaded_b["review_round"] == 3 and loaded_b["total_tokens"] == 999

    checkpoint.clear(tmp_path, workflow="creative-director")
    assert checkpoint.load_state(tmp_path, spec_path.read_text(encoding="utf-8"), workflow="creative-director") is None
    # Clearing one workflow's checkpoint must never touch the other's.
    still_there = checkpoint.load_state(tmp_path, spec_path.read_text(encoding="utf-8"), workflow="text-team")
    assert still_there["review_round"] == 3


def test_wipe_n8n_artifacts_is_scoped_per_workflow_and_does_not_touch_others(tmp_path):
    for wf in ("creative-director", "text-team"):
        wf_dir = tmp_path / "n8n" / wf
        wf_dir.mkdir(parents=True)
        (wf_dir / "workflow.json").write_text("{}", encoding="utf-8")

    removed = wipe_n8n_artifacts(tmp_path, workflow="creative-director")
    assert any("creative-director" in r for r in removed)
    assert not (tmp_path / "n8n" / "creative-director" / "workflow.json").exists()
    # The other workflow's own artifact must survive untouched.
    assert (tmp_path / "n8n" / "text-team" / "workflow.json").exists()


def test_conventions_and_production_triggers_paths_are_scoped_per_workflow(tmp_path):
    assert _conventions_path(tmp_path) == tmp_path / "n8n" / ".conventions.json"
    assert _conventions_path(tmp_path, workflow="text-team") == tmp_path / "n8n" / "text-team" / ".conventions.json"
    assert _production_triggers_path(tmp_path, workflow="text-team") == (
        tmp_path / "n8n" / "text-team" / ".production_triggers.json"
    )


def test_augment_spec_tells_builder_which_workflow_to_build_when_multiple_exist(tmp_path):
    _write_architect_decision(tmp_path, [
        {"name": "Creative Director", "scope": "s1", "platform": "n8n"},
        {"name": "Text Team", "scope": "s2", "platform": "n8n"},
    ])
    augmented = _augment_spec_with_scale_adaptive_context("ORIGINAL SPEC", tmp_path, workflow="text-team")
    assert "ORIGINAL SPEC" in augmented
    assert "text-team" in augmented
    assert "TARGETS EXACTLY ONE WORKFLOW" in augmented.upper()


def test_augment_spec_has_no_workflow_scoping_note_when_workflow_not_given(tmp_path):
    _write_architect_decision(tmp_path, [
        {"name": "Creative Director", "scope": "s1", "platform": "n8n"},
        {"name": "Text Team", "scope": "s2", "platform": "n8n"},
    ])
    augmented = _augment_spec_with_scale_adaptive_context("ORIGINAL SPEC", tmp_path)
    assert "THIS BUILD TARGETS" not in augmented.upper()


def test_deploy_one_legacy_single_workflow_path(tmp_path):
    (tmp_path / "code" / "n8n").mkdir(parents=True)
    (tmp_path / "code" / "n8n" / "workflow.json").write_text(
        json.dumps({"name": "Only Workflow", "nodes": [], "connections": {}}), encoding="utf-8",
    )
    assert _deploy_one(tmp_path, None) is True
    assert (tmp_path / "code" / "n8n" / "deploy" / "workflow.json").exists()
    assert (tmp_path / "code" / "n8n" / "deploy" / "bootstrap_manifest.json").exists()


def test_deploy_one_scoped_workflow_path(tmp_path):
    wf_dir = tmp_path / "code" / "n8n" / "text-team"
    wf_dir.mkdir(parents=True)
    (wf_dir / "workflow.json").write_text(
        json.dumps({"name": "Text Team", "nodes": [], "connections": {}}), encoding="utf-8",
    )
    assert _deploy_one(tmp_path, "text-team") is True
    assert (wf_dir / "deploy" / "workflow.json").exists()
    # A sibling workflow that was never built must not get a deploy folder from this call.
    assert not (tmp_path / "code" / "n8n" / "creative-director").exists()


def test_deploy_one_returns_false_when_nothing_built_yet(tmp_path):
    assert _deploy_one(tmp_path, "not-built-yet") is False


def _write_pm_epics(project_dir, epic_names):
    (project_dir / ".pm_epics.json").write_text(
        json.dumps({"summary": "s", "epics": [
            {"name": n, "description": f"desc for {n}", "stories": [{"story": "s1", "priority": "high"}]}
            for n in epic_names
        ]}),
        encoding="utf-8",
    )


def test_load_pm_epics_is_empty_with_no_pm_file(tmp_path):
    assert load_pm_epics(tmp_path) == []


def test_load_pm_epics_reads_ordered_epics(tmp_path):
    _write_pm_epics(tmp_path, ["Creative Director", "Text Team", "Image Team"])
    epics = load_pm_epics(tmp_path)
    assert [e["name"] for e in epics] == ["Creative Director", "Text Team", "Image Team"]


def test_next_unbuilt_stage_index_is_zero_with_no_prior_run(tmp_path):
    assert next_unbuilt_stage_index(tmp_path, 4) == 0


def test_next_unbuilt_stage_index_advances_past_what_was_already_built(tmp_path):
    (tmp_path / "run_result.json").write_text(
        json.dumps({"status": "pass_partial", "slices": {"total": 4, "built": ["A", "B"], "remaining": ["C", "D"]}}),
        encoding="utf-8",
    )
    assert next_unbuilt_stage_index(tmp_path, 4) == 2


def test_next_unbuilt_stage_index_resets_when_stage_count_no_longer_matches(tmp_path):
    # The epics were re-planned since the last run (different stage count) - a stale "already
    # built" count could otherwise skip real, never-actually-built work.
    (tmp_path / "run_result.json").write_text(
        json.dumps({"status": "pass_partial", "slices": {"total": 3, "built": ["A", "B"], "remaining": ["C"]}}),
        encoding="utf-8",
    )
    assert next_unbuilt_stage_index(tmp_path, 5) == 0


def test_augment_spec_no_longer_accepts_stage_scope(tmp_path):
    # 2026-09-28, Douglas's direct instruction: python's PM-epic/checkpoint stage gating is retired
    # entirely, replaced by Architect-led build_units decomposition - confirming the retirement is
    # real, not just unused.
    import inspect
    assert "stage_scope" not in inspect.signature(builder_node_module._augment_spec_with_scale_adaptive_context).parameters


def test_format_unit_task_foregrounds_the_concrete_unit_and_its_contract():
    unit = {
        "name": "storage", "purpose": "Persistence layer.", "files": ["storage.py"],
        "interface": {"exports": [{"name": "InMemoryStorage", "kind": "class", "signature": "class InMemoryStorage: ...", "description": "..."}]},
        "verification_hint": "save() then load() of the same key returns equal data",
    }
    task = builder_node_module._format_unit_task("ORIGINAL SPEC", unit, {}, "")
    assert "ORIGINAL SPEC" in task
    assert "YOUR UNIT THIS ROUND: 'storage'" in task
    assert "InMemoryStorage" in task
    assert "save() then load()" in task


def test_format_unit_task_hands_forward_a_dependencys_real_interface():
    # The actual mechanism that prevents the InMemoryStorage/InMemoryStorageBackend class of bug -
    # a later unit is HANDED the earlier unit's real shape, not left to read its file and guess.
    unit = {"name": "validation", "purpose": "...", "depends_on": ["storage"], "interface": {"exports": []}, "verification_hint": "..."}
    prior_interfaces = {"storage": {"exports": [{"name": "InMemoryStorage", "kind": "class", "signature": "...", "description": "..."}]}}
    task = builder_node_module._format_unit_task("SPEC", unit, prior_interfaces, "")
    assert "INTERFACES OF UNITS THIS ONE DEPENDS ON" in task
    assert "InMemoryStorage" in task
    assert "never invent your own name/shape" in task


def test_format_unit_task_includes_relevant_journal_text_when_present():
    task = builder_node_module._format_unit_task("SPEC", {"name": "storage", "interface": {"exports": []}}, {}, "BUILD JOURNAL (...)\n- [storage, builder] Built it.")
    assert "BUILD JOURNAL" in task


def test_format_unit_task_puts_the_unit_block_and_journal_before_the_full_spec():
    # The real incident (2026-09-29): journal_text used to sit at the very end of the prompt, after
    # the whole spec (tens of thousands of characters on a real project) - a peer flagged this as the
    # most concrete lead for why a precisely diagnosed, verified one-line fix kept not landing.
    huge_spec = "SPEC-CONTENT-MARKER-" + ("x" * 5000)
    unit = {"name": "image_team_subgraph", "purpose": "...", "interface": {"exports": []}, "verification_hint": "..."}
    task = builder_node_module._format_unit_task(huge_spec, unit, {}, "BUILD JOURNAL - READ THIS FIRST\n- known issue")
    unit_pos = task.index("YOUR UNIT THIS ROUND")
    journal_pos = task.index("BUILD JOURNAL")
    spec_pos = task.index("SPEC-CONTENT-MARKER")
    assert unit_pos < journal_pos < spec_pos


def test_check_unit_interface_compliance_passes_when_the_export_exists(tmp_path):
    (tmp_path / "storage.py").write_text("class InMemoryStorage:\n    pass\n", encoding="utf-8")
    unit = {"files": ["storage.py"], "interface": {"exports": [{"name": "InMemoryStorage", "kind": "class"}]}}
    assert builder_node_module._check_unit_interface_compliance(tmp_path, unit) == []


def test_check_unit_interface_compliance_catches_the_real_incident(tmp_path):
    # The exact real incident this exists to catch: the builder passed its own verification but
    # built a DIFFERENT class name than the contract declared.
    (tmp_path / "storage.py").write_text("class InMemoryStorageBackend:\n    pass\n", encoding="utf-8")
    unit = {"files": ["storage.py"], "interface": {"exports": [{"name": "InMemoryStorage", "kind": "class"}]}}
    problems = builder_node_module._check_unit_interface_compliance(tmp_path, unit)
    assert len(problems) == 1
    assert "InMemoryStorage" in problems[0]


def test_check_unit_interface_compliance_falls_back_to_scanning_the_whole_tree_without_files(tmp_path):
    (tmp_path / "somewhere_else.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    unit = {"interface": {"exports": [{"name": "helper", "kind": "function"}]}}  # no "files" declared
    assert builder_node_module._check_unit_interface_compliance(tmp_path, unit) == []


def test_check_unit_interface_compliance_falls_back_when_declared_files_dont_exist_at_that_path(tmp_path):
    # The real creative-team incident (2026-09-28): the Architect's hint gave bare filenames, but the
    # builder correctly nested them under a package directory - the export is really there, just not
    # at the literal hinted path. Must NOT be reported as missing.
    package_dir = tmp_path / "creative_studio"
    package_dir.mkdir()
    (package_dir / "storage_interface.py").write_text(
        "from abc import ABC\n\nclass StorageBackend(ABC):\n    pass\n", encoding="utf-8"
    )
    unit = {
        "files": ["storage_interface.py", "in_memory_storage.py"],  # wrong - no package prefix
        "interface": {"exports": [{"name": "StorageBackend", "kind": "class"}]},
    }
    assert builder_node_module._check_unit_interface_compliance(tmp_path, unit) == []


def test_check_unit_interface_compliance_still_catches_a_real_miss_when_declared_files_are_wrong(tmp_path):
    # Same wrong-hint shape as above, but this time the export genuinely doesn't exist anywhere -
    # the tree-scan fallback must not paper over an actually-missing export.
    package_dir = tmp_path / "creative_studio"
    package_dir.mkdir()
    (package_dir / "storage_interface.py").write_text("class SomethingUnrelated:\n    pass\n", encoding="utf-8")
    unit = {
        "files": ["storage_interface.py"],  # wrong - no package prefix, and the export isn't there anyway
        "interface": {"exports": [{"name": "StorageBackend", "kind": "class"}]},
    }
    problems = builder_node_module._check_unit_interface_compliance(tmp_path, unit)
    assert len(problems) == 1
    assert "StorageBackend" in problems[0]


def test_check_unit_interface_compliance_catches_an_uncompiled_signature_that_actually_compiles(tmp_path):
    # The real creative-team incident (2026-09-29): content_writer.py declared "-> StateGraph"
    # (uncompiled) but actually returned graph.compile() (a CompiledStateGraph) - burned a full
    # round's 3 retries never converging. image_team.py had the identical, still-latent bug.
    (tmp_path / "content_writer.py").write_text(
        "from langgraph.graph import StateGraph\n\n"
        "def build_content_writer_subgraph() -> StateGraph:\n"
        "    graph = StateGraph(dict)\n"
        "    return graph.compile()\n",
        encoding="utf-8",
    )
    unit = {
        "files": ["content_writer.py"],
        "interface": {"exports": [{
            "name": "build_content_writer_subgraph", "kind": "function",
            "signature": "def build_content_writer_subgraph() -> StateGraph",
        }]},
    }
    problems = builder_node_module._check_unit_interface_compliance(tmp_path, unit)
    assert len(problems) == 1
    assert "build_content_writer_subgraph" in problems[0]
    assert "CompiledStateGraph" in problems[0]


def test_check_unit_interface_compliance_passes_a_correctly_uncompiled_stategraph(tmp_path):
    # The two sibling units that got it right (writer_gatekeeper.py, creative_director.py in the
    # real incident) - must not be flagged just because the signature mentions StateGraph.
    (tmp_path / "creative_director.py").write_text(
        "from langgraph.graph import StateGraph\n\n"
        "def build_creative_director_graph() -> StateGraph:\n"
        "    graph = StateGraph(dict)\n"
        "    return graph\n",
        encoding="utf-8",
    )
    unit = {
        "files": ["creative_director.py"],
        "interface": {"exports": [{
            "name": "build_creative_director_graph", "kind": "function",
            "signature": "def build_creative_director_graph() -> StateGraph",
        }]},
    }
    assert builder_node_module._check_unit_interface_compliance(tmp_path, unit) == []


def test_check_unit_interface_compliance_allows_compile_when_the_signature_says_so(tmp_path):
    # A signature that itself declares "-> CompiledStateGraph" is correctly satisfied by a
    # .compile() call - the check must only fire on the specific uncompiled-but-compiles mismatch.
    (tmp_path / "image_team.py").write_text(
        "from langgraph.graph import StateGraph\n\n"
        "def build_image_team_graph() -> CompiledStateGraph:\n"
        "    graph = StateGraph(dict)\n"
        "    return graph.compile()\n",
        encoding="utf-8",
    )
    unit = {
        "files": ["image_team.py"],
        "interface": {"exports": [{
            "name": "build_image_team_graph", "kind": "function",
            "signature": "def build_image_team_graph() -> CompiledStateGraph",
        }]},
    }
    assert builder_node_module._check_unit_interface_compliance(tmp_path, unit) == []


def _real_build_unit(name, depends_on=None):
    return {
        "name": name, "purpose": f"{name} module", "files": [f"{name}.py"],
        "interface": {"exports": [{"name": name.title(), "kind": "class", "signature": "...", "description": "..."}]},
        "depends_on": depends_on or [], "verification_hint": "a real behavioral check",
    }


def test_build_python_units_builds_in_order_and_hands_forward_interfaces(tmp_path, monkeypatch):
    # End-to-end proof of the actual mechanism that prevents the InMemoryStorage/
    # InMemoryStorageBackend class of bug: the second unit's task must be HANDED the first unit's
    # real, declared interface directly - not left to read its file and guess.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    build_units = [_real_build_unit("storage"), _real_build_unit("validation", depends_on=["storage"])]
    captured_tasks = []

    class _CapturingLLM:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages, config=None):
            self.calls += 1
            captured_tasks.append(messages[1].content)
            if self.calls == 1:
                return _FakeToolResponse(tool_calls=[_tool_call("write_file", "1", path="storage.py", content="class Storage:\n    pass\n")])
            if self.calls == 3:
                return _FakeToolResponse(tool_calls=[_tool_call("write_file", "2", path="validation.py", content="class Validation:\n    pass\n")])
            return _FakeToolResponse(content="Done.")

    llm = _CapturingLLM()
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    state = _agentic_state(tmp_path, spec_path, code_dir)
    result = builder_node_module._build_python_units(state, llm, "config", "BASE SPEC", code_dir, build_units, None)

    assert result["builder_status"] == "done"
    assert result["python_units_built"] == ["storage", "validation"]
    assert (code_dir / "storage.py").exists() and (code_dir / "validation.py").exists()
    # captured_tasks[2] is "validation"'s own first turn - it must have been handed "storage"'s
    # real declared interface directly.
    assert "Storage" in captured_tasks[2]
    assert "INTERFACES OF UNITS THIS ONE DEPENDS ON" in captured_tasks[2]


def test_build_python_units_skips_an_already_built_unit_not_named_in_feedback(tmp_path, monkeypatch):
    # The n8n-style "reuse the existing artifact, rebuild only what was actually named" principle,
    # adapted to units - re-verifying (and re-paying tokens for) an untouched unit every round would
    # be exactly the waste this whole mechanism exists to avoid.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "storage.py").write_text("class Storage:\n    pass\n", encoding="utf-8")

    build_units = [_real_build_unit("storage"), _real_build_unit("validation", depends_on=["storage"])]

    llm = _ScriptedAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call("write_file", "1", path="validation.py", content="class Validation:\n    pass\n")]),
        _FakeToolResponse(content="Done."),
    ])
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    state = _agentic_state(tmp_path, spec_path, code_dir, python_units_built=["storage"], review_problems=[])
    result = builder_node_module._build_python_units(state, llm, "config", "BASE SPEC", code_dir, build_units, None)

    assert result["builder_status"] == "done"
    assert result["python_units_built"] == ["storage", "validation"]
    # llm was only ever asked to build "validation" - "storage" was skipped entirely, not re-verified.
    assert llm._calls == 2


def test_build_python_units_does_not_persist_a_unit_that_just_failed_its_rebuild(tmp_path, monkeypatch):
    # The real creative-team incident (2026-09-29): content_writer_subgraph passed round 1; the
    # reviewer later flagged a real problem with it; round 2's rebuild attempt FAILED after
    # exhausting its attempts. newly_built was seeded from already_built (this unit's OWN prior-round
    # success) at the top of the function, and the failure return path never removed the unit
    # currently being rebuilt - so python_units_built still claimed it was built. A resume then
    # skipped it entirely, even though the whole reason it was rebuilding was that it wasn't fine.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "content_writer.py").write_text("class ContentWriter:\n    pass\n", encoding="utf-8")

    build_units = [_real_build_unit("content_writer")]

    class _NeverStopsLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages, config=None):
            return _FakeToolResponse(tool_calls=[_tool_call("list_files", "x", path=".")])

    pytest_calls = []

    def _fake_run_pytest(code_dir):
        pytest_calls.append(1)
        return False, f"failure variant {len(pytest_calls)}"  # different every call - no dedup short-circuit

    monkeypatch.setattr(builder_node_module, "_run_pytest", _fake_run_pytest)

    state = _agentic_state(
        tmp_path, spec_path, code_dir,
        python_units_built=["content_writer"],  # already built in an earlier round
        review_problems=["content_writer: real bug found by reviewer"],  # this round rebuilds it
    )
    result = builder_node_module._build_python_units(
        state, _NeverStopsLLM(), "config", "BASE SPEC", code_dir, build_units, None,
    )

    assert result["builder_status"] == "ambiguous"
    assert "content_writer" not in result["python_units_built"]


def test_build_python_units_does_not_persist_a_unit_that_fails_compliance_after_a_rebuild(tmp_path, monkeypatch):
    # Same stale-success bug, via the OTHER failure path (_check_unit_interface_compliance instead of
    # builder_status != "done") - must be fixed on both return paths, not just one.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    build_units = [_real_build_unit("storage")]  # declares export "Storage"

    llm = _ScriptedAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call("write_file", "1", path="storage.py", content="class WrongName:\n    pass\n")]),
        _FakeToolResponse(content="Done."),
    ])
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    state = _agentic_state(
        tmp_path, spec_path, code_dir,
        python_units_built=["storage"],  # already built in an earlier round
        review_problems=["storage: interface mismatch, please fix"],  # this round rebuilds it
    )
    result = builder_node_module._build_python_units(state, llm, "config", "BASE SPEC", code_dir, build_units, None)

    assert result["builder_status"] == "ambiguous"
    assert "storage" not in result["python_units_built"]


def test_problem_node_extracts_the_leading_node_before_first_colon():
    assert builder_node_module._problem_node("image_team.py: some description with more: colons") == "image_team.py"


def test_node_belongs_to_unit_matches_exact_name_and_file_but_not_a_bare_mention():
    assert builder_node_module._node_belongs_to_unit("storage", "storage", ["storage.py"]) is True
    assert builder_node_module._node_belongs_to_unit("storage.py", "storage", ["storage.py"]) is True
    # Basename match - a real on-disk path nested under a package dir the Architect's hint didn't
    # anticipate must still be recognized as belonging to this unit.
    assert builder_node_module._node_belongs_to_unit("creative_studio/storage.py", "storage", ["storage.py"]) is True
    assert builder_node_module._node_belongs_to_unit("image_team.py", "storage", ["storage.py"]) is False


def test_node_belongs_to_unit_matches_a_function_name_defined_in_a_declared_file(tmp_path):
    # Real incident (2026-09-29, creative-team): a reviewer's final-round findings named
    # call_llm/content_writer_node/_validate_request/image_composition_node - function names, never
    # file paths or a unit's own name - so neither existing check ever matched, and the units that
    # actually contain them were silently skipped as "not named in reviewer feedback."
    (tmp_path / "content_writer.py").write_text("def content_writer_node():\n    pass\n", encoding="utf-8")
    assert builder_node_module._node_belongs_to_unit(
        "content_writer_node", "content_writer", ["content_writer.py"], tmp_path,
    ) is True
    assert builder_node_module._node_belongs_to_unit(
        "content_writer_node", "storage", ["storage.py"], tmp_path,
    ) is False


def test_node_belongs_to_unit_finds_a_function_name_via_a_files_real_nested_location(tmp_path):
    # Same nested-path lenience _check_unit_interface_compliance already applies - the declared file
    # hint may not match where the builder actually put it.
    package_dir = tmp_path / "creative_studio"
    package_dir.mkdir()
    (package_dir / "content_writer.py").write_text("def content_writer_node():\n    pass\n", encoding="utf-8")
    assert builder_node_module._node_belongs_to_unit(
        "content_writer_node", "content_writer", ["content_writer.py"], tmp_path,
    ) is True


def test_node_belongs_to_unit_without_code_dir_is_unchanged(tmp_path):
    # Omitting code_dir (the default) must preserve the exact prior behavior - no new false positive
    # for a caller that hasn't been updated to pass it.
    (tmp_path / "content_writer.py").write_text("def content_writer_node():\n    pass\n", encoding="utf-8")
    assert builder_node_module._node_belongs_to_unit("content_writer_node", "content_writer", ["content_writer.py"]) is False


def test_node_belongs_to_unit_matches_a_dot_qualified_method_name(tmp_path):
    # Real creative-team incident (2026-09-29): a reviewer's finding named "Config.get_api_key"
    # specifically - a METHOD, not a top-level name - so _declared_names_in_file (top-level defs
    # only) never found it literally. The prefix before the first dot ("Config") IS top-level and
    # IS what should be checked.
    (tmp_path / "context.py").write_text(
        "class Config:\n    def get_api_key(self):\n        pass\n", encoding="utf-8",
    )
    assert builder_node_module._node_belongs_to_unit(
        "Config.get_api_key", "config", ["context.py"], tmp_path,
    ) is True


def test_build_python_units_does_not_rebuild_a_unit_merely_named_in_an_unrelated_findings_text(tmp_path):
    # The real creative-team incident (2026-09-29): a test-coverage-gap finding about image_team.py
    # lists EVERY built unit's name for context ("despite epic(s) reported built as of this round:
    # ...storage..."). The old crude `name in review_problems_text` substring check concluded
    # "storage" had been flagged - nothing was wrong with it. Consequence: storage got rebuilt for no
    # reason, burning the round's entire token budget and its own 2-identical-failures dedup cap on a
    # unit that was never broken, while the ACTUALLY broken unit never got a dedicated retry at all.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "storage.py").write_text("class Storage:\n    pass\n", encoding="utf-8")

    build_units = [_real_build_unit("storage")]

    class _ShouldNeverBeCalledLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages, config=None):
            raise AssertionError("storage was never broken - it should have been skipped, not rebuilt")

    state = _agentic_state(
        tmp_path, spec_path, code_dir,
        python_units_built=["storage"],
        review_problems=[
            "creative_studio/image_team.py: reviewer did not engage this mechanical finding (no "
            "correctness_bug reported, no verified dismissal) - creative_studio/image_team.py: "
            "contains real function/class definitions, but no test_image_team.py exists and "
            "'image_team' is not referenced in any test file's content - no apparent test coverage "
            "for this module, despite epic(s) reported built as of this round: content_writer_subgraph, "
            "context, creative_director_graph, image_team_subgraph, storage, writer_gatekeeper_subgraph."
        ],
    )
    result = builder_node_module._build_python_units(
        state, _ShouldNeverBeCalledLLM(), "config", "BASE SPEC", code_dir, build_units, None,
    )

    assert result["builder_status"] == "done"
    # Contract breakage is now caught during the attempts (retry-first), so neither the unit whose
    # attempt failed nor the damaged one survives as built.
    assert "content_writer" not in result["python_units_built"]  # correctly left untouched


def test_build_python_units_still_rebuilds_a_unit_genuinely_named_in_feedback(tmp_path, monkeypatch):
    # The precise match must still correctly trigger a real rebuild - not just correctly reject the
    # false positive above.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "storage.py").write_text("class Storage:\n    pass\n", encoding="utf-8")

    build_units = [_real_build_unit("storage")]
    llm = _ScriptedAgentLLM([_FakeToolResponse(content="Done.")])
    monkeypatch.setattr(builder_node_module, "_finding_change_problems", lambda code_dir, baseline: [])
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    state = _agentic_state(
        tmp_path, spec_path, code_dir,
        python_units_built=["storage"],
        review_problems=["storage.py: genuine bug, wrong return value"],
    )
    result = builder_node_module._build_python_units(state, llm, "config", "BASE SPEC", code_dir, build_units, None)

    assert result["builder_status"] == "done"
    assert llm._calls == 1  # it WAS actually invoked to rebuild storage


def test_build_python_units_mechanically_recognizes_an_already_built_unit_with_empty_state(tmp_path, monkeypatch):
    # Real creative-team incident (2026-09-29): --until-unit N re-scopes the checkpoint FILE per N
    # (checkpoint.py's own _stage_scope_key), so a --until-unit 4 invocation can never find a
    # --until-unit 3 invocation's own checkpoint and starts with python_units_built genuinely EMPTY -
    # re-attempting units 1-3 from scratch at ~370K-500K tokens apiece for zero real work, even
    # though their real, on-disk code already, genuinely satisfies their own declared contracts.
    # Requires a REAL build-journal entry (not just a name match - see the next test for why).
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "storage.py").write_text("class Storage:\n    pass\n", encoding="utf-8")  # already built for real
    build_journal_module.record_journal_entry(tmp_path, "storage", "builder", "Built Storage for real.")

    build_units = [_real_build_unit("storage"), _content_writer_unit()]
    # Scripted for exactly ONE unit's own build (content_writer) - if storage were wrongly
    # re-attempted too, it would consume these same responses meant for content_writer instead.
    llm = _ScriptedAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call("write_file", "1", path="content_writer.py", content="def content_writer_node():\n    pass\n")]),
        _FakeToolResponse(content="Done."),
    ])
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    # NOTHING in state - exactly the "no checkpoint found across --until-unit N values" scenario.
    state = _agentic_state(tmp_path, spec_path, code_dir)
    result = builder_node_module._build_python_units(state, llm, "config", "BASE SPEC", code_dir, build_units, None)

    assert result["builder_status"] == "done"
    assert result["python_units_built"] == ["content_writer", "storage"]
    assert llm._calls == 2  # exactly ONE unit's own build (write_file + "Done.") - storage was skipped


def test_build_python_units_does_not_recognize_a_coincidental_name_match_with_no_journal_entry(tmp_path, monkeypatch):
    # The real, second creative-team incident, same day (2026-09-29): an Architect re-split moved
    # "config" out of an earlier, monolithic unit into its own dedicated unit with real, new work
    # (spend-cap fields, a get_api_key fix, output_root) - none of which was ever done. But the OLD
    # context.py's Config class, which PRE-DATED this unit's own existence entirely, already
    # satisfied the name-level check by pure coincidence - so interface compliance ALONE wrongly
    # marked "config" built and skipped it every round. A bare name match must never be trusted
    # without a real build-journal entry proving the pipeline itself actually worked on this unit.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    # Config already exists - but from an EARLIER, unrelated unit, never for "config" itself.
    (code_dir / "context.py").write_text("class Config:\n    pass\n", encoding="utf-8")
    # NO journal entry for "config" - it was never actually, genuinely attempted.

    build_units = [{
        "name": "config", "purpose": "spend caps, API key derivation, output_root", "files": ["config.py"],
        "interface": {"exports": [{"name": "Config", "kind": "class", "signature": "...", "description": "..."}]},
        "depends_on": [], "verification_hint": "a real behavioral check",
    }]
    llm = _ScriptedAgentLLM([_FakeToolResponse(content="Done.")])
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    state = _agentic_state(tmp_path, spec_path, code_dir)
    result = builder_node_module._build_python_units(state, llm, "config", "BASE SPEC", code_dir, build_units, None)

    assert llm._calls >= 1  # it WAS actually invoked - not skipped via a coincidental name match


def test_build_python_units_does_not_mechanically_recognize_a_unit_with_no_real_exports(tmp_path, monkeypatch):
    # Must not over-fire: a unit whose declared interface has no exports at all can never be
    # mechanically verified as built (an empty check trivially "passes" without checking anything
    # real) - it still gets a real attempt.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "storage.py").write_text("class Storage:\n    pass\n", encoding="utf-8")

    build_units = [{
        "name": "storage", "purpose": "storage module", "files": ["storage.py"],
        "interface": {"exports": []}, "depends_on": [], "verification_hint": "a real behavioral check",
    }]
    llm = _ScriptedAgentLLM([_FakeToolResponse(content="Done.")])
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    state = _agentic_state(tmp_path, spec_path, code_dir)
    result = builder_node_module._build_python_units(state, llm, "config", "BASE SPEC", code_dir, build_units, None)

    assert llm._calls == 1  # it WAS actually invoked - not skipped via a trivial empty-exports "pass"


def test_build_python_units_final_return_carries_partial_build_forward(tmp_path, monkeypatch):
    # The real creative-team incident (2026-09-29, video-generator-78): unit_stage_info computed
    # slices_total/slices_built/slices_remaining/partial_build every round but only ever handed
    # them INTO _build_python_agentic, never back OUT via this function's own final return - so
    # reviewer_node.py's own "if state.get('partial_build')" check never fired for a python
    # --until-unit run. The reviewer judged the WHOLE spec against only the built unit's code,
    # burning all 3 rounds (755K tokens, real incident) on complaints naming units that were never
    # in scope, with no code changed at all. The n8n path already does this correctly via its own
    # slice_info - this brings the python path to parity.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    build_units = [_real_build_unit("storage"), _content_writer_unit()]
    llm = _ScriptedAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call("write_file", "1", path="storage.py", content="class Storage:\n    pass\n")]),
        _FakeToolResponse(content="Done."),
    ])
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    state = _agentic_state(tmp_path, spec_path, code_dir)
    result = builder_node_module._build_python_units(state, llm, "config", "BASE SPEC", code_dir, build_units, 0)

    assert result["builder_status"] == "done"
    assert result["partial_build"] is True
    assert result["slices_total"] == 2
    assert result["slices_built"] == ["storage"]
    assert result["slices_remaining"] == ["content_writer"]


def test_build_python_units_final_return_is_not_partial_when_everything_is_built(tmp_path, monkeypatch):
    # Must not over-fire: a genuinely complete run (--until-unit all, or omitted) is not partial.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    build_units = [_real_build_unit("storage")]
    llm = _ScriptedAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call("write_file", "1", path="storage.py", content="class Storage:\n    pass\n")]),
        _FakeToolResponse(content="Done."),
    ])
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    state = _agentic_state(tmp_path, spec_path, code_dir)
    result = builder_node_module._build_python_units(state, llm, "config", "BASE SPEC", code_dir, build_units, None)

    assert result["partial_build"] is False
    assert result["slices_remaining"] == []


def test_reviewer_prompt_contains_partial_build_scope_for_a_python_until_unit_run(tmp_path, monkeypatch):
    # End-to-end, exactly as reported: feed a real _build_python_units partial result into
    # reviewer_node and confirm the DELIBERATE PARTIAL BUILD text actually reaches the reviewer's
    # own prompt - not just that the state dict carries the right keys.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild storage, then content_writer.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    build_units = [_real_build_unit("storage"), _content_writer_unit()]
    builder_llm = _ScriptedAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call("write_file", "1", path="storage.py", content="class Storage:\n    pass\n")]),
        _FakeToolResponse(content="Done."),
    ])
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    state = _agentic_state(tmp_path, spec_path, code_dir)
    build_result = builder_node_module._build_python_units(
        state, builder_llm, "config", "BASE SPEC", code_dir, build_units, 0,
    )

    captured_prompts = []

    class _CapturingReviewerLLM:
        def invoke(self, messages, config=None):
            captured_prompts.append(messages[-1][1])
            return _FakeResponse('{"verdict": "pass", "problems": []}')

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _CapturingReviewerLLM())

    reviewer_node_module.reviewer_node(build_result)

    assert any("DELIBERATE PARTIAL BUILD" in p for p in captured_prompts)
    assert any("storage" in p and "content_writer" in p for p in captured_prompts)


def test_build_python_units_catches_collateral_damage_to_a_different_already_built_unit(tmp_path, monkeypatch):
    # The real creative-team incident (2026-09-29), the third time in a row the model reached into a
    # DIFFERENT unit's file while a different one was nominally in scope - this time outright
    # deleting a required export. Checking only the nominal unit's own contract left this completely
    # undetected: the round reported storage as fine, and content_writer's own broken contract was
    # never checked until (if ever) ITS turn came around again. ~1M tokens spent that round, net
    # negative: a working test suite regressed to a hard collection error.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "storage.py").write_text("class Storage:\n    pass\n", encoding="utf-8")
    (code_dir / "content_writer.py").write_text("def content_writer_node():\n    pass\n", encoding="utf-8")

    build_units = [
        _real_build_unit("storage"),
        {
            "name": "content_writer", "purpose": "content writer", "files": ["content_writer.py"],
            "interface": {"exports": [{
                "name": "content_writer_node", "kind": "function", "signature": "...", "description": "...",
            }]},
            "depends_on": [], "verification_hint": "a real behavioral check",
        },
    ]

    llm = _ScriptedAgentLLM([
        # Nominally rebuilding "storage" - but reaches into content_writer.py instead and deletes
        # its required export.
        _FakeToolResponse(tool_calls=[_tool_call(
            "edit_file", "1", path="content_writer.py",
            old_string="def content_writer_node():\n    pass\n",
            new_string="def content_writer_graph():\n    pass\n",
        )]),
        _FakeToolResponse(content="Done."),
    ])
    monkeypatch.setattr(builder_node_module, "_finding_change_problems", lambda code_dir, baseline: [])
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    state = _agentic_state(
        tmp_path, spec_path, code_dir,
        python_units_built=["storage", "content_writer"],
        review_problems=["storage.py: minor cleanup needed"],  # triggers storage's own rebuild
    )
    result = builder_node_module._build_python_units(state, llm, "config", "BASE SPEC", code_dir, build_units, None)

    assert result["builder_status"] == "ambiguous"
    assert "collateral damage" in result["ambiguity_question"]
    assert "content_writer_node" in result["ambiguity_question"]
    # storage's own contract is genuinely fine and must stay marked built; content_writer's is
    # broken and must NOT survive as built, or a later resume would skip a unit now known to be
    # broken - the exact same bug the earlier stale-success fix closed, via a different path.
    # Contract breakage is now caught during the attempts (retry-first), so neither the unit whose
    # attempt failed nor the damaged one survives as built.
    assert "content_writer" not in result["python_units_built"]


def test_augment_spec_has_no_stage_scoping_note_when_stage_scope_not_given(tmp_path):
    _write_pm_epics(tmp_path, ["Creative Director", "Text Team"])
    augmented = _augment_spec_with_scale_adaptive_context("ORIGINAL SPEC", tmp_path)
    assert "CHECKPOINTED STAGE" not in augmented.upper()


def test_checkpoint_path_is_scoped_per_python_stage(tmp_path):
    assert checkpoint.checkpoint_path(tmp_path, stage="stage0") == tmp_path / ".build_checkpoint.stage0.json"
    assert checkpoint.checkpoint_path(tmp_path, stage="stage1") == tmp_path / ".build_checkpoint.stage1.json"
    assert checkpoint.checkpoint_path(tmp_path) == tmp_path / ".build_checkpoint.json"


def test_unit2_never_resumes_unit1_checkpoint_the_real_incident(tmp_path):
    # Recreates the actual reported incident (originally with PM-epic stages, same mechanism now
    # scoped to Architect-decomposed units instead): unit 1 completes (builder_status "done", no
    # review problems - exactly what resume_stage_for reads as "re-enter at reviewer"), its
    # checkpoint is never scoped, and a later --until-unit 2 invocation loads THAT checkpoint and
    # skips the builder entirely, reporting a clean pass having built nothing.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("shared spec for both units", encoding="utf-8")

    unit1_done_state = {
        "project_dir": str(tmp_path), "spec_path": str(spec_path), "n8n_workflow": None,
        "python_until_unit_index": 0, "builder_status": "done", "review_problems": [],
        "review_round": 1, "total_tokens": 274541,
    }
    checkpoint.save_state(unit1_done_state)

    # Unit 2's own invocation must find NOTHING under its own scope - unit 1's completed
    # checkpoint must be invisible to it.
    loaded_for_unit2 = checkpoint.load_state(
        tmp_path, spec_path.read_text(encoding="utf-8"), workflow=None, stage="unit1",
    )
    assert loaded_for_unit2 is None

    # Unit 1's own re-invocation (same scope) still correctly resumes its own state - the fix is
    # isolation between DIFFERENT scopes, not breaking legitimate same-scope resume.
    loaded_for_unit1 = checkpoint.load_state(
        tmp_path, spec_path.read_text(encoding="utf-8"), workflow=None, stage="unit0",
    )
    assert loaded_for_unit1["builder_status"] == "done"
    assert checkpoint.resume_stage_for(loaded_for_unit1) == "reviewer"


def test_clear_only_removes_its_own_unit_checkpoint(tmp_path):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("spec", encoding="utf-8")
    checkpoint.save_state({
        "project_dir": str(tmp_path), "spec_path": str(spec_path), "python_until_unit_index": 0,
    })
    checkpoint.save_state({
        "project_dir": str(tmp_path), "spec_path": str(spec_path), "python_until_unit_index": 1,
    })
    checkpoint.clear(tmp_path, stage="unit0")
    assert checkpoint.load_state(tmp_path, "spec", stage="unit0") is None
    # Clearing unit 0 must never remove unit 1's own, unrelated checkpoint.
    assert checkpoint.load_state(tmp_path, "spec", stage="unit1") is not None


def test_stub_marker_is_stable_and_names_the_epic():
    marker = stub_marker("Writer Gatekeeper")
    assert "Writer Gatekeeper" in marker
    assert marker == stub_marker("Writer Gatekeeper")  # deterministic, not regenerated differently


def test_check_stage_stub_regressions_the_real_incident(tmp_path):
    # Recreates the actual reported incident: stage 2's headline epic ("Phase 1 Text Production and
    # Independent Review") was reported built, but teams/text_team.py was still, verbatim, the stub
    # written back in stage 1 when that epic was deferred.
    teams_dir = tmp_path / "teams"
    teams_dir.mkdir()
    (teams_dir / "text_team.py").write_text(
        f'"""{stub_marker("Phase 1 Text Production and Independent Review")}"""\n',
        encoding="utf-8",
    )
    findings = check_stage_stub_regressions(tmp_path, ["Phase 1 Text Production and Independent Review"])
    assert len(findings) == 1
    assert findings[0]["epic"] == "Phase 1 Text Production and Independent Review"
    assert findings[0]["confidence"] == "high"
    assert findings[0]["file"] == "teams\\text_team.py" or findings[0]["file"] == "teams/text_team.py"


def test_check_stage_stub_regressions_clean_when_stub_was_actually_replaced(tmp_path):
    teams_dir = tmp_path / "teams"
    teams_dir.mkdir()
    (teams_dir / "text_team.py").write_text(
        "def build_subgraph():\n    return real_content_here()\n", encoding="utf-8",
    )
    assert check_stage_stub_regressions(tmp_path, ["Phase 1 Text Production and Independent Review"]) == []


def test_check_stage_stub_regressions_noop_with_no_epics_or_missing_dir(tmp_path):
    assert check_stage_stub_regressions(tmp_path, []) == []
    assert check_stage_stub_regressions(tmp_path / "nonexistent", ["Epic A"]) == []


def test_check_stage_stub_regressions_catches_the_legacy_stub_that_motivated_the_fix(tmp_path):
    # This recreates the actual real-world gap reported live (2026-09-26): the exact stub that
    # motivated the whole check predates the standardized marker, is semantically identical but
    # textually different, and spans two lines - an exact-substring check alone missed it entirely.
    teams_dir = tmp_path / "teams"
    teams_dir.mkdir()
    (teams_dir / "text_team.py").write_text(
        '"""Content Writer + Writer Gatekeeper subgraph - STUB.\n'
        'Out of scope for this build stage (epic: "Phase 1 Text Production and Independent Review")."""\n',
        encoding="utf-8",
    )
    findings = check_stage_stub_regressions(tmp_path, ["Phase 1 Text Production and Independent Review"])
    assert len(findings) == 1
    assert "text_team.py" in findings[0]["file"]
    assert findings[0]["confidence"] == "low"


def test_check_stage_stub_regressions_exact_marker_is_reported_as_higher_confidence(tmp_path):
    teams_dir = tmp_path / "teams"
    teams_dir.mkdir()
    (teams_dir / "text_team.py").write_text(
        f'"""{stub_marker("Epic A")}"""\n', encoding="utf-8",
    )
    findings = check_stage_stub_regressions(tmp_path, ["Epic A"])
    assert len(findings) == 1
    assert findings[0]["confidence"] == "high"


def test_check_stage_stub_regressions_ignores_unrelated_stub_mentions_far_from_the_epic_name(tmp_path):
    # "stub" is common, legitimate vocabulary in real test code (mocks/stubs) - the window-based
    # check must not flag every file that happens to mention both "stub" and an epic's name
    # somewhere far apart in a large file.
    (tmp_path / "test_helpers.py").write_text(
        "def make_stub_client():\n    return FakeClient()\n\n\n"
        + ("\n" * 50)
        + "# This module also happens to discuss Epic A in an unrelated changelog note far below.\n",
        encoding="utf-8",
    )
    assert check_stage_stub_regressions(tmp_path, ["Epic A"]) == []


def test_check_stage_test_coverage_gaps_flags_a_real_untested_module(tmp_path):
    # Recreates the actual reported incident (2026-09-26/27, creative-team): a real module with
    # genuine logic, shipped with zero corresponding test coverage, across multiple build rounds.
    (tmp_path / "text_team.py").write_text(
        "def build_subgraph():\n    return real_gatekeeper_wiring()\n\n\n"
        "class TextTeam:\n    def run(self, brief):\n        return self.build_subgraph()\n",
        encoding="utf-8",
    )
    findings = check_stage_test_coverage_gaps(tmp_path, ["Text Team"])
    assert len(findings) == 1
    assert findings[0]["file"] == "text_team.py"
    assert findings[0]["confidence"] == "high"


def test_check_stage_test_coverage_gaps_clean_when_a_same_named_test_file_exists(tmp_path):
    (tmp_path / "config.py").write_text("def load():\n    return {}\n", encoding="utf-8")
    (tmp_path / "test_config.py").write_text("def test_load():\n    assert True\n", encoding="utf-8")
    assert check_stage_test_coverage_gaps(tmp_path, ["Epic A"]) == []


def test_check_stage_test_coverage_gaps_clean_when_referenced_in_a_test_files_content(tmp_path):
    # Real case observed on creative-team: conftest.py imports a module by name without a dedicated
    # test_<module>.py existing for it - still a genuine coverage signal, not a gap.
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tmp_path / "director.py").write_text("def build_runtime():\n    return object()\n", encoding="utf-8")
    (tests_dir / "conftest.py").write_text("from director import build_runtime\n", encoding="utf-8")
    assert check_stage_test_coverage_gaps(tmp_path, ["Epic A"]) == []


def test_check_stage_test_coverage_gaps_skips_docstring_only_init_files(tmp_path):
    # A package __init__.py that's purely a docstring/re-export marker has nothing to test - no
    # separate size heuristic needed, the "has a real def/class" check already excludes it.
    pkg = tmp_path / "teams"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        '"""Team entry points.\n\nJust documentation here, nothing to import directly.\n"""\n',
        encoding="utf-8",
    )
    assert check_stage_test_coverage_gaps(tmp_path, ["Epic A"]) == []


def test_check_stage_test_coverage_gaps_flags_an_init_file_with_real_logic(tmp_path):
    # The opposite case: an __init__.py that DOES contain real factory logic must still be checked,
    # not blanket-excluded just for being an __init__.py (real incident: creative-team's own
    # storage/__init__.py and teams/__init__.py both contain real factory functions).
    pkg = tmp_path / "storage"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "def get_storage(config):\n    return LocalDiskStorage(config.path)\n", encoding="utf-8",
    )
    findings = check_stage_test_coverage_gaps(tmp_path, ["Epic A"])
    assert len(findings) == 1
    assert "__init__.py" in findings[0]["file"]


def test_check_stage_test_coverage_gaps_ignores_test_files_and_conftest_themselves(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "conftest.py").write_text("def fixture_thing():\n    return 1\n", encoding="utf-8")
    (tests_dir / "test_something.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    assert check_stage_test_coverage_gaps(tmp_path, ["Epic A"]) == []


def test_check_stage_test_coverage_gaps_noop_with_no_epics_or_missing_dir(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    pass\n", encoding="utf-8")
    assert check_stage_test_coverage_gaps(tmp_path, []) == []
    assert check_stage_test_coverage_gaps(tmp_path / "nonexistent", ["Epic A"]) == []


def test_check_stage_test_coverage_gaps_caps_findings_per_round(tmp_path):
    # Real incident this exists to catch (2026-09-27, creative-team): all 22 accumulated gaps were
    # force-injected as problems in one round, and the builder blew a 500,000-token run budget
    # trying to write real tests for all 22 at once - zero tests saved. A single call must never
    # return more than the per-round cap, no matter how many real gaps actually exist.
    for i in range(9):
        (tmp_path / f"module_{i}.py").write_text(
            f"def do_thing_{i}():\n    return {i}\n", encoding="utf-8",
        )
    findings = check_stage_test_coverage_gaps(tmp_path, ["Epic A"])
    assert len(findings) == 5


def test_check_stage_test_coverage_gaps_prioritizes_real_branching_logic_first(tmp_path):
    # A thin one-line wrapper is a real gap, but a module with real decision logic is more likely to
    # hide an actual bug if it ships untested - that one must win a scarce per-round slot over a
    # bunch of trivial wrappers even when it sorts last alphabetically (proving priority, not just
    # tiebreak order, decided the cut).
    for i in range(7):
        (tmp_path / f"a_trivial_{i}.py").write_text(
            f"def noop_{i}():\n    return {i}\n", encoding="utf-8",
        )
    (tmp_path / "zz_real_logic.py").write_text(
        "def classify(value):\n"
        "    if value is None:\n"
        "        raise ValueError('missing value')\n"
        "    for candidate in range(value):\n"
        "        if candidate % 2 == 0:\n"
        "            continue\n"
        "        try:\n"
        "            assert candidate > 0\n"
        "        except AssertionError:\n"
        "            return None\n"
        "    return value\n",
        encoding="utf-8",
    )
    findings = check_stage_test_coverage_gaps(tmp_path, ["Epic A"])
    assert len(findings) == 5
    assert "zz_real_logic.py" in {f["file"] for f in findings}


def test_check_stage_test_coverage_gaps_notes_deferred_count_in_message(tmp_path):
    for i in range(7):
        (tmp_path / f"module_{i}.py").write_text(
            f"def do_thing_{i}():\n    return {i}\n", encoding="utf-8",
        )
    findings = check_stage_test_coverage_gaps(tmp_path, ["Epic A"])
    assert len(findings) == 5
    assert all("2 additional module" in f["message"] for f in findings)


def test_check_stage_test_coverage_gaps_no_deferred_note_when_under_the_cap(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    pass\n", encoding="utf-8")
    findings = check_stage_test_coverage_gaps(tmp_path, ["Epic A"])
    assert len(findings) == 1
    assert "additional module" not in findings[0]["message"]


def test_check_stage_test_coverage_gaps_shrinks_naturally_as_modules_get_covered(tmp_path):
    # No separate "already queued" bookkeeping needed - re-running after a module gets a real test
    # must show one fewer gap and let the next-highest-priority one take its place, for free.
    for i in range(6):
        (tmp_path / f"module_{i}.py").write_text(
            f"def do_thing_{i}():\n    return {i}\n", encoding="utf-8",
        )
    first_round = check_stage_test_coverage_gaps(tmp_path, ["Epic A"])
    assert len(first_round) == 5
    covered_file = first_round[0]["file"]
    module_stem = covered_file.rsplit(".py", 1)[0]
    (tmp_path / f"test_{module_stem}.py").write_text(
        "def test_it():\n    assert True\n", encoding="utf-8",
    )
    second_round = check_stage_test_coverage_gaps(tmp_path, ["Epic A"])
    assert len(second_round) == 5
    assert covered_file not in {f["file"] for f in second_round}


def test_check_stage_test_coverage_gaps_against_the_real_creative_team_layout(tmp_path):
    # Real-world validation, same standard as the stub check: recreate the actual reported file
    # layout (2 covered modules, N genuinely uncovered ones with real logic) rather than only
    # synthetic single-file cases, so the heuristic is proven against a realistic multi-package tree.
    cs = tmp_path / "creative_studio"
    cs.mkdir()
    (cs / "config.py").write_text("class Config:\n    pass\n", encoding="utf-8")
    (cs / "preflight.py").write_text("def run_preflight():\n    return True\n", encoding="utf-8")
    (cs / "delivery.py").write_text("def deliver(job):\n    return job\n", encoding="utf-8")
    vendors = cs / "vendors"
    vendors.mkdir()
    (vendors / "__init__.py").write_text('"""Vendor package."""\n', encoding="utf-8")
    (vendors / "base.py").write_text("class VendorClient:\n    pass\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_config.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    (tests_dir / "test_preflight.py").write_text("def test_y():\n    assert True\n", encoding="utf-8")

    findings = check_stage_test_coverage_gaps(tmp_path, ["Foundation", "Text Team"])
    flagged_files = {f["file"] for f in findings}
    assert "creative_studio\\delivery.py" in flagged_files or "creative_studio/delivery.py" in flagged_files
    assert any("base.py" in f for f in flagged_files)
    assert not any("config.py" in f for f in flagged_files)
    assert not any("preflight.py" in f for f in flagged_files)
    assert not any(f.endswith("__init__.py") and "vendors" in f for f in flagged_files)


def test_reviewer_node_forces_a_fail_for_an_unengaged_test_coverage_gap(tmp_path, monkeypatch):
    # End-to-end: the test-coverage check routes through the SAME enforcement path as the stub
    # check - a silently-ignored coverage finding must also force a fail, not just a stub finding.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild the Text Team subgraph, with tests.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "text_team.py").write_text(
        "def build_subgraph():\n    return real_wiring()\n", encoding="utf-8",
    )

    class _FakeLLM:
        def invoke(self, messages, config=None):
            return _FakeResponse('{"verdict": "pass", "problems": []}')

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _FakeLLM())

    state = {
        "project_dir": str(tmp_path),
        "spec_path": str(spec_path),
        "code_dir": str(code_dir),
        "platform": "python",
        "python_units_built": ["Text Team"],
        "review_round": 0,
        "total_tokens": 0,
    }
    result = reviewer_node_module.reviewer_node(state)

    assert result["review_verdict"] == "fail"
    assert any("text_team.py" in p for p in result["review_problems"])


def test_reviewer_node_includes_the_builder_account_section_when_present(tmp_path, monkeypatch):
    # The safer version of "let the coder and reviewer talk" (Douglas's own proposal): a
    # mechanical file list plus the builder's own labeled, unverified explanation.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("x = 1\n", encoding="utf-8")

    captured_prompts = []

    class _FakeLLM:
        def invoke(self, messages, config=None):
            captured_prompts.append(messages[1][1])  # ("user", user_prompt)
            return _FakeResponse('{"verdict": "pass", "problems": []}')

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _FakeLLM())

    state = {
        "project_dir": str(tmp_path),
        "spec_path": str(spec_path),
        "code_dir": str(code_dir),
        "platform": "python",
        "review_round": 0,
        "total_tokens": 0,
        "builder_files_touched": [{"path": "a.py", "action": "write_file"}],
        "builder_account": "Created a.py to hold the constant the spec asked for.",
    }
    reviewer_node_module.reviewer_node(state)

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "BUILDER'S OWN ACCOUNT" in prompt
    assert "write_file: a.py" in prompt
    assert "Created a.py to hold the constant the spec asked for." in prompt


def test_reviewer_node_records_a_tech_quirk_from_its_own_response(tmp_path, monkeypatch):
    # 2026-09-28, Douglas's direct instruction: the reviewer can also write to the shared,
    # cross-project ledger - via an optional structured-output field, since the reviewer isn't a
    # tool-calling loop the way the python builder is.
    quirks_path = tmp_path / "tech-quirks.md"
    monkeypatch.setattr(tech_quirks_module, "TECH_QUIRKS_PATH", quirks_path)

    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("x = 1\n", encoding="utf-8")

    class _FakeLLM:
        def invoke(self, messages, config=None):
            return _FakeResponse(json.dumps({
                "verdict": "pass",
                "problems": [],
                "tech_quirks": ["LangGraph Runtime: has no .config, use context instead."],
            }))

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _FakeLLM())

    state = {
        "project_dir": str(tmp_path),
        "spec_path": str(spec_path),
        "code_dir": str(code_dir),
        "platform": "python",
        "review_round": 0,
        "total_tokens": 0,
    }
    reviewer_node_module.reviewer_node(state)

    content = quirks_path.read_text(encoding="utf-8")
    assert "LangGraph Runtime: has no .config, use context instead." in content
    assert "(reviewer, " in content


def test_reviewer_node_omits_the_builder_account_section_when_absent(tmp_path, monkeypatch):
    # No behavior change for an n8n build, or any build that never reached "done" via the agentic
    # path - the section must not appear at all, not appear empty.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    captured_prompts = []

    class _FakeLLM:
        def invoke(self, messages, config=None):
            captured_prompts.append(messages[1][1])
            return _FakeResponse('{"verdict": "pass", "problems": []}')

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _FakeLLM())

    state = {
        "project_dir": str(tmp_path),
        "spec_path": str(spec_path),
        "code_dir": str(code_dir),
        "platform": "python",
        "review_round": 0,
        "total_tokens": 0,
    }
    reviewer_node_module.reviewer_node(state)

    assert "BUILDER'S OWN ACCOUNT" not in captured_prompts[0]


def test_builder_node_agentic_returns_a_mechanical_file_list_and_the_agents_account(tmp_path, monkeypatch):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()

    llm = _ScriptedAgentLLM([
        _FakeToolResponse(tool_calls=[_tool_call(
            "write_file", "1", path="module_a.py", content="def do_thing():\n    return 1\n",
        )]),
        _FakeToolResponse(tool_calls=[_tool_call(
            "write_file", "2", path="smoke_run.py",
            content="import json\nprint(json.dumps({'ran': True, 'output_summary': 'ok'}))\n",
        )]),
        _FakeToolResponse(content="Wrote module_a.py implementing the spec's constant, plus the required smoke test."),
    ])
    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))

    result = builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    assert result["builder_status"] == "done"
    assert {"path": "module_a.py", "action": "write_file"} in result["builder_files_touched"]
    assert {"path": "smoke_run.py", "action": "write_file"} in result["builder_files_touched"]
    assert "Wrote module_a.py" in result["builder_account"]


def test_route_entry_and_route_after_reviewer_key_on_round_budget_exhausted_not_review_round():
    # Douglas's explicit direction (2026-09-27): review_round can legitimately climb well past
    # MAX_ROUNDS while coverage-debt batches are still making progress - routing must key on the
    # authoritative round_budget_exhausted flag reviewer_node computes, never a raw review_round
    # comparison, or a real per-batch reset upstream would be silently overridden here.
    from langgraph.graph import END

    high_review_round_but_not_exhausted = {"review_round": 9, "round_budget_exhausted": False}
    assert graph_module.route_entry(high_review_round_but_not_exhausted) == "builder"
    assert graph_module.route_after_reviewer(
        {**high_review_round_but_not_exhausted, "review_verdict": "fail"}
    ) == "builder"

    low_review_round_but_exhausted = {"review_round": 1, "round_budget_exhausted": True}
    assert graph_module.route_entry(low_review_round_but_exhausted) == END
    assert graph_module.route_after_reviewer(
        {**low_review_round_but_exhausted, "review_verdict": "fail"}
    ) == END

    # A pass always ends the run regardless of the budget flag.
    assert graph_module.route_after_reviewer(
        {"review_verdict": "pass", "round_budget_exhausted": True}
    ) == END


def test_reviewer_node_resets_the_round_budget_when_a_new_coverage_batch_starts(tmp_path, monkeypatch):
    # The actual fix for the 2026-09-27 incident: 6 real untested modules means one 5-module batch
    # plus 1 deferred. A reviewer that only ever sees coverage debt (never a genuine bug) must get a
    # FRESH 3-round budget for each distinct batch, not one flat 3-round budget for the whole project.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild the creative studio modules.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    for i in range(6):
        (code_dir / f"module_{i}.py").write_text(
            f"def do_thing_{i}():\n    return {i}\n", encoding="utf-8",
        )

    class _FakeLLM:
        def invoke(self, messages, config=None):
            # Never engages the coverage evidence at all - forces every round into pure_coverage_fail.
            return _FakeResponse('{"verdict": "pass", "problems": []}')

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _FakeLLM())

    state = {
        "project_dir": str(tmp_path),
        "spec_path": str(spec_path),
        "code_dir": str(code_dir),
        "platform": "python",
        "python_units_built": ["Epic A"],
        "review_round": 0,
        "total_tokens": 0,
    }

    round1 = reviewer_node_module.reviewer_node(state)
    assert round1["review_verdict"] == "fail"
    assert round1["coverage_batch_rounds"] == 1
    assert round1["round_budget_exhausted"] is False

    round2 = reviewer_node_module.reviewer_node(round1)
    assert round2["coverage_batch_rounds"] == 2
    assert round2["round_budget_exhausted"] is False
    assert round2["coverage_batch_key"] == round1["coverage_batch_key"]  # same batch, real progress

    round3 = reviewer_node_module.reviewer_node(round2)
    assert round3["review_round"] == 3
    assert round3["coverage_batch_rounds"] == 3
    assert round3["round_budget_exhausted"] is True  # this batch itself is genuinely stuck

    # Now simulate real progress: one of the current batch's modules gets a real test between
    # rounds - a genuinely different top-5 batch must get its OWN fresh budget, even though
    # review_round keeps climbing past MAX_ROUNDS.
    covered_file = None
    for f in check_stage_test_coverage_gaps(code_dir, ["Epic A"]):
        covered_file = f["file"]
        break
    module_stem = covered_file.rsplit(".py", 1)[0]
    (code_dir / f"test_{module_stem}.py").write_text("def test_it():\n    assert True\n", encoding="utf-8")

    round4 = reviewer_node_module.reviewer_node(round3)
    assert round4["review_round"] == 4
    assert round4["coverage_batch_key"] != round3["coverage_batch_key"]
    assert round4["coverage_batch_rounds"] == 1
    assert round4["round_budget_exhausted"] is False


def test_reviewer_node_is_not_choked_by_the_builders_own_legitimate_token_spend(tmp_path, monkeypatch):
    # The real creative-team incident (2026-09-29): building all 6 real units alone cost ~2.24M
    # tokens before the reviewer ever ran once. The OLD check compared this same shared total_tokens
    # against max_run_tokens (1M for that project) - so the reviewer's own designed-in MAX_ROUNDS=3
    # was dead on arrival, always exhausted after exactly 1 round no matter what it found. A modest
    # reviewer call must NOT be blocked just because the builder's own (legitimate) spend is huge.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild the creative studio modules.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "module_a.py").write_text("def do_thing():\n    return 1\n", encoding="utf-8")

    class _FakeLLM:
        def invoke(self, messages, config=None):
            resp = _FakeResponse('{"verdict": "pass", "problems": []}')
            resp.usage_metadata = {"total_tokens": 10_000}
            return resp

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _FakeLLM())

    state = {
        "project_dir": str(tmp_path),
        "spec_path": str(spec_path),
        "code_dir": str(code_dir),
        "platform": "python",
        "python_units_built": ["storage", "validation", "vendors", "content_writer", "image_team", "graph"],
        "review_round": 0,
        "total_tokens": 2_240_000,  # the builder's own real, legitimate multi-unit spend
        "reviewer_tokens_used": 0,  # the reviewer itself hasn't spent anything yet
    }

    result = reviewer_node_module.reviewer_node(state)
    assert result["round_budget_exhausted"] is False
    assert result["reviewer_tokens_used"] == 10_000  # isolated the reviewer's own delta only
    assert result["total_tokens"] == 2_250_000  # still correctly accumulates for reporting


def test_reviewer_node_stops_once_its_OWN_cumulative_spend_exceeds_its_dedicated_ceiling(tmp_path, monkeypatch):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild the creative studio modules.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "module_a.py").write_text("def do_thing():\n    return 1\n", encoding="utf-8")

    class _FakeLLM:
        def invoke(self, messages, config=None):
            resp = _FakeResponse('{"verdict": "pass", "problems": []}')
            resp.usage_metadata = {"total_tokens": 200}
            return resp

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _FakeLLM())

    state = {
        "project_dir": str(tmp_path),
        "spec_path": str(spec_path),
        "code_dir": str(code_dir),
        "platform": "python",
        "python_units_built": ["storage"],
        "review_round": 2,
        "total_tokens": 0,
        "reviewer_tokens_used": (reviewer_node_module.MAX_REVIEWER_TOKENS_PER_ROUND * MAX_ROUNDS) - 100,
    }

    result = reviewer_node_module.reviewer_node(state)
    assert result["round_budget_exhausted"] is True
    assert result["reviewer_tokens_used"] >= (reviewer_node_module.MAX_REVIEWER_TOKENS_PER_ROUND * MAX_ROUNDS)
    assert any("reviewer's own cumulative spend" in p for p in result["review_problems"])


def test_reviewer_node_keeps_the_flat_round_cap_for_a_genuine_correctness_bug(tmp_path, monkeypatch):
    # A real, verified correctness bug - alone, or alongside unresolved coverage debt - must NOT get
    # the per-batch relaxation: only a fail driven PURELY by coverage debt gets a fresh budget per
    # batch. Mixing in a genuine bug falls back to the original flat per-run cap, unchanged.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild the creative studio modules.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "module_a.py").write_text("def do_thing():\n    return 1\n", encoding="utf-8")

    class _FakeLLM:
        def invoke(self, messages, config=None):
            return _FakeResponse(json.dumps({
                "verdict": "fail",
                "problems": [{
                    "node": "module_a.py",
                    "claim_type": "correctness_bug",
                    "description": "does the wrong thing",
                }],
            }))

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _FakeLLM())

    state = {
        "project_dir": str(tmp_path),
        "spec_path": str(spec_path),
        "code_dir": str(code_dir),
        "platform": "python",
        "python_units_built": ["Epic A"],
        "review_round": MAX_ROUNDS - 1,  # one round away from the flat cap
        "total_tokens": 0,
    }
    result = reviewer_node_module.reviewer_node(state)

    assert result["review_verdict"] == "fail"
    assert result["review_round"] == MAX_ROUNDS
    assert result["round_budget_exhausted"] is True  # flat cap applied, no per-batch reset
    assert result["coverage_batch_key"] is None


def test_max_rounds_defaults_to_the_global_constant_and_honors_a_project_override(tmp_path):
    from pipeline.config_loader import ConfigError, max_rounds
    assert max_rounds(None) == MAX_ROUNDS
    assert max_rounds(str(tmp_path)) == MAX_ROUNDS  # no models.yaml at all
    (tmp_path / "models.yaml").write_text("max_rounds: 1\n", encoding="utf-8")
    assert max_rounds(str(tmp_path)) == 1
    (tmp_path / "models.yaml").write_text("max_rounds: 0\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        max_rounds(str(tmp_path))


def test_one_round_project_with_must_fix_hands_findings_to_the_builder_for_one_fix_pass_then_ends(tmp_path, monkeypatch):
    (tmp_path / "models.yaml").write_text("max_rounds: 1\n", encoding="utf-8")
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "module_a.py").write_text("def do_thing():\n    return 1\n", encoding="utf-8")

    class _FakeLLM:
        def invoke(self, messages, config=None):
            return _FakeResponse(json.dumps({"verdict": "fail", "problems": [{
                "node": "module_a.py", "claim_type": "correctness_bug",
                "description": "does the wrong thing", "severity": "must_fix"}]}))

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _FakeLLM())
    result = reviewer_node_module.reviewer_node(_severity_state(tmp_path, spec_path, code_dir))
    assert result["review_verdict"] == "fail"
    assert result["round_budget_exhausted"] is False
    assert result["final_fix_pass"] is True
    assert result["review_must_fix"] == ["module_a.py: does the wrong thing"]
    from langgraph.graph import END
    from pipeline.graph import route_after_builder, route_after_reviewer
    assert route_after_reviewer(result) == "builder"
    # After the builder's fix pass: accepted if the builder finished (no second review); a builder
    # that could not verify (ambiguous) also ends the run, and is reported as such.
    assert route_after_builder({**result, "builder_status": "done"}) == END
    assert route_after_builder({**result, "builder_status": "ambiguous"}) == END
    # A 3-round project never takes this path.
    (tmp_path / "models.yaml").write_text("max_rounds: 3\n", encoding="utf-8")
    three = reviewer_node_module.reviewer_node(_severity_state(tmp_path, spec_path, code_dir))
    assert three["final_fix_pass"] is False
    assert route_after_builder({**three, "builder_status": "done"}) == "reviewer"


def _severity_state(tmp_path, spec_path, code_dir):
    return {
        "project_dir": str(tmp_path),
        "spec_path": str(spec_path),
        "code_dir": str(code_dir),
        "platform": "python",
        "review_round": 0,
        "total_tokens": 0,
    }


def test_reviewer_node_passes_when_only_should_fix_and_nitpick_findings_remain(tmp_path, monkeypatch):
    # Douglas's explicit direction (2026-09-29, relayed via video-generator-78): "the reviewer
    # ranking its finding as well, priority must-haves, good to have, nitpicks etc. And must haves
    # must be delivered." A round with no must_fix must pass, even with real should_fix/nitpick
    # findings still recorded.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "module_a.py").write_text("def do_thing():\n    return 1\n", encoding="utf-8")

    class _FakeLLM:
        def invoke(self, messages, config=None):
            return _FakeResponse(json.dumps({
                "verdict": "fail",  # deliberately wrong/inconsistent - must be mechanically overridden
                "problems": [
                    {"node": "module_a.py", "claim_type": "correctness_bug", "severity": "should_fix", "description": "a real but non-blocking gap"},
                    {"node": "module_a.py", "claim_type": "correctness_bug", "severity": "nitpick", "description": "a cosmetic observation"},
                ],
            }))

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _FakeLLM())
    result = reviewer_node_module.reviewer_node(_severity_state(tmp_path, spec_path, code_dir))

    assert result["review_verdict"] == "pass"
    assert result["review_problems"] == []  # nothing loops - no must_fix
    assert result["review_should_fix"] == ["module_a.py: a real but non-blocking gap"]
    assert result["review_nitpicks"] == ["module_a.py: a cosmetic observation"]


def test_reviewer_node_fails_and_rides_should_fix_along_when_a_must_fix_is_present(tmp_path, monkeypatch):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "module_a.py").write_text("def do_thing():\n    return 1\n", encoding="utf-8")

    class _FakeLLM:
        def invoke(self, messages, config=None):
            return _FakeResponse(json.dumps({
                "verdict": "fail",
                "problems": [
                    {"node": "module_a.py", "claim_type": "correctness_bug", "severity": "must_fix", "description": "a genuine blocking bug"},
                    {"node": "module_a.py", "claim_type": "correctness_bug", "severity": "should_fix", "description": "a real but non-blocking gap"},
                ],
            }))

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _FakeLLM())
    result = reviewer_node_module.reviewer_node(_severity_state(tmp_path, spec_path, code_dir))

    assert result["review_verdict"] == "fail"
    # should_fix rides along ONLY because a round is already happening - never triggers one alone.
    assert result["review_problems"] == [
        "module_a.py: a genuine blocking bug",
        "module_a.py: a real but non-blocking gap",
    ]
    assert result["review_must_fix"] == ["module_a.py: a genuine blocking bug"]
    assert result["review_should_fix"] == ["module_a.py: a real but non-blocking gap"]


def test_reviewer_node_never_sends_a_nitpick_to_the_builder_even_during_a_real_round(tmp_path, monkeypatch):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "module_a.py").write_text("def do_thing():\n    return 1\n", encoding="utf-8")

    class _FakeLLM:
        def invoke(self, messages, config=None):
            return _FakeResponse(json.dumps({
                "verdict": "fail",
                "problems": [
                    {"node": "module_a.py", "claim_type": "correctness_bug", "severity": "must_fix", "description": "a genuine blocking bug"},
                    {"node": "module_a.py", "claim_type": "correctness_bug", "severity": "nitpick", "description": "a cosmetic observation"},
                ],
            }))

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _FakeLLM())
    result = reviewer_node_module.reviewer_node(_severity_state(tmp_path, spec_path, code_dir))

    assert result["review_verdict"] == "fail"
    assert result["review_problems"] == ["module_a.py: a genuine blocking bug"]  # nitpick excluded
    assert result["review_nitpicks"] == ["module_a.py: a cosmetic observation"]  # but still recorded


def test_reviewer_node_pure_coverage_fail_still_resets_alongside_a_should_fix_finding(tmp_path, monkeypatch):
    # The fix's own subtlety: verified_must_fix_before_force must be must_fix-ONLY, or a real
    # should_fix finding sitting alongside a forced coverage gap would wrongly disqualify the
    # per-batch coverage-round-budget relaxation, even though the should_fix item isn't what's
    # actually driving this round at all. module_b has its own real test file specifically so its
    # should_fix mention doesn't ALSO accidentally "engage" (by file-name match) module_a's own,
    # separate, genuinely-unaddressed coverage gap.
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild the creative studio modules.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "module_a.py").write_text("def do_thing():\n    return 1\n", encoding="utf-8")
    (code_dir / "module_b.py").write_text("def other_thing():\n    return 2\n", encoding="utf-8")
    (code_dir / "test_module_b.py").write_text(
        "from module_b import other_thing\ndef test_it():\n    assert other_thing() == 2\n", encoding="utf-8",
    )

    class _FakeLLM:
        def invoke(self, messages, config=None):
            # Never engages module_a's coverage evidence, but DOES report a real, unrelated
            # should_fix finding about module_b.
            return _FakeResponse(json.dumps({
                "verdict": "fail",
                "problems": [
                    {"node": "module_b.py", "claim_type": "correctness_bug", "severity": "should_fix", "description": "a real but non-blocking gap"},
                ],
            }))

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _FakeLLM())

    state = _severity_state(tmp_path, spec_path, code_dir)
    state["python_units_built"] = ["Epic A"]
    result = reviewer_node_module.reviewer_node(state)

    assert result["review_verdict"] == "fail"  # the forced, unengaged coverage gap is must_fix
    assert result["coverage_batch_rounds"] == 1  # per-batch relaxation still engaged
    assert "module_b.py: a real but non-blocking gap" in result["review_should_fix"]


def test_reviewer_node_mechanically_drops_a_problem_tagged_with_a_real_waiver_id(tmp_path, monkeypatch):
    # Real incident (2026-09-29): a real, already-decided waiver (a symlink-escape edge case
    # Douglas explicitly declined to fix) kept coming back as a must_fix finding every round,
    # burning a whole run's budget on something already settled.
    monkeypatch.setattr(decision_log_module, "DECISION_LOG_PATH", tmp_path / "decision_log.jsonl")
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "disk_storage.py").write_text("def resolve_path():\n    pass\n", encoding="utf-8")
    review_waivers_module.add_waiver(
        tmp_path, text="Symlink escape in disk_storage.py bypasses namespace check",
        reason="Douglas: unlikely to happen, waived for now.", id="symlink-escape",
    )

    class _FakeLLM:
        def invoke(self, messages, config=None):
            return _FakeResponse(json.dumps({
                "verdict": "fail",
                "problems": [{
                    "node": "disk_storage.py", "claim_type": "correctness_bug", "severity": "must_fix",
                    "description": "symlink escape bypasses the namespace check",
                    "waiver_id": "symlink-escape",
                }],
            }))

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _FakeLLM())
    result = reviewer_node_module.reviewer_node(_severity_state(tmp_path, spec_path, code_dir))

    assert result["review_verdict"] == "pass"  # dropped before verify_problems even sees it
    assert result["review_problems"] == []
    assert result["review_must_fix"] == []


def test_reviewer_node_does_not_drop_an_untagged_problem_that_merely_resembles_a_waiver(tmp_path, monkeypatch):
    # The "otherwise just flagged" half: a kept finding that happens to share wording with a real
    # waiver, but was NOT tagged with its waiver_id, must NOT be auto-dropped - only flagged.
    monkeypatch.setattr(decision_log_module, "DECISION_LOG_PATH", tmp_path / "decision_log.jsonl")
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "disk_storage.py").write_text("def resolve_path():\n    pass\n", encoding="utf-8")
    review_waivers_module.add_waiver(
        tmp_path, text="Symlink escape in disk_storage.py bypasses namespace check",
        reason="Douglas: unlikely to happen, waived for now.", id="symlink-escape",
    )

    class _FakeLLM:
        def invoke(self, messages, config=None):
            return _FakeResponse(json.dumps({
                "verdict": "fail",
                "problems": [{
                    "node": "disk_storage.py", "claim_type": "correctness_bug", "severity": "must_fix",
                    "description": "symlink escape bypasses the namespace check in disk_storage",
                    # no waiver_id at all
                }],
            }))

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _FakeLLM())
    result = reviewer_node_module.reviewer_node(_severity_state(tmp_path, spec_path, code_dir))

    assert result["review_verdict"] == "fail"  # NOT auto-dropped - only a mechanical tag drops it
    assert len(result["review_must_fix"]) == 1


def test_reviewer_node_prompt_includes_the_waiver_section_when_waivers_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_log_module, "DECISION_LOG_PATH", tmp_path / "decision_log.jsonl")
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    review_waivers_module.add_waiver(
        tmp_path, text="Symlink escape in disk_storage.py", reason="Douglas waived it.", id="symlink-escape",
    )

    captured_prompts = []

    class _CapturingLLM:
        def invoke(self, messages, config=None):
            captured_prompts.append(messages[-1][1])
            return _FakeResponse('{"verdict": "pass", "problems": []}')

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _CapturingLLM())
    reviewer_node_module.reviewer_node(_severity_state(tmp_path, spec_path, code_dir))

    assert any("symlink-escape" in p and "Douglas waived it" in p for p in captured_prompts)


def test_run_concluded_treats_partial_statuses_as_concluded():
    # Real incident this covers: "pass_partial"/"failed_after_max_rounds_partial" never matched the
    # old `status in ("pass", "failed_after_max_rounds")` check, so a genuinely concluded stage's
    # checkpoint was never cleared and stuck around to be wrongly resumed by a later, different
    # stage's invocation.
    assert _run_concluded("pass") is True
    assert _run_concluded("failed_after_max_rounds") is True
    assert _run_concluded("pass_partial") is True
    assert _run_concluded("failed_after_max_rounds_partial") is True
    assert _run_concluded("ambiguous") is False








def test_edit_file_not_found_shows_closest_real_line(tmp_path):
    (tmp_path / "context.py").write_text(
        "class Config:\n    @staticmethod\n    def get_api_key(vendor: str) -> str | None:\n        return None\n",
        encoding="utf-8",
    )
    tools = {t.name: t for t in make_agent_tools(tmp_path)}
    result = tools["edit_file"].invoke(
        {"path": "context.py", "old_string": "def get_api_key(self, vendor: str) -> str:", "new_string": "x"}
    )
    assert result.startswith("ERROR")
    assert "line 3" in result and "def get_api_key(vendor: str) -> str | None:" in result


def test_edit_file_ambiguous_match_reports_line_numbers(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\ny = 2\nx = 1\n", encoding="utf-8")
    tools = {t.name: t for t in make_agent_tools(tmp_path)}
    result = tools["edit_file"].invoke({"path": "a.py", "old_string": "x = 1", "new_string": "x = 3"})
    assert "2 times" in result and "1, 3" in result


def test_format_unit_task_includes_existing_file_content(tmp_path):
    (tmp_path / "context.py").write_text("class Config:\n    pass\n", encoding="utf-8")
    unit = {"name": "config", "files": ["context.py", "missing.py"], "interface": {"exports": []}}
    task = builder_node_module._format_unit_task("SPEC", unit, {}, "", tmp_path)
    assert "CURRENT CONTENT OF YOUR UNIT'S EXISTING FILE(S)" in task
    assert "class Config:" in task
    assert task.index("class Config:") < task.index("SPEC")
    assert "CURRENT CONTENT" not in builder_node_module._format_unit_task("SPEC", unit, {}, "")


def test_format_unit_task_skips_oversized_existing_file(tmp_path):
    (tmp_path / "big.py").write_text("x = 1\n" * 10_000, encoding="utf-8")
    unit = {"name": "big", "files": ["big.py"], "interface": {"exports": []}}
    assert "CURRENT CONTENT" not in builder_node_module._format_unit_task("SPEC", unit, {}, "", tmp_path)


def test_build_python_agentic_refreshes_the_task_text_for_each_later_attempt(tmp_path, monkeypatch):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    llm = _MessageCapturingAgentLLM([_FakeToolResponse(content="Done."), _FakeToolResponse(content="Done.")])
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (False, _REAL_PYTEST_TWO_FAILURES_OUTPUT))
    renders = iter(["SECOND-ATTEMPT-FRESH-TEXT"])
    builder_node_module._build_python_agentic(
        _agentic_state(tmp_path, spec_path, code_dir), llm, {}, "FIRST-ATTEMPT-TEXT", code_dir, {},
        refresh_spec_text=lambda: next(renders),
    )
    assert "FIRST-ATTEMPT-TEXT" in llm.invoke_message_contents[0][1]
    assert "SECOND-ATTEMPT-FRESH-TEXT" in llm.invoke_message_contents[1][1]
    assert "FIRST-ATTEMPT-TEXT" not in llm.invoke_message_contents[1][1]


def test_new_other_failures_feedback_allows_updating_callers_of_a_changed_interface():
    source = Path(builder_node_module.__file__).read_text(encoding="utf-8")
    assert "do NOT revert your intended" in source
    assert "YOU INTRODUCED A NEW FAILURE" not in source


def test_unit_baseline_roundtrip_and_clear(tmp_path):
    m = builder_node_module
    assert m._load_unit_baseline(tmp_path, "config") is None
    m._save_unit_baseline(tmp_path, "config", {"tests/a.py::t1"})
    m._save_unit_baseline(tmp_path, "other", set())
    assert m._load_unit_baseline(tmp_path, "config") == {"tests/a.py::t1"}
    assert m._load_unit_baseline(tmp_path, "other") == set()
    m._clear_unit_baseline(tmp_path, "config")
    assert m._load_unit_baseline(tmp_path, "config") is None
    m._clear_unit_baseline(tmp_path, "other")
    assert not (tmp_path / m._UNIT_BASELINE_FILENAME).exists()


def test_resumed_unit_uses_persisted_baseline_so_unit_caused_failures_count_as_new(tmp_path, monkeypatch):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    # An earlier unfinished run recorded a clean baseline; the tree is now red in ANOTHER unit's tests.
    builder_node_module._save_unit_baseline(tmp_path, "config", set())
    llm = _MessageCapturingAgentLLM([_FakeToolResponse(content="Done."), _FakeToolResponse(content="Done.")])
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (False, _REAL_PYTEST_TWO_FAILURES_OUTPUT))
    state = _agentic_state(tmp_path, spec_path, code_dir)
    state["project_dir"] = str(tmp_path)
    result = builder_node_module._build_python_agentic(
        state, llm, {}, "TASK", code_dir, {}, unit_files=["context.py"], unit_name="config",
    )
    assert result["builder_status"] != "done"
    assert any("PASSED BEFORE YOUR CHANGES" in str(m) for m in llm.invoke_message_contents[1])


def test_unit_with_persisted_baseline_is_not_treated_as_already_built(tmp_path, monkeypatch):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    unit = {"name": "modu", "purpose": "p", "files": ["mod.py"], "depends_on": [],
            "interface": {"exports": [{"name": "f", "kind": "function", "signature": "f()", "description": "d"}]}}
    calls = []
    monkeypatch.setattr(builder_node_module, "_build_python_agentic",
                        lambda *a, **k: calls.append(1) or {"builder_status": "done", "total_tokens": 0})
    state = _agentic_state(tmp_path, spec_path, code_dir)
    state["project_dir"] = str(tmp_path)
    state["python_units_built"] = ["modu"]
    args = (state, None, {}, "SPEC", code_dir, [unit], None)
    builder_node_module._build_python_units(*args)
    assert calls == []
    builder_node_module._save_unit_baseline(tmp_path, "modu", set())
    builder_node_module._build_python_units(*args)
    assert calls == [1]


def test_extract_json_object_error_shows_the_tail_so_truncation_is_visible():
    from pipeline.json_extract import extract_json_object
    truncated = "```json\n{\"a\": [1, 2, \"unfinished"
    try:
        extract_json_object(truncated, context="x")
        raise AssertionError("should have raised")
    except ValueError as exc:
        assert "unfinished" in str(exc) and "truncated" in str(exc)


def test_resumed_unit_first_attempt_is_told_about_new_failures_up_front(tmp_path, monkeypatch):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    builder_node_module._save_unit_baseline(tmp_path, "config", set())
    llm = _MessageCapturingAgentLLM([_FakeToolResponse(content="Done."), _FakeToolResponse(content="Done.")])
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (False, _REAL_PYTEST_TWO_FAILURES_OUTPUT))
    state = _agentic_state(tmp_path, spec_path, code_dir)
    state["project_dir"] = str(tmp_path)
    builder_node_module._build_python_agentic(
        state, llm, {}, "TASK", code_dir, {}, unit_files=["context.py"], unit_name="config",
    )
    first_prompt = llm.invoke_message_contents[0][1]
    assert "PASSED BEFORE YOUR CHANGES" in first_prompt
    assert "Do NOT report done" in first_prompt


def test_load_role_clamps_max_tokens_to_the_models_real_ceiling(tmp_path, capsys):
    from pipeline.config_loader import load_role

    (tmp_path / "models.yaml").write_text(
        "roles:\n  builder:\n    model: \"openai/gpt-6-sol\"\n    max_tokens: 300000\n", encoding="utf-8"
    )
    assert load_role("builder", project_dir=str(tmp_path)).max_tokens == 128000
    assert "exceeds" in capsys.readouterr().err

    (tmp_path / "models.yaml").write_text(
        "roles:\n  builder:\n    model: \"openai/gpt-6-sol\"\n    max_tokens: 50000\n", encoding="utf-8"
    )
    assert load_role("builder", project_dir=str(tmp_path)).max_tokens == 50000


def test_read_file_ranged_and_capped_at_400_lines(tmp_path):
    (tmp_path / "big.py").write_text("\n".join(f"line{i}" for i in range(1, 1001)), encoding="utf-8")
    tools = {t.name: t for t in make_agent_tools(tmp_path)}
    first = tools["read_file"].invoke({"path": "big.py"})
    assert "400\tline400" in first and "401\tline401" not in first
    assert "start_line=401" in first
    ranged = tools["read_file"].invoke({"path": "big.py", "start_line": 500, "end_line": 502})
    assert ranged.splitlines() == ["500\tline500", "501\tline501", "502\tline502", "... (lines 503-1000 not shown; call read_file again with start_line=503 to continue)"]


def test_slice_spec_for_unit_keeps_relevant_sections_only():
    from pipeline.nodes.builder_node import _slice_spec_for_unit

    filler = "x" * 3000
    spec = (
        "Intro line\n# Storage\nstorage_backend details " + filler + "\n"
        "# Billing\nbilling stuff " + filler + "\n# Email\nemail stuff " + filler + "\n"
    )
    unit = {"name": "storage_backend", "files": ["storage.py"], "interface": {"exports": []}, "depends_on": []}
    out = _slice_spec_for_unit(spec, unit, {})
    assert "storage_backend details" in out and "billing stuff" not in out
    assert "# Billing" in out  # omitted sections are listed by heading
    small = "# A\nstorage_backend\n# B\nother\n# C\nmore"
    assert _slice_spec_for_unit(small, unit, {}) == small
    assert _slice_spec_for_unit(spec, {"name": "zzz_none", "files": []}, {}) == spec


def test_format_unit_task_preloads_files_named_in_reviewer_findings(tmp_path):
    from pipeline.nodes.builder_node import _format_unit_task

    (tmp_path / "other.py").write_text("y = 2\n", encoding="utf-8")
    unit = {"name": "u", "files": ["u.py"], "interface": {}, "depends_on": []}
    task = _format_unit_task("spec", unit, {}, "", tmp_path, ["u: crashes because other.py sets y wrongly"])
    assert "=== other.py ===" in task and "y = 2" in task
    assert "=== other.py ===" not in _format_unit_task("spec", unit, {}, "", tmp_path, None)


def test_roles_default_to_their_vendors_own_host_and_unlisted_vendors_are_unpinned():
    from pipeline.provider_pin import default_provider_for

    assert default_provider_for("anthropic/claude-sonnet-5.5") == ["anthropic"]
    assert default_provider_for("openai/gpt-6-astra") == ["openai"]
    assert default_provider_for("google/gemini-3.8-flash") == ["google-ai-studio"]
    assert default_provider_for("deepseek/deepseek-v4-pro") is None
    assert load_role("architect").provider_order == ["openai"]


def test_provider_override_wins_but_only_for_the_model_it_was_chosen_for(tmp_path):
    from pipeline.provider_pin import set_provider_override

    set_provider_override(tmp_path, "builder", "openai/gpt-6-sol", "azure")
    assert load_role("builder", project_dir=str(tmp_path)).provider_order == ["azure"]
    assert load_role("builder", model_override="openai/gpt-6-luna", project_dir=str(tmp_path)).provider_order == ["openai"]


def test_provider_failure_without_a_terminal_stops_and_asks_instead_of_switching(tmp_path, monkeypatch):
    import pytest
    from pipeline import model_fallback
    from pipeline.provider_pin import ProviderFailureError

    class Boom(Exception):
        status_code = 503

    monkeypatch.setattr("pipeline.provider_pin.alternative_providers", lambda m, c: ["azure"])
    monkeypatch.setattr("sys.stdin", None)
    with pytest.raises(ProviderFailureError) as exc:
        model_fallback.attempt_fallback("builder", lambda llm: None, project_dir=tmp_path, primary_error=Boom("down"))
    assert "ENDS the prompt cache" in str(exc.value) and "azure" in str(exc.value)
    assert not (tmp_path / ".provider_override_state.json").exists()
    assert not (tmp_path / ".model_fallback_state.json").exists()


class _CountingLLM:
    def __init__(self, make_response):
        self.make_response = make_response
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, config=None):
        self.calls += 1
        return self.make_response(self.calls)


def _run_rounds_scenario(tmp_path, monkeypatch, make_response):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    llm = _CountingLLM(make_response)
    monkeypatch.setattr(builder_node_module, "resolve_llm_for_role", lambda *a, **k: llm)

    def _pytest(code_dir):
        llm.calls_at_first_verification = getattr(llm, "calls_at_first_verification", llm.calls)
        return True, ""

    monkeypatch.setattr(builder_node_module, "_run_pytest", _pytest)
    builder_node_module.builder_node(_agentic_state(tmp_path, spec_path, code_dir))
    return llm


def test_agent_round_without_writes_ends_the_attempt(tmp_path, monkeypatch):
    llm = _run_rounds_scenario(
        tmp_path, monkeypatch, lambda n: _FakeToolResponse(tool_calls=[_tool_call("list_files", f"c{n}", path=".")])
    )
    assert llm.calls_at_first_verification == builder_node_module.MAX_AGENT_TURNS


def test_agent_round_with_writes_continues_up_to_max_rounds(tmp_path, monkeypatch):
    llm = _run_rounds_scenario(
        tmp_path,
        monkeypatch,
        lambda n: _FakeToolResponse(tool_calls=[_tool_call("write_file", f"c{n}", path=f"f{n}.py", content=f"x = {n}\n")]),
    )
    assert llm.calls_at_first_verification == builder_node_module.MAX_AGENT_TURNS * builder_node_module.MAX_ROUNDS_PER_ATTEMPT


def test_agent_reading_round_then_writing_round_continues(tmp_path, monkeypatch):
    turns = builder_node_module.MAX_AGENT_TURNS

    def respond(n):
        if n <= turns:
            return _FakeToolResponse(tool_calls=[_tool_call("write_file", f"c{n}", path=f"f{n}.py", content="x = 1\n")])
        return _FakeToolResponse(content="done")

    llm = _run_rounds_scenario(tmp_path, monkeypatch, respond)
    assert llm.calls_at_first_verification == turns + 1


def test_parse_pytest_failures_counts_collection_errors():
    out = (
        "=== short test summary info ===\n"
        "ERROR tests/test_research_envelopes.py - ModuleNotFoundError: No module named 'pandas'\n"
        "FAILED tests/test_a.py::test_x - assert 1 == 2\n"
    )
    found = builder_node_module._parse_pytest_failures(out)
    assert [f["test"] for f in found] == ["tests/test_research_envelopes.py", "tests/test_a.py::test_x"]
    assert "pandas" in found[0]["reason"]
    assert builder_node_module._test_belongs_to_unit(found[0]["test"], ["src/research_envelopes.py"])


def test_project_env_uses_build_agent_python_without_requirements(tmp_path):
    from pipeline.project_env import ensure_project_python

    (tmp_path / "code").mkdir()
    assert ensure_project_python(tmp_path / "code") == (sys.executable, None)


def test_project_env_creates_venv_and_reinstalls_only_when_requirements_change(tmp_path, monkeypatch):
    import pipeline.project_env as pe

    code = tmp_path / "code"
    code.mkdir()
    (code / "requirements.txt").write_text("SQLAlchemy\n", encoding="utf-8")
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        if argv[1:3] == ["-m", "venv"]:
            py = pe._venv_python(Path(argv[3]))
            py.parent.mkdir(parents=True)
            py.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(pe.subprocess, "run", fake_run)
    python, err = pe.ensure_project_python(code)
    assert err is None and python.startswith(str(tmp_path.resolve() / ".venv"))
    assert any("pip" in c for c in calls[1:]) and len(calls) == 2
    pe.ensure_project_python(code)
    assert len(calls) == 2  # unchanged requirements: no reinstall
    (code / "requirements.txt").write_text("SQLAlchemy\npandas\n", encoding="utf-8")
    pe.ensure_project_python(code)
    assert len(calls) == 3


def test_verified_unit_that_ran_out_of_turns_still_gets_a_journal_entry(tmp_path, monkeypatch):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild a thing.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))
    llm = _CountingLLM(
        lambda n: _FakeToolResponse(tool_calls=[_tool_call(
            "write_file", f"c{n}", path="storage.py" if n == 1 else f"extra{n}.py", content="class Storage:\n    pass\n")])
    )
    result = builder_node_module._build_python_units(
        _agentic_state(tmp_path, spec_path, code_dir), llm, "config", "BASE SPEC", code_dir,
        [_real_build_unit("storage")], None,
    )
    assert result["builder_status"] == "done" and "storage" in result["python_units_built"]
    entries = build_journal_module.load_relevant_entries(tmp_path, "storage")
    assert any(e["role"] == "builder" and e["unit"] == "storage" for e in entries)


def test_format_unit_task_includes_dependency_definitions_digest(tmp_path):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "migrations.py").write_text(
        'DEDUPE_COLUMN = "dedupe_key"\n\n'
        'class Row:\n    dedupe_key: str = ""\n\n    def method(self):\n        return SECRET_BODY\n\n'
        'def migrate(db):\n    db.execute("CREATE TABLE delivery_outbox (id INTEGER, dedupe_key TEXT)")\n',
        encoding="utf-8",
    )
    dep = {"name": "db", "files": ["migrations.py"], "interface": {"exports": []}}
    unit = {"name": "queue", "files": ["queue.py"], "depends_on": ["db"], "interface": {"exports": []}}
    task = builder_node_module._format_unit_task(
        "SPEC", unit, {"db": {}}, "", code_dir, None, {"db": dep, "queue": unit},
    )
    assert "CONCRETE DEFINITIONS" in task
    assert 'DEDUPE_COLUMN = "dedupe_key"' in task and "dedupe_key: str" in task
    assert "CREATE TABLE delivery_outbox" in task
    assert "SECRET_BODY" not in task


def test_undeclared_third_party_imports_flags_missing_requirements(tmp_path):
    from pipeline.requirements_check import undeclared_third_party_imports

    code = tmp_path / "code"
    (code / "pkg").mkdir(parents=True)
    (code / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (code / "pkg" / "mod.py").write_text(
        "import os, yaml, pandas\nfrom zoneinfo import ZoneInfo\nfrom pkg import other\nimport pytest\n", encoding="utf-8",
    )
    assert undeclared_third_party_imports(code) == []  # no requirements.txt: not checked
    (code / "requirements.txt").write_text("PyYAML>=6\n", encoding="utf-8")
    problems = " ".join(undeclared_third_party_imports(code))
    assert "'pandas'" in problems and "tzdata" in problems
    assert "'yaml'" not in problems and "'os'" not in problems and "pytest" not in problems
    (code / "requirements.txt").write_text("PyYAML\npandas\ntzdata\n", encoding="utf-8")
    assert undeclared_third_party_imports(code) == []


def test_definitions_digest_includes_config_files(tmp_path):
    code_dir = tmp_path / "code"
    (code_dir / "config").mkdir(parents=True)
    (code_dir / "config" / "defaults.yaml").write_text("sources:\n  per_field_asof_threshold_minutes: 30\n", encoding="utf-8")
    digest = builder_node_module._definitions_digest(code_dir, ["config/defaults.yaml"])
    assert "per_field_asof_threshold_minutes" in digest


def test_load_carried_findings(tmp_path):
    import json
    from pipeline.build_and_review import _load_carried_findings

    assert _load_carried_findings(tmp_path) == {}
    (tmp_path / "run_result.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    assert _load_carried_findings(tmp_path) == {}
    (tmp_path / "run_result.json").write_text(json.dumps({
        "status": "pass_partial", "problems": ["ignored"],
        "review_findings": {"must_fix": [], "should_fix": ["s: clock"], "nitpick": ["n"]},
    }), encoding="utf-8")
    assert _load_carried_findings(tmp_path) == {"must_fix": [], "should_fix": ["s: clock"]}
    (tmp_path / "run_result.json").write_text(json.dumps({
        "status": "failed_after_max_rounds",
        "review_findings": {"must_fix": ["a: bad"], "should_fix": ["b: meh"], "nitpick": ["c"]},
    }), encoding="utf-8")
    assert _load_carried_findings(tmp_path) == {"must_fix": ["a: bad"], "should_fix": ["b: meh"]}
    (tmp_path / "run_result.json").write_text(json.dumps({
        "status": "failed_after_max_rounds", "problems": ["x: y"],
    }), encoding="utf-8")
    assert _load_carried_findings(tmp_path) == {"must_fix": ["x: y"], "should_fix": []}
    (tmp_path / "run_result.json").write_text("not json", encoding="utf-8")
    assert _load_carried_findings(tmp_path) == {}


def test_method_name_finding_matches_unit_and_finding_change_gate(tmp_path):
    from pipeline.nodes.builder_node import (
        _node_belongs_to_unit, _py_snapshot, _finding_change_problems,
    )

    (tmp_path / "recipients.py").write_text(
        "class RecipientRepository:\n    def bootstrap_owner(self):\n        return 1\n", encoding="utf-8")
    assert _node_belongs_to_unit("bootstrap_owner", "Recipient Directory", ["recipients.py"], tmp_path)
    assert _node_belongs_to_unit("RecipientRepository.bootstrap_owner", "Recipient Directory", ["recipients.py"], tmp_path)
    assert not _node_belongs_to_unit("other_method", "Recipient Directory", ["recipients.py"], tmp_path)

    base = _py_snapshot(tmp_path)
    assert len(_finding_change_problems(tmp_path, base)) == 2
    (tmp_path / "recipients.py").write_text(
        "class RecipientRepository:\n    def bootstrap_owner(self):\n        return 2\n", encoding="utf-8")
    assert len(_finding_change_problems(tmp_path, base)) == 1
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_recipients.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    assert _finding_change_problems(tmp_path, base) == []


def test_ambiguity_answers_roundtrip_and_carry_after_ambiguous(tmp_path):
    import json
    from pipeline import ambiguity_answers as aa
    from pipeline.build_and_review import _load_carried_findings

    assert aa.load_answers(tmp_path) == []
    aa.record_answer(tmp_path, "Which Yahoo defect?", "FALSE POSITIVE, fix only the error text", ["Yahoo news"])
    answers = aa.load_answers(tmp_path)
    assert len(answers) == 1
    assert "FALSE POSITIVE" in aa.rulings_block(aa.format_rulings(answers))
    assert aa.rulings_block([]) == ""
    assert aa.drop_dismissed(["x: Yahoo news stale", "y: other"], answers) == ["y: other"]

    (tmp_path / "run_result.json").write_text(json.dumps({
        "status": "ambiguous_partial", "question": "q",
        "review_findings": {"must_fix": ["a: dedupe TypeError"], "should_fix": [], "nitpick": []},
    }), encoding="utf-8")
    assert _load_carried_findings(tmp_path) == {"must_fix": ["a: dedupe TypeError"], "should_fix": []}


def test_question_already_ruled_matches_normalized_question():
    from pipeline.build_and_review import _question_already_ruled
    answers = [{"question": "Should  X   be Fixed?", "answer": "no"}]
    assert _question_already_ruled("should x be fixed?", answers)
    assert not _question_already_ruled("something else", answers)
    assert not _question_already_ruled("", answers)


def test_rulings_block_forbids_re_asking_and_dismissal_covers_all_finding_lists(tmp_path):
    from pipeline.ambiguity_answers import drop_dismissed, record_answer, load_answers, rulings_block, format_rulings
    record_answer(tmp_path, "q", "FALSE POSITIVE", ["reads Yahoo news rows"])
    answers = load_answers(tmp_path)
    assert "NEVER reply AMBIGUOUS" in rulings_block(format_rulings(answers))
    state = {"review_problems": ["EventCollector.collect reads Yahoo news rows via x", "b: real"]}
    assert drop_dismissed(state["review_problems"], answers) == ["b: real"]


def test_external_findings_round_trip(tmp_path):
    from pipeline.external_findings import dismiss_matching, load_open_findings, record_finding, resolve_finding, finding_line
    a = record_finding(tmp_path, "dedupe raises TypeError on mixed kind", "Agent 1 concept check", "events.py")
    record_finding(tmp_path, "dedupe raises TypeError on mixed kind", "Agent 1 concept check", "events.py")
    b = record_finding(tmp_path, "other bug", "Douglas")
    assert [e["id"] for e in load_open_findings(tmp_path)] == [a["id"], b["id"]]
    assert finding_line(a) == "events.py: dedupe raises TypeError on mixed kind"
    assert dismiss_matching(tmp_path, ["OTHER bug"]) == [b["id"]]
    resolve_finding(tmp_path, a["id"], "reviewer", "q")
    assert load_open_findings(tmp_path) == []


def _external_review_state(tmp_path, code):
    from pipeline.external_findings import record_finding
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec\nBuild events.\n", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "events.py").write_text(code, encoding="utf-8")
    e = record_finding(tmp_path, "dedupe raises TypeError", "Agent 1", "events.py")
    state = {
        "project_dir": str(tmp_path), "spec_path": str(spec_path), "code_dir": str(code_dir),
        "platform": "python", "review_round": 0, "total_tokens": 0,
        "external_findings": [{**e, "line": "events.py: dedupe raises TypeError"}],
    }
    return state, e


def test_reviewer_reopens_external_finding_when_silent(tmp_path, monkeypatch):
    state, e = _external_review_state(tmp_path, "def dedupe(x):\n    return sorted(x)\n")

    class _LLM:
        def invoke(self, messages, config=None):
            return _FakeResponse('{"verdict": "pass", "problems": []}')

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _LLM())
    result = reviewer_node_module.reviewer_node(state)
    from pipeline.external_findings import load_open_findings
    assert result["review_verdict"] == "fail"
    assert any("did not confirm it fixed" in p for p in result["review_must_fix"])
    assert [f["id"] for f in load_open_findings(tmp_path)] == [e["id"]]


def test_reviewer_clears_external_finding_only_with_verified_quote(tmp_path, monkeypatch):
    state, e = _external_review_state(tmp_path, "def dedupe(x):\n    return sorted(x, key=str)\n")
    from pipeline.external_findings import load_open_findings
    answers = iter([
        '{"verdict": "pass", "problems": [], "finding_confirmations": {"%s": "not in the code"}}' % e["id"],
        '{"verdict": "pass", "problems": [], "finding_confirmations": {"%s": "return sorted(x, key=str)"}}' % e["id"],
    ])

    class _LLM:
        def invoke(self, messages, config=None):
            return _FakeResponse(next(answers))

    monkeypatch.setattr(reviewer_node_module, "resolve_llm_for_role", lambda *a, **k: _LLM())
    assert reviewer_node_module.reviewer_node(state)["review_verdict"] == "fail"
    assert load_open_findings(tmp_path)
    result = reviewer_node_module.reviewer_node(state)
    assert result["review_verdict"] == "pass"
    assert load_open_findings(tmp_path) == []


def test_seam_test_problems_flags_consumer_tested_without_real_producer(tmp_path):
    from pipeline.nodes.builder_node import _seam_test_problems
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pkg" / "yahoo.py").write_text("class YahooClient:\n    def market(self): ...\n", encoding="utf-8")
    (tmp_path / "pkg" / "coord.py").write_text("def run(frames): ...\n", encoding="utf-8")
    (tmp_path / "tests" / "test_coord.py").write_text("from pkg.coord import run\n", encoding="utf-8")
    units = {
        "Yahoo": {"name": "Yahoo", "files": ["pkg/yahoo.py"]},
        "Coord": {"name": "Coord", "files": ["pkg/coord.py"], "depends_on": ["Yahoo"]},
    }
    problems = _seam_test_problems(tmp_path, units["Coord"], units)
    assert problems and problems[0].startswith("Coord:")
    (tmp_path / "tests" / "test_coord.py").write_text(
        "from pkg.coord import run\nfrom pkg.yahoo import YahooClient\n", encoding="utf-8"
    )
    assert _seam_test_problems(tmp_path, units["Coord"], units) == []
    assert _seam_test_problems(tmp_path, units["Yahoo"], units) == []


def test_missing_declared_files_flags_a_unit_that_wrote_nothing(tmp_path):
    from pipeline.nodes.builder_node import _missing_declared_files
    unit = {"name": "Opt", "files": ["pkg/options_buying.py", "tests/test_options_buying.py"]}
    assert _missing_declared_files(tmp_path, unit)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "options_buying.py").write_text("", encoding="utf-8")
    assert _missing_declared_files(tmp_path, unit)  # empty does not count
    (tmp_path / "pkg" / "options_buying.py").write_text("def scan(): ...", encoding="utf-8")
    assert _missing_declared_files(tmp_path, unit) == []
    assert _missing_declared_files(tmp_path, {"name": "x"}) == []


def test_unit_that_produces_nothing_is_retried_with_contract_feedback_before_going_ambiguous(tmp_path, monkeypatch):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("# Spec Build a thing.", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    build_units = [_real_build_unit("options")]
    llm = _ScriptedAgentLLM([_FakeToolResponse(content="Done.")])
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))
    result = builder_node_module._build_python_units(
        _agentic_state(tmp_path, spec_path, code_dir), llm, "config", "BASE SPEC", code_dir, build_units, None,
    )
    assert result["builder_status"] == "ambiguous"
    assert "UNIT CONTRACT NOT MET" in result["ambiguity_question"]
    assert "you did not create them" in result["ambiguity_question"] or "declared export" in result["ambiguity_question"]
    assert llm._calls >= 2  # retried, not ended on the first attempt


def _already_fixed_setup(tmp_path):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("spec", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "storage.py").write_text("class Storage:\n    def load(self, x):\n        return x.removeprefix('US.')\n", encoding="utf-8")
    return spec_path, code_dir, [_real_build_unit("storage")]


def test_already_fixed_claim_with_valid_quote_completes_without_change_or_ambiguity(tmp_path, monkeypatch):
    spec_path, code_dir, units = _already_fixed_setup(tmp_path)
    finding = "storage.py: does not strip the 'US.' prefix"
    ref = builder_node_module._finding_ref(finding)
    llm = _ScriptedAgentLLM([_FakeToolResponse(
        content=f"Already handled.\nALREADY_FIXED: {ref} | return x.removeprefix('US.')")])
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))
    state = _agentic_state(tmp_path, spec_path, code_dir, python_units_built=["storage"], review_problems=[finding])
    result = builder_node_module._build_python_units(state, llm, "config", "BASE SPEC", code_dir, units, None)
    assert result["builder_status"] == "done"
    assert [c["id"] for c in result["claimed_fixed"]] == [ref]


def test_already_fixed_claim_with_fake_quote_is_rejected(tmp_path):
    spec_path, code_dir, _ = _already_fixed_setup(tmp_path)
    finding = "storage.py: does not strip the 'US.' prefix"
    ref = builder_node_module._finding_ref(finding)
    valid, rejected = builder_node_module._parse_already_fixed_claims(
        f"ALREADY_FIXED: {ref[:4]} | return nothing_like_this()\nALREADY_FIXED: f-zzzzzz | x", [finding], code_dir)
    assert valid == []
    assert len(rejected) == 2


def test_reviewer_prompt_tracks_claimed_fixed_and_reopens_when_unconfirmed():
    src = (Path(__file__).resolve().parent.parent / "pipeline" / "nodes" / "reviewer_node.py").read_text(encoding="utf-8")
    assert "claimed_fixed" in src and "claimed-" in src


def _skeleton_units(flag_index=None):
    units = [_real_unit("models"), _real_unit("core", depends_on=["models"]), _real_unit("cli", depends_on=["core"])]
    if flag_index is not None:
        units[flag_index]["walking_skeleton"] = True
    return units


def test_validate_build_units_requires_one_walking_skeleton_in_first_three_from_three_units():
    assert any("walking_skeleton" in p for p in validate_build_units(_skeleton_units()))
    assert validate_build_units(_skeleton_units(1)) == []
    four = _skeleton_units(1) + [_real_unit("late", depends_on=["cli"])]
    four[3]["walking_skeleton"] = True
    assert any("walking_skeleton" in p for p in validate_build_units(four))
    late_only = _skeleton_units() + [_real_unit("late", depends_on=["cli"])]
    late_only[3]["walking_skeleton"] = True
    assert any("walking_skeleton" in p for p in validate_build_units(late_only))


def test_unit_task_gives_skeleton_and_later_units_the_end_to_end_rules(tmp_path):
    units = _skeleton_units(1)
    by_name = {u["name"]: u for u in units}
    def task(u):
        return builder_node_module._format_unit_task("SPEC", u, {}, "", tmp_path, None, by_name)
    assert "WALKING SKELETON" in task(units[1])
    assert "END-TO-END TEST" not in task(units[0])
    assert "END-TO-END TEST" in task(units[2])


def test_agent_rounds_and_read_first_round_extend_one_attempt(tmp_path, monkeypatch, capsys):
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("spec", encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(builder_node_module, "_run_pytest", lambda code_dir: (True, ""))
    monkeypatch.setattr(builder_node_module, "_run_smoke", lambda code_dir, mode="dry": (True, ""))

    def scripted():
        # round 1 (3 turns): reads only; then rounds 2 and 3 write, then done
        calls = [_FakeToolResponse(tool_calls=[_tool_call("read_file", str(i), path="a.py")]) for i in range(3)]
        calls.append(_FakeToolResponse(tool_calls=[_tool_call("write_file", "w", path="b.py", content="def f():\n    return 1\n")]))
        calls.append(_FakeToolResponse(content="Done."))
        return _ScriptedAgentLLM(calls)

    # Override: exploration-only round 1 is allowed and the builder reaches its write in round 2.
    state = _agentic_state(tmp_path, spec_path, code_dir, agent_rounds=5, read_first_round=True)
    r = builder_node_module._build_python_agentic(state, scripted(), "config", "SPEC", code_dir, {})
    assert (code_dir / "b.py").exists()
    assert r["builder_status"] == "done"
    assert "attempt 2/" not in capsys.readouterr().err  # done within attempt 1, across rounds


def test_requirements_check_catches_string_literal_dynamic_imports(tmp_path):
    (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")
    (tmp_path / "q.py").write_text(
        "import importlib\nimportlib.import_module('moomoo')\n__import__('yaml')\n"
        "importlib.util.find_spec('os')\nimportlib.import_module(name)\nimportlib.import_module('.rel')\n",
        encoding="utf-8",
    )
    from pipeline.requirements_check import undeclared_third_party_imports
    problems = " ".join(undeclared_third_party_imports(tmp_path))
    assert "'moomoo'" in problems and "'yaml'" in problems
    assert "'os'" not in problems and "rel" not in problems
    (tmp_path / "requirements.txt").write_text("requests\nmoomoo-api\npyyaml\n", encoding="utf-8")
    assert undeclared_third_party_imports(tmp_path) == []


def test_dismiss_matches_ids_and_reports_unmatched(tmp_path):
    from pipeline.external_findings import dismiss_matching, load_open_findings, record_finding, unmatched_needles
    a = record_finding(tmp_path, "replay_backends.py: delete per ruling")
    before = load_open_findings(tmp_path)
    assert unmatched_needles(before, [a["id"], "nothing like this"]) == ["nothing like this"]
    assert dismiss_matching(tmp_path, [a["id"].upper()]) == [a["id"]]
    assert load_open_findings(tmp_path) == []


def test_unmatched_file_finding_is_routed_via_journal_else_last_unit(tmp_path):
    from pipeline.nodes.builder_node import _route_unmatched_problems
    from pipeline.build_journal import record_journal_entry
    units = [{"name": "A", "files": ["a.py"]}, {"name": "B", "files": ["b.py"]}, {"name": "C", "files": ["c.py"]}]
    record_journal_entry(tmp_path, "B", "builder", "did it. Files touched: pkg/testing/replay.py")
    p_journal = "pkg/testing/replay.py: delete per ruling"
    (tmp_path / "pkg" / "other").mkdir(parents=True)
    (tmp_path / "pkg" / "other" / "gone.py").write_text("x = 1", encoding="utf-8")
    p_none = "pkg/other/gone.py: delete per ruling"
    p_missing = "pkg/nope.py: stale"
    p_named = "a.py: bug"
    routed = _route_unmatched_problems([p_journal, p_none, p_named, p_missing], units, tmp_path, tmp_path)
    assert routed[id(p_journal)] == "B"
    assert routed[id(p_none)] == "C"
    assert id(p_named) not in routed and id(p_missing) not in routed


def test_provisional_fixed_claims_roundtrip_and_clear(tmp_path):
    from pipeline.provisional_fixed import clear_provisional, load_provisional, save_provisional
    save_provisional(tmp_path, [{"id": "f-abc123", "finding": "u: x", "quote": "q"}])
    assert load_provisional(tmp_path)["f-abc123"]["quote"] == "q"
    clear_provisional(tmp_path)
    assert load_provisional(tmp_path) == {}
