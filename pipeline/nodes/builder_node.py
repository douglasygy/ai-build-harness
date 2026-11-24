from __future__ import annotations

import ast
import concurrent.futures
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..agent_tools import _SYNTAX_ERROR_MARKER, make_agent_tools
from ..ambiguity_answers import rulings_block
from ..provisional_fixed import load_provisional, save_provisional
from ..build_journal import format_journal_for_prompt, load_all_entries, load_relevant_entries, record_journal_entry
from ..build_log import record_build
from ..build_scribe import summarize_build_run
from ..constants import (
    MAX_AGENT_TURNS,
    MAX_ROUNDS_PER_ATTEMPT,
    PRUNE_STALE_EXPLORATION_MID_ATTEMPT,
    MAX_INTERNAL_FIX_ATTEMPTS,
    MAX_TOKENS_PER_ATTEMPT,
    MAX_TOKENS_PER_INVOCATION,
)
from ..json_extract import extract_json_object
from ..langfuse_setup import new_run_id, run_config
from ..project_env import ensure_project_python
from ..requirements_check import undeclared_third_party_imports
from ..model_fallback import attempt_fallback, resolve_llm_for_role
from ..n8n_client import N8nError
from ..n8n_workflow_target import n8n_subdir
from ..n8n_verify import (
    VERIFY_ADAPTER_JS,
    adapter_name_for,
    verify_native_workflow,
    verify_webhook_path,
)
from ..retry import with_retries_nonempty, with_retries_tool_call
from ..scale_roster import architect_decisions_path, load_python_build_units, pm_epics_path
from ..state import BuildState
from ..tech_quirks import make_tech_quirk_tool, record_tech_quirk_entry, with_tech_quirks
from ..token_guard import extract_tokens, extract_usage, over_budget

# Retired 2026-09-27 (replaced by AGENT_SYSTEM_PROMPT + pipeline/agent_tools.py, below): the python
# builder used to be a single one-shot completion asked to regenerate "the complete set of files"
# in this delimited format every attempt - the actual root cause of a real incident (creative-team,
# 5,033,372 tokens, a crash present since stage 1 shipped undetected because nothing ever ran the
# code). n8n's own builder is untouched by this change and does not use this format or this prompt.
_MODEL_IDENTIFIER_RULE = """MODEL IDENTIFIERS ARE NEVER YOURS TO CHOOSE OR SUBSTITUTE. Agent/model selection is reserved to
Douglas alone - this is a hard rule of the pipeline you are building for, not a style preference.
When the spec names a specific model for a role (e.g. "Claude Sonnet", "Gemini Deep Think", a
specific OpenRouter slug), any hardcoded value or default you write for that role - in config,
env-var fallbacks, API call code, anywhere - MUST be the exact identifier the spec states, copied
verbatim. Never substitute a different version, a similar-sounding model, an older/newer release,
or "whatever you'd expect that name to map to" - a real incident produced anthropic/claude-3.5-sonnet
and an experimental gemini-2.0-flash-thinking-exp as fallback defaults when the spec said "Claude
Sonnet" and "Gemini Deep Think" (this pipeline's own established identifiers for those are
anthropic/claude-sonnet-5 and a specific Gemini Deep Think slug Douglas confirms - check
config/models.yaml's own role entries for precedent on exact current identifiers before guessing
at a mapping). If the spec names a role only generically (e.g. just "GPT" with no version), do NOT
invent a specific version either - read the identifier from a required environment variable with
NO hardcoded fallback default, so an unconfigured deployment fails loudly and immediately instead
of silently running on a wrong or lower-tier model."""

AGENT_SYSTEM_PROMPT = f"""You are a careful software builder agent working on a real python project. You
have tools to explore, search, read, write, and edit real files, and to run real commands (list_files,
search_files, read_file, write_file, edit_file, run_command) - use them. Do not guess at what a file
currently contains or assume what you wrote earlier is still there; read it first. write_file is
REJECTED for a file that already has real content - use edit_file for a targeted change instead, even
for a large change, made as one or more precise replacements rather than a full-file rewrite. This is
not just a style preference: rewriting a whole file from scratch has no memory of what some OTHER file
already depends on it for (a class name, a function signature), and can silently break it - fixing a
specific verification failure means changing exactly what's broken, not regenerating everything nearby.

In an existing codebase you don't already have memorized, PREFER search_files over reading whole files
to find something - search for a function name, a call site, an import, an error message, then read
only the specific file(s) that turned up. Reading every file in full to locate one thing is expensive
and slow; that is what search_files is for. Every file you read stays in this conversation for the rest
of this attempt, so read what you need, not everything that might be relevant.

Every path you pass to these tools is relative to the project's code root, which is where these tools
already operate - do NOT prefix paths with "code/". If the spec describes a file as e.g.
"code/wordstats/graph.py", the path you actually pass to these tools is "wordstats/graph.py"; a path
like "smoke_run.py" belongs directly at that root, not inside a "code/" subfolder you create yourself.
Use list_files(".") first if you're ever unsure what's already there and at what depth.

If an installed dependency (a framework, a library) appears to be behaving unexpectedly - an error
that doesn't make sense from your own code alone, a value that doesn't match what you passed in, a
parameter silently being replaced - read that library's OWN source instead of guessing at why: prefix
a path with "site-packages/" (e.g. "site-packages/langgraph/graph/state.py", or search_files with
path="site-packages/langgraph" for a term you'd expect near the behavior) to look inside it, exactly
like you'd look inside your own project. This is read-only - write_file/edit_file/run_command never
touch anything outside the project - but read access to your own dependencies is often the fastest,
most certain way to understand a real bug instead of forming an untested theory about it.

Work iteratively: locate what's relevant (search_files, list_files, read_file) before changing anything,
make your change (edit_file/write_file), then verify it for real with run_command - run pytest, and run
smoke_run.py - before considering anything done. Include at least one real pytest test file under a
tests/ subfolder that exercises the spec's behavior, in addition to the smoke test below - they check
different things.

MANDATORY smoke test - every build must include a smoke_run.py that actually runs the real
system's entry point once, end to end (not a mocked/stubbed version of it - the real thing), proving
it genuinely works rather than just that whatever tests you wrote don't fail. This is what catches the
kind of bug that survives every unit test because none of them actually exercise the real path.
Requirements:
- Runnable standalone as `python smoke_run.py` with no CLI arguments. Reuse a real example input under
  examples/ if this project already has one; otherwise construct a minimal realistic one inline.
- Calls the real top-level entry point for whatever is built and in scope THIS round. If only part of
  the system is built so far (a staged/partial build), exercise as much of the currently in-scope,
  already-built work as you genuinely can - never invoke functionality that belongs to a later,
  not-yet-built stage.
- Reads the SMOKE_MODE environment variable. When SMOKE_MODE=dry (the default, and the one that MUST
  always work, at zero real cost): stub/mock any call this project would make to a real, paid external
  vendor - reuse this project's own vendor-abstraction seam if it has one, or add a minimal one if it
  doesn't. When SMOKE_MODE=live: make the real calls. Support for live mode where the project's own
  architecture reasonably allows it.
- On success, print exactly one line of JSON to stdout as the LAST line of output:
  {{"ran": true, "output_summary": "<a short, real, specific description of what was actually produced -
  never a generic placeholder like 'success' or 'done'>"}}
- On a genuine failure, let the exception propagate (non-zero exit) rather than catching it and printing
  a fake success.
Run `python smoke_run.py` yourself via run_command before reporting done, and check that the JSON line
it prints is a real, specific summary - not a stub.

AUTHORITATIVE NAMES - never guess. Any table/column name, config key, constant or class field that
this unit reads or writes must come from the authoritative definition (a dependency's migration or
schema, the config file and the config loader's real output, the dependency's class fields shown in the
task or read with read_file). Never probe for alternative names at runtime, never read a key the
authoritative definition does not produce, and never silently filter or drop fields to make them fit an
existing schema. If a needed column, key or field does not exist, add it properly (a migration, a config
entry) or stop with AMBIGUOUS. Tests must exercise the REAL loader/schema where one exists, not a stub
that happens to have the keys your code expects.

DEPENDENCIES - every third-party package your code imports (including tzdata when zoneinfo is used)
must be listed in code/requirements.txt; verification fails otherwise. You cannot install packages
yourself; the pipeline installs requirements.txt into the project's environment.

When you are confident the work is complete and genuinely verified (pytest passes, and smoke_run.py
really ran), respond with a short plain-text summary of what you did and do NOT call any more tools -
that absence of a tool call is how you signal you're finished. If you are genuinely blocked on a
product-intent decision only a human can make (not a technical detail you can reasonably decide
yourself), respond with ONLY this single line instead, with nothing else:
AMBIGUOUS: <the exact question for the human>

{_MODEL_IDENTIFIER_RULE}"""

_STUB_JS_BODIES = {"return items;", "return $input.all();", "return $input.all()", "return items"}

# Built-in n8n node types seen in practice. Used ONLY for typo detection (see _validate_node_type):
# an unrecognised type is allowed through, but one that is a near-miss of a real type is rejected.
# That way this list never has to be exhaustive to be useful, and can't block legitimate node types
# it simply hasn't heard of.
_KNOWN_N8N_TYPES = {
    "n8n-nodes-base.cron",
    "n8n-nodes-base.scheduleTrigger",
    "n8n-nodes-base.code",
    "n8n-nodes-base.function",
    "n8n-nodes-base.if",
    "n8n-nodes-base.switch",
    "n8n-nodes-base.merge",
    "n8n-nodes-base.set",
    "n8n-nodes-base.noOp",
    "n8n-nodes-base.wait",
    "n8n-nodes-base.httpRequest",
    "n8n-nodes-base.webhook",
    "n8n-nodes-base.telegram",
    "n8n-nodes-base.telegramTrigger",
    "n8n-nodes-base.emailSend",
    "n8n-nodes-base.gmail",
    "n8n-nodes-base.googleSheets",
    "n8n-nodes-base.dataTable",
    "n8n-nodes-base.splitInBatches",
    "n8n-nodes-base.itemLists",
    "n8n-nodes-base.dateTime",
    "n8n-nodes-base.manualTrigger",
    "n8n-nodes-base.errorTrigger",
    "n8n-nodes-base.stopAndError",
    "n8n-nodes-base.filter",
}

# Identifiers whose correct casing is fixed by the JavaScript language. Wrong casing is valid
# SYNTAX but fails at runtime, so a syntax check can't see it - seen repeatedly in generated
# output (Math.random().tostring(36), .toisostring(), .getfullyear()).
#
# These are auto-corrected rather than sent back to the model: the correction is fully determined
# (there is exactly one right spelling), so re-prompting spends tokens to maybe get it right. In
# practice deepseek-v4-pro produced '.toisostring' on six consecutive attempts across two runs for
# the same node, ignoring the feedback each time. Mechanical problems get mechanical fixes; LLM
# retries are reserved for problems that actually need generation.
_JS_KNOWN_IDENTIFIERS = (
    "toString",
    "toISOString",
    "toLocaleDateString",
    "toLocaleTimeString",
    "toLowerCase",
    "toUpperCase",
    "toFixed",
    "getFullYear",
    "getMonth",
    "getDate",
    "getHours",
    "getMinutes",
    "getTime",
    "parseInt",
    "parseFloat",
    "JSON.stringify",
    "JSON.parse",
    "Math.random",
    "Math.floor",
    "Math.round",
    "Math.abs",
    "Number.isInteger",
    "Array.isArray",
    "Object.keys",
    "Object.values",
    "Object.entries",
)

# Anchored on word boundaries so only whole identifiers match, and a match is only rewritten when
# its actual spelling differs from the canonical one. Dotted globals (JSON.parse, Math.random, ...)
# additionally require that nothing precedes them, so a genuine property path is never rewritten -
# n8n code is full of `item.json.<field>`, and turning that into `item.JSON.<field>` would break it.
_JS_IDENTIFIER_PATTERNS = tuple(
    (
        correct,
        re.compile(
            (r"(?<![.\w])" if "." in correct else r"\b") + re.escape(correct) + r"\b",
            re.IGNORECASE,
        ),
    )
    for correct in _JS_KNOWN_IDENTIFIERS
)


def _autofix_js_casing(js: str):
    """Correct wrong-cased known JavaScript identifiers in place.

    Returns (fixed_js, [descriptions of what changed]). Safe because every occurrence of a given
    identifier is normalised to the same canonical spelling: even in the pathological case where a
    variable genuinely was named `tostring`, it gets renamed consistently throughout and the code
    still behaves the same.
    """
    if not js:
        return js, []
    fixes = []
    for correct, pattern in _JS_IDENTIFIER_PATTERNS:
        def replace(match, correct=correct):
            found = match.group(0)
            if found != correct:
                fixes.append(f"{found} -> {correct}")
                return correct
            return found

        js = pattern.sub(replace, js)
    return js, fixes


def _find_js_casing_problem(js: str) -> str | None:
    """Detection counterpart of the auto-fix, for JS that arrived from somewhere else (e.g. an
    older generated workflow) and never went through the builder's repair path.
    """
    _, fixes = _autofix_js_casing(js or "")
    if fixes:
        wrong, right = fixes[0].split(" -> ")
        return f"uses '{wrong}' - JavaScript is case-sensitive, this must be '{right}'"
    return None

_NODE_EXE_CHECKED = False
_NODE_EXE_AVAILABLE = False


def _node_available() -> bool:
    """Whether a Node.js binary is on PATH, so generated Code-node JS can be syntax-checked for
    real. Cached; absence is not an error - the check simply degrades to the cheaper heuristics.
    """
    global _NODE_EXE_CHECKED, _NODE_EXE_AVAILABLE
    if not _NODE_EXE_CHECKED:
        _NODE_EXE_CHECKED = True
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=15)
            _NODE_EXE_AVAILABLE = result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _NODE_EXE_AVAILABLE = False
    return _NODE_EXE_AVAILABLE


def _check_js_syntax(js: str) -> str | None:
    """Parse the JS with Node itself. Catches genuinely broken output that the stub check can't -
    e.g. a mangled escape producing `});\\return items;`, which is a hard syntax error that would
    fail on import. Nothing is executed, only parsed.

    The body is wrapped in an async function first, because that is how n8n actually runs a Code
    node: `return items;` at the top level of the snippet is correct there, but is a syntax error
    in a bare script/module. Without this wrapper the check rejected perfectly valid n8n code.
    """
    if not _node_available():
        return None
    wrapped = "async function __n8nCodeNode() {\n" + js + "\n}\n"
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8") as handle:
            handle.write(wrapped)
            temp_path = handle.name
    except OSError:
        return None
    try:
        result = subprocess.run(["node", "--check", temp_path], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip().splitlines()
            message = next((line for line in detail if "Error" in line or "error" in line), "")
            return f"is not valid JavaScript: {message or 'syntax error'}"
        return None
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        try:
            Path(temp_path).unlink()
        except OSError:
            pass


def _is_stub_js(js: str) -> str | None:
    """Returns a problem description if the JS looks like a stub or is broken, else None."""
    js = (js or "").strip()
    if len(js) < 20:
        return "empty or trivially short JS - looks like a stub"
    if js in _STUB_JS_BODIES:
        return "just passes input through unchanged - not implemented"
    casing_problem = _find_js_casing_problem(js)
    if casing_problem:
        return casing_problem
    api_problem = _find_unavailable_api(js)
    if api_problem:
        return api_problem
    return _check_js_syntax(js)


# Helpers a model confidently invented that do not exist in n8n. See docs/n8n-ground-truth.md.
# Cheap to check, and the failure is otherwise silent until live execution.
_FABRICATED_N8N_APIS = (
    "$getWorkflowDataTable",
    "$httpNode",
    "$http.get",
    "$http.request",
    "$credentials.",
    "$telegram.",
    "$cancelScheduledExecution",
)

_REAL_N8N_CODE_APIS = "$getWorkflowStaticData, $helpers.httpRequest, $input, $json, $('Node Name'), $jmespath, DateTime (Luxon, global)"


def _find_unavailable_api(js: str) -> str | None:
    """Reject calls to things that simply don't exist in an n8n Code node, plus require(), which
    n8n disallows outright. All were produced confidently by a model and all fail only at runtime.
    """
    for fake in _FABRICATED_N8N_APIS:
        if fake in js:
            return (
                f"uses '{fake}', which does not exist in n8n. Available in a Code node: "
                f"{_REAL_N8N_CODE_APIS}. Data tables are reachable ONLY via the "
                f"n8n-nodes-base.dataTable node, never from code."
            )
    if re.search(r"\brequire\s*\(", js):
        return (
            "calls require(), which n8n Code nodes disallow. Luxon's DateTime is already a global; "
            "no other module can be imported."
        )
    if "process.env" in js:
        return (
            "reads process.env directly - n8n's Code node sandbox explicitly blocks this and fails "
            "at runtime with 'access to env vars denied'. Use $env.VARIABLE_NAME instead."
        )
    return None


def _validate_n8n_versioning(node: dict) -> str | None:
    """Catch the version/syntax mismatch that silently matches nothing at runtime.

    Filter-style conditions need Switch typeVersion 3+ / IF typeVersion 2+. A Switch declared
    typeVersion 1 carrying v3-style rules passes structural validation, saves and renders fine, and
    then matches nothing - every item falls through with no error. Only live execution revealed it.
    """
    node_type = str(node.get("type", ""))
    params = node.get("parameters")
    if not isinstance(params, dict):
        return None

    def _uses_filter_syntax(blob) -> bool:
        if isinstance(blob, dict):
            if "conditions" in blob and ("combinator" in blob or "options" in blob):
                return True
            return any(_uses_filter_syntax(v) for v in blob.values())
        if isinstance(blob, list):
            return any(_uses_filter_syntax(v) for v in blob)
        return False

    try:
        type_version = float(node.get("typeVersion") or 0)
    except (TypeError, ValueError):
        type_version = 0.0

    if node_type.endswith(".switch"):
        if _uses_filter_syntax(params) and type_version < 3:
            return (
                f"uses filter-style conditions but declares typeVersion {node.get('typeVersion')!r} - "
                f"that syntax requires Switch typeVersion 3 or higher, or it silently matches nothing "
                f"at runtime."
            )
        fallback = params.get("fallbackOutput")
        if fallback is not None and not isinstance(fallback, str):
            return (
                f"has fallbackOutput={fallback!r} - it must be the string \"extra\" to create a real "
                f"fallback port; a number just routes to an existing rule output."
            )
    if node_type.endswith(".if") and _uses_filter_syntax(params) and type_version < 2:
        return (
            f"uses filter-style conditions but declares typeVersion {node.get('typeVersion')!r} - "
            f"that syntax requires IF typeVersion 2 or higher."
        )
    if node_type in ("n8n-nodes-base.telegram", "n8n-nodes-base.telegramTrigger") and type_version >= 2:
        return (
            f"declares typeVersion {node.get('typeVersion')!r} - real incident: typeVersion 2 for this "
            f"node type passed structural validation AND executed correctly at runtime, but n8n's own "
            f"publish/activation validation rejected it as having 'missing or invalid required "
            f"parameters' and silently wiped the node's parameters to {{}} the next time the workflow "
            f"was opened, breaking it with no warning until someone tried to (re)publish. Use "
            f"typeVersion 1 instead - proven correct end-to-end (build, execute, AND publish) with flat "
            f"parameters: {{chatId, text, additionalFields}} for n8n-nodes-base.telegram, "
            f"{{updates: [...], additionalFields: {{}}}} for n8n-nodes-base.telegramTrigger."
        )
    return None


def _expected_output_count(node: dict) -> int | None:
    """How many outputs a branching node's own parameters actually define, or None if it can't be
    determined confidently. Used to catch connections wired to outputs that can never fire - seen
    in practice: a switch defining 2 outcomes but wired to 4 downstream nodes, so two branches were
    silently dead. Returns None rather than guessing, so an unfamiliar parameter shape is never
    reported as a false problem.
    """
    node_type = str(node.get("type", ""))
    if node_type.endswith(".if"):
        return 2  # an IF node is always true/false
    if not node_type.endswith(".switch"):
        return None

    params = node.get("parameters")
    if not isinstance(params, dict):
        return None

    rules = params.get("rules")
    rule_count = None
    if isinstance(rules, dict):
        for key in ("rules", "values", "conditions"):
            candidate = rules.get(key)
            if isinstance(candidate, list):
                rule_count = len(candidate)
                break
    elif isinstance(rules, list):
        rule_count = len(rules)

    if rule_count is None:
        for key in ("numberOutputs", "outputsAmount"):
            if isinstance(params.get(key), int):
                return params[key]
        return None

    options = params.get("options") if isinstance(params.get("options"), dict) else {}
    has_fallback = bool(params.get("fallbackOutput") is not None or options.get("fallbackOutput") is not None)
    return rule_count + (1 if has_fallback else 0)


def _validate_node_type(node_type: str) -> str | None:
    """Rejects a node type that is a near-miss of a real one (e.g. 'n8n-nodes-base.codee'), which
    n8n would refuse to load. Types it simply doesn't recognise are allowed through - this list
    isn't exhaustive and shouldn't block legitimate nodes.
    """
    node_type = (node_type or "").strip()
    if not node_type or node_type in _KNOWN_N8N_TYPES:
        return None
    close = difflib.get_close_matches(node_type, _KNOWN_N8N_TYPES, n=1, cutoff=0.9)
    if close:
        return f"type {node_type!r} is not a real n8n node type - did you mean {close[0]!r}?"
    return None


_TRIGGER_TYPE_HINTS = ("trigger", "cron", "webhook", "schedule", "interval")


def _looks_like_trigger(node_type: str) -> bool:
    """True for anything that's exempt from needing an incoming connection - by definition nothing
    feeds a trigger. Same hint list n8n_verify.py uses to spot trigger-shaped nodes, kept local
    here rather than imported to avoid reaching into that module's private names.
    """
    lowered = (node_type or "").lower()
    return any(hint in lowered for hint in _TRIGGER_TYPE_HINTS)


def _run_pytest(code_dir: Path):
    # cwd=str(code_dir) added 2026-09-28 (Douglas's direct instruction, found while verifying the
    # previous commit): this call passed str(code_dir) as pytest's TARGET argument but never set
    # the subprocess's own working directory - unlike _run_smoke below, which always has. A relative
    # path resolved by the PROJECT's own code (e.g. a config's storage_root defaulting to "./data")
    # resolved against wherever build_and_review.py itself was launched from, not the project being
    # built. Real, evidenced incident: a creative-team smoke-test run wrote build-agent/data/
    # acme-property/... - real job output landing in the wrong repo entirely, since every invocation
    # that day had been launched from build-agent's own directory. Harmless so far (untracked,
    # nothing sensitive) but would only get worse with more than one project running through this
    # pipeline. Swept every other subprocess.run/Popen call in this file and agent_tools.py for the
    # same gap - run_command and _run_smoke already had it; _check_package_still_imports (added the
    # same day) already had it; the two `node` subprocess calls check an installed binary's version
    # and an isolated temp-file JS snippet respectively, neither ever resolving anything relative to
    # a project directory, so they don't need it. This was the only genuine gap.
    #
    # Target argument is "." (not str(code_dir)) now that cwd is pinned to code_dir - code_dir
    # itself is NOT guaranteed to be absolute (build_and_review.py's project_dir = Path(args.
    # project_dir) never calls .resolve()), so passing the same possibly-relative string as BOTH
    # cwd and the target would double-resolve it against the new cwd, e.g. "creative-team/code"
    # would become cwd's own "creative-team/code/creative-team/code" - a real regression this
    # avoids entirely rather than depending on code_dir happening to be absolute.
    python, env_error = ensure_project_python(code_dir)
    if env_error:
        return False, env_error
    result = subprocess.run(
        [python, "-m", "pytest", ".", "-q"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(code_dir),
    )
    passed = result.returncode == 0
    return passed, (result.stdout + "\n" + result.stderr)[-6000:]


# Real gap this closes (2026-09-27, Douglas's own diagnosis): pytest passing only proves whatever
# tests already exist don't fail - zero tests is a pass. That's the actual root cause behind the
# test-coverage-gap incident chain earlier the same day. n8n gets a genuine live-execution check
# (pipeline/n8n_verify.py pushes the real workflow to a real instance); python never had an
# equivalent - everything rested on the builder's own self-selected, self-graded test suite. This is
# that equivalent: code/smoke_run.py must actually import and run the real built entry point once,
# end to end, independent of whatever pytest file the builder chose to write. The contract itself
# (dry/live modes, the required JSON summary line) is documented once, in AGENT_SYSTEM_PROMPT above
# - no longer a separate opt-in addendum, since running it is now a mandatory, unconditional part of
# every agentic build (pipeline/smoke_gate.py's old per-project opt-in marker is retired for exactly
# this reason - see AGENTS.md).
SMOKE_RUN_FILENAME = "smoke_run.py"


def _run_smoke(code_dir: Path, mode: str = "dry"):
    """Executes code/smoke_run.py as an isolated subprocess - same pattern and same risk profile as
    _run_pytest above (a real subprocess, a real timeout, no new exposure category): this does not
    run untrusted code any more than the existing pytest gate already does, it just checks a
    different thing (does the real system actually run, not just does an existing test not fail).

    mode: "dry" (default, zero real external cost - see the smoke-test contract in AGENT_SYSTEM_PROMPT)
    or "live" (real calls to real, paid vendors - NEVER run without fresh, per-invocation consent; see
    pipeline/build_and_review.py's --live-smoke flag and AGENTS.md's live-smoke hard rule). This
    function itself does not decide whether live mode is authorized - the caller must only ever pass
    mode="live" after that consent has actually been obtained for this exact run.
    """
    smoke_path = code_dir / SMOKE_RUN_FILENAME
    if not smoke_path.exists():
        return False, (
            f"{SMOKE_RUN_FILENAME} is missing - required so a build round can be verified by "
            "actually running the real system, not just reading the code or trusting your own "
            "pytest suite. See the smoke-test requirements above."
        )
    env = {**os.environ, "SMOKE_MODE": mode}
    python, env_error = ensure_project_python(code_dir)
    if env_error:
        return False, env_error
    try:
        result = subprocess.run(
            [python, SMOKE_RUN_FILENAME],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"{SMOKE_RUN_FILENAME} ({mode} mode) did not finish within 120 seconds - timed out."

    if result.returncode != 0:
        return False, (
            f"{SMOKE_RUN_FILENAME} crashed in {mode} mode (exit code {result.returncode}):\n"
            f"{(result.stdout + result.stderr)[-4000:]}"
        )

    stdout_lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    if not stdout_lines:
        return False, f"{SMOKE_RUN_FILENAME} ({mode} mode) exited cleanly but printed no output at all."
    try:
        payload = json.loads(stdout_lines[-1])
    except (ValueError, TypeError):
        return False, (
            f"{SMOKE_RUN_FILENAME} ({mode} mode) exited cleanly but its last output line was not the "
            f"required JSON summary: {stdout_lines[-1][:500]}"
        )
    if not isinstance(payload, dict) or not payload.get("ran") or not str(payload.get("output_summary") or "").strip():
        return False, (
            f"{SMOKE_RUN_FILENAME} ({mode} mode) ran but its own summary reports no real output - "
            "this looks like a stub, not a genuine run."
        )
    return True, ""


def _find_unresolved_expression_strings(value, path: str = "parameters") -> list[str]:
    """Recursively finds string values inside a node's `parameters` that contain a `{{` mustache
    expression but don't start with a leading `=` - n8n only evaluates `{{ }}` inside a parameter
    when the ENTIRE string value is prefixed with `=` (the same convention already used correctly
    for simple fields like a URL: `"={{ $env.X }}"`). Without it, the whole string - including the
    unresolved `{{ ... }}` text - is sent through completely literally.

    Real incident this fixes: an HTTP Request node's multi-line jsonBody mixed literal JSON text
    with several `{{ $json.x }}` expressions but was missing the leading `=`, so n8n never evaluated
    any of them - the literal `{{ $json.user_id }}` text landed in the outgoing request body, which
    then failed n8n's own "is this valid JSON" check with an error pointing exactly at the unresolved
    `{{`. It reached a live Telegram test before being caught, because build-time live verification
    exercised the authorization-gate branch, not the branch that calls this node.
    """
    problems = []
    if isinstance(value, str):
        if "{{" in value and not value.startswith("="):
            problems.append(
                f"{path}: contains a {{{{ }}}} expression but the value doesn't start with '=', so "
                f"n8n treats the WHOLE string as literal text and never evaluates the expression - "
                f"prefix the entire value with '=' (e.g. \"={{{{ $json.x }}}}\" for a pure expression, "
                f"or \"=literal text {{{{ $json.x }}}} more text\" for mixed content)."
            )
    elif isinstance(value, dict):
        for key, sub in value.items():
            problems.extend(_find_unresolved_expression_strings(sub, f"{path}.{key}"))
    elif isinstance(value, list):
        for i, sub in enumerate(value):
            problems.extend(_find_unresolved_expression_strings(sub, f"{path}[{i}]"))
    return problems


def _validate_n8n_workflow(workflow_path: Path):
    """Structural self-check for a generated n8n workflow - no n8n instance required. Catches the
    failure class actually seen in practice: stub Code nodes, and connections referencing nodes that
    don't exist. Not a substitute for opening it in real n8n, but a cheap automated gate before that.

    Takes the exact workflow file path rather than guessing (e.g. "largest .json under code_dir") -
    that heuristic broke in practice once checkpointing added .n8n_build_checkpoint.json alongside
    it, which is often larger than the real workflow and got picked by mistake.
    """
    if not workflow_path.exists():
        return False, f"No workflow file found at {workflow_path}."
    try:
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"{workflow_path.name} is not valid JSON: {exc}"

    problems = []
    nodes = workflow.get("nodes")
    connections = workflow.get("connections")
    if not isinstance(nodes, list) or not nodes:
        problems.append(f"{workflow_path.name}: 'nodes' is missing, not a list, or empty.")
        nodes = []
    if not isinstance(connections, dict):
        problems.append(f"{workflow_path.name}: 'connections' is missing or not an object.")
        connections = {}

    node_names = set()
    branch_node_types = {}
    branch_nodes = {}
    ids_seen = {}
    for node in nodes:
        if not isinstance(node, dict):
            problems.append(f"A node entry is not an object: {node!r}")
            continue
        name = node.get("name")
        if not name:
            problems.append(f"A node is missing its 'name' field: {node}")
            continue
        node_names.add(name)
        node_type = str(node.get("type", ""))
        type_problem = _validate_node_type(node_type)
        if type_problem:
            problems.append(f"Node '{name}': {type_problem}")
        version_problem = _validate_n8n_versioning(node)
        if version_problem:
            problems.append(f"Node '{name}' {version_problem}")
        node_id = node.get("id")
        if node_id:
            ids_seen.setdefault(node_id, []).append(name)
        if node_type.endswith(".if") or node_type.endswith(".switch"):
            branch_node_types[name] = node_type
            branch_nodes[name] = node
        if "code" in node_type.lower():
            params = node.get("parameters") or {}
            js = params.get("jsCode") or params.get("functionCode") or ""
            stub_problem = _is_stub_js(js)
            if stub_problem:
                problems.append(f"Code node '{name}' {stub_problem}.")
            language = params.get("language")
            if language is not None and language not in ("javaScript", "pythonNative"):
                problems.append(
                    f"Code node '{name}' sets language: {language!r}, which n8n doesn't recognize "
                    f"and fails to activate with 'Unsupported language' - valid values are exactly "
                    f"'javaScript' or 'pythonNative' (case-sensitive), or omit the field entirely "
                    f"(defaults to javaScript)."
                )
            if "$input.item" in js and params.get("mode") != "runOnceForEachItem":
                problems.append(
                    f"Code node '{name}' uses $input.item (the 'Run Once for Each Item' convention) "
                    f"but parameters.mode is not set to 'runOnceForEachItem', so it defaults to "
                    f"'Run Once for All Items' instead - a real mismatch that produces no error, just "
                    f"silently wrong data (every field present, every value null/undefined). Either "
                    f"add \"mode\": \"runOnceForEachItem\" to parameters, or rewrite the code to use "
                    f"$input.all() and return an array of {{json: {{...}}}} objects instead."
                )

        for expr_problem in _find_unresolved_expression_strings(node.get("parameters") or {}):
            problems.append(f"Node '{name}' {expr_problem}")

    # Branching coherence. n8n runs EVERY target attached to a given output, so an IF node with
    # only its true branch wired silently drops the else-path, and a decision node whose outcomes
    # all hang off one output fires them together instead of choosing between them.
    # n8n requires node ids to be unique; a duplicate makes the whole workflow fail to import.
    # Assembly assigns these mechanically, so this is a safety net for workflows produced elsewhere.
    for node_id, owners in ids_seen.items():
        if len(owners) > 1:
            problems.append(
                f"Node id {node_id!r} is used by {len(owners)} nodes ({', '.join(owners[:5])}"
                f"{', ...' if len(owners) > 5 else ''}) - n8n requires ids to be unique."
            )

    # Reachability. A node can pass every other check and still never run if nothing actually
    # feeds it - neither the branch-wiring check above (which only looks at a branch node's own
    # OUTGOING wiring) nor live verification (which only reports errors from nodes that actually
    # executed - an unreached node just silently never runs, which looks identical to "no problems"
    # ) catches this. Confirmed in practice: half a workflow's nodes were disconnected islands and
    # both checks reported clean; only a full reviewer read caught it. Triggers are exempt - by
    # definition nothing feeds them.
    targeted_names = set()
    for outputs in connections.values():
        for branch in (outputs or {}).get("main") or []:
            for target in branch or []:
                target_name = (target or {}).get("node")
                if target_name:
                    targeted_names.add(target_name)
    for node in nodes:
        if not isinstance(node, dict):
            continue
        name = node.get("name")
        node_type = str(node.get("type", ""))
        if not name or _looks_like_trigger(node_type):
            continue
        if name not in targeted_names:
            problems.append(
                f"Node '{name}' has no incoming connection - nothing in the workflow ever reaches "
                f"it, so it will never run. Wire something to it, or remove it if it's not needed."
            )

    for name, node_type in branch_node_types.items():
        outputs = (connections.get(name) or {}).get("main") or []
        wired = [i for i, targets in enumerate(outputs) if targets]

        # Outputs wired beyond what the node's own rules define can never fire.
        expected = _expected_output_count(branch_nodes.get(name) or {})
        if expected is not None and wired and max(wired) >= expected:
            dead = [i for i in wired if i >= expected]
            problems.append(
                f"{'IF' if node_type.endswith('.if') else 'Switch'} node '{name}' defines "
                f"{expected} output(s) but has connections on output(s) {dead} - those branches "
                f"can never fire. Add matching rules, or rewire them."
            )

        if node_type.endswith(".if") and len(wired) < 2:
            problems.append(
                f"IF node '{name}' has only output(s) {wired or 'none'} wired - both the true (0) "
                f"and false (1) branches must go somewhere, or one outcome is silently dropped."
            )
        elif node_type.endswith(".switch") and len(wired) < 2:
            problems.append(
                f"Switch node '{name}' has only output(s) {wired or 'none'} wired - a switch with "
                f"fewer than two used outputs isn't choosing between anything."
            )

    for source_name, outputs in connections.items():
        if source_name not in node_names:
            problems.append(f"'connections' has source node '{source_name}' which doesn't exist in 'nodes'.")
        for output_list in (outputs or {}).values():
            for branch in output_list or []:
                for target in branch or []:
                    target_name = (target or {}).get("node")
                    if target_name and target_name not in node_names:
                        problems.append(
                            f"'connections' has target node '{target_name}' which doesn't exist in 'nodes'."
                        )

    if problems:
        return False, "\n".join(problems)
    return True, f"{workflow_path.name}: nodes/connections present, Code nodes appear implemented."


# --- Chunked n8n build: plan -> one node per LLM call, validated individually -> assemble -> final
# validate/targeted repair. Replaces asking for one giant workflow JSON in a single completion, which
# in practice triggered runaway reasoning-token consumption on large specs (empty/truncated output
# regardless of model token ceiling). Each call here is small and independently bounded/retryable, so
# a bad piece costs a cheap retry of just that piece, not the whole build. n8n-platform only - the
# python platform's single-shot loop below is untouched.

_N8N_PLAN_SYSTEM_PROMPT = """You are planning an n8n workflow's structure from a product spec - not writing
any node content yet, just the plan.

Output using EXACTLY this delimited format and nothing else - no JSON, no markdown fences, no
commentary. Free text inside these blocks is raw: never escape quotes, newlines or braces.

<<<WORKFLOW: the workflow's name>>>

<<<CONVENTIONS>>>
timezone: the one timezone everything uses
- a shared rule every node must follow
- another shared rule
<<<END_CONVENTIONS>>>

<<<TRIGGER: Exact Node Name>>>
production_type: the REAL n8n trigger type this becomes once deployed - n8n-nodes-base.telegramTrigger,
n8n-nodes-base.cron, n8n-nodes-base.webhook, n8n-nodes-base.scheduleTrigger, etc, whichever the spec
actually calls for
production_parameters: a single-line JSON object of that trigger's real configuration (e.g. a cron
expression, or {"updates": ["message"]} for Telegram)
<<<END_TRIGGER>>>

(one <<<TRIGGER: ...>>> block per distinct trigger the spec needs - a workflow with a daily check-in
AND a Telegram command needs two. Do NOT also write a <<<NODE: ...>>> block for a trigger - it is
built for you, from this declaration alone. Reference a trigger by its exact name in <<<CONNECTIONS>>>
exactly like any other node.

A trigger named "X" automatically gets a companion node called "[Trigger Adapter] X" inserted
immediately after it - this ONLY unwraps the raw webhook envelope ({headers, params, query, body} ->
just body), nothing more. You never write this node yourself, and connecting FROM "X" in your
<<<CONNECTIONS>>> already accounts for it (it's inserted transparently in between).

That adapter does NOT do any product-level reshaping of the payload - if the spec needs the
trigger's raw data reshaped into a different shape (e.g. flattening nested fields, renaming keys,
extracting one field from several), that is real product logic and you MUST write your own
<<<NODE: ...>>> for it, exactly like any other node. Real incident this note exists because of:
a plan named its own reshaping node "Normalize: Telegram Trigger" - which collided with this exact
reserved adapter-name pattern used to be ("Normalize: {trigger name}") - and the mechanical adapter
silently won the name collision, discarding the plan's real reshaping logic with no error at all;
downstream nodes then read fields that were never actually populated. The adapter's name was changed
specifically so this can't recur, but never name one of your own nodes "[Trigger Adapter] ..." or
reuse a trigger's own exact name for a different node.)

<<<NODE: Exact Node Name>>>
type: n8n-nodes-base.xxx
slice: the functional slice this node belongs to
purpose: a full paragraph - exactly what this node must do, its inputs/outputs, and, for Code nodes,
the precise logic/algorithm/thresholds it must implement, in enough detail that someone could write
it later without re-reading the spec. May run over several lines.
<<<END_NODE>>>

(repeat one <<<NODE: ...>>> block per non-trigger node)

<<<CONNECTIONS>>>
Source Node Name -> Target Node Name [0]
Some IF Node -> Node For True Case [0]
Some IF Node -> Node For False Case [1]
<<<END_CONNECTIONS>>>

The number in [brackets] on each connection is the source node's OUTPUT INDEX, and it is how
branching is expressed:
- A normal node has one output: use [0].
- An IF node ("n8n-nodes-base.if") has TWO outputs: [0] = the true branch, [1] = the false branch.
  You MUST wire both, to different targets.
- A Switch node ("n8n-nodes-base.switch") has one output per rule, indexed [0], [1], [2]... in the
  order its rules are defined.

CONDITIONAL LOGIC IS MANDATORY WHERE THE SPEC HAS IT.
Anywhere the spec says something happens only under a condition - success vs failure, answered vs
timed out, one tier vs another, suppressed vs sent - you MUST insert a real
"n8n-nodes-base.if" or "n8n-nodes-base.switch" node to make that decision and wire each outcome to
its own output index. Do NOT connect several mutually-exclusive targets to the same output of the
same node: in n8n every target on one output runs, every time, in parallel. A workflow that lists
"handle success", "handle failure" and "handle no response" on one node's output 0 is wrong - it
would run all three at once.

Also look for ONE-TIME SETUP the spec requires: if the spec provides concrete seed/reference data
that must exist before the product can work (a starting library, a lookup table, default records),
plan an explicit node that loads it, rather than assuming it is already there.

When the spec supplies CONCRETE DATA - named records, specific values, fixed lists - copy it into
that node's purpose VERBATIM, character for character, every field of every record. Whoever
implements that node later sees only its purpose and never the spec: anything you summarise as
"the seed routes from the spec" or "the standard set" will be invented from scratch instead. In
practice a seed node described that way produced nine plausible made-up records in place of the
seven real ones the spec listed. Long is fine here - reproducing the data exactly is the point.

Each node's "slice" is the functionally coherent chunk it belongs to - a capability a human would
recognise ("daily check-in", "route generation", "run logging"), not an arbitrary batch. Order your
node blocks so that all of a slice's nodes appear together, and so earlier slices stand on their own
as far as possible. This exists so a build can stop at a working functional base and be reviewed
before continuing, instead of only being judgeable once all 40 nodes exist.

"conventions" is CRITICAL and must be filled in carefully. Each node is later implemented in isolation by
someone who will NOT see the spec - only its own "purpose" plus these conventions. Anything that must be
consistent across nodes has to appear here or it WILL be implemented inconsistently. Extract from the spec
and state explicitly, as short imperative rules in "shared_rules": the exact timezone to use everywhere;
every measurement unit; the exact numeric range/scale of every score or rating; the exact names and column
schemas of every Data Table; the exact names of every credential/API key; date and time formats; ID
formats; any shared constants, thresholds or tier boundaries; and any naming conventions for fields passed
between nodes. Be specific and literal (e.g. "All cron schedules and date maths use Asia/Singapore, never
UTC or any other zone", "Feel-good score is an integer 1-10 inclusive, never 0-100"). Do not invent values
the spec doesn't state - if the spec doesn't say, leave it out rather than guessing.

Be thorough: every distinct piece of decision logic, every Data Table read/write, every message sent,
needs its own <<<NODE>>> with a fully self-contained purpose description (triggers are declared
separately above, not as nodes). Use real n8n built-in node types (n8n-nodes-base.telegram,
n8n-nodes-base.httpRequest, n8n-nodes-base.code, n8n-nodes-base.dataTable, etc.) for everything else.
If, and only if, the spec is genuinely ambiguous about what the product should do (not a technical detail
you can reasonably decide yourself), output ONLY: AMBIGUOUS: <the exact question for the human>
"""

_N8N_NODE_SYSTEM_PROMPT = """You are implementing exactly ONE node of a larger n8n workflow someone else
already planned. Output ONLY the single JSON object for this one node - real field values, no
placeholders: "id", "name" (must match exactly what was given), "type", "typeVersion", "position"
([x, y] - any simple numbers), "parameters". If this is a Code node ("n8n-nodes-base.code"), "parameters"
must include "jsCode" containing COMPLETE, CORRECT, working JavaScript that fully implements the stated
purpose - never a stub, never a placeholder, never something that just returns its input unchanged. For
every other node type, "parameters" must be realistic and complete for that node type's real purpose.
Output nothing except that one JSON object - no markdown fences, no commentary.

If, while building this node, you confirm something genuinely reusable about n8n ITSELF (not this
node's own business purpose) that isn't already covered below - a real API behavior, a version quirk,
something you had to work out by trial and error - you may put exactly one line before the JSON
object: `TECH_QUIRK: <the lesson, self-contained>`. This is rare; most nodes have nothing new to
report, and this is never a substitute for the JSON object itself. Every future build on any project
using this pipeline will see it, so only write one for something you've actually confirmed.

HARD n8n FACTS - all of these were got wrong before and failed only at runtime:

If any node in this workflow calls out to an LLM (an HTTP Request node hitting a model API, an AI
node, a hardcoded model string in a Code node), and the spec names a specific model for that role
("Claude Sonnet", "Gemini Deep Think", a specific OpenRouter slug), use EXACTLY that identifier,
copied verbatim - never substitute a different version or a similar-sounding one. Model selection
is reserved to Douglas alone; a real incident produced wrong, lower-tier model defaults this way
(anthropic/claude-3.5-sonnet and an experimental gemini-2.0-flash-thinking-exp substituted for
"Claude Sonnet" and "Gemini Deep Think"). If the spec names a role only generically with no pinned
version, don't invent a specific one either - read it from a required parameter/credential/env
value with no hardcoded fallback, so a misconfiguration fails loudly instead of silently running
on the wrong model.

Inside a Code node you may use ONLY: $getWorkflowStaticData('global'), $helpers.httpRequest({...}),
$input, $json, $('Node Name'), $jmespath, $env, and Luxon's DateTime as a global. require() is
DISALLOWED - no require('luxon'), no require('moment-timezone'). These DO NOT EXIST and must never
appear: $getWorkflowDataTable, $httpNode, $http.get, $credentials.<name>, $telegram.sendMessage,
$cancelScheduledExecution. Data tables are reachable ONLY through the n8n-nodes-base.dataTable node -
there is no way to read or write one from code.

A Code node's execution mode defaults to "Run Once for All Items" unless "parameters.mode" is
explicitly set to "runOnceForEachItem". These two modes use DIFFERENT conventions and mixing them
produces silently wrong output, not an error:
- Default (All Items): read via $input.all() (an array); return an ARRAY of {json: {...}} objects
  (one per output item), e.g. `return $input.all().map(item => ({ json: {...} }));` even for a
  single-item workflow.
- Each Item (only if parameters.mode is explicitly "runOnceForEachItem"): read the current item via
  $input.item.json; return ONE bare object (or {json: {...}}), not an array.
Never write code using $input.item.json (the Each-Item convention) without ALSO setting
parameters.mode to "runOnceForEachItem" - real incident: a node did exactly this, with mode left
unset (defaulting to All Items), and $input.item resolved to something with the right shape but
none of the real field values, silently producing an object with every field present but every
value null - no error, no crash, just wrong data flowing to every node downstream of it.

PREFER checking an environment variable directly in an ordinary node's own expression field (e.g.
an IF node's condition: {{ $env.VARIABLE_NAME.split(',').includes(...) }}) over reading it inside a
Code node's JS at all, whenever the logic is simple enough to express that way (a membership/
equality/threshold check feeding a branch). Real incident: a Code node using process.env.X - and,
after correcting that, the equally-correct $env.X - both failed identically at runtime with "access
to env vars denied", because that specific n8n instance disables ALL env var access inside the Code
node sandbox as a security setting. An ordinary node's condition field uses n8n's general
expression evaluator, a different code path from the Code node sandbox, and isn't subject to that
restriction - so this isn't just a fallback for when the block is hit, it sidesteps the whole class
of failure whether or not any given instance has that setting enabled. Only fall back to reading
$env inside a Code node (never process.env - see above) when the logic genuinely can't be expressed
as a single node's condition.

If a Code node's "language" parameter is set at all, it must be exactly "javaScript" or
"pythonNative" (case-sensitive) - "javascript", "JavaScript", "js", etc. all save and validate fine
but fail at runtime with "Unsupported language". Simplest: omit the field entirely, it defaults to
javaScript.

Version and syntax must match. Filter-style conditions ({options, conditions: [{leftValue,
rightValue, operator}], combinator}) require Switch typeVersion 3 or higher and IF typeVersion 2 or
higher. A Switch on typeVersion 1 with that syntax saves and renders fine but matches NOTHING at
runtime. Switch "fallbackOutput" must be the string "extra" to create a real fallback port; a number
routes to an existing output instead. The operator is "notEmpty" with singleValue: true, never
"isNotEmpty".

Portability - the workflow must survive being copied to a different n8n host. Reference data tables
BY NAME via the resourceLocator ({mode: "name", value: "Table Name"}), never by an instance-specific
id. Never hardcode a chat id or user id: derive it from the trigger payload, e.g.
$('Trigger').item.json.message.chat.id.

Remember that an HTTP Request node REPLACES the item json with its response, so downstream nodes must
recover upstream fields via $('Node Name'), and that nodes run once per input item, so chaining two
"fetch all rows" steps multiplies the rows.

For n8n-nodes-base.telegram and n8n-nodes-base.telegramTrigger, always use typeVersion 1 with flat
parameters - {chatId, text, additionalFields} for a send node, {updates: [...], additionalFields: {}}
for the trigger. Real incident: typeVersion 2 for these two node types passed structural validation
AND executed correctly at runtime (a real message was sent), but n8n's own publish/activation
validation rejected it as having missing/invalid required parameters, and the next time the workflow
was opened its parameters were silently wiped to {} - it broke with no warning until someone tried to
publish it. typeVersion 1 is proven correct end-to-end (build, execute, and publish) on real instances.

A parameter value is only evaluated as an expression if the ENTIRE string starts with a leading "=" -
this applies to every field, not just short ones. A multi-line field that mixes literal text with
several {{ }} expressions (e.g. an HTTP Request node's JSON Body) needs that SAME leading "=" on the
whole string, exactly like a short field such as a URL does. Real incident: an HTTP Request node's
jsonBody was written as `"{\n  \"user_id\": {{ $json.user_id }},\n  ...\n}"` - correct-looking
mustache syntax, but missing the leading "=" - so n8n treated the whole thing as literal text and
never evaluated any of it. The literal, unresolved "{{ $json.user_id }}" text then landed in the
outgoing request body and broke n8n's own JSON-body parser at exactly that position. It reached a
live Telegram test before being caught, because build-time verification happened to exercise a
different branch. Always write it as `"=\n{\n  \"user_id\": {{ $json.user_id }},\n  ...\n}"` instead
- the leading "=" belongs on the string as a whole, never repeated per-expression inside it.
"""


def _looks_like_n8n_workflow(data) -> bool:
    """Content-sniffed, not path-sniffed - a hybrid project's own file layout for its n8n shell
    isn't guaranteed to be any particular path, so this checks the actual JSON shape (top-level
    'nodes' list + 'connections' dict) rather than assuming a filename convention."""
    return isinstance(data, dict) and isinstance(data.get("nodes"), list) and isinstance(data.get("connections"), dict)


def _validate_written_n8n_files(code_dir: Path) -> list[str]:
    """Runs the same n8n structural safety net used for pure n8n-platform builds
    (_validate_n8n_workflow - the '=' -prefix expression check, the Telegram typeVersion check,
    the branch-wiring/reachability checks, etc.) against any n8n workflow JSON a python-platform
    (hybrid) build happens to produce anywhere under code_dir.

    Real incident this fixes: _validate_n8n_workflow was only ever called from the platform=="n8n"
    code path (see builder_node() below). A hybrid project (an n8n shell + a Python/LangGraph
    service, one project, one SPEC.md) is built entirely via the platform=="python" path instead -
    the python builder faithfully produces the n8n shell file too, but nothing n8n-specific ever
    checked it. It shipped with unresolved {{ }} expressions missing their leading "=" (silently
    sent as literal text) and Telegram nodes at a typeVersion that passes validation and executes
    fine, then gets its own parameters silently wiped the next time the workflow is opened for
    publish - both real, previously-fixed incidents that this same hybrid project's OWN later
    rebuilds kept silently reintroducing, because every fresh python-platform build regenerates its
    n8n file from scratch with no n8n-aware check ever looking at it. Caught only much later, by a
    live deployment review that happened to read the actual file - exactly the kind of gap a
    mechanical, always-on check should have caught immediately, at build time, like it already does
    for pure n8n builds.

    Scans the whole tree rather than taking a "files written this round" list (2026-09-27, changed
    alongside the agentic builder rewrite): with real file tools, an n8n shell file may have been
    edited in an earlier attempt/turn and left untouched this round, but it still needs checking -
    this independent verification step should never depend on the agent's own account of what it
    touched.
    """
    problems = []
    for full_path in sorted(code_dir.rglob("*.json")):
        try:
            parsed = json.loads(full_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not _looks_like_n8n_workflow(parsed):
            continue
        passed, output = _validate_n8n_workflow(full_path)
        if not passed:
            problems.append(f"{full_path.relative_to(code_dir)}: {output}")
    return problems


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("AMBIGUOUS:"):
        return {"ambiguous_question": stripped[len("AMBIGUOUS:"):].strip()}
    return extract_json_object(text, context="builder response")


_TECH_QUIRK_LINE_RE = re.compile(r"^TECH_QUIRK:[ \t]*(?P<lesson>.+?)[ \t]*$\r?\n?", re.MULTILINE)


def _extract_and_record_tech_quirk(text: str) -> str:
    """Strips an optional leading 'TECH_QUIRK: <lesson>' line (see _N8N_NODE_SYSTEM_PROMPT's own
    instruction) before JSON parsing, recording it to the shared ledger as a side effect - attributed
    to "builder (n8n)", never the model's own claim of who/what. Returns the text with that one line
    removed so the rest still parses as the node's JSON exactly as before; text with no such line is
    returned completely unchanged (the common case - most node responses have nothing to report)."""
    match = _TECH_QUIRK_LINE_RE.search(text)
    if not match:
        return text
    record_tech_quirk_entry(match.group("lesson"), source="builder (n8n)")
    return text[: match.start()] + text[match.end():]


def _llm_json_call(llm, config, system_prompt: str, user_prompt: str, total_tokens: list, project_dir=None) -> dict:
    """One LLM call expecting a single JSON object back. Retries on transient API errors and on an
    empty/whitespace response (with_retries_nonempty), falling back to the standing fallback model
    (see model_fallback.py) if the primary is exhausted, and on malformed JSON (feeds the parse
    error back) - up to MAX_INTERNAL_FIX_ATTEMPTS total. Mutates total_tokens[0] as a running
    counter. Raises ValueError with the last error if all attempts fail. Always the "builder_n8n"
    role - this helper's only caller is the n8n per-node generation path.
    """
    last_exc: Exception | None = None
    for _attempt in range(1, MAX_INTERNAL_FIX_ATTEMPTS + 1):
        messages = [("system", system_prompt), ("user", user_prompt)]
        try:
            response = with_retries_nonempty(lambda: llm.invoke(messages, config=config))
        except Exception as exc:
            response = attempt_fallback(
                "builder_n8n", lambda fb_llm: fb_llm.invoke(messages, config=config),
                project_dir=project_dir, platform="n8n", primary_error=exc,
            )
        total_tokens[0] += extract_tokens(response)
        try:
            return _extract_json_object(_extract_and_record_tech_quirk(response.content))
        except (ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
            user_prompt += f"\n\nYour previous response could not be parsed: {exc}\nOutput ONLY the single JSON object specified."
    raise ValueError(str(last_exc))


_PLAN_WORKFLOW_RE = re.compile(r"<{2,4}WORKFLOW:\s*(?P<name>[^\n>]+?)\s*>{2,4}")
# The trailing (?:\r?\n)? on all four block regexes below - OPTIONAL, not required - is a real-
# incident fix: a genuinely empty body has no newline to satisfy a REQUIRED one, so the regex
# silently expanded past the next block's own markers instead of failing loudly, merging unrelated
# content under the wrong name. A truly empty CONVENTIONS/NODE/TRIGGER/CONNECTIONS body is less
# likely than an empty file, but the fragility is identical and cheap to close everywhere at once
# rather than wait for a second real incident to find it again in a different one of these four.
#
# {2,4} in place of every literal triple-angle-bracket - fixes a second, worse real incident: a
# mis-counted OPENING delimiter (<<<NODE: Name>> with only two ">") left "name" with no valid
# nearby stopping point, so it expanded forward under DOTALL until it found some LATER ">>>" -
# typically the next node's own opening delimiter - swallowing that entire node's real fields into
# the first node's name and cascading into "node doesn't exist" connection errors for everything
# downstream.
_PLAN_CONVENTIONS_RE = re.compile(r"<{2,4}CONVENTIONS>{2,4}\r?\n(?P<body>.*?)(?:\r?\n)?<{2,4}END_CONVENTIONS>{2,4}", re.DOTALL)
_PLAN_NODE_RE = re.compile(r"<{2,4}NODE:\s*(?P<name>.+?)\s*>{2,4}\r?\n(?P<body>.*?)(?:\r?\n)?<{2,4}END_NODE>{2,4}", re.DOTALL)
_PLAN_TRIGGER_RE = re.compile(r"<{2,4}TRIGGER:\s*(?P<name>.+?)\s*>{2,4}\r?\n(?P<body>.*?)(?:\r?\n)?<{2,4}END_TRIGGER>{2,4}", re.DOTALL)
_PLAN_TRIGGER_FIELD_RE = re.compile(r"^(?P<key>production_type|production_parameters)\s*:\s*(?P<value>.*)$", re.IGNORECASE)
_PLAN_CONNECTIONS_RE = re.compile(r"<{2,4}CONNECTIONS>{2,4}\r?\n(?P<body>.*?)(?:\r?\n)?<{2,4}END_CONNECTIONS>{2,4}", re.DOTALL)
_PLAN_CONNECTION_LINE_RE = re.compile(r"^(?P<src>.+?)\s*->\s*(?P<dst>.+?)\s*(?:\[(?P<out>\d+)\])?$")
_PLAN_FIELD_RE = re.compile(r"^(?P<key>type|slice|purpose)\s*:\s*(?P<value>.*)$", re.IGNORECASE)


def _parse_n8n_plan(text: str) -> dict:
    """Parse the delimited plan format into the same dict shape the rest of the builder expects.

    Delimited rather than JSON because the plan is mostly long free-text purpose paragraphs, and
    asking a model to escape those correctly inside a ~30KB JSON blob failed in practice - one
    unescaped quote 29,000 characters in cost the entire plan and all three of its retries. Raw
    delimited text has nothing to escape. Falls back to the older JSON form so a plan produced
    before this change (or a model that answers in JSON anyway) still parses.
    """
    stripped = text.strip()
    if stripped.startswith("AMBIGUOUS:"):
        return {"ambiguous_question": stripped[len("AMBIGUOUS:"):].strip()}

    node_matches = list(_PLAN_NODE_RE.finditer(text))
    trigger_matches = list(_PLAN_TRIGGER_RE.finditer(text))
    if not node_matches and not trigger_matches:
        if stripped.startswith("{"):
            return _extract_json_object(text)  # legacy JSON plan
        raise ValueError(f"Plan had no <<<NODE: ...>>> or <<<TRIGGER: ...>>> blocks: {text[:400]}")

    triggers = []
    for match in trigger_matches:
        name = match.group("name").strip()
        fields = {"production_type": "", "production_parameters": "{}"}
        for line in match.group("body").splitlines():
            field = _PLAN_TRIGGER_FIELD_RE.match(line.strip())
            if field:
                fields[field.group("key").lower()] = field.group("value").strip()
        try:
            params = json.loads(fields["production_parameters"]) if fields["production_parameters"] else {}
        except json.JSONDecodeError:
            params = {}
        triggers.append({
            "name": name,
            "production_type": fields["production_type"] or "n8n-nodes-base.webhook",
            "production_parameters": params if isinstance(params, dict) else {},
        })

    nodes = []
    slice_order = []
    slice_members: dict = {}
    for match in node_matches:
        name = match.group("name").strip()
        body = match.group("body")
        fields = {"type": "", "slice": "", "purpose": ""}
        current = None
        for line in body.splitlines():
            field = _PLAN_FIELD_RE.match(line.strip())
            if field:
                current = field.group("key").lower()
                fields[current] = field.group("value").strip()
            elif current == "purpose":
                # Purpose is free text and may run over many lines.
                fields["purpose"] += ("\n" + line.rstrip())
        if not fields["type"]:
            raise ValueError(f"Node block {name!r} is missing its 'type:' line")
        nodes.append({"name": name, "type": fields["type"], "purpose": fields["purpose"].strip()})
        slice_name = fields["slice"] or "unassigned"
        if slice_name not in slice_members:
            slice_members[slice_name] = []
            slice_order.append(slice_name)
        slice_members[slice_name].append(name)

    connections = []
    conn_block = _PLAN_CONNECTIONS_RE.search(text)
    if conn_block:
        for line in conn_block.group("body").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parsed = _PLAN_CONNECTION_LINE_RE.match(line)
            if parsed:
                connections.append({
                    "from": parsed.group("src").strip(),
                    "to": parsed.group("dst").strip(),
                    "from_output": int(parsed.group("out") or 0),
                })

    conventions: dict = {}
    conv_block = _PLAN_CONVENTIONS_RE.search(text)
    if conv_block:
        rules = []
        for line in conv_block.group("body").splitlines():
            line = line.strip()
            if line.lower().startswith("timezone:"):
                conventions["timezone"] = line.split(":", 1)[1].strip()
            elif line.startswith("-"):
                rule = line.lstrip("- ").strip()
                if rule:
                    rules.append(rule)
        if rules:
            conventions["shared_rules"] = rules

    workflow_match = _PLAN_WORKFLOW_RE.search(text)
    return {
        "workflow_name": workflow_match.group("name").strip() if workflow_match else "Workflow",
        "conventions": conventions,
        "triggers": triggers,
        "nodes": nodes,
        "connections": connections,
        # Slices are derived from each node's own "slice:" field, in first-appearance order, so
        # every node is necessarily in exactly one slice - the planner can't leave one out.
        "slices": [{"name": s, "purpose": "", "nodes": slice_members[s]} for s in slice_order],
    }


def _build_n8n_plan(llm, config, spec_text: str, total_tokens: list, project_dir=None) -> dict:
    """Plan the workflow, retrying with the parse error fed back if the format comes out wrong."""
    # Computed once per call (this function itself only ever runs once per build - see its one call
    # site, gated on "no checkpoint exists yet") - see with_tech_quirks' own docstring for why this
    # matters even though it wouldn't change again within this one call's own retry loop regardless.
    system_prompt = with_tech_quirks(_N8N_PLAN_SYSTEM_PROMPT)
    user_prompt = f"SPEC:\n{spec_text}"
    last_exc: Exception | None = None
    for _attempt in range(1, MAX_INTERNAL_FIX_ATTEMPTS + 1):
        messages = [("system", system_prompt), ("user", user_prompt)]
        try:
            response = with_retries_nonempty(lambda: llm.invoke(messages, config=config))
        except Exception as exc:
            response = attempt_fallback(
                "builder_n8n", lambda fb_llm: fb_llm.invoke(messages, config=config),
                project_dir=project_dir, platform="n8n", primary_error=exc,
            )
        total_tokens[0] += extract_tokens(response)
        try:
            return _parse_n8n_plan(response.content or "")
        except (ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
            user_prompt = (
                f"SPEC:\n{spec_text}\n\nYour previous plan could not be parsed: {exc}\n"
                "Output ONLY the delimited block format specified - <<<WORKFLOW: ...>>>, "
                "<<<CONVENTIONS>>>, one <<<TRIGGER: ...>>> block per trigger, one <<<NODE: ...>>> "
                "block per non-trigger node, and <<<CONNECTIONS>>>. Do not output JSON."
            )
    raise ValueError(str(last_exc))


def _repair_node_in_place(node: dict) -> list:
    """Apply mechanical, fully-determined corrections to a freshly generated node before it is
    validated, so a deterministic typo never costs an LLM round-trip. Returns what was changed.
    """
    if not isinstance(node, dict):
        return []
    if "code" not in str(node.get("type", "")).lower():
        return []
    params = node.get("parameters")
    if not isinstance(params, dict):
        return []
    applied = []
    for key in ("jsCode", "functionCode"):
        original = params.get(key)
        if isinstance(original, str) and original:
            fixed, fixes = _autofix_js_casing(original)
            if fixes:
                params[key] = fixed
                applied.extend(fixes)
    return applied


def _validate_node_json(node: dict, expected_name: str) -> str | None:
    """Returns a problem description if invalid, else None."""
    if not isinstance(node, dict):
        return "response was not a JSON object"
    if node.get("name") != expected_name:
        return f"'name' field was {node.get('name')!r}, expected {expected_name!r}"
    if not node.get("type"):
        return "missing 'type' field"
    node_type = str(node.get("type", ""))
    type_problem = _validate_node_type(node_type)
    if type_problem:
        return type_problem
    version_problem = _validate_n8n_versioning(node)
    if version_problem:
        return version_problem
    if "code" in node_type.lower():
        params = node.get("parameters") or {}
        js = params.get("jsCode") or params.get("functionCode") or ""
        return _is_stub_js(js)
    return None


def _format_conventions(conventions: dict) -> str:
    """The small shared-context block every node build gets. Deliberately compact (a few hundred
    tokens) - it is NOT the spec. Without it, each node was implemented blind to project-wide
    decisions and invented its own: the reviewer kept (correctly) reporting nodes hardcoding the
    wrong timezone, or using a 0-100 score scale where the spec said 1-10, purely because nothing
    in that node's prompt ever said otherwise.
    """
    if not isinstance(conventions, dict) or not conventions:
        return ""
    lines = []
    timezone = conventions.get("timezone")
    if timezone:
        lines.append(f"- Timezone for ALL dates, times and cron schedules: {timezone}")
    for rule in conventions.get("shared_rules") or []:
        if isinstance(rule, str) and rule.strip():
            lines.append(f"- {rule.strip()}")
    if not lines:
        return ""
    return (
        "PROJECT-WIDE CONVENTIONS - these are decided for the whole workflow and are NOT optional. "
        "Follow them exactly; never substitute your own defaults for anything covered here:\n"
        + "\n".join(lines)
    )


def _build_n8n_node(llm, config, node_spec: dict, all_names: list, total_tokens: list, node_system_prompt: str, conventions: dict | None = None, project_dir=None) -> dict:
    # Deliberately NOT re-sending the full spec here - the plan step's whole point was writing each
    # node's "purpose" as a fully self-contained description precisely so this call doesn't need it.
    # Re-sending the full spec on every single node call was the actual driver of a 500K-token
    # budget exhaustion in practice (spec repeated once per node, on top of per-node output).
    # The conventions block below is the deliberate exception: a few hundred tokens of project-wide
    # rules that every node must share, which the per-node "purpose" descriptions were never a
    # reliable place to repeat.
    conventions_block = _format_conventions(conventions or {})
    base_prompt = (
        (conventions_block + "\n\n" if conventions_block else "")
        + f"Other node names in this workflow (for reference only): {all_names}\n\n"
        f"Node to build:\nName: {node_spec['name']}\nType: {node_spec['type']}\n"
        f"Purpose: {node_spec['purpose']}"
    )
    user_prompt = base_prompt
    last_problem = None
    for _attempt in range(1, MAX_INTERNAL_FIX_ATTEMPTS + 1):
        try:
            node = _llm_json_call(llm, config, node_system_prompt, user_prompt, total_tokens, project_dir=project_dir)
        except ValueError as exc:
            # _llm_json_call already exhausted its own internal retries on malformed JSON - without
            # catching this here, it propagated straight out of this loop uncaught, silently
            # skipping the rest of THIS function's attempts entirely (a real bug: a node that failed
            # to parse got zero benefit from the outer retry loop). Now it gets a genuinely fresh
            # attempt instead.
            last_problem = str(exc)
            user_prompt = base_prompt + f"\n\nYour previous attempt at this node could not be parsed as JSON: {last_problem}\nOutput ONLY the corrected JSON object for this one node, valid and correctly escaped."
            continue
        repairs = _repair_node_in_place(node)
        if repairs:
            print(
                f"DEBUG: auto-corrected node '{node_spec['name']}' JS casing: {'; '.join(repairs)}",
                file=sys.stderr,
            )
        problem = _validate_node_json(node, node_spec["name"])
        if not problem:
            return node
        last_problem = problem
        user_prompt = base_prompt + f"\n\nYour previous attempt at this node was rejected: {problem}\nFix that and output ONLY the corrected JSON object for this one node."
    raise ValueError(f"Node '{node_spec['name']}' still invalid after {MAX_INTERNAL_FIX_ATTEMPTS} attempts: {last_problem}")


def _slices_from_plan(plan: dict, node_specs: list) -> list:
    """Ordered slices covering every planned node. Any node the plan forgot to place in a slice is
    appended to a trailing catch-all rather than silently dropped from the build.
    """
    raw_slices = plan.get("slices") if isinstance(plan, dict) else None
    all_names = [n.get("name") for n in node_specs if isinstance(n, dict) and n.get("name")]
    slices = []
    placed = set()
    for entry in raw_slices or []:
        if not isinstance(entry, dict):
            continue
        names = [n for n in (entry.get("nodes") or []) if n in all_names and n not in placed]
        if not names:
            continue
        placed.update(names)
        slices.append({"name": entry.get("name") or f"slice-{len(slices) + 1}",
                       "purpose": entry.get("purpose") or "",
                       "nodes": names})
    leftover = [n for n in all_names if n not in placed]
    if leftover:
        slices.append({"name": "unassigned", "purpose": "Nodes the plan did not place in a slice.",
                       "nodes": leftover})
    if not slices:
        slices = [{"name": "all", "purpose": "Whole workflow.", "nodes": all_names}]
    return slices


def _resolve_until_slice(slices: list, until_slice) -> int:
    """Index of the last slice to build, from a name or 1-based number. -1 means build everything."""
    if until_slice in (None, "", "all"):
        return len(slices) - 1
    text = str(until_slice).strip()
    for i, entry in enumerate(slices):
        if entry["name"].lower() == text.lower():
            return i
    if text.isdigit():
        return max(0, min(len(slices) - 1, int(text) - 1))
    return len(slices) - 1


def _normalize_connection(entry) -> tuple | None:
    """Accepts either the current {"from","to","from_output"} object form or the older
    ["Source","Target"] pair (treated as output 0, so plans/checkpoints written before branching
    existed still assemble). Returns (source, target, output_index) or None if unusable.
    """
    if isinstance(entry, dict):
        source, target = entry.get("from"), entry.get("to")
        try:
            output_index = int(entry.get("from_output", 0) or 0)
        except (TypeError, ValueError):
            output_index = 0
    elif isinstance(entry, (list, tuple)) and len(entry) == 2:
        source, target = entry
        output_index = 0
    else:
        return None
    if not source or not target:
        return None
    return str(source), str(target), max(0, output_index)


# --- Mechanical trigger construction ---------------------------------------------------------
#
# A trigger is built as a webhook FROM THE START, under its real declared name, forever - not
# retrofitted onto a finished workflow later. This is what makes live verification possible without
# the impersonation logic that caused every verification bug found tonight: there is no swap to get
# wrong and no reference to break, because the name a node like $('Receive Telegram Message') depends
# on was never anything else. Only the trigger's TYPE and its adapter's code ever change, once, at
# deploy time (pipeline/n8n_deploy.py, not yet built) - nothing downstream needs to know or care.
#
# adapter_name_for/verify_webhook_path/default_test_payload/VERIFY_ADAPTER_JS live in n8n_verify.py,
# not here, so this module can import verify_native_workflow from there without a circular import.


def build_trigger_and_adapter_nodes(triggers: list, project_name: str) -> dict:
    """One webhook node plus one adapter node per declared trigger, keyed by name. The webhook keeps
    the trigger's own declared name (see module note above); the adapter reshapes n8n's
    {headers, params, query, body} webhook envelope into the payload shape a real trigger would have
    produced, which downstream nodes are written to expect. Node ids are left for
    _assign_unique_node_ids to fill in - it overwrites every node's id unconditionally anyway.
    """
    nodes = {}
    for i, trigger in enumerate(triggers):
        name = (trigger or {}).get("name")
        if not name:
            continue
        nodes[name] = {
            "name": name,
            "type": "n8n-nodes-base.webhook",
            "typeVersion": 2,
            "position": [0, i * 200],
            "parameters": {
                "path": verify_webhook_path(project_name, name),
                "httpMethod": "POST",
                "responseMode": "onReceived",
            },
        }
        adapter_name = adapter_name_for(name)
        nodes[adapter_name] = {
            "name": adapter_name,
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [200, i * 200],
            "parameters": {"jsCode": VERIFY_ADAPTER_JS},
        }
    return nodes


def insert_adapters_into_connections(connections_plan: list, triggers: list) -> list:
    """Rewires the plan's connections so each trigger's outgoing edges leave from its adapter
    instead, and adds trigger -> adapter. The plan's author writes connections as if the trigger
    feeds the rest of the workflow directly, exactly as a real trigger would - this is the one place
    that knowledge is corrected for the adapter's existence, so nothing else needs to know about it.
    """
    trigger_names = {t.get("name") for t in (triggers or []) if t.get("name")}
    if not trigger_names:
        return list(connections_plan or [])

    adjusted = []
    for entry in connections_plan or []:
        normalized = _normalize_connection(entry)
        if normalized and normalized[0] in trigger_names:
            source, target, output_index = normalized
            adjusted.append({"from": adapter_name_for(source), "to": target, "from_output": output_index})
        else:
            adjusted.append(entry)
    for name in trigger_names:
        adjusted.append({"from": name, "to": adapter_name_for(name), "from_output": 0})
    return adjusted


def _production_triggers_path(code_dir: Path, workflow: str | None = None) -> Path:
    """Sidecar recording each trigger's real production type/parameters, for the deploy step to
    apply. Kept outside the workflow file itself, same reasoning as .conventions.json - an n8n
    workflow must stay import-clean, no custom top-level keys. Scoped per n8n workflow (see
    pipeline/n8n_workflow_target.py) - each workflow has its own triggers.
    """
    return n8n_subdir(code_dir, workflow) / ".production_triggers.json"


def save_production_triggers(code_dir: Path, triggers: list, workflow: str | None = None) -> None:
    if not triggers:
        return
    path = _production_triggers_path(code_dir, workflow=workflow)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(triggers, indent=2), encoding="utf-8")


_NODE_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "build-agent/n8n-node-id")


def _assign_unique_node_ids(workflow_name: str, nodes_by_name: dict) -> None:
    """Overwrite every node's "id" with one derived from the workflow and node name.

    Node ids must be unique or n8n rejects the workflow on import, but each node is generated in
    isolation with no knowledge of the others, so the model cannot pick a globally-unique value -
    in practice it emitted the same well-known placeholder UUID for 11 different nodes in one
    build. An id carries no meaning, so it is assigned mechanically here rather than trusted from
    the model, the same way JS casing is corrected rather than re-prompted.

    Derived (uuid5) rather than random so ids stay stable across rebuilds of the same workflow -
    re-importing produces the same ids instead of churning them every run.
    """
    for name, node in nodes_by_name.items():
        if isinstance(node, dict):
            node["id"] = str(uuid.uuid5(_NODE_ID_NAMESPACE, f"{workflow_name}:{name}"))


def _assemble_n8n_workflow(workflow_name: str, nodes_by_name: dict, connections_plan: list) -> dict:
    """Builds n8n's connections structure, honouring each connection's source OUTPUT INDEX.

    This is what makes conditional branching possible at all: n8n runs every target attached to a
    given output, so an IF node's true/false paths must land on outputs 0 and 1 respectively. The
    earlier version put every connection on output 0, which silently turned every intended branch
    into an unconditional parallel fan-out - mutually exclusive paths (success / failure / no
    response) all firing together on import.
    """
    _assign_unique_node_ids(workflow_name, nodes_by_name)
    connections: dict = {}
    for entry in connections_plan or []:
        normalized = _normalize_connection(entry)
        if not normalized:
            continue
        source, target, output_index = normalized
        outputs = connections.setdefault(source, {}).setdefault("main", [])
        while len(outputs) <= output_index:
            outputs.append([])
        outputs[output_index].append({"node": target, "type": "main", "index": 0})
    return {
        "name": workflow_name,
        "nodes": list(nodes_by_name.values()),
        "connections": connections,
    }


# Matches _validate_n8n_workflow's own under-wired-branch message text exactly - see the two
# f-strings that produce "IF node '<name>' has only output(s) ... wired" / "Switch node '<name>'
# has only output(s) ... wired" above.
_BRANCH_UNDERWIRED_RE = re.compile(r"(?:IF|Switch) node '(?P<name>[^']+)' has only output\(s\)")


def _repair_branch_connections(
    llm, config, problems_text: str, connections_plan: list, node_specs: list,
    nodes_by_name: dict, all_names: list, total_tokens: list,
) -> tuple[list, bool]:
    """When the validator reports an IF/Switch node with missing outgoing connections, fix the
    actual defect (ask for just the missing wiring) instead of rebuilding that node's JSON body.

    Real incident this fixes: a node's own body only ever holds ITS OWN parameters (e.g. an IF
    node's condition) - it has nothing to do with the workflow's connections object, which is
    built once from the plan before the repair loop ever runs. The pre-existing repair loop's only
    move was "rebuild whatever node the validator named", which for this failure class rebuilds a
    node that was never broken and never touches the actual missing wire - guaranteeing identical
    failures across all repair rounds every time the plan under-specifies a branch node's outputs
    (seen repeatedly in practice: the plan-generation model reliably writes every other connection
    correctly but intermittently drops one or both outgoing edges of an IF/Switch node specifically).
    Returns (possibly-extended connections_plan, whether anything was actually added).
    """
    specs_by_name = {n.get("name"): n for n in node_specs if isinstance(n, dict)}
    changed = False
    for match in _BRANCH_UNDERWIRED_RE.finditer(problems_text):
        name = match.group("name")
        node = nodes_by_name.get(name)
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("type", ""))
        is_if = node_type.endswith(".if")
        output_hint = (
            "This is an IF node: output 0 is the true branch, output 1 is the false branch. Both "
            "must be wired to a target, even if the false branch just leads to an ending node."
            if is_if else
            "This is a Switch node: each rule gets its own output index (0, 1, 2, ...) in the order "
            "its rules are defined. At least two distinct outputs must be wired to different targets."
        )
        purpose = (specs_by_name.get(name) or {}).get("purpose", "")
        user_prompt = (
            f"Node '{name}' (type {node_type}) is missing some of its outgoing connections in an "
            f"n8n workflow plan. {output_hint}\n\nThis node's purpose: {purpose}\n\n"
            f"All valid node names in this workflow: {', '.join(all_names)}\n\n"
            f"Output ONLY the missing connection line(s) for '{name}', one per line, in exactly this "
            f"format and nothing else:\n{name} -> Target Node Name [output_index]"
        )
        try:
            response = with_retries_nonempty(lambda: llm.invoke(
                [
                    ("system", "You fix missing wiring in an n8n workflow plan. Output only the "
                               "requested connection lines - no commentary, no markdown fences."),
                    ("user", user_prompt),
                ],
                config=config,
            ))
        except Exception:
            continue
        total_tokens[0] += extract_tokens(response)
        for line in (response.content or "").splitlines():
            line = line.strip()
            if not line:
                continue
            parsed = _PLAN_CONNECTION_LINE_RE.match(line)
            if parsed and parsed.group("src").strip() == name:
                connections_plan.append({
                    "from": name,
                    "to": parsed.group("dst").strip(),
                    "from_output": int(parsed.group("out") or 0),
                })
                changed = True
    return connections_plan, changed


# Matches _validate_n8n_workflow's own "no incoming connection" reachability message exactly.
_DISCONNECTED_NODE_RE = re.compile(r"Node '(?P<name>[^']+)' has no incoming connection")


def _repair_disconnected_nodes(
    llm, config, problems_text: str, connections_plan: list, node_specs: list,
    nodes_by_name: dict, all_names: list, total_tokens: list,
) -> tuple[list, bool]:
    """When the validator reports a node with no incoming connection at all (see the reachability
    check in _validate_n8n_workflow), ask for just the missing upstream wire, the mirror image of
    _repair_branch_connections (which fixes a node's missing OUTGOING wires). Same reasoning
    applies: nothing about the disconnected node's own body is wrong, so rebuilding it fixes
    nothing - only adding the actual missing connection does.
    """
    specs_by_name = {n.get("name"): n for n in node_specs if isinstance(n, dict)}
    changed = False
    for match in _DISCONNECTED_NODE_RE.finditer(problems_text):
        name = match.group("name")
        if name not in nodes_by_name:
            continue
        purpose = (specs_by_name.get(name) or {}).get("purpose", "")
        user_prompt = (
            f"Node '{name}' has no incoming connection in an n8n workflow plan - nothing feeds it, "
            f"so it never runs.\n\nThis node's purpose: {purpose}\n\n"
            f"All valid node names in this workflow: {', '.join(all_names)}\n\n"
            f"Which upstream node should connect INTO '{name}', and from which of that upstream "
            f"node's output indices (0 for a node with a single output; 0 or 1 for an IF node's "
            f"true/false branch; 0, 1, 2... for a Switch node's rules)? Output ONLY one connection "
            f"line in exactly this format, nothing else:\nUpstream Node Name -> {name} [output_index]"
        )
        try:
            response = with_retries_nonempty(lambda: llm.invoke(
                [
                    ("system", "You fix missing wiring in an n8n workflow plan. Output only the "
                               "requested connection line - no commentary, no markdown fences."),
                    ("user", user_prompt),
                ],
                config=config,
            ))
        except Exception:
            continue
        total_tokens[0] += extract_tokens(response)
        for line in (response.content or "").splitlines():
            line = line.strip()
            if not line:
                continue
            parsed = _PLAN_CONNECTION_LINE_RE.match(line)
            if parsed and parsed.group("dst").strip() == name:
                connections_plan.append({
                    "from": parsed.group("src").strip(),
                    "to": name,
                    "from_output": int(parsed.group("out") or 0),
                })
                changed = True
    return connections_plan, changed


# Matches _validate_n8n_workflow's own dead-branch message text exactly - see the "wired beyond
# what the node's own rules define" check above.
_DEAD_BRANCH_RE = re.compile(
    r"(?:IF|Switch) node '(?P<name>[^']+)' defines \d+ output\(s\) but has connections on "
    r"output\(s\) \[(?P<dead>[\d,\s]+)\]"
)


def _prune_dead_branches(problems_text: str, connections_plan: list) -> tuple[list, bool]:
    """When the validator reports a branch node with connections on an output index it doesn't
    even define, remove exactly those connections - a purely mechanical fix, no LLM call needed,
    since "those branches can never fire" already means removing them changes nothing about what
    actually runs today.

    Deliberately does NOT try to guess whether the real intent was "add a matching rule" instead of
    "remove this connection" (the validator's own suggested alternative) - the LLM would have to
    edit the node's own rule/condition definition to do that, not just connections_plan, which is
    out of scope for a connections-only repair helper. Pruning is always safe and composes cleanly
    with _repair_disconnected_nodes: whatever node the dead connection pointed to just becomes
    reachable-from-nowhere instead, which that helper already knows how to fix with a real,
    LLM-informed routing decision on the next validation pass - two narrow fixes covering the full
    problem better than one fix guessing at intent it can't actually see.
    """
    changed = False
    for match in _DEAD_BRANCH_RE.finditer(problems_text):
        name = match.group("name")
        dead_indices = {int(d.strip()) for d in match.group("dead").split(",") if d.strip()}
        before = len(connections_plan)
        connections_plan = [
            e for e in connections_plan
            if not (e.get("from") == name and int(e.get("from_output", 0) or 0) in dead_indices)
        ]
        if len(connections_plan) != before:
            changed = True
    return connections_plan, changed


def _connections_dict_to_plan(connections: dict) -> list:
    """Inverse of _assemble_n8n_workflow's connection-building: turns an already-assembled
    connections object back into the flat (from, to, from_output) list form the repair helpers and
    _assemble_n8n_workflow itself both work with. Needed by the incremental reviewer-feedback path,
    which inherits an already-assembled workflow rather than a fresh connections_plan.
    """
    plan = []
    for source, outputs in (connections or {}).items():
        for output_index, branch in enumerate((outputs or {}).get("main") or []):
            for target in branch or []:
                target_name = (target or {}).get("node")
                if target_name:
                    plan.append({"from": source, "to": target_name, "from_output": output_index})
    return plan


def _checkpoint_path(code_dir: Path, workflow: str | None = None) -> Path:
    # Scoped per n8n workflow (see pipeline/n8n_workflow_target.py) so two workflows in the same
    # multi-workflow project never share, and can't clobber, each other's per-node build progress.
    # `workflow` falsy (every project today) keeps the exact original filename.
    filename = ".n8n_build_checkpoint.json" if not workflow else f".n8n_build_checkpoint.{workflow}.json"
    return code_dir / filename


def _conventions_path(code_dir: Path, workflow: str | None = None) -> Path:
    """Sidecar holding the project-wide conventions the plan phase extracted. Kept next to the
    workflow rather than inside it (an n8n workflow file must stay importable, so no custom keys)
    and kept separate from the build checkpoint, which is deleted once a build completes - later
    incremental repairs still need these conventions to avoid reintroducing the very
    inconsistencies they were added to prevent. Scoped per n8n workflow (see
    pipeline/n8n_workflow_target.py) - each workflow in a multi-workflow project gets its own
    conventions, since "project-wide" only ever meant "wide across one workflow's own nodes".
    """
    return n8n_subdir(code_dir, workflow) / ".conventions.json"


def _save_conventions(code_dir: Path, conventions: dict, workflow: str | None = None) -> None:
    if not conventions:
        return
    path = _conventions_path(code_dir, workflow=workflow)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(conventions, indent=2), encoding="utf-8")


def _load_conventions(code_dir: Path, workflow: str | None = None) -> dict:
    path = _conventions_path(code_dir, workflow=workflow)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def wipe_n8n_artifacts(code_dir: Path, workflow: str | None = None) -> list:
    """Deletes every n8n build artifact for THIS workflow (see pipeline/n8n_workflow_target.py -
    `workflow` falsy means the legacy single-workflow project): the workflow itself, the builder's
    per-node checkpoint, saved conventions, and the production-triggers sidecar. Returns the paths
    actually removed. A multi-workflow project's other workflows are never touched by this call.

    Why this needs to exist as its own thing, separate from --fresh: --fresh only clears the OUTER
    graph-level round checkpoint (pipeline/checkpoint.py) - it was never about n8n's own per-node
    "frozen once built" state, which lives entirely in these files. A node already sitting in
    workflow.json is deliberately never rebuilt just because a new run started (so a manual fix
    made to the file between runs survives) - normally exactly what you want, but it means a node
    that was built wrong for a reason that's since been fixed elsewhere (a real incident: a
    reserved-name collision silently produced a semantically wrong node body, which no ordinary
    --fresh run would ever touch since the file was structurally valid) can persist indefinitely
    with nothing to signal it needs rebuilding. Call this when that staleness itself is the problem.
    """
    removed = []
    for path in (
        n8n_subdir(code_dir, workflow) / "workflow.json",
        _checkpoint_path(code_dir, workflow=workflow),
        _conventions_path(code_dir, workflow=workflow),
        _production_triggers_path(code_dir, workflow=workflow),
    ):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def wipe_python_code(code_dir: Path) -> list:
    """Deletes every top-level file/directory under code_dir - the python analog of
    wipe_n8n_artifacts above, needed for the same reason (2026-09-28, added after a peer verifying a
    creative-team rebuild found no --fresh-python-style flag existed at all - --fresh alone only
    clears the OUTER round-level checkpoint, pipeline/checkpoint.py, never touching real source
    files already on disk). A file already on disk is deliberately never rebuilt just because a new
    run started - exactly what you want normally (real, working progress from an earlier attempt
    survives), but it means there was no way to force a genuine clean-slate rebuild without manually
    deleting code_dir yourself. Returns the top-level paths actually removed (not every nested file
    - keeps the report short, matching wipe_n8n_artifacts' own "what got removed" shape). Caller is
    responsible for also clearing this project's build journal (pipeline/build_journal.py's
    clear_journal) - a stale journal entry referencing a unit that "was built" and code that no
    longer exists would mislead a later unit otherwise.
    """
    removed = []
    if not code_dir.exists():
        return removed
    for entry in sorted(code_dir.iterdir()):
        removed.append(str(entry))
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    return removed


def _spec_hash(spec_text: str) -> str:
    return hashlib.sha256(spec_text.encode("utf-8")).hexdigest()


def _load_checkpoint(code_dir: Path, spec_text: str, workflow: str | None = None):
    """Returns (plan, nodes_by_name) if a valid, matching-spec checkpoint exists, else None. This is
    what lets a build survive a process kill (e.g. the OS killing it for memory) partway through
    building nodes - resume with already-built nodes intact instead of replanning+rebuilding
    everything from a brand-new plan. Ignored (treated as absent) if the spec text has changed since
    the checkpoint was written, so an edited spec never silently resumes stale, mismatched work.
    """
    path = _checkpoint_path(code_dir, workflow=workflow)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("spec_hash") != _spec_hash(spec_text):
        return None
    return data.get("plan"), data.get("nodes_by_name", {})


def _save_checkpoint(code_dir: Path, spec_text: str, plan: dict, nodes_by_name: dict, workflow: str | None = None) -> None:
    path = _checkpoint_path(code_dir, workflow=workflow)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"spec_hash": _spec_hash(spec_text), "plan": plan, "nodes_by_name": nodes_by_name}, indent=2),
        encoding="utf-8",
    )


def _clear_checkpoint(code_dir: Path, workflow: str | None = None) -> None:
    path = _checkpoint_path(code_dir, workflow=workflow)
    if path.exists():
        path.unlink()


# Nodes built at once. Each per-node call only needs the plan's node NAMES (for cross-reference),
# never another node's generated CONTENT - there is no ordering dependency between them, only
# between the plan phase and every node. Running them concurrently only shortens wall-clock time; it
# changes nothing about what gets built. Kept modest rather than unbounded, to stay polite to the
# provider's rate limits rather than to protect anything about correctness.
_NODE_BUILD_CONCURRENCY = 4


def _build_nodes_concurrently(llm, config, node_specs: list, all_names: list, conventions, total_tokens: list, project_dir, on_built) -> str | None:
    """Builds node_specs (already filtered to whatever still needs building) in small concurrent
    batches. Calls on_built(name, node) on the MAIN thread as each one finishes - not from inside a
    worker thread - so checkpoint saves and nodes_by_name mutation never need their own locking.

    Returns an ambiguity message if the run should stop (budget exceeded, or a node failed after its
    own internal retries), else None.
    """
    for batch_start in range(0, len(node_specs), _NODE_BUILD_CONCURRENCY):
        if over_budget(total_tokens[0], project_dir):
            return (
                f"Run exceeded the token budget ({total_tokens[0]} tokens) while building nodes "
                f"({batch_start}/{len(node_specs)} done) - stopped to control cost."
            )
        batch = node_specs[batch_start : batch_start + _NODE_BUILD_CONCURRENCY]

        # Each concurrent call gets its OWN token-counter cell rather than sharing total_tokens - two
        # threads doing `total_tokens[0] += x` at once is a real race that can silently drop tokens
        # from the budget check. Cells are merged into the shared total on the MAIN thread as each
        # future completes, so the merge itself is never concurrent.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
            future_to_spec = {}
            for spec in batch:
                cell = [0]
                future = executor.submit(_build_n8n_node, llm, config, spec, all_names, cell, conventions)
                future_to_spec[future] = (spec, cell)

            results: dict = {}
            batch_error = None
            for future in concurrent.futures.as_completed(future_to_spec):
                spec, cell = future_to_spec[future]
                total_tokens[0] += cell[0]  # merged regardless of outcome - failed attempts still cost tokens
                try:
                    results[spec["name"]] = future.result()
                except ValueError as exc:
                    batch_error = f"Could not build node '{spec['name']}': {exc}"

        if batch_error:
            return batch_error
        for spec in batch:
            if spec["name"] in results:
                on_built(spec["name"], results[spec["name"]])
    return None


def _log_repair_round(project_dir, project_name: str, repair_round: int, problems_text: str, llm=None) -> None:
    """Best-effort: logs ONE failing repair round (structural validation or live verification) via
    the Descriptive Scribe - the friction worth learning from, not every round. A round that passes
    triggers no call at all, so a clean build costs nothing extra beyond the existing per-run
    summary; this is deliberately bounded by MAX_INTERNAL_FIX_ATTEMPTS, never by node count. Never
    allowed to affect the actual repair loop - any failure here is logged and swallowed.

    `llm`: the actual model instance used to build this round, if the caller has one in scope -
    read its `.model_name` (already-resolved, no extra config lookup needed) so the logged entry
    names the real model, not just the "builder_n8n" role, enabling the model-scorecard synthesis
    in pipeline/review_decisions.py.
    """
    try:
        round_result = {"round": repair_round, "role": "builder_n8n", "status": "fail", "problems": problems_text}
        summary, _ = summarize_build_run(round_result, project_dir=project_dir)
        model = getattr(llm, "model_name", None) if llm is not None else None
        record_build(
            project_name, "fail", summary, scope="round", round_num=repair_round,
            role="builder_n8n", model=model,
        )
    except Exception as exc:  # noqa: BLE001 - a logging side channel must never break a real build
        print(f"WARNING: round-level memory logging failed: {exc}", file=sys.stderr)


def _build_n8n_chunked(state: BuildState, llm, config, spec_text: str, code_dir: Path) -> BuildState:
    total_tokens = [state.get("total_tokens", 0)]
    project_dir = state.get("project_dir")
    workflow = state.get("n8n_workflow")
    workflow_path = n8n_subdir(code_dir, workflow) / "workflow.json"
    review_problems = state.get("review_problems") or []
    conventions = _load_conventions(code_dir, workflow=workflow)
    # Computed ONCE for this whole run (not once per node - potentially dozens of _build_n8n_node
    # calls below) - both for with_tech_quirks' own "stable prefix across the run" reasoning, and so
    # a quirk this same run records mid-way (see _llm_json_call's TECH_QUIRK: line handling) doesn't
    # change what later nodes in THIS run see - it becomes visible starting next run instead, which
    # is all Douglas actually asked for (durable across runs, not mid-run reactivity).
    node_system_prompt = with_tech_quirks(_N8N_NODE_SYSTEM_PROMPT)

    existing_workflow = None
    if workflow_path.exists():
        try:
            existing_workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_workflow = None

    if review_problems and isinstance((existing_workflow or {}).get("nodes"), list):
        # Incremental fix: a previous round already produced a workflow and the reviewer found
        # problems with it. Reuse it entirely (nodes, connections) and rebuild ONLY whichever
        # node(s) the reviewer's problems actually name - never replan/rebuild everything just
        # because one round found an issue. Confirmed in practice this was the actual cost driver:
        # each review round used to throw away all prior work and regenerate a brand-new,
        # different-every-time 40-node plan from scratch (~150-200K tokens per round, ~500K+ for
        # one run) - the reviewer's specific feedback was never even being used.
        nodes_by_name = {n["name"]: n for n in existing_workflow["nodes"] if isinstance(n, dict) and n.get("name")}
        all_names = list(nodes_by_name.keys())
        workflow_name = existing_workflow.get("name", "Workflow")
        connections_plan = _connections_dict_to_plan(existing_workflow.get("connections") or {})
        node_specs_proxy = [{"name": n, "purpose": ""} for n in all_names]
        problems_to_fix = review_problems

        for _incremental_round in range(1, MAX_INTERNAL_FIX_ATTEMPTS + 1):
            combined_feedback = "; ".join(problems_to_fix)

            # Missing-wire problems first, same reasoning as the main repair loop below: a node
            # named only because it's missing a connection has nothing wrong with its own body -
            # rebuilding it fixes nothing. See _repair_branch_connections / _repair_disconnected_nodes.
            connections_plan, branch_fixed = _repair_branch_connections(
                llm, config, combined_feedback, connections_plan, node_specs_proxy, nodes_by_name, all_names, total_tokens,
            )
            connections_plan, unreachable_fixed = _repair_disconnected_nodes(
                llm, config, combined_feedback, connections_plan, node_specs_proxy, nodes_by_name, all_names, total_tokens,
            )
            connections_plan, dead_pruned = _prune_dead_branches(combined_feedback, connections_plan)
            wire_fixed_names = set()
            if branch_fixed:
                wire_fixed_names |= {m.group("name") for m in _BRANCH_UNDERWIRED_RE.finditer(combined_feedback)}
            if unreachable_fixed:
                wire_fixed_names |= {m.group("name") for m in _DISCONNECTED_NODE_RE.finditer(combined_feedback)}
            if dead_pruned:
                wire_fixed_names |= {m.group("name") for m in _DEAD_BRANCH_RE.finditer(combined_feedback)}

            rebuilt_any = branch_fixed or unreachable_fixed or dead_pruned
            for name, node in list(nodes_by_name.items()):
                if name in wire_fixed_names:
                    continue
                if any(name in p for p in problems_to_fix):
                    node_spec = {
                        "name": name,
                        "type": node.get("type", "n8n-nodes-base.code"),
                        "purpose": f"This node already exists in the workflow; rebuild it correctly. Reviewer feedback to address: {combined_feedback}",
                    }
                    try:
                        nodes_by_name[name] = _build_n8n_node(
                            llm, config, node_spec, all_names, total_tokens, node_system_prompt,
                            conventions=conventions, project_dir=project_dir,
                        )
                        rebuilt_any = True
                        print(f"DEBUG: incrementally rebuilt node '{name}' per reviewer feedback", file=sys.stderr)
                    except ValueError as exc:
                        print(f"DEBUG: could not incrementally rebuild node '{name}': {exc}", file=sys.stderr)

            if not rebuilt_any:
                # Feedback didn't name any specific existing node, and no wire fix applied either -
                # can't target-fix blindly, fall through to a fresh plan+rebuild below.
                break

            # _assemble_n8n_workflow re-stamps every node with a guaranteed-unique id as a side
            # effect (see _assign_unique_node_ids) - the actual fix for a real incident where this
            # incremental path rebuilt node bodies (each carrying whatever id the model invented)
            # and wrote them straight to disk with NO id re-stamp and NO validation at all,
            # producing a real duplicate id between two unrelated nodes.
            workflow = _assemble_n8n_workflow(workflow_name, nodes_by_name, connections_plan)
            workflow_path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
            passed, output = _validate_n8n_workflow(workflow_path)
            if passed:
                return {**state, "builder_status": "done", "ambiguity_question": None, "total_tokens": total_tokens[0]}
            print(f"DEBUG: incremental fix round {_incremental_round} still has structural problems: {output[:500]}", file=sys.stderr)
            problems_to_fix = output.splitlines()
        # Exhausted the incremental rounds (or found nothing left to target-fix) without reaching a
        # structurally clean workflow - fall through to a fresh plan+rebuild below as a last resort.
        print("DEBUG: reviewer problems didn't name any existing node - falling back to full replan", file=sys.stderr)

    checkpoint = _load_checkpoint(code_dir, spec_text, workflow=workflow)
    if checkpoint:
        plan, nodes_by_name = checkpoint
        print(f"DEBUG: resuming from checkpoint - {len(nodes_by_name)} node(s) already built", file=sys.stderr)
        # workflow.json on disk WINS over the checkpoint's copy of the same node.
        #
        # The checkpoint is this builder's private memory of what it generated; workflow.json is the
        # artifact everyone else works on. Someone (or Claude Code, or a live n8n instance export)
        # may have corrected nodes there between runs. Regenerating from the checkpoint threw those
        # corrections away and reintroduced the very bugs that had just been fixed - which made the
        # obvious workflow, "pipeline drafts / human fixes / pipeline continues", structurally
        # impossible. Anything already present on disk is treated as frozen and is never rebuilt.
        disk_nodes = {
            n["name"]: n
            for n in ((existing_workflow or {}).get("nodes") or [])
            if isinstance(n, dict) and n.get("name")
        }
        if disk_nodes:
            overridden = [name for name in disk_nodes if name in nodes_by_name]
            nodes_by_name = {**nodes_by_name, **disk_nodes}
            print(
                f"DEBUG: workflow.json supplied {len(disk_nodes)} node(s); "
                f"{len(overridden)} took precedence over the checkpoint (external edits preserved)",
                file=sys.stderr,
            )
    else:
        try:
            plan = _build_n8n_plan(llm, config, spec_text, total_tokens, project_dir=project_dir)
        except ValueError as exc:
            return {**state, "builder_status": "ambiguous", "ambiguity_question": f"Could not produce a valid n8n plan: {exc}", "total_tokens": total_tokens[0]}
        if plan.get("ambiguous_question"):
            return {**state, "builder_status": "ambiguous", "ambiguity_question": plan["ambiguous_question"], "total_tokens": total_tokens[0]}
        print(f"DEBUG: plan phase used {total_tokens[0]} tokens so far", file=sys.stderr)
        nodes_by_name = {}
        _save_checkpoint(code_dir, spec_text, plan, nodes_by_name, workflow=workflow)

    # Conventions come from whichever plan is in play (fresh or resumed from checkpoint), and are
    # persisted so later incremental repairs - which never reload a plan - still honour them.
    plan_conventions = plan.get("conventions") if isinstance(plan, dict) else None
    if plan_conventions:
        conventions = plan_conventions
        _save_conventions(code_dir, conventions, workflow=workflow)
    print(f"DEBUG: conventions in play: {json.dumps(conventions)[:400]}", file=sys.stderr)

    # Trigger nodes are mechanically constructed, never LLM-authored - seeded directly into
    # nodes_by_name so the per-node build loop below never sees or touches them. setdefault, not
    # assignment: if resumed from a checkpoint or already present on disk, that copy wins (same
    # "frozen once built" rule already applied to every other node).
    triggers = plan.get("triggers") or [] if isinstance(plan, dict) else []

    # Safety net for the exact incident adapter_name_for's docstring describes: if a plan-authored
    # node's name collides with a trigger's own name or its mechanical adapter's reserved name, the
    # collision below (nodes_by_name.setdefault) would silently keep the mechanical node and
    # discard the plan's real one with no error at all. Reserved names changed to make this very
    # unlikely going forward, but catch it explicitly rather than trust that alone.
    if triggers:
        reserved_names = {t.get("name") for t in triggers if t.get("name")} | {
            adapter_name_for(t.get("name")) for t in triggers if t.get("name")
        }
        colliding = [n.get("name") for n in (plan.get("nodes") or []) if n.get("name") in reserved_names]
        if colliding:
            return {
                **state, "builder_status": "ambiguous",
                "ambiguity_question": (
                    f"The plan named one of its own nodes the same as a trigger or its reserved "
                    f"adapter name: {colliding}. These names are reserved for mechanically-built "
                    f"infrastructure and would silently discard the plan's real node if allowed "
                    f"through - rename the colliding node(s) in the spec/plan and rebuild."
                ),
                "total_tokens": total_tokens[0],
            }
        for name, node in build_trigger_and_adapter_nodes(triggers, plan.get("workflow_name", "Workflow")).items():
            nodes_by_name.setdefault(name, node)
        save_production_triggers(code_dir, triggers, workflow=workflow)

    node_specs = plan.get("nodes") or []
    all_names = [n.get("name") for n in node_specs] + list({
        n for t in triggers for n in (t.get("name"), adapter_name_for(t.get("name") or "")) if n
    })
    print(f"DEBUG: plan has {len(node_specs)} nodes and {len(triggers)} trigger(s): {all_names}", file=sys.stderr)

    # Build in slice order, so stopping partway leaves a coherent functional base rather than an
    # arbitrary prefix of the node list.
    slices = _slices_from_plan(plan, node_specs)
    last_index = _resolve_until_slice(slices, state.get("until_slice"))
    slices_to_build = slices[: last_index + 1]
    partial = last_index < len(slices) - 1
    wanted_names = [name for entry in slices_to_build for name in entry["nodes"]]
    specs_by_name = {n.get("name"): n for n in node_specs if isinstance(n, dict)}
    node_specs = [specs_by_name[name] for name in wanted_names if name in specs_by_name]
    print(
        f"DEBUG: {len(slices)} slice(s): {[s['name'] for s in slices]}; building through "
        f"'{slices[last_index]['name']}' ({len(node_specs)} node(s)){' - PARTIAL' if partial else ''}",
        file=sys.stderr,
    )

    still_needed = [spec for spec in node_specs if spec.get("name") not in nodes_by_name]

    def _on_node_built(name: str, node: dict) -> None:
        nodes_by_name[name] = node
        _save_checkpoint(code_dir, spec_text, plan, nodes_by_name, workflow=workflow)
        print(f"DEBUG: built node '{name}' - running total {total_tokens[0]} tokens", file=sys.stderr)

    build_error = _build_nodes_concurrently(
        llm, config, still_needed, all_names, conventions, total_tokens, project_dir, _on_node_built
    )
    if build_error:
        return {**state, "builder_status": "ambiguous", "ambiguity_question": build_error, "total_tokens": total_tokens[0]}

    slice_info = {
        "slices_total": len(slices),
        "slices_built": [entry["name"] for entry in slices_to_build],
        "slices_remaining": [entry["name"] for entry in slices[last_index + 1:]],
        "partial_build": partial,
    }

    # Rewire each trigger's outgoing connections to leave from its adapter instead - the plan is
    # written as if the trigger feeds the workflow directly, exactly as a real trigger would; this is
    # the one place that gets corrected for the adapter's existence, so nothing else needs to know.
    connections_plan = insert_adapters_into_connections(plan.get("connections") or [], triggers)

    # On a partial build, connections pointing at nodes from slices that haven't been built yet
    # would be dangling references - drop them so what's on disk is a valid standalone workflow.
    if partial:
        built = set(nodes_by_name)
        kept = []
        for entry in connections_plan:
            normalized = _normalize_connection(entry)
            if normalized and normalized[0] in built and normalized[1] in built:
                kept.append(entry)
        connections_plan = kept

    for _repair_round in range(1, MAX_INTERNAL_FIX_ATTEMPTS + 1):
        workflow = _assemble_n8n_workflow(plan.get("workflow_name", "Workflow"), nodes_by_name, connections_plan)
        workflow_path.parent.mkdir(parents=True, exist_ok=True)
        workflow_path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")

        passed, output = _validate_n8n_workflow(workflow_path)

        # Live verification only runs once structural validation is clean, and only on a complete
        # (non-partial) build - a partial build is a deliberate stop, not yet a whole thing to fire.
        # This is the actual oracle: everything above only checks that the workflow LOOKS right,
        # this checks that it RUNS right, against Douglas's real dev instance.
        if passed and not partial and triggers:
            print("DEBUG: structural validation clean - running live verification against dev n8n instance", file=sys.stderr)
            try:
                live_passed, live_problems = verify_native_workflow(
                    workflow, plan.get("workflow_name", "Workflow"), triggers, project_dir=project_dir
                )
            except N8nError as exc:
                live_passed, live_problems = False, [f"WORKFLOW: live verification could not run: {exc}"]
            # verify_native_workflow may have rewritten node credential references in place
            # (resolve_credential_references) - re-persist so the artifact on disk matches what was
            # actually verified, not the pre-resolution version written a few lines up.
            workflow_path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
            if not live_passed:
                passed = False
                output = "\n".join(live_problems)
                print(f"DEBUG: live verification found {len(live_problems)} problem(s): {output[:500]}", file=sys.stderr)
            else:
                print("DEBUG: live verification passed - workflow actually executed cleanly", file=sys.stderr)

        if passed:
            if not partial:
                _clear_checkpoint(code_dir, workflow=workflow)  # done - the workflow file is now the durable artifact
            return {**state, "builder_status": "done", "ambiguity_question": None, "total_tokens": total_tokens[0], **slice_info}

        _log_repair_round(
            project_dir, Path(project_dir).name if project_dir else "unknown", _repair_round, output, llm=llm,
        )

        # Missing-wire problems first: a node named in one of these two ways has nothing wrong
        # with its own body (see _repair_branch_connections / _repair_disconnected_nodes) - fix
        # the actual defect, and skip rebuilding that node's body below purely because its name
        # happened to appear in the same error text.
        connections_plan, branch_fixed = _repair_branch_connections(
            llm, config, output, connections_plan, node_specs, nodes_by_name, all_names, total_tokens,
        )
        connections_plan, unreachable_fixed = _repair_disconnected_nodes(
            llm, config, output, connections_plan, node_specs, nodes_by_name, all_names, total_tokens,
        )
        connections_plan, dead_pruned = _prune_dead_branches(output, connections_plan)
        connections_fixed = branch_fixed or unreachable_fixed or dead_pruned
        wire_fixed_names = set()
        if branch_fixed:
            wire_fixed_names |= {m.group("name") for m in _BRANCH_UNDERWIRED_RE.finditer(output)}
        if unreachable_fixed:
            wire_fixed_names |= {m.group("name") for m in _DISCONNECTED_NODE_RE.finditer(output)}
        if dead_pruned:
            wire_fixed_names |= {m.group("name") for m in _DEAD_BRANCH_RE.finditer(output)}
        if connections_fixed:
            # Drop every existing entry touching a fixed/pruned node, then re-add exactly what
            # connections_plan currently says for it - correct for both additions (branch/
            # unreachable fixes) and removals (dead-branch pruning) without risking duplicates from
            # trying to diff the two shapes against each other.
            plan["connections"] = [
                e for e in (plan.get("connections") or [])
                if e.get("from") not in wire_fixed_names and e.get("to") not in wire_fixed_names
            ] + [
                {"from": e["from"], "to": e["to"], "from_output": e["from_output"]}
                for e in connections_plan
                if e.get("from") in wire_fixed_names or e.get("to") in wire_fixed_names
            ]
            _save_checkpoint(code_dir, spec_text, plan, nodes_by_name, workflow=workflow)

        # Targeted repair: rebuild only the node(s) the validator actually named (and that aren't
        # already accounted for by the connections fix above), not everything.
        rebuilt_any = connections_fixed
        for name in list(nodes_by_name.keys()):
            if name in wire_fixed_names:
                continue
            if name in output:
                node_spec = next((n for n in node_specs if n.get("name") == name), None)
                if not node_spec:
                    continue
                try:
                    node_spec = {**node_spec, "purpose": node_spec["purpose"] + f"\n\nA previous attempt at this node was rejected during whole-workflow validation: {output}"}
                    nodes_by_name[name] = _build_n8n_node(
                        llm, config, node_spec, all_names, total_tokens, node_system_prompt,
                        conventions=conventions, project_dir=project_dir,
                    )
                    _save_checkpoint(code_dir, spec_text, plan, nodes_by_name, workflow=workflow)
                    rebuilt_any = True
                except ValueError:
                    pass
        if not rebuilt_any:
            # Validator found a problem not attributable to a specific named node (e.g. a dangling
            # connection to a name that was never a real node) - nothing more to target-repair.
            break

    return {**state, "builder_status": "ambiguous", "ambiguity_question": f"n8n workflow still failed validation after {MAX_INTERNAL_FIX_ATTEMPTS} repair rounds:\n{output[-1500:]}", "total_tokens": total_tokens[0]}


def _augment_spec_with_scale_adaptive_context(
    spec_text: str, project_dir, workflow: str | None = None, strip_build_units: bool = False
) -> str:
    """Folds in Douglas-approved PM epics/stories and Architect decisions, if this project has
    them, as binding context - the "approve-then-binding" rule (AGENTS.md "Scale-adaptive roles")
    actually taking effect, not just documented. Both files only ever exist because Agent 1 wrote
    them after Douglas approved that exact content (see pipeline/review_pm.py,
    pipeline/review_architect.py) - this function never calls either role itself, it only reads
    what was already approved. A project with neither file (the common case) gets its spec back
    completely unchanged.

    `workflow`: which n8n workflow this specific build invocation targets, when the Architect
    decision names more than one (see pipeline/n8n_workflow_target.py). Without an explicit "you
    are building ONLY this one, right now" instruction, the builder sees every workflow's scope in
    the same context with no signal which one it's actually meant to produce this call - real risk
    of conflating two workflows' nodes into one file, or building the wrong one entirely.

    No longer takes a `stage_scope` parameter (2026-09-28, Douglas's direct instruction: python's
    old PM-epic/checkpoint stage gating is retired in favor of Architect-led build_units
    decomposition - see _build_python_units, which constructs its OWN per-unit section and appends
    it to whatever this function returns, once per unit, inside its own loop, rather than once here
    for a single whole stage the way the old code did). n8n's own --until-slice is a completely
    separate mechanism and is untouched by this.

    `strip_build_units` (2026-09-29, real incident): pass True ONLY when the caller ALSO hands the
    unit's own contract and its dependencies' contracts forward separately (i.e. the python
    unit-decomposed path via _format_unit_task) - every OTHER unit's own full build_units entry is
    then pure, redundant weight that gets resent on EVERY turn of EVERY attempt for the rest of
    this call (it lives in the FIRST message, which is never pruned the way exploration results
    are). Real cost: ~55-62K tokens per turn on a real project, tripping the 500K per-attempt cap in
    9 turns for a change of a few lines. Everything else in the Architect decision (data_placement,
    credential_architecture, interface_contracts, regression_strategy, tech_quirks) is NOT
    unit-specific and stays exactly as-is - only the large, per-unit-redundant array is removed.
    """
    if not project_dir:
        return spec_text
    sections = [spec_text]
    architect_path = architect_decisions_path(project_dir)
    has_architect = architect_path.exists()
    if has_architect:
        architect_text = architect_path.read_text(encoding="utf-8")
        if strip_build_units:
            try:
                parsed = json.loads(architect_text)
                any_stripped = False
                for wf in ((parsed.get("system_shape") or {}).get("workflows") or []):
                    if isinstance(wf, dict) and wf.pop("build_units", None) is not None:
                        any_stripped = True
                if any_stripped:
                    architect_text = json.dumps(parsed, indent=2) + (
                        "\n(build_units omitted here - see \"YOUR UNIT THIS ROUND\" below for this "
                        "unit's own contract, and \"INTERFACES OF UNITS THIS ONE DEPENDS ON\" for "
                        "its dependencies' - never invent or assume another unit's shape from what "
                        "used to be shown here.)"
                    )
            except (json.JSONDecodeError, AttributeError):
                pass  # malformed/unexpected shape - fall back to the raw text, never crash on it
        sections.append(
            "--- ARCHITECT DECISIONS (Douglas-approved, binding - see AGENTS.md "
            "\"Scale-adaptive roles\") ---\n" + architect_text
        )
        if workflow:
            sections.append(
                f"--- THIS BUILD TARGETS EXACTLY ONE WORKFLOW: {workflow!r} ---\n"
                f"The ARCHITECT DECISIONS above name more than one n8n workflow. Build ONLY the one "
                f"named {workflow!r} (matching its 'name' field in system_shape.workflows, case/"
                f"punctuation-insensitive) - its own scope, not the others'. Where it calls another "
                f"workflow in this system, wire that as an HTTP Request node to that other workflow's "
                f"own trigger (per any interface_contracts entry naming the two), never by inlining "
                f"the other workflow's nodes into this one."
            )
    pm_path = pm_epics_path(project_dir)
    if pm_path.exists():
        sections.append(
            "--- PM EPICS & STORIES (Douglas-approved, binding - see AGENTS.md "
            "\"Scale-adaptive roles\") ---\n" + pm_path.read_text(encoding="utf-8")
        )
    if len(sections) > 1:
        # Context-clash guard: a real (if rare) disagreement between the original spec's prose and
        # a later, more structurally-informed decision needs an explicit precedence rule, not silent
        # accumulation - see AGENTS.md "Wiring PM/Architect into the build".
        precedence = ["--- PRECEDENCE, if any of the above conflicts with SPEC.md's prose ---"]
        if has_architect:
            precedence.append(
                "On structural questions (workflow count, data placement, credential architecture), "
                "ARCHITECT DECISIONS take precedence over SPEC.md prose - it was written specifically "
                "to resolve exactly this kind of ambiguity. On product-intent questions (what the "
                "thing should actually do), SPEC.md remains authoritative."
            )
        precedence.append(
            "If a conflict doesn't fit either rule above, stop and ask - do not guess which side wins."
        )
        sections.append("\n".join(precedence))
    return "\n\n".join(sections)


# The only two tools that actually change code - see _MAX_EXPLORATION_CALLS_BEFORE_FORCED_WRITE
# below. Deliberately narrower than "not list_files/search_files/read_file": run_command doesn't
# change anything either, so a read/run/read/run diagnostic loop that never once writes or edits a
# file is just as unproductive as pure reading.
_CODE_CHANGING_TOOLS = {"write_file", "edit_file"}
_EXPLORATION_TOOLS = {"list_files", "search_files", "read_file"}

# Filenames (stem, case-insensitive, leading/trailing underscores ignored) that read as disposable
# rather than real project code. Real incident this guards against (2026-09-28, Douglas reviewing a
# transcript himself): a 98-byte throwaway `_probe.py` satisfied the bare "wrote/edited something"
# condition, lifting escalation and resetting the exploration-call counter, which let the model go
# straight back to reading instead of doing the actual work the escalation was meant to force.
_SCRATCH_FILE_STEMS = {"probe", "scratch", "tmp", "temp", "wip", "throwaway", "debug", "test_probe"}


# 2026-09-28, Douglas's direct instruction (relayed, after being surprised the "compounding turn
# fetch" problem was still open once he learned prompt caching only cuts dollar cost, not the
# resent-every-turn token COUNT that a long attempt actually compounds): a chat-completions API is
# stateless - the WHOLE messages list is resent every single turn, so an early read_file/list_files/
# search_files result never stops costing tokens once appended, no matter how irrelevant it's become
# by turn 30. Deliberately the client-side, OpenRouter-compatible option, not Anthropic's native
# context-editing feature - that would mean bypassing OpenRouter for the builder role specifically, a
# new SDK dependency, and (found while actually comparing the two) breaking attempt_fallback's
# assumption that every role's primary and fallback paths both go through the same OpenRouter
# model_factory - directly against Douglas's own repeatedly-stated provider-agnostic architecture
# principle, not just a bigger lift. This is a real, evidenced problem (a real transcript hit
# 900K+ tokens in a single attempt) with a real, bounded fix: never deletes a message (would break
# the tool_call_id <-> AIMessage.tool_calls pairing every chat-completions API requires, producing a
# malformed conversation), only replaces stale content with a short marker once real progress makes
# it safe to assume the model doesn't need the full detail resent forever - it can always re-run the
# call if it turns out to still matter.
_PRUNED_EXPLORATION_PLACEHOLDER = (
    "[pruned - this result is from earlier in this attempt, before the most recent real write/edit. "
    "Re-run the tool call if you genuinely need this content again; it may also be stale by now.]"
)


def _prune_stale_exploration_results(tool_messages: dict) -> int:
    """Replaces the .content of each given ToolMessage with a short placeholder IN PLACE (confirmed
    safe - langchain_core's ToolMessage is a mutable pydantic model, not frozen) rather than removing
    the message from the conversation, which would break the tool_call_id <-> AIMessage.tool_calls
    pairing every chat-completions API requires. `tool_messages` is exploration-tool (list_files/
    search_files/read_file) ToolMessage objects accumulated from EARLIER turns this attempt - never
    the current turn's own results, see the call site in the main loop. Returns how many were
    actually changed, for DEBUG logging - a message already carrying the placeholder from an earlier
    prune doesn't count again."""
    pruned = 0
    for tool_message in tool_messages.values():
        if tool_message.content != _PRUNED_EXPLORATION_PLACEHOLDER:
            tool_message.content = _PRUNED_EXPLORATION_PLACEHOLDER
            pruned += 1
    return pruned


def _normalize_dir_path(path: str) -> str:
    """Treats '', '.', '/', './' and similar trivial spellings of "the project root" as the same
    key. Real transcript evidence (2026-09-28, a real creative-team rebuild): list_files('.'),
    list_files(''), list_files('/'), then list_files('.') again - four calls in the first four turns
    of one attempt, all meaning exactly the same thing, none of them caught by any duplicate guard
    because each one was a cosmetically different string."""
    normalized = path.strip().rstrip("/")
    return normalized if normalized else "."


def _exploration_call_signature(name: str, args: dict):
    """A normalized signature for a list_files/search_files call, for the duplicate-call guard below
    - None for any other tool, including read_file (which already has its own simpler, dedicated
    path-based guard, unchanged). Two calls with the same signature are guaranteed to return the
    same result, since nothing about the filesystem could have changed between them within one
    guard's lifetime (cleared on any real write/edit, same as read_paths_this_attempt)."""
    if name == "list_files":
        return ("list_files", _normalize_dir_path(args.get("path", ".")))
    if name == "search_files":
        return ("search_files", args.get("pattern", ""), _normalize_dir_path(args.get("path", ".")))
    return None


# 2026-09-28, Douglas's direct instruction (discussed with him and relayed via video-generator-78):
# loop 2 (the builder<->reviewer round loop) hands the builder STRUCTURED findings
# ({"node", "claim_type", "description", "spec_quote"}) to react to - loop 1 (this function's own
# internal MAX_INTERNAL_FIX_ATTEMPTS retry loop) only ever fed forward one giant blob of raw pytest
# stdout/stderr text. Give loop 1 the same discipline, mechanically - no new LLM call, no new LLM
# role: pytest's own default output already reliably distinguishes which test failed (the "short
# test summary info" section's "FAILED <nodeid>" lines) from why (the "E   ..." line in that test's
# own FAILURES block above it) - this just parses a format pytest already produces, every time,
# without being asked to. Best-effort by design: a pytest version difference, or a failure shape
# with no FAILURES section at all (e.g. a pure collection error), simply yields [] - callers must
# always show the full raw output too, never rely on this extraction alone.
_PYTEST_FAILED_SUMMARY_RE = re.compile(r"^(?:FAILED|ERROR) (?P<nodeid>\S+)(?: - (?P<msg>.*))?$", re.MULTILINE)
_PYTEST_FAILURE_HEADER_RE = re.compile(r"^_{3,} (?P<test_name>.+?) _{3,}\s*$", re.MULTILINE)
_PYTEST_E_LINE_RE = re.compile(r"^E\s+(?P<msg>.+)$", re.MULTILINE)


def _parse_pytest_failures(output: str) -> list[dict]:
    """[{"test": "<nodeid>", "reason": "<first E-line in that test's own block, or ''>"}, ...] - see
    the module comment above. Order matches the short-summary section's own order (pytest's, not
    reconstructed)."""
    summary = list(_PYTEST_FAILED_SUMMARY_RE.finditer(output))
    if not summary:
        return []
    failed_nodeids = [m.group("nodeid") for m in summary]
    # A collection/import error (e.g. ModuleNotFoundError) prints "ERROR <file> - <reason>" and has
    # no FAILURES block; count it as a failure of that file's unit instead of a silent pass.
    summary_reason = {m.group("nodeid"): (m.group("msg") or "").strip() for m in summary}

    reason_by_test_name: dict = {}
    headers = list(_PYTEST_FAILURE_HEADER_RE.finditer(output))
    for i, header in enumerate(headers):
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(output)
        block = output[header.end():block_end]
        e_match = _PYTEST_E_LINE_RE.search(block)
        if e_match:
            reason_by_test_name[header.group("test_name").strip()] = e_match.group("msg").strip()

    findings = []
    for nodeid in failed_nodeids:
        test_name = nodeid.rsplit("::", 1)[-1]
        findings.append({"test": nodeid, "reason": reason_by_test_name.get(test_name) or summary_reason.get(nodeid, "")})
    return findings


def _test_belongs_to_unit(test_nodeid: str, files: list) -> bool:
    """Whether a pytest nodeid's own test file corresponds to one of this unit's declared files -
    the same test_<module_stem>.py convention pipeline/build_stages.py's own coverage-gap check
    already assumes (check_stage_test_coverage_gaps), applied here to attribute a FAILING test to a
    unit instead of to a coverage GAP. Basename-based, tolerant of a nested tests/ directory - a test
    file's real location, like its source counterpart, is a hint's business, not a literal
    requirement."""
    test_file = test_nodeid.split("::", 1)[0]
    test_stem = Path(test_file).stem
    if test_stem.startswith("test_"):
        test_stem = test_stem[len("test_"):]
    return any(f and Path(f).stem == test_stem for f in files)


def _owner_unit_name(test_nodeid: str, build_units: list) -> str:
    """Name of the unit whose declared files own this test's file, else the test file path."""
    for u in build_units:
        if _test_belongs_to_unit(test_nodeid, u.get("files") or []):
            return u.get("name") or test_nodeid.split("::", 1)[0]
    return test_nodeid.split("::", 1)[0]


def _read_back(code_dir: Path, path: str) -> str:
    """Re-reads a file's real current content straight from disk. Only needed for edit_file, whose
    own arguments carry just the changed fragment (old_string/new_string) - write_file's own
    `content` argument already IS the whole file, so it never needs this."""
    try:
        return (code_dir / path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _is_substantive_write(path: str, current_full_content: str, tool_result: str) -> bool:
    """Whether a successful write_file/edit_file call represents real progress, for the
    escalation-lifting / exploration-counter-reset decision ONLY. touched_files (the mechanical
    record shown to the reviewer's builder-account section) stays unconditional on any successful
    write/edit regardless of this check - that record's job is "what did the builder actually touch",
    never "was it any good."

    Deliberately does NOT require a def/class in .py content - an early version of this did, and a
    real test caught it rejecting a legitimate flat driver script (no function needed to just run some
    steps and print a result) as if it were fake filler. Byte-count doesn't discriminate well either:
    the real incident's throwaway `_probe.py` (98 bytes) and a legitimate small smoke_run.py are the
    same order of magnitude - length alone can't tell them apart. What the real incident actually
    evidenced was the FILENAME (`_probe.py` reads as disposable on its face), so that's the check kept
    here, plus two other zero-ambiguity signals: a write that doesn't even parse (agent_tools.py's own
    immediate ast.parse check, surfaced via _SYNTAX_ERROR_MARKER), and content that's empty or
    whitespace-only. Anything past that bar counts - this is a backstop against obvious gaming, not a
    code-quality judgment call.
    """
    if _SYNTAX_ERROR_MARKER in tool_result:
        return False
    if Path(path).stem.lower().strip("_") in _SCRATCH_FILE_STEMS:
        return False
    return bool(current_full_content.strip())


# Redesigned 2026-09-28 (Douglas's own reaction to a real transcript: "one of my 3 max rounds could
# end up 1 full round purely read-only... it should just know it's supposed to write code" - he was
# right, this was a real design flaw, not expected model behavior to work around). Two problems
# with the original version of this cap:
#
# (1) It counted TURNS, not tool CALLS - the same real transcript had NINE list_files calls in one
# single turn, so a threshold measured in turns let a large amount of real exploring hide inside
# what looked like a small number of turns. This counts every individual exploration call instead.
#
# (2) It gave a polite, ignorable text nudge before doing anything mechanical, and the real
# transcript proved that doesn't work - the nudge fired and was ignored, twice, before the run died
# on the token budget with nothing ever written. There is no longer a nudge-first phase: the FIRST
# time this threshold is crossed, the tool is mechanically taken away immediately - see "escalated"
# in _build_python_agentic. The model never gets the chance to knowingly ignore a request, because
# there is no request - only the tools it can actually use.
#
# Deliberately LESS than MAX_AGENT_TURNS, not equal to it: in the worst case (one exploration call
# per turn, no batching), hitting this threshold in the attempt's very LAST turn would escalate
# with zero turns left to act on it, wasting the whole mechanism - the attempt would just end via
# "ran out of turns" instead. This guarantees real turns remain after escalation even in that
# pathological case, regardless of how either constant gets tuned later.
_MAX_EXPLORATION_CALLS_BEFORE_FORCED_WRITE = max(1, MAX_AGENT_TURNS - 3)


def _dispatch_tool_call(tools_by_name: dict, tool_call: dict) -> str:
    """Executes one tool call, returning its result as a string. A tool's own errors already come
    back as a string (see pipeline/agent_tools.py - a rejected path, a missing file, a disallowed
    command all return an error message rather than raising) - the broad except here is only for
    something truly unexpected (a bug in the tool itself), fed back to the model as an error so one
    bad tool call can't crash the whole build.
    """
    name = tool_call.get("name")
    tool_fn = tools_by_name.get(name)
    if tool_fn is None:
        return f"ERROR: unknown tool '{name}'."
    try:
        return tool_fn.invoke(tool_call.get("args") or {})
    except Exception as exc:  # noqa: BLE001 - a tool bug must feed back to the model, never crash the build
        return f"ERROR: tool '{name}' raised an unexpected error: {exc}"


def _describe_tool_call(tool_call: dict) -> str:
    """A short, per-tool-appropriate description of a tool call for the DEBUG log line below - NOT
    just "the path argument" for every tool. Real bug this fixes (2026-09-27, creative-team's actual
    from-scratch build attempt): the original version of this logging only ever printed a "path"
    key, so every run_command call showed up as the useless `run_command('')` (its real argument is
    named "command", not "path") and every search_files call hid its actual pattern - exactly the
    turn-by-turn visibility this logging exists to provide, silently missing for two of six tools.
    """
    args = tool_call.get("args") or {}
    name = tool_call.get("name")
    if name == "run_command":
        return repr(args.get("command", ""))
    if name == "search_files":
        return f"pattern={args.get('pattern', '')!r} path={args.get('path', '.')!r}"
    if name == "write_file":
        return f"path={args.get('path', '')!r} ({len(str(args.get('content', '')))} chars)"
    if name == "edit_file":
        return f"path={args.get('path', '')!r} old_string={str(args.get('old_string', ''))[:60]!r}"
    if name == "record_tech_quirk":
        return repr(str(args.get("lesson", ""))[:100])
    return repr(args.get("path", ""))


_UNIT_BASELINE_FILENAME = ".unit_test_baselines.json"


def _new_other_failures_feedback(other_block: str) -> str:
    return (
        "THESE TESTS OUTSIDE YOUR UNIT'S OWN FILES PASSED BEFORE YOUR CHANGES AND "
        "FAIL NOW. If they fail because YOUR intended change altered an interface "
        "(a new/reordered required parameter, a renamed field, a changed return "
        "shape), update those callers/tests minimally so they match the new "
        "interface - that is part of your job, and do NOT revert your intended "
        "change to make them pass. Only if the failure is an unintended side "
        "effect, revert or fix that specifically. Do not weaken or delete tests to "
        "get green - the ONE exception: tests that cover a feature/module a HUMAN RULING "
        "explicitly retires or removes may be deleted, or rewritten to the replacement path "
        "the ruling names. Do NOT report done while any of these still fail - if you truly "
        "cannot fix them, reply 'AMBIGUOUS:' with the reason instead.\n" + other_block
    )


def _load_unit_baseline(project_dir, unit_name) -> set | None:
    if not project_dir or not unit_name:
        return None
    try:
        data = json.loads((Path(project_dir) / _UNIT_BASELINE_FILENAME).read_text(encoding="utf-8"))
        entry = data.get(unit_name)
        return set(entry) if isinstance(entry, list) else None
    except (OSError, ValueError, AttributeError):
        return None


def _save_unit_baseline(project_dir, unit_name, failures: set) -> None:
    if not project_dir or not unit_name:
        return
    path = Path(project_dir) / _UNIT_BASELINE_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data[unit_name] = sorted(failures)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except (OSError, ValueError):
        pass


def _clear_unit_baseline(project_dir, unit_name) -> None:
    if not project_dir or not unit_name:
        return
    path = Path(project_dir) / _UNIT_BASELINE_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if unit_name in data:
            del data[unit_name]
            if data:
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            else:
                path.unlink()
    except (OSError, ValueError, AttributeError):
        pass


def _build_python_agentic(
    state: BuildState, llm, config, spec_text: str, code_dir: Path, stage_info: dict, unit_files: list | None = None,
    refresh_spec_text=None, unit_name: str | None = None, finding_baseline: dict | None = None,
    contract_check=None,
) -> BuildState:
    """The agentic python builder (2026-09-27, replacing the old one-shot <<<FILE>>> completion that
    was the confirmed root cause of a real, expensive incident - see AGENT_SYSTEM_PROMPT and
    pipeline/agent_tools.py for the full story). Given real tools instead of one shot at everything,
    the builder explores, edits targeted files, and runs its own verification as it goes - the same
    way n8n's builder plans and live-verifies one node at a time, and n8n's build path is completely
    untouched by this function.

    Two nested caps: MAX_INTERNAL_FIX_ATTEMPTS fresh agent conversations (outer - a fresh message
    history each time, so context never grows unboundedly across attempts), each up to
    MAX_AGENT_TURNS tool-call round-trips (inner). Independent verification (pytest, the n8n-shape
    check, smoke_run.py) runs fresh after EVERY attempt, including one that ran out of turns
    mid-tool-call - never trusting the agent's own in-loop tool calls OR its mere absence as the
    final word, the same "mechanical check, not self-report" principle behind every other fix built
    the same day. (Running verification even when the attempt ran out of turns is itself a
    2026-09-28 fix: a real transcript showed 6 genuine files written - independently confirmed by
    reading them - discarded unchecked because the old code skipped verification entirely on that
    path. Mechanical truth wins over an incomplete self-report either way: if what's already on disk
    happens to pass, that's done, whether or not the agent itself got to say so.) Two consecutive
    attempts ending in the IDENTICAL verification failure stop the run immediately (Douglas's
    explicit direction) rather than spending a third attempt on something already proven not to
    change - this rule now actually applies to a ran-out-of-turns attempt too, not only one the
    agent voluntarily stopped.

    Within one attempt, too many cumulative exploration calls (list_files/search_files/read_file)
    with nothing SUBSTANTIVELY written or edited IMMEDIATELY restricts the next call to only
    write_file/edit_file - the API itself is never offered any other tool, so the model cannot choose
    to keep exploring even if it wants to. Deliberately not a nudge-first, escalate-second design
    (2026-09-28, Douglas's own reaction to a real transcript: a whole attempt spent purely reading is
    a real design flaw, not something to ask the model nicely to stop doing) - see
    _MAX_EXPLORATION_CALLS_BEFORE_FORCED_WRITE's own comment for the full incident and why a
    turn-based, nudge-first version of this already proved insufficient.

    "Substantively" (see _is_substantive_write) matters because escalation is otherwise gameable:
    Douglas found a real transcript where a 98-byte throwaway `_probe.py` satisfied a bare
    "wrote/edited something" check and lifted the restriction straight back off. A scratch-named file,
    or a write that fails agent_tools.py's own immediate ast.parse check, or content that's empty or
    whitespace-only, does NOT lift escalation or reset the counter - though it's still recorded
    unconditionally in touched_files, since that record's job is showing the reviewer what the builder
    actually touched, never vouching for whether it was any good.
    """
    project_dir = state.get("project_dir")
    tools = make_agent_tools(code_dir)
    # record_tech_quirk (2026-09-28, Douglas's direct instruction - see pipeline/tech_quirks.py)
    # is added alongside the six code_dir-scoped tools, not inside make_agent_tools itself: it
    # writes to memory/tech-quirks.md, a repo-level, cross-project file, not anything scoped to
    # this one project's code_dir - keeping it separate keeps make_agent_tools' own contract
    # ("everything here is scoped to one project") true without an exception buried inside it.
    tools = tools + [make_tech_quirk_tool(source="builder (python, agentic)")]
    tools_by_name = {t.name: t for t in tools}
    agent_llm = llm.bind_tools(tools)
    # Escalation toolset (see nudge_count below) - the two code-changing tools plus
    # record_tech_quirk, so once escalated the model cannot keep exploring even if it wants to (the
    # API itself never offers it any other tool to call), but can still record something it already
    # confirmed - record_tech_quirk doesn't count as exploration OR as the substantive write that
    # would lift escalation (see _is_substantive_write - it only ever looks at write_file/edit_file
    # calls), so including it here creates no way to game the restriction. Bound once, reused across
    # attempts.
    restricted_tools = [tools_by_name["write_file"], tools_by_name["edit_file"], tools_by_name["record_tech_quirk"]]
    restricted_agent_llm = llm.bind_tools(restricted_tools)

    # Loaded ONCE per call (not per attempt/turn) - tech-quirks.md doesn't change mid-run, and
    # computing this once keeps the system prompt a stable, cacheable prefix across every attempt
    # and every turn this call makes (see model_factory.py's _EXPLICIT_CACHE_CONTROL_PREFIXES and
    # with_tech_quirks' own docstring).
    system_prompt_text = with_tech_quirks(AGENT_SYSTEM_PROMPT)

    total_tokens = state.get("total_tokens", 0)  # lifetime cumulative - reporting only, see MAX_TOKENS_PER_ATTEMPT's own comment in constants.py
    invocation_tokens = 0  # resets to 0 at the start of every call to this function - see MAX_TOKENS_PER_INVOCATION
    problems = state.get("review_problems") or []
    failure_output = ""
    prior_failure_signature = None
    # Mechanical record of every file write_file/edit_file actually touched, across ALL attempts
    # (files persist on disk between attempts even though each attempt's conversation is fresh) -
    # ground truth for reviewer_node.py's builder-account section below, never the agent's own
    # claim about what it changed. Order preserved, latest action per path kept.
    touched_files: dict = {}
    builder_account = ""

    # Real incident this closes (2026-09-29): a unit's own verification required the WHOLE test
    # suite to pass - meaning a unit with nothing wrong of its own could never complete while a
    # DIFFERENT, unrelated unit's own already-known bug kept some other test red, so the sequence
    # never advanced far enough to give the ACTUALLY broken unit its own dedicated turn again. The
    # model's only lever to pass ITS OWN gate was to keep reaching into the other unit's file - not
    # confusion, the only option available (confirmed live: ~1M tokens spent this way with zero
    # progress). Captured ONCE, before this call's first attempt, so every attempt is judged against
    # the SAME "what was already broken before I started" baseline, not a moving target across
    # retries. unit_files is only passed by the unit-decomposed path (_build_python_units) - the
    # non-decomposed direct call (unit_files=None, its default) is completely unaffected, still
    # requires the whole suite to pass exactly as before, since it has no separate "someone else's
    # problem" to tolerate.
    baseline_other_failures: set = set()
    if unit_files:
        # A baseline persisted by an EARLIER, unfinished invocation of this same unit wins over a
        # fresh capture: a resumed unit starts on a tree its own previous run already changed (e.g.
        # a Config signature edited but its callers' tests not yet updated), so capturing now would
        # bake that unit-caused breakage in as "pre-existing" and never ask the builder to fix it.
        stored_baseline = _load_unit_baseline(project_dir, unit_name)
        if stored_baseline is not None:
            baseline_other_failures = stored_baseline
        else:
            _, baseline_pytest_output = _run_pytest(code_dir)
            baseline_other_failures = {
                f["test"] for f in _parse_pytest_failures(baseline_pytest_output)
                if not _test_belongs_to_unit(f["test"], unit_files)
            }
            _save_unit_baseline(project_dir, unit_name, baseline_other_failures)
        if stored_baseline is not None:
            # A resumed unit's FIRST attempt otherwise gets no hint that failures already on disk
            # are its responsibility (only later attempts get post-verification feedback) - the
            # model ran pytest, saw them red, and reported done. State them up front.
            _, start_output = _run_pytest(code_dir)
            start_new_other = [
                f for f in _parse_pytest_failures(start_output)
                if not _test_belongs_to_unit(f["test"], unit_files) and f["test"] not in baseline_other_failures
            ]
            if start_new_other:
                failure_output = _new_other_failures_feedback("\n".join(
                    f"- {f['test']}" + (f": {f['reason']}" if f['reason'] else "") for f in start_new_other
                ))

    rounds_cap = max(1, int(state.get("agent_rounds") or MAX_ROUNDS_PER_ATTEMPT))
    read_first_round = bool(state.get("read_first_round"))
    for _attempt in range(1, MAX_INTERNAL_FIX_ATTEMPTS + 1):
        if _attempt > 1 and refresh_spec_text is not None:
            # Re-rendered per attempt so the inlined current file contents reflect what the
            # previous attempt already changed on disk (a stale copy made attempts 2/3 re-try
            # an edit that had already been applied).
            spec_text = refresh_spec_text()
        task_prompt = f"SPEC:\n{spec_text}"
        if state.get("rulings"):
            task_prompt += "\n\n" + rulings_block(state["rulings"])
        if problems:
            task_prompt += "\n\nREVIEWER PROBLEMS TO FIX:\n" + "\n".join(f"- [{_finding_ref(p)}] {p}" for p in problems)
        if finding_baseline is not None:
            task_prompt += (
                "\n\nRULES FOR FIXING REVIEWER FINDINGS: (1) For each finding, change the real source "
                "AND add or change a test that reproduces it (fails on the old code, passes on your fix); "
                "tests that were already green prove nothing, and a rebuild with no source change or no "
                "new test is rejected. (2) If fixing a finding requires changing this unit's "
                "Architect-approved interface (a signature or export listed in the interface block), do "
                "NOT change it and do NOT silently skip the finding - reply 'AMBIGUOUS: <finding> requires "
                "changing the approved interface <old> to <new>' so a human can decide on an amendment. "
                "(3) If a finding describes a defect that is ALREADY fixed in the current code (check the "
                "code, do not assume), do not ask anyone: end your final reply with one line per such "
                "finding, exactly `ALREADY_FIXED: <finding id> | <verbatim quote from the current code "
                "that proves the fix>` (the id is the bracketed [f-...] tag; the quote must appear "
                "verbatim in the code or the claim is rejected). The reviewer then confirms or re-opens "
                "it. Fix every other finding normally. (4) Only if you genuinely disagree that a finding "
                "is a defect and cannot prove it fixed, reply AMBIGUOUS with the evidence."
            )
        if failure_output:
            task_prompt += f"\n\nYOUR LAST ATTEMPT'S VERIFICATION FAILED (fix this):\n{failure_output}"

        print(
            f"DEBUG: agent attempt {_attempt}/{MAX_INTERNAL_FIX_ATTEMPTS} starting (total_tokens="
            f"{total_tokens}, invocation_tokens={invocation_tokens})",
            file=sys.stderr,
        )
        messages = [SystemMessage(system_prompt_text), HumanMessage(task_prompt)]
        ambiguity_question = None
        final_text = ""
        ran_out_of_turns = True
        attempt_tokens = 0  # resets to 0 for THIS attempt - see MAX_TOKENS_PER_ATTEMPT
        attempt_token_budget_exceeded = False
        # True the moment ANY turn this attempt makes a substantive write/edit (see
        # _is_substantive_write) - 2026-09-29, real incident: a unit ran out of turns in every one
        # of its 3 attempts, writing NOTHING (its own file byte-identical to what was already there
        # before the attempt started), yet still got reported "done" because pytest/smoke happened
        # to pass on that untouched, pre-existing code - a pass that was already true before this
        # attempt even began proves nothing about what THIS attempt did. See the verification gate
        # below for how this is used - scoped specifically to ran_out_of_turns, not a voluntary
        # "done" report, matching the real incident's own shape exactly.
        attempt_made_substantive_progress = False
        # Cumulative COUNT of exploration tool calls this attempt (list_files/search_files/
        # read_file - NOT turns, see _MAX_EXPLORATION_CALLS_BEFORE_FORCED_WRITE's own comment for
        # why), reset to 0 on any real write_file/edit_file. The moment it reaches the threshold,
        # escalation is immediate and mechanical - no nudge-first phase, see "escalated" below.
        exploration_calls_this_attempt = 0
        escalated = False
        # Paths already read via read_file this attempt, cleared on any real code change. The same
        # transcript showed the exact same site-packages/langgraph/graph/state.py read TWICE,
        # 46,000+ tokens apart, with nothing having changed in between - short-circuits a second
        # read of an unchanged file instead of paying to resend its content again.
        read_paths_this_attempt: set = set()
        # Normalized (tool, args...) signatures for list_files/search_files calls made this attempt
        # - see _exploration_call_signature. Cleared alongside read_paths_this_attempt on any real
        # write/edit (the same "nothing could have changed" reasoning, just for a different pair of
        # tools).
        exploration_signatures_this_attempt: set = set()
        # Exploration-tool ToolMessage objects from turns BEFORE the most recent substantive write
        # this attempt - see _prune_stale_exploration_results above. Reset fresh each attempt, same
        # lifecycle as read_paths_this_attempt (a fresh attempt is a fresh conversation).
        exploration_tool_messages_this_attempt: dict = {}

        _turn = 0
        round_made_progress = False
        while True:
            if _turn and _turn % MAX_AGENT_TURNS == 0:
                _round = _turn // MAX_AGENT_TURNS
                if (not round_made_progress and not (read_first_round and _round == 1)) or _round >= rounds_cap:
                    break
                print(
                    f"DEBUG: agent attempt {_attempt} round {_round} made progress - continuing the "
                    f"same conversation into round {_round + 1}/{rounds_cap}",
                    file=sys.stderr,
                )
                round_made_progress = False
            _turn += 1
            current_llm = restricted_agent_llm if escalated else agent_llm
            current_tools = restricted_tools if escalated else tools
            try:
                response = with_retries_tool_call(lambda: current_llm.invoke(messages, config=config))
            except Exception as exc:
                response = attempt_fallback(
                    "builder", lambda fb_llm: fb_llm.bind_tools(current_tools).invoke(messages, config=config),
                    project_dir=project_dir, platform="python", primary_error=exc,
                )
            tokens_this_call = extract_tokens(response)
            total_tokens += tokens_this_call
            attempt_tokens += tokens_this_call
            invocation_tokens += tokens_this_call
            usage = extract_usage(response)
            print(
                f"DEBUG: agent attempt {_attempt} turn {_turn} usage: input={usage['input']} "
                f"(cached={usage['cached']}) output={usage['output']}",
                file=sys.stderr,
            )
            if invocation_tokens >= MAX_TOKENS_PER_INVOCATION:
                # Per-INVOCATION cap - ends the WHOLE invocation, not just this attempt. Loop 2's
                # own existing MAX_ROUNDS is the real backstop now (see constants.py) - a resumed
                # round always starts a FRESH _build_python_agentic call with invocation_tokens=0.
                return {
                    **state,
                    "builder_status": "ambiguous",
                    "ambiguity_question": (
                        f"This invocation exceeded its {MAX_TOKENS_PER_INVOCATION}-token budget "
                        f"({invocation_tokens} tokens across {MAX_INTERNAL_FIX_ATTEMPTS} attempts) - "
                        "stopped to control cost. If the reviewer sends this back for another round, "
                        "that round gets its own fresh budget."
                    ),
                    "total_tokens": total_tokens,
                }
            if attempt_tokens >= MAX_TOKENS_PER_ATTEMPT:
                # Per-ATTEMPT cap - ends only THIS attempt, same as running out of turns: mechanical
                # truth (independent verification) still runs on whatever's on disk below, and the
                # NEXT attempt (if any) starts with a completely fresh attempt_tokens budget.
                attempt_token_budget_exceeded = True
                print(
                    f"DEBUG: agent attempt {_attempt} turn {_turn} exceeded its {MAX_TOKENS_PER_ATTEMPT}-token "
                    f"attempt budget ({attempt_tokens} tokens) - ending this attempt early",
                    file=sys.stderr,
                )
                messages.append(response)
                break

            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                ran_out_of_turns = False
                text = (response.content or "").strip()
                if text.startswith("AMBIGUOUS:"):
                    ambiguity_question = text[len("AMBIGUOUS:"):].strip()
                else:
                    # The agent's own final account of this attempt - kept for reviewer_node.py's
                    # builder-account section, clearly labeled there as an unverified claim, never
                    # a substitute for the reviewer's own read of the actual code. Overwritten by a
                    # later attempt's own final text if this attempt still doesn't pass verification.
                    builder_account = text
                    final_text = text
                print(
                    f"DEBUG: agent attempt {_attempt} turn {_turn} - no more tool calls, "
                    f"{'ambiguous' if ambiguity_question else 'reporting done'} "
                    f"(total_tokens={total_tokens})",
                    file=sys.stderr,
                )
                break

            any_code_changed = False
            new_exploration_calls = 0
            new_exploration_tool_messages: dict = {}
            for tool_call in tool_calls:
                name = tool_call.get("name")
                args = tool_call.get("args") or {}
                read_path = (
                    (args.get("path"), args.get("start_line") or 1, args.get("end_line") or 0)
                    if name == "read_file" and args.get("path") is not None
                    else None
                )
                exploration_signature = _exploration_call_signature(name, args)

                substantive = False
                if read_path is not None and read_path in read_paths_this_attempt:
                    result = (
                        f"You already read '{read_path[0]}' (same line range) earlier this attempt and haven't changed "
                        "it since, so its content hasn't changed either - look back in this "
                        "conversation for what it said instead of reading it again."
                    )
                elif exploration_signature is not None and exploration_signature in exploration_signatures_this_attempt:
                    result = (
                        f"You already called {name} with these same (or equivalent) arguments "
                        "earlier this attempt, and nothing on disk has changed since - the result "
                        "would be identical. Look back in this conversation for what it returned "
                        "instead of calling it again."
                    )
                else:
                    result = _dispatch_tool_call(tools_by_name, tool_call)
                    if read_path is not None and not str(result).startswith("ERROR"):
                        read_paths_this_attempt.add(read_path)
                    if exploration_signature is not None and not str(result).startswith("ERROR"):
                        exploration_signatures_this_attempt.add(exploration_signature)
                    if name in _CODE_CHANGING_TOOLS and not str(result).startswith("ERROR"):
                        # Mechanical ground truth, not the agent's claim - see touched_files above.
                        # Unconditional on any SUCCESSFUL write/edit regardless of substantiveness -
                        # this record's job is "what did the builder actually touch", not "was it good".
                        write_path = args.get("path", "")
                        touched_files[write_path] = name
                        current_content = (
                            args.get("content", "") if name == "write_file" else _read_back(code_dir, write_path)
                        )
                        substantive = _is_substantive_write(write_path, current_content, str(result))

                tool_message = ToolMessage(content=result, tool_call_id=tool_call["id"])
                messages.append(tool_message)
                # Real incident this print exists to catch (2026-09-27, creative-team's actual bug):
                # two consecutive real attempts burned ~580K and ~500K tokens with zero files ever
                # written, discovered only afterward via file mtimes because nothing printed what
                # the agent was actually doing turn by turn - the same lesson as every stderr DEBUG
                # print already elsewhere in this pipeline, just missing here until this incident.
                # _describe_tool_call (not a bare "path" lookup) - a SEPARATE real bug found in the
                # very first transcript this logging produced: run_command's real argument is named
                # "command", not "path", so every run_command call printed as the useless
                # run_command('') until this was fixed.
                print(
                    f"DEBUG: agent attempt {_attempt} turn {_turn} called {name}({_describe_tool_call(tool_call)}) "
                    f"(total_tokens={total_tokens})",
                    file=sys.stderr,
                )
                if str(result).startswith("ERROR"):
                    print(
                        f"DEBUG: agent attempt {_attempt} turn {_turn} {name} was rejected: "
                        f"{str(result)[:400]!r}",
                        file=sys.stderr,
                    )
                if substantive:
                    # A non-substantive write/edit (see _is_substantive_write) deliberately falls
                    # through here doing nothing - it doesn't reset the counter (that would let a
                    # throwaway probe file "buy" more exploration budget, the exact incident this
                    # exists to close) and it isn't an exploration call either.
                    any_code_changed = True
                    attempt_made_substantive_progress = True
                    round_made_progress = True
                elif name in _EXPLORATION_TOOLS:
                    new_exploration_calls += 1
                    new_exploration_tool_messages[tool_call["id"]] = tool_message

            if any_code_changed:
                exploration_calls_this_attempt = 0
                if escalated:
                    print(f"DEBUG: agent attempt {_attempt} turn {_turn} - wrote/edited, lifting the tool restriction", file=sys.stderr)
                escalated = False
                read_paths_this_attempt.clear()
                exploration_signatures_this_attempt.clear()
                # Prune what accumulated BEFORE this turn's real progress - never this turn's own
                # new exploration results (new_exploration_tool_messages), which get at least one
                # full turn of visibility before they're themselves eligible next time.
                pruned = (
                    _prune_stale_exploration_results(exploration_tool_messages_this_attempt)
                    if PRUNE_STALE_EXPLORATION_MID_ATTEMPT else 0
                )
                if pruned:
                    print(
                        f"DEBUG: agent attempt {_attempt} turn {_turn} - pruned {pruned} stale "
                        "exploration result(s) from earlier turns after real progress",
                        file=sys.stderr,
                    )
                exploration_tool_messages_this_attempt = dict(new_exploration_tool_messages)
            else:
                exploration_calls_this_attempt += new_exploration_calls
                exploration_tool_messages_this_attempt.update(new_exploration_tool_messages)

            if (
                not escalated and exploration_calls_this_attempt >= _MAX_EXPLORATION_CALLS_BEFORE_FORCED_WRITE
                and not (read_first_round and _turn <= MAX_AGENT_TURNS)
            ):
                # Immediate, mechanical, no nudge-first phase - see _MAX_EXPLORATION_CALLS_BEFORE_
                # FORCED_WRITE's own comment for why a polite request already proved insufficient.
                escalated = True
                # SECOND pruning trigger, alongside "a substantive write happened" above - added
                # 2026-09-28 after a real transcript showed why the write-triggered one alone isn't
                # enough: a real creative-team rebuild spent 3 full attempts (30 turns, 798,485
                # tokens) never writing anything substantive at all, so the write-triggered prune
                # above never fired even once - tokens compounded, completely unchecked, for the
                # entire run. Escalating IS itself proof this stretch was unproductive (that's what
                # the threshold means), so it's safe to compress everything accumulated so far right
                # here too, not just wait for a write that may never come.
                pruned = (
                    _prune_stale_exploration_results(exploration_tool_messages_this_attempt)
                    if PRUNE_STALE_EXPLORATION_MID_ATTEMPT else 0
                )
                exploration_tool_messages_this_attempt = {}
                print(
                    f"DEBUG: agent attempt {_attempt} turn {_turn} - escalating: restricting to "
                    f"write_file/edit_file only after {exploration_calls_this_attempt} exploration "
                    f"calls with nothing written or edited (pruned {pruned} stale exploration "
                    "result(s) at the same time)",
                    file=sys.stderr,
                )
                messages.append(HumanMessage(
                    f"You've made {exploration_calls_this_attempt} exploration calls (list_files/"
                    "search_files/read_file) without writing or editing anything. Only write_file "
                    "and edit_file are available to you now - use what you've already learned to "
                    "make the change. If you are genuinely still blocked on a product-intent "
                    "decision, say so as your final answer instead of calling a tool."
                ))

        if ambiguity_question:
            return {
                **state,
                "builder_status": "ambiguous",
                "ambiguity_question": ambiguity_question,
                "total_tokens": total_tokens,
            }

        if ran_out_of_turns:
            reason = (
                f"exceeded its {MAX_TOKENS_PER_ATTEMPT}-token attempt budget ({attempt_tokens} tokens)"
                if attempt_token_budget_exceeded
                else f"ran out of turns ({_turn} across {-(-_turn // MAX_AGENT_TURNS)} round(s) of {MAX_AGENT_TURNS})"
            )
            print(
                f"DEBUG: agent attempt {_attempt} {reason} without reporting done or blocked - "
                "running independent verification anyway on whatever is on disk, rather than "
                "discarding it unchecked",
                file=sys.stderr,
            )

        # The agent believes it's done (or ran out of turns trying) - never trust either alone.
        # Independent verification runs EITHER way, not only on the "agent said done" path
        # (2026-09-28, fixed after a real transcript: a real creative-team rebuild wrote 6 genuine
        # files - config.py, storage.py, validation.py, platform_specs.py, context.py, __init__.py,
        # independently confirmed by reading them, not just a byte count - but ran out of turns
        # before ever calling run_command itself. The OLD code below this comment never even ran on
        # that path at all: it hit `continue` immediately on ran_out_of_turns, discarding whatever
        # real, possibly-working state was on disk in favor of a generic "be faster" message, and
        # the SAME static message every time meant the failure-signature dedup two lines below never
        # got a chance to fire either - a run could exhaust all 3 attempts this way even if every
        # attempt hit the identical mechanical failure, silently bypassing the "two identical
        # failures stop early" rule this same function documents as one of its own design points.
        # Mechanical truth wins over an incomplete self-report either way: if the on-disk state
        # already passes despite the model never getting to say so, that's still done.
        passed, output = _run_pytest(code_dir)
        own_failures: list = []
        new_other_failures: list = []
        tolerated_failures: list = []
        verification_passed = passed
        if not passed and unit_files:
            all_failures = _parse_pytest_failures(output)
            own_failures = [f for f in all_failures if _test_belongs_to_unit(f["test"], unit_files)]
            new_other_failures = [
                f for f in all_failures
                if not _test_belongs_to_unit(f["test"], unit_files) and f["test"] not in baseline_other_failures
            ]
            # A pre-existing failure in another unit's OWN tests (present in the baseline, not this
            # attempt's doing) is tolerated - fixing it isn't this unit's job, and forcing this unit
            # to keep re-attempting a problem that was never its own can never converge. A NEW
            # failure this attempt introduced anywhere is never tolerated either way (collateral
            # damage), and this unit's OWN tests must genuinely pass regardless of the baseline.
            tolerated_failures = [
                {"test": f["test"], "reason": f.get("reason") or "", "tolerated_for": unit_name}
                for f in all_failures
                if not _test_belongs_to_unit(f["test"], unit_files) and f["test"] in baseline_other_failures
            ]
            verification_passed = not own_failures and not new_other_failures
        n8n_problems = _validate_written_n8n_files(code_dir)
        requirement_problems = undeclared_third_party_imports(code_dir)
        claims, rejected_claims = (
            _parse_already_fixed_claims(final_text, list(state.get("review_problems") or []), code_dir)
            if finding_baseline is not None else ([], [])
        )
        all_claimed = bool(claims) and len(claims) == len(state.get("review_problems") or [])
        finding_change_problems = (
            _finding_change_problems(code_dir, finding_baseline)
            if finding_baseline is not None and not all_claimed else []
        )
        contract_problems = contract_check() if contract_check is not None else []
        smoke_passed, smoke_output = _run_smoke(code_dir, mode="dry")
        if smoke_passed and state.get("live_smoke"):
            smoke_passed, smoke_output = _run_smoke(code_dir, mode="live")
        # Real incident (2026-09-29): a unit ran out of turns in every attempt, writing NOTHING - its
        # own file byte-identical to what was already there before the attempt started - yet still
        # got reported "done" because pytest/smoke happened to pass on that untouched, pre-existing
        # code. A pass that was already true BEFORE this attempt even began is not evidence of
        # anything THIS attempt did - crediting it risks marking a unit "built" that was never
        # actually completed by any real work. Scoped specifically to ran_out_of_turns (never a
        # voluntary "done" report, which the agent only gives when it genuinely believes nothing more
        # is needed) - matches the real incident's own shape exactly, and does not revisit the
        # separate, earlier 2026-09-28 fix for genuine-but-unreported progress (that attempt DID
        # write 6 real files, just never got to say so).
        zero_progress_exhaustion = ran_out_of_turns and not attempt_made_substantive_progress
        print(
            f"DEBUG: agent attempt {_attempt} verification - pytest={passed} "
            f"(own_failures={len(own_failures)} new_other_failures={len(new_other_failures)}) "
            f"n8n_problems={len(n8n_problems)} smoke={smoke_passed} "
            f"zero_progress_exhaustion={zero_progress_exhaustion}",
            file=sys.stderr,
        )

        # Final fix pass (1-round budget): green tests cannot show the reviewer's findings were
        # addressed (real incident: a fix pass ran out of turns, tests stayed green on code that
        # still had both must_fix bugs, and it was accepted). Acceptance there requires the builder
        # itself to conclude - reporting done, not being cut off - since only then has it claimed to
        # have worked through the findings.
        fix_pass_unconfirmed = bool(state.get("final_fix_pass")) and ran_out_of_turns
        if (
            verification_passed and not n8n_problems and not requirement_problems and smoke_passed and not zero_progress_exhaustion
            and not fix_pass_unconfirmed and not finding_change_problems and not contract_problems
        ):
            _clear_unit_baseline(project_dir, unit_name)
            if not passed:
                print(
                    f"DEBUG: WARNING - unit '{unit_name}' accepted with the full suite still RED: "
                    f"{len(baseline_other_failures)} failure(s) in other units' tests were already failing "
                    "at this unit's first start and are tolerated (not attributed to this unit).",
                    file=sys.stderr,
                )
            return {
                **state,
                "builder_status": "done",
                "ambiguity_question": None,
                "total_tokens": total_tokens,
                # Mechanical file list (ground truth) + the agent's own final account (an
                # unverified claim) - see reviewer_node.py's builder-account section, which labels
                # and treats these very differently from each other.
                "builder_files_touched": [{"path": p, "action": a} for p, a in touched_files.items()],
                "builder_account": builder_account,
                "tolerated_failures": tolerated_failures,
                "claimed_fixed": claims,
                **stage_info,
            }

        feedback_parts = []
        if attempt_token_budget_exceeded:
            feedback_parts.append(
                f"Exceeded this attempt's own {MAX_TOKENS_PER_ATTEMPT}-token budget without reporting "
                "done or blocked - you were still calling tools when it ran out. Work in a more "
                "focused way: verify with run_command sooner, and prefer edit_file over rewriting "
                "whole files (large rewrites cost far more tokens per turn than targeted edits). The "
                "verification results below are from whatever was actually on disk when the budget ran out."
            )
        elif ran_out_of_turns:
            feedback_parts.append(
                f"Ran out of turns ({MAX_AGENT_TURNS} per round, up to {rounds_cap} rounds while progressing) without reporting done or blocked - you were "
                "still calling tools when the turn budget ran out. Work in a more focused way: verify "
                "with run_command sooner, and prefer edit_file over rewriting whole files. The "
                "verification results below are from whatever was actually on disk when time ran out."
            )
        if fix_pass_unconfirmed:
            feedback_parts.append(
                "THIS IS THE FINAL FIX PASS: passing tests do NOT show the reviewer's findings are fixed, "
                "so being cut off does not count as done. Address EACH reviewer finding listed above "
                "specifically with edit_file, then stop and report done - do not spend turns re-exploring."
            )
        if zero_progress_exhaustion:
            feedback_parts.append(
                "YOU RAN OUT OF TURNS WITHOUT WRITING OR EDITING ANYTHING AT ALL THIS ATTEMPT - "
                "whatever passed verification below was already there before you started, so it "
                "does not count as this unit being done. You must actually call write_file/edit_file "
                "and make a real, substantive change before this can be considered complete - "
                "exploring more without ever writing is not progress."
            )
        if not verification_passed:
            if unit_files:
                # Only this unit's own failures and any NEW regression are real feedback - a
                # pre-existing, unrelated failure elsewhere was already tolerated above and must not
                # be shown as something THIS attempt needs to fix (see the baseline comment above).
                if own_failures:
                    findings_block = "\n".join(
                        f"- {f['test']}" + (f": {f['reason']}" if f['reason'] else "")
                        for f in own_failures
                    )
                    feedback_parts.append(
                        "VERIFICATION FAILURES TO FIX (targeted - fix each of these specifically; "
                        "prefer edit_file for any file that already has real content, not a full "
                        f"rewrite that risks silently breaking what some OTHER file already expects "
                        f"from it):\n{findings_block}"
                    )
                if new_other_failures:
                    other_block = "\n".join(
                        f"- {f['test']}" + (f": {f['reason']}" if f['reason'] else "")
                        for f in new_other_failures
                    )
                    feedback_parts.append(_new_other_failures_feedback(other_block))
            elif not passed:
                pytest_findings = _parse_pytest_failures(output)
                if pytest_findings:
                    findings_block = "\n".join(
                        f"- {f['test']}" + (f": {f['reason']}" if f['reason'] else "")
                        for f in pytest_findings
                    )
                    feedback_parts.append(
                        "VERIFICATION FAILURES TO FIX (targeted - fix each of these specifically; "
                        "prefer edit_file for any file that already has real content, not a full "
                        "rewrite that risks silently breaking what some OTHER file already expects "
                        f"from it):\n{findings_block}\n\nFull pytest output:\n{output}"
                    )
                else:
                    feedback_parts.append(output)
        if n8n_problems:
            feedback_parts.append(
                "N8N WORKFLOW STRUCTURAL PROBLEMS (fix these too - this file is part of the same "
                "build even though the platform is 'python'):\n" + "\n".join(f"- {p}" for p in n8n_problems)
            )
        if requirement_problems:
            feedback_parts.append(
                "UNDECLARED DEPENDENCIES (fix these - add each to code/requirements.txt):\n"
                + "\n".join(f"- {p}" for p in requirement_problems)
            )
        if rejected_claims:
            feedback_parts.append("ALREADY_FIXED CLAIMS REJECTED:\n" + "\n".join(f"- {r}" for r in rejected_claims))
        if finding_change_problems:
            feedback_parts.append(
                "REVIEWER FINDING NOT ADDRESSED (passing tests alone do not count):\n"
                + "\n".join(f"- {p}" for p in finding_change_problems)
                + "\nIf the finding needs an approved-interface change, reply 'AMBIGUOUS: ...' instead."
            )
        if contract_problems:
            feedback_parts.append(
                "UNIT CONTRACT NOT MET (passing tests do not count - the Architect's declared files and "
                "exports must really exist; create/write them with write_file, then run the tests):\n"
                + "\n".join(f"- {p}" for p in contract_problems)
            )
        if not smoke_passed:
            feedback_parts.append(f"SMOKE-TEST FAILURE (fix this):\n{smoke_output}")
        failure_output = "\n\n".join(feedback_parts)
        # Only the first attempt carries reviewer problems (later ones react to the verification
        # failure) - except the final fix pass, whose whole point IS those findings.
        if not state.get("final_fix_pass") and finding_baseline is None:
            problems = []

        failure_signature = hashlib.sha256(failure_output.encode("utf-8")).hexdigest()
        if failure_signature == prior_failure_signature:
            return {
                **state,
                "builder_status": "ambiguous",
                "ambiguity_question": (
                    "Two consecutive attempts produced the identical verification failure - stopping "
                    f"rather than repeating it a third time. Last failure:\n{failure_output[-1500:]}"
                ),
                "total_tokens": total_tokens,
            }
        prior_failure_signature = failure_signature

    return {
        **state,
        "builder_status": "ambiguous",
        "ambiguity_question": (
            f"Builder could not get verification passing after {MAX_INTERNAL_FIX_ATTEMPTS} attempts. "
            f"Last failure:\n{failure_output[-1500:]}"
        ),
        "total_tokens": total_tokens,
    }


def _declared_method_names(path: Path) -> set:
    """'Class.method' and bare 'method' for every method defined directly in a top-level class of a
    .py file - lets a finding that names a method (e.g. 'bootstrap_owner') reach the unit whose file
    holds it. {} if the file can't be parsed."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return set()
    names: set = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(f"{node.name}.{item.name}")
                    names.add(item.name)
    return names


_TEST_PATH_RE = re.compile(r"(^|/)(tests?/|test_[^/]*$|[^/]*_test\.py$|conftest\.py$)")


def _py_snapshot(code_dir: Path) -> dict:
    """{relative posix path: sha256} of every project .py file (venv/caches excluded)."""
    snap: dict = {}
    for path in Path(code_dir).rglob("*.py"):
        rel = path.relative_to(code_dir)
        if set(rel.parts) & {".venv", "venv", "__pycache__", "site-packages", ".git"}:
            continue
        try:
            snap[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
    return snap


_ALREADY_FIXED_RE = re.compile(r"^\s*ALREADY_FIXED:\s*(\S+)\s*\|\s*(.+?)\s*$", re.MULTILINE)


def _finding_ref(text: str) -> str:
    return "f-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]


def _code_text_normalized(code_dir: Path) -> str:
    parts = []
    for path in sorted(Path(code_dir).rglob("*.py")):
        if set(path.relative_to(code_dir).parts) & {".venv", "venv", "__pycache__", "site-packages", ".git"}:
            continue
        try:
            parts.append(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError):
            continue
    return " ".join(" ".join(parts).split())


def _parse_already_fixed_claims(text: str, findings: list, code_dir: Path) -> tuple:
    """Returns (valid, rejected). valid: [{'id', 'finding', 'quote'}] where the id (or unique id
    prefix) names one of `findings` and the verbatim quote really exists in the current code.
    rejected: human-readable reasons, fed back to the builder."""
    refs = {_finding_ref(f): f for f in findings}
    valid, rejected, seen = [], [], set()
    norm_code = None
    for ref, quote in _ALREADY_FIXED_RE.findall(text or ""):
        ref = ref.strip("[]")
        matches = [r for r in refs if r == ref or r.startswith(ref)]
        if len(matches) != 1:
            rejected.append(f"ALREADY_FIXED {ref!r} does not identify exactly one finding id listed above")
            continue
        quote = " ".join(quote.strip().strip("`").split())
        if norm_code is None:
            norm_code = _code_text_normalized(code_dir)
        if not quote or quote not in norm_code:
            rejected.append(
                f"ALREADY_FIXED {matches[0]}: the quote does not appear verbatim in the current code - "
                "quote the exact line(s) that fix it, or fix the finding"
            )
            continue
        if matches[0] not in seen:
            seen.add(matches[0])
            valid.append({"id": matches[0], "finding": refs[matches[0]], "quote": quote[:300]})
    return valid, rejected


def _finding_change_problems(code_dir: Path, baseline: dict) -> list:
    """For a unit rebuilt because of reviewer findings: passing tests that were already green prove
    nothing, so require at least one changed/new non-test source file AND one changed/new test file
    relative to `baseline` (a _py_snapshot taken before the unit's build started)."""
    now = _py_snapshot(code_dir)
    changed = [p for p, h in now.items() if baseline.get(p) != h]
    src = [p for p in changed if not _TEST_PATH_RE.search(p)]
    tests = [p for p in changed if _TEST_PATH_RE.search(p)]
    out = []
    if not src:
        out.append("no non-test source file was changed or added, so the reviewer finding(s) cannot have been fixed")
    if not tests:
        out.append(
            "no test was added or changed - add a test that reproduces the finding (fails on the old "
            "code, passes on your fix)"
        )
    return out


def _imported_module_names(path: Path) -> set:
    """Every module/name a python file imports, as dotted strings plus bare imported names."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return set()
    out: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module)
            out.update(a.name for a in node.names)
    return out


def _module_matches(imported: set, file_path: str) -> bool:
    stem = Path(file_path).stem
    return any(m == stem or m.endswith("." + stem) for m in imported)


def _seam_test_problems(code_dir: Path, unit: dict, units_by_name: dict) -> list:
    """Heuristic seam check: if a unit depends on units that export public functions/classes, at
    least one of the unit's own test files should import from one of those dependency modules (so
    the consumer is exercised against the REAL producer, not only hand-built stand-in data).
    Returns [] when the unit has no dependencies with producers, or no identifiable test files."""
    unit_files = [f for f in (unit.get("files") or []) if f and f.endswith(".py")]
    if not unit_files:
        return []
    dep_files: dict = {}
    for dep in unit.get("depends_on") or []:
        for f in ((units_by_name.get(dep) or {}).get("files") or []):
            if not f or not f.endswith(".py"):
                continue
            direct = Path(code_dir) / f
            candidates = [direct] if direct.exists() else list(Path(code_dir).rglob(Path(f).name))
            for path in candidates[:1]:
                names = _declared_names_in_file(path)
                if any(kind in ("function", "class") and not n.startswith("_") for n, kind in names.items()):
                    dep_files.setdefault(dep, []).append(f)
    if not dep_files:
        return []
    own_tests = []
    for path in Path(code_dir).rglob("*.py"):
        rel = path.relative_to(code_dir).as_posix()
        if set(path.relative_to(code_dir).parts) & {".venv", "venv", "__pycache__", "site-packages", ".git"}:
            continue
        if not _TEST_PATH_RE.search(rel) or path.name == "conftest.py":
            continue
        imported = _imported_module_names(path)
        if any(_module_matches(imported, f) for f in unit_files):
            own_tests.append(imported)
    if not own_tests:
        return []
    if any(_module_matches(imp, f) for imp in own_tests for fs in dep_files.values() for f in fs):
        return []
    deps = ", ".join(sorted(dep_files))
    return [
        f"{unit.get('name', '')}: none of this unit's tests import anything from its dependency unit(s) "
        f"({deps}), so it appears to be tested only against hand-built producer data. Add at least one "
        "test that gets the input by calling the dependency's REAL producing function/class, faking only "
        "the network/SDK boundary with realistic raw payloads."
    ]


def _declared_names_in_file(path: Path) -> dict:
    """{export_name: 'class'|'function'|'constant'} for every top-level definition/assignment in a
    real .py file - {} if it can't be read/parsed (a syntax-broken file already fails pytest/smoke
    independently elsewhere; this doesn't need to duplicate that with a confusing message of its
    own here)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return {}
    names: dict = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            names[node.name] = "class"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names[node.name] = "function"
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names[target.id] = "constant"
    return names


_UNCOMPILED_STATEGRAPH_RETURN_RE = re.compile(r"->\s*StateGraph\b")


def _returns_a_compiled_graph(path: Path, function_name: str) -> bool:
    """True if a top-level function named `function_name` anywhere in `path` has any return
    statement whose value is directly a `<something>.compile(...)` call. AST-based, same "by
    pattern, not full type inference" spirit as _declared_names_in_file above - built for one
    concrete, real, evidenced pattern (see _check_unit_interface_compliance's own docstring), not
    general type-checking. A return that compiles indirectly (e.g. `g = graph.compile(); return g`)
    is NOT caught - a false negative, not a false positive, and an acceptable gap for the same "far
    more machinery for much less certain payoff" reason the sibling by-name check already accepts."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Return)
                    and isinstance(inner.value, ast.Call)
                    and isinstance(inner.value.func, ast.Attribute)
                    and inner.value.func.attr == "compile"
                ):
                    return True
    return False


def _check_unit_interface_compliance(code_dir: Path, unit: dict) -> list:
    """Mechanical (ast-based, no LLM call): does the unit's ACTUAL on-disk code contain what the
    Architect's contract declared it must export? 2026-09-28, going further than Douglas's own
    requirement (b) literally asked - closes the loop fully: not just HANDING the contract forward
    to later units, but VERIFYING the builder actually conformed to it, never trusting a pytest/
    smoke pass alone as proof it built the right shape. Checked by NAME only (not full signature
    matching - real signature verification would need far more machinery for much less certain
    payoff), with ONE narrow, evidenced exception below; `files` is a hint, never a literal
    requirement (_format_unit_task's own prompt to the builder says so explicitly) - so a declared
    path that doesn't exist, or doesn't contain an expected name, falls back to a whole-code_dir tree
    scan for that name specifically, the same fallback already used when `files` is absent/empty
    entirely. Real incident this closes (2026-09-28): the Architect's hint named bare filenames
    ("storage_interface.py") while the builder correctly followed normal package layout and nested it
    under a package directory (creative_studio/storage_interface.py) - the declared export was really
    there, but the old code only ever looked at the literal hinted path, found nothing, and reported
    every export missing - a wrong hint must never make this check MORE restrictive than specifying no
    hint at all.

    The one narrow exception (2026-09-29, real evidenced incident, creative-team's own
    content_writer.py AND, still latent/undetected, image_team.py): a declared signature saying
    "-> StateGraph" (uncompiled) but the function's actual return is `<x>.compile(...)` - LangGraph's
    StateGraph.compile() returns a CompiledStateGraph, a DIFFERENT type than declared. This burned a
    full round's 3 retries (3,168,995 tokens) never converging, because raw pytest failure text alone
    never pointed the builder at "your sibling unit solved this exact contract differently" - two of
    four sibling units in the same round got this right. Checked textually via AST (see
    _returns_a_compiled_graph), not full type inference - one concrete, now-evidenced pattern, not a
    step toward general signature verification."""
    exports = (unit.get("interface") or {}).get("exports") or []
    if not exports:
        return []
    declared_files = unit.get("files") or []
    hint_paths = [code_dir / f for f in declared_files]
    hint_names: dict = {}
    for path in hint_paths:
        if path.exists() and path.is_file():
            hint_names.update(_declared_names_in_file(path))

    problems = []
    tree_names: dict | None = None  # computed lazily - only when a hint miss actually needs it
    tree_paths: list | None = None
    for export in exports:
        name = export.get("name") if isinstance(export, dict) else None
        if not name:
            continue
        if name in hint_names:
            search_paths = hint_paths
        else:
            if tree_names is None:
                tree_paths = list(code_dir.rglob("*.py"))
                tree_names = {}
                for path in tree_paths:
                    tree_names.update(_declared_names_in_file(path))
            if name not in tree_names:
                problems.append(f"declared export '{name}' ({export.get('kind', '?')}) does not appear anywhere in this unit's code")
                continue
            search_paths = tree_paths

        signature = str(export.get("signature") or "")
        if export.get("kind") == "function" and _UNCOMPILED_STATEGRAPH_RETURN_RE.search(signature):
            if any(p.exists() and p.is_file() and _returns_a_compiled_graph(p, name) for p in search_paths):
                problems.append(
                    f"declared export '{name}' has signature '{signature.strip()}' (uncompiled) but "
                    f"its actual return is a .compile() call - that returns a CompiledStateGraph, a "
                    f"different type than the declared, uncompiled StateGraph"
                )
    return problems


_MAX_EXISTING_FILE_CHARS = 20_000


def _existing_files_block(unit: dict, code_dir: Path | None, extra_files: list | None = None) -> str:
    """Current content (line-numbered) of this unit's already-existing declared files, so an
    extend-in-place unit doesn't spend 6-9 turns rediscovering code the builder must edit anyway.
    Skips missing/oversized files; empty when nothing applies."""
    if code_dir is None:
        return ""
    blocks = []
    seen: set = set()
    for f in list(unit.get("files") or []) + list(extra_files or []):
        if not f or f in seen:
            continue
        seen.add(f)
        direct = code_dir / f
        target = direct if direct.is_file() else next(iter(sorted(code_dir.rglob(Path(f).name))), None)
        if target is None or not target.is_file():
            continue
        try:
            text = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not text.strip() or len(text) > _MAX_EXISTING_FILE_CHARS:
            continue
        rel = target.relative_to(code_dir).as_posix()
        numbered = "\n".join(f"{i + 1:>5}\t{ln}" for i, ln in enumerate(text.splitlines()))
        blocks.append(f"=== {rel} ===\n{numbered}")
    if not blocks:
        return ""
    return (
        "--- CURRENT CONTENT OF YOUR UNIT'S EXISTING FILE(S) (already read for you - edit these with "
        "edit_file using exact text from below; no need to read_file them again) ---\n"
        + "\n\n".join(blocks)
    )


_PY_PATH_IN_TEXT_RE = re.compile(r"[\w./\\-]+\.py\b")
_SPEC_SLICE_MIN_CHARS = 6_000


def _problem_file_mentions(review_problems: list, code_dir: Path | None) -> list:
    """Existing .py files a reviewer's findings name in their text, so the first prompt already
    carries them and the builder doesn't spend a turn reading each (turn-cost item 4, 2026-09-30)."""
    if code_dir is None:
        return []
    found: list = []
    for p in review_problems or []:
        for m in _PY_PATH_IN_TEXT_RE.findall(str(p)):
            m = m.replace("\\", "/").removeprefix("./")
            if m and m not in found and (code_dir / m).is_file():
                found.append(m)
    return found


def _slice_spec_for_unit(spec_text: str, unit: dict, prior_interfaces: dict) -> str:
    """Sends only the spec sections relevant to this unit (turn-cost item 3, 2026-09-30, Douglas):
    every turn resends the whole prompt, and a large spec dominated it. Keeps the preamble (text
    before the first heading), and every '#'-heading section that mentions this unit's name, file
    stems, exported names or a dependency's name. Falls back to the FULL spec when the spec is small,
    has no headings, nothing matches, or the slice would save under 25%. Omitted sections are listed
    by heading so the builder knows what exists."""
    if len(spec_text) < _SPEC_SLICE_MIN_CHARS:
        return spec_text
    parts = re.split(r"(?m)^(?=#{1,6}\s)", spec_text)
    if len(parts) < 3:
        return spec_text
    preamble = "" if re.match(r"#{1,6}\s", parts[0]) else parts[0]
    sections = parts[1:] if preamble else parts
    interface = unit.get("interface") or {}
    needles = {str(unit.get("name") or "")}
    needles.update(Path(f).stem for f in (unit.get("files") or []) if f)
    needles.update(str(e.get("name") if isinstance(e, dict) else e) for e in (interface.get("exports") or []))
    needles.update(str(d) for d in (unit.get("depends_on") or []))
    needles = {n.lower() for n in needles if len(n) >= 3}
    if not needles:
        return spec_text
    kept, omitted = [], []
    for sec in sections:
        low = sec.lower()
        (kept if any(n in low for n in needles) else omitted).append(sec)
    if not kept:
        return spec_text
    body = preamble + "".join(kept)
    if len(body) > 0.75 * len(spec_text):
        return spec_text
    titles = [sec.splitlines()[0].strip() for sec in omitted if sec.strip()]
    note = (
        "\n\n[Spec trimmed to the sections relevant to your unit to save tokens. Other sections "
        "exist but are not shown: " + "; ".join(titles[:40]) + "]"
    )
    return body + note


_DIGEST_MAX_CHARS = 10000
_DIGEST_CONFIG_SUFFIXES = (".yaml", ".yml", ".json", ".toml", ".ini", ".cfg")
_DIGEST_CONFIG_MAX_CHARS = 2500
_DIGEST_ITEM_MAX_CHARS = 1500
_SQL_DDL_RE = re.compile(r"\b(?:CREATE|ALTER)\s+(?:TABLE|INDEX)\b", re.IGNORECASE)


def _definitions_digest(code_dir: Path | None, files: list) -> str:
    """Concrete data definitions from an already-built dependency's own source files: module-level
    constants/tables, class fields (dataclass/enum/pydantic/ORM - fields only, never methods), and
    any CREATE/ALTER TABLE SQL strings. Exists because signatures alone (the Architect's interface)
    don't tell a dependent unit the real column/field names, so it guessed them (trading-team unit 6
    guessed an outbox column that unit 4 had defined as dedupe_key, 2026-09-30). Best-effort: files
    that don't exist or don't parse yield nothing."""
    if not code_dir or not files:
        return ""
    parts: list = []
    for rel in files:
        if not rel:
            continue
        path = code_dir / rel
        if not path.exists():
            matches = [m for m in code_dir.rglob(Path(rel).name) if m.is_file()]
            if not matches:
                continue
            path = matches[0]
        if str(path).endswith(_DIGEST_CONFIG_SUFFIXES):
            try:
                text = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                continue
            if text:
                parts.append(f"# {path.relative_to(code_dir).as_posix()}\n" + text[:_DIGEST_CONFIG_MAX_CHARS])
            continue
        if not str(path).endswith(".py"):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue

        def seg(node) -> str:
            return (ast.get_source_segment(source, node) or "")[:_DIGEST_ITEM_MAX_CHARS]

        items: list = []
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                items.append(seg(node))
            elif isinstance(node, ast.ClassDef):
                header = source.splitlines()[node.lineno - 1].rstrip()
                fields = [
                    "    " + seg(n).replace("\n", "\n    ")
                    for n in node.body if isinstance(n, (ast.Assign, ast.AnnAssign))
                ]
                items.append(header + ("\n" + "\n".join(fields) if fields else "\n    ..."))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and _SQL_DDL_RE.search(node.value):
                items.append("SQL: " + node.value.strip()[:_DIGEST_ITEM_MAX_CHARS])
        if items:
            parts.append(f"# {path.relative_to(code_dir).as_posix()}\n" + "\n".join(items))
    digest = "\n\n".join(parts)
    if len(digest) > _DIGEST_MAX_CHARS:
        digest = digest[:_DIGEST_MAX_CHARS] + "\n... (digest truncated - read the file for the rest)"
    return digest


def _format_unit_task(
    base_spec_text: str, unit: dict, prior_interfaces: dict, journal_text: str, code_dir: Path | None = None,
    review_problems: list | None = None, units_by_name: dict | None = None,
) -> str:
    """The per-unit task handed to _build_python_agentic - foregrounds exactly what THIS unit must
    build, and hands forward every dependency's ALREADY-BUILT interface directly. This is the actual
    mechanism that prevents the InMemoryStorage/InMemoryStorageBackend class of bug: a later unit is
    HANDED an earlier one's real shape here, instead of having to read its file and guess (or, worse,
    invent a plausible-but-different one across a fresh conversation with no memory of the first).

    Section order: unit block, then journal, then dependency interfaces, then the full spec LAST
    (2026-09-29, real incident - reordered from base_spec_text FIRST/journal LAST). A precisely
    diagnosed, verified one-line fix sat correctly tagged in the journal and still hadn't landed
    across 3 dedicated attempts on image_team_subgraph; a peer flagged that journal_text sat at the
    very end of a prompt after the WHOLE spec (creative-team's own runs tens of thousands of
    characters) with no distinguishing emphasis, as the most concrete lead for why. Not certain this
    is the full explanation on its own (the same attempt also referenced code that doesn't exist in
    the real file, which general prompt position doesn't obviously explain either - see
    AGENT_SYSTEM_PROMPT's own "read it first" instruction, already prominent, already not reliably
    followed here), but this reorder is cheap, low-risk, and independently sensible regardless: tell
    the agent exactly what it's building and what's known to block it BEFORE the large reference
    document, not after."""
    depends_on = unit.get("depends_on") or []
    files = unit.get("files") or []
    sections = [
        (
            "--- THIS BUILD IS DECOMPOSED INTO UNITS (Architect-approved, binding) ---\n"
            f"YOUR UNIT THIS ROUND: '{unit.get('name', '')}'\n"
            f"Purpose: {unit.get('purpose', '')}\n"
            "Suggested file(s) (your own reasonable judgment on internals - a hint, not a literal "
            f"requirement): {', '.join(files) if files else '(none specified)'}\n\n"
            "YOU MUST EXPORT EXACTLY THIS INTERFACE - the Architect's own binding contract, "
            "mechanically checked once you're done. Build precisely this, using these exact names:\n"
            f"{json.dumps(unit.get('interface') or {}, indent=2)}\n\n"
            f"How this unit's real behavior will be verified: {unit.get('verification_hint', '')}"
        ),
    ]
    existing_block = _existing_files_block(unit, code_dir, _problem_file_mentions(review_problems, code_dir))
    if existing_block:
        sections.append(existing_block)
    if journal_text:
        sections.append(journal_text)
    dep_blocks = [
        f"UNIT '{dep}' (already built - use EXACTLY this interface; never invent your own name/shape "
        f"for it, and never read its source file to guess when this contract already tells you):\n"
        f"{json.dumps(prior_interfaces[dep], indent=2)}"
        for dep in depends_on if dep in prior_interfaces
    ]
    if dep_blocks:
        sections.append("--- INTERFACES OF UNITS THIS ONE DEPENDS ON (binding, not to be rediscovered) ---\n" + "\n\n".join(dep_blocks))
    digest_blocks = []
    for dep in depends_on:
        digest = _definitions_digest(code_dir, ((units_by_name or {}).get(dep) or {}).get("files") or [])
        if digest:
            digest_blocks.append(f"UNIT '{dep}':\n{digest}")
    skeleton = None
    for u in (units_by_name or {}).values():
        if u.get("name") == unit.get("name"):
            break  # only units AFTER the skeleton have an end-to-end test to extend
        if u.get("walking_skeleton") is True:
            skeleton = u
    if unit.get("walking_skeleton") is True:
        sections.append(
            "--- WALKING SKELETON (binding rule) ---\n"
            "This unit is the project's FIRST end-to-end path. Build the thinnest real slice that runs the "
            "real entry point through every stage of the system to its real final output, with only the "
            "external network/SDK boundary faked. Stages may be thin or hard-coded inside, but each must be "
            "a real function/class with its final name and signature so later units fill it in rather than "
            "replace it. Write ONE end-to-end test (name it tests/test_integration_e2e.py) that runs this "
            "whole path and asserts on the real final output. Later units will extend that test."
        )
    elif skeleton is not None:
        sections.append(
            "--- END-TO-END TEST (binding rule) ---\n"
            "The project has an end-to-end test (tests/test_integration_e2e.py, from the walking-skeleton unit "
            f"'{skeleton.get('name')}') that runs the real assembled system. Your unit's own unit tests are not "
            "enough: run that end-to-end test, keep it green, and extend it (or add a case beside it) so this "
            "unit's new behavior is exercised through the real assembled path. If it cannot pass yet because a "
            "stage this unit needs is still a placeholder, replace that placeholder with your real implementation."
        )
    if depends_on:
        sections.append(
            "--- TESTING AGAINST DEPENDENCIES (binding rule) ---\n"
            "When this unit consumes data produced by a unit it depends on, at least one test MUST obtain "
            "that data by calling the dependency's REAL producing function or class, with only the "
            "network/SDK boundary faked (a fake HTTP transport or fake SDK client returning realistic RAW "
            "payloads of the kind the real service sends). Hand-built DataFrames, dicts or config objects "
            "may supplement that test but never replace it: a stand-in shaped the way THIS unit expects "
            "proves nothing about what the producer really emits (index/column names, drop=True vs "
            "drop=False, value labels, fields the producer never fills). Read the producer's code (see the "
            "definitions below) to see what it really returns, and make this unit consume that."
        )
    if digest_blocks:
        sections.append(
            "--- CONCRETE DEFINITIONS FROM UNITS THIS ONE DEPENDS ON (authoritative names: table/column "
            "names, constants, class fields) ---\n"
            "Use these EXACT names. Never guess or probe for alternative names at runtime. Anything not "
            "shown here (a function body, a field beyond the cap) must be READ from that unit's file "
            "with read_file before you rely on it - never assumed.\n\n" + "\n\n".join(digest_blocks)
        )
    sections.append(base_spec_text)
    return "\n\n".join(sections)


def _problem_node(entry: str) -> str:
    """Extracts the leading 'node' portion of one review_problems entry - every entry
    reviewer_node.py produces (verify_problems' own kept.append, and the forced-mechanical-finding
    append) is shaped 'node: description', so this is exact, not a guess."""
    node, _, _ = entry.partition(": ")
    return node.strip()


def _route_unmatched_problems(review_problems: list, units: list, project_dir, code_dir) -> dict:
    """{id(problem): unit_name} for findings no unit claims by name/file/symbol. A file-named finding
    (e.g. 'pkg/mod.py: delete per ruling') is routed to the unit whose builder journal entry lists
    that path as touched; otherwise to the LAST unit in range, with a loud warning - never silently
    dropped, which used to burn review-only rounds with no builder."""
    routed: dict = {}
    if not units:
        return routed
    entries = [e for e in load_all_entries(project_dir) if e.get("role") == "builder"]
    for p in review_problems:
        node = _problem_node(p)
        if any(_node_belongs_to_unit(node, u.get("name") or "", u.get("files") or [], code_dir) for u in units):
            continue
        norm = node.replace("\\", "/")
        owner = None
        if "/" in norm or norm.endswith(".py"):
            for e in reversed(entries):
                if e.get("unit") in {u.get("name") for u in units} and norm in (e.get("summary") or "").replace("\\", "/"):
                    owner = e["unit"]
                    break
        if owner is None:
            on_disk = ("/" in norm or norm.endswith(".py")) and (Path(code_dir) / norm).is_file()
            if not on_disk:
                print(f"WARNING: finding matches no unit and names no existing file; NOT routed to any "
                      f"builder: {p[:160]}", file=sys.stderr)
                continue
            owner = units[-1].get("name")
            print(f"WARNING: finding names an existing file no unit owns; routing to last unit "
                  f"'{owner}': {p[:160]}", file=sys.stderr)
        routed[id(p)] = owner
    return routed


def _node_belongs_to_unit(node: str, name: str, files: list, code_dir: Path | None = None) -> bool:
    """True if `node` (one finding's own claimed subject) genuinely names THIS unit - its exact unit
    name, one of its declared files exactly or by basename (a file hint may legitimately be nested
    under a package directory the Architect's hint didn't anticipate - same lenient-hint philosophy
    _check_unit_interface_compliance already applies), OR a function/class/constant NAME actually
    defined inside one of those declared files (2026-09-29 fix, real incident: a reviewer's final-
    round findings named call_llm/content_writer_node/_validate_request/image_composition_node -
    function names, never file paths or a unit's own name - so none of the checks above ever
    matched, and the units that actually contain them were silently skipped as "not named in
    reviewer feedback"). code_dir is optional and only needed for this third check - omitting it (the
    default) preserves the exact prior behavior. Deliberately narrow: a node that merely CONTAINS
    this unit's name as a substring does NOT count - see _build_python_units' own docstring for the
    real incident this distinction exists to prevent."""
    if node == name:
        return True
    normalized_node = node.replace("\\", "/")
    for f in files:
        if not f:
            continue
        normalized_f = f.replace("\\", "/")
        if normalized_node == normalized_f or normalized_node.rsplit("/", 1)[-1] == normalized_f.rsplit("/", 1)[-1]:
            return True
    if code_dir is not None:
        # A dot-qualified node ("Config.get_api_key") names a METHOD on a class, not a top-level
        # name itself - _declared_names_in_file only records top-level defs, so it never finds
        # "Config.get_api_key" literally. Its own PREFIX (before the first dot) is the real class
        # name, which IS top-level and IS what gets checked here (2026-09-29 fix, real incident: a
        # reviewer's finding named "Config.get_api_key" specifically and never matched anything).
        node_prefix = node.split(".", 1)[0]
        for f in files:
            if not f:
                continue
            direct = code_dir / f
            candidates = [direct] if direct.exists() else list(code_dir.rglob(Path(f).name))
            for path in candidates:
                if not path.exists() or not path.is_file():
                    continue
                declared = _declared_names_in_file(path)
                if node in declared or node_prefix in declared:
                    return True
                methods = _declared_method_names(path)
                if node in methods or node.rsplit(".", 1)[-1] in methods:
                    return True
    return False


def _missing_declared_files(code_dir: Path, unit: dict) -> list:
    """Problems when NONE of a unit's declared files exists (by path or basename anywhere under
    code_dir) with content - i.e. the builder produced nothing for it. `files` stays a hint: some
    missing while others exist is not flagged."""
    declared = [f for f in (unit.get("files") or []) if f]
    if not declared:
        return []
    for f in declared:
        direct = code_dir / f
        candidates = [direct] if direct.exists() else list(code_dir.rglob(Path(f).name))
        for path in candidates:
            try:
                if path.is_file() and path.stat().st_size > 0:
                    return []
            except OSError:
                continue
    return [
        f"none of this unit's declared file(s) exist with content: {', '.join(declared)} - you did not "
        "create them, so nothing was built"
    ]


def _check_all_units_interface_compliance(code_dir: Path, build_units: list, built_names: set, current_name: str) -> tuple[list, set]:
    """Runs _check_unit_interface_compliance for the CURRENT unit AND every OTHER already-built unit
    (2026-09-29 fix, real incident: the agentic builder is not actually confined to the nominal
    unit's own files - `files` is explicitly documented as a hint, not binding - and has repeatedly
    reached into a DIFFERENT unit's files while that different unit was not the one nominally in
    scope, the third time outright deleting a required export. Checking only the nominal unit left
    this completely undetected: the round reported success, and nothing would have caught it until
    - if ever - the damaged unit got its own dedicated turn later. ~1M tokens were spent that round
    with a net-negative result: a working test suite regressed to a hard collection error.

    Returns (problems, affected_units) - problems is every violation found, each one about an
    OTHER unit prefixed to say so; affected_units is every unit name (possibly including
    current_name, possibly others, possibly both) whose contract just failed - callers must treat
    ALL of them as no longer reliably built, not just current_name, or a later resume would skip a
    unit now known to be broken the exact same way the earlier stale-success bug did."""
    units_by_name = {u.get("name") or f"unit_{j}": u for j, u in enumerate(build_units)}
    problems: list = []
    affected: set = set()

    current_unit = units_by_name.get(current_name)
    if current_unit is not None:
        current_problems = _check_unit_interface_compliance(code_dir, current_unit)
        if current_problems:
            affected.add(current_name)
            problems.extend(current_problems)

    for other_name in sorted(built_names):
        if other_name == current_name:
            continue
        other_unit = units_by_name.get(other_name)
        if other_unit is None:
            continue
        other_problems = _check_unit_interface_compliance(code_dir, other_unit)
        if other_problems:
            affected.add(other_name)
            problems.extend(
                f"[collateral damage to already-built unit '{other_name}', not the one nominally "
                f"being built this round] {p}" for p in other_problems
            )

    return problems, affected


def _build_python_units(
    state: BuildState, llm, config, base_spec_text: str, code_dir: Path, build_units: list, until_unit_index,
) -> BuildState:
    """Builds an Architect-decomposed sequence of units, one at a time, IN ORDER - the python analog
    of n8n's own per-node plan-and-verify discipline (2026-09-28, Douglas's firm direction, replacing
    PM-epic/checkpoint stage gating for the python path entirely - n8n's own --until-slice is
    untouched, separate, not part of this).

    Each unit is built via _build_python_agentic - every mechanism that function already has (the
    per-attempt/per-invocation token caps, context pruning, the duplicate-call guard, structured
    pytest findings, the write_file/edit_file substantive-content checks) applies automatically to
    each unit's own build, since this function only ever changes WHAT gets asked for (a narrowed,
    unit-scoped task built by _format_unit_task), never HOW it gets built - with ONE exception: this
    function also passes the unit's own declared `files` as `unit_files`, which lets
    _build_python_agentic tolerate a PRE-EXISTING failure in some OTHER unit's own tests (2026-09-29
    fix - see _build_python_agentic's own comment on baseline_other_failures for the real incident:
    without this, a unit with nothing wrong of its own could never complete while a different,
    unrelated unit's own already-known bug kept the whole suite red, so the sequence could never
    advance far enough to give the actually-broken unit its own turn again).

    An already-built unit not named in this round's reviewer feedback (review_problems) is SKIPPED,
    not rebuilt - the same "reuse the existing artifact, rebuild only what was actually named"
    principle n8n's own incremental repair path already uses (_build_n8n_chunked), adapted from node
    names to unit names/files. A skipped unit's own declared interface still gets handed forward to
    any later unit that depends on it - the CONTRACT is what's binding, regardless of whether this
    specific invocation touched that unit's code. "Named" is checked per-finding via
    _node_belongs_to_unit (2026-09-29 fix, real incident) - NOT a substring scan of every problem's
    text concatenated together, which is what it was before: a test-coverage-gap message about
    image_team.py happened to LIST every built unit's name for context ("despite epic(s) reported
    built as of this round: ...storage..."), and the old crude `name in review_problems_text` check
    concluded "storage" had been flagged, when nothing was wrong with it. Consequence: storage got
    rebuilt for no reason, burning the round's ENTIRE token budget and its own attempt/dedup cap on a
    unit that was never broken, while the actually-broken unit (content_writer_subgraph, an unrelated
    real bug) never got its own dedicated retry this round at all - not merely wasteful, but actively
    starved the genuinely broken unit of the attempt it needed. A whole-artifact ("WORKFLOW") finding
    now deliberately matches no single unit here (it was never a reliable way to target one anyway -
    just an accidental side effect of the same substring bug being fixed).

    A unit that fails its own internal verification, or passes it but does not actually export what
    its contract declared (_check_unit_interface_compliance), stops the WHOLE sequence immediately -
    the same "don't push forward past a genuine block" principle used everywhere else in this
    pipeline, rather than building later units on top of a known-bad one. On EITHER failure path, the
    unit currently being (re)built is explicitly excluded from the returned python_units_built
    (`newly_built - {name}`), never just `newly_built` as-is - real incident (2026-09-29): newly_built
    is seeded from already_built (a PRIOR round's success) before this loop starts, so a unit whose
    reviewer-triggered REBUILD just failed would otherwise still be reported as built (its stale prior
    success was never cleared), and a resume would then skip a unit known to be broken instead of
    retrying it.
    """
    project_dir = state.get("project_dir")
    total_tokens = state.get("total_tokens", 0)
    already_built = set(state.get("python_units_built") or [])
    review_problems = state.get("review_problems") or []
    last_index = until_unit_index if until_unit_index is not None else len(build_units) - 1

    # ALSO mechanically re-derive "already built" from the real, on-disk code (2026-09-29, real
    # incident: a --until-unit N invocation re-scopes python_until_unit_index for EVERY different N,
    # and checkpoint.py's own _stage_scope_key deliberately scopes the checkpoint FILE per unit-index
    # (a real, earlier fix for a DIFFERENT incident - collapsing that scoping would reintroduce IT,
    # see checkpoint_path's own docstring) - so a --until-unit 4 invocation can never find a
    # --until-unit 3 invocation's own checkpoint, and starts with python_units_built genuinely empty
    # even though units 1-3 were already, genuinely built - re-attempting each one from scratch at
    # ~370K-500K tokens apiece for zero real work.
    #
    # Interface compliance ALONE is NOT enough (2026-09-29, second real incident, SAME DAY: an
    # Architect re-split moved "config" out of an earlier, monolithic unit into its own dedicated
    # unit with real, new required work - three spend-cap fields, a get_api_key fix, output_root -
    # none of which was ever done. But the OLD context.py's Config class, which PRE-DATED this
    # unit's own existence entirely, already satisfied the name-level check by pure coincidence, so
    # this mechanism wrongly marked "config" built and skipped it every round). A bare name match
    # can be satisfied by unrelated, older code that happens to share a name - it proves nothing
    # about whether THIS unit's own contract was ever actually, deliberately fulfilled. A build
    # journal entry (role="builder", tagged with this exact unit name) DOES prove that: it is only
    # ever written right after _build_python_units itself completed a real attempt for this unit
    # (see the record_journal_entry call below) - it cannot be satisfied by coincidence the way a
    # name can. Require BOTH signals together. Only ADDS to already_built, never removes what this
    # round's own state already reports; a unit with no declared exports is never (re-)derived this
    # way either, since an empty interface check trivially "passes" without verifying anything real.
    for _u in build_units[: last_index + 1]:
        _u_name = _u.get("name")
        if _u_name and _u_name not in already_built and (_u.get("interface") or {}).get("exports"):
            has_builder_journal_entry = any(
                e.get("role") == "builder" and e.get("unit") == _u_name
                for e in load_relevant_entries(project_dir, _u_name)
            )
            if has_builder_journal_entry and not _check_unit_interface_compliance(code_dir, _u):
                already_built.add(_u_name)

    # A persisted other-tests baseline means the unit's own earlier build never concluded cleanly (or
    # was explicitly flagged with --rebuild-unit) - it is NOT built, whatever the interface and
    # journal say. Cleared the moment the unit is accepted as done.
    already_built = {n for n in already_built if _load_unit_baseline(project_dir, n) is None}

    interfaces_by_unit: dict = {}
    units_by_name = {u.get("name"): u for u in build_units if u.get("name")}
    newly_built = set(already_built)
    touched_files_all: dict = {}
    builder_account_last = ""
    tolerated_all: dict = {}
    seam_all: dict = {}
    claimed_all: list = []

    # Findings the builder already claimed fixed (quote verified) on an earlier invocation, whose quote
    # is still in the code: the reviewer confirms them at the end, but they must not make a built
    # unit rebuild (and burn budget) on every rerun. Carried to the reviewer as claims instead.
    _prov = load_provisional(project_dir)
    _provisional_lines: set = set()
    if _prov and review_problems:
        _norm_code = _code_text_normalized(code_dir)
        for _p in review_problems:
            _c = _prov.get(_finding_ref(_p))
            if _c and _c.get("quote") and _c["quote"] in _norm_code:
                _provisional_lines.add(_p)
                claimed_all.append({"id": _finding_ref(_p), "finding": _p, "quote": _c["quote"]})
    if _provisional_lines:
        print(f"DEBUG: {len(_provisional_lines)} finding(s) already claimed fixed with a still-present quote; "
              "not rebuilding units for them (reviewer confirms)", file=sys.stderr)
    review_problems = [_p for _p in review_problems if _p not in _provisional_lines]

    _unmatched_route = _route_unmatched_problems(review_problems, build_units[: last_index + 1], project_dir, code_dir)

    for i, unit in enumerate(build_units):
        if i > last_index:
            break
        name = unit.get("name") or f"unit_{i}"
        depends_on = unit.get("depends_on") or []
        files = unit.get("files") or []

        mentioned_in_feedback = any(
            _node_belongs_to_unit(_problem_node(p), name, files, code_dir) or _unmatched_route.get(id(p)) == name
            for p in review_problems
        )
        needs_build = name not in already_built or mentioned_in_feedback

        if needs_build:
            journal_text = format_journal_for_prompt(load_relevant_entries(project_dir, name, depends_on))
            unit_task = _format_unit_task(
                base_spec_text, unit, interfaces_by_unit, journal_text, code_dir,
                review_problems if mentioned_in_feedback else None, units_by_name,
            )

            def _refresh_unit_task(
                unit=unit, journal_text=journal_text, interfaces_by_unit=interfaces_by_unit,
            ) -> str:
                return _format_unit_task(
                    base_spec_text, unit, interfaces_by_unit, journal_text, code_dir,
                    review_problems if mentioned_in_feedback else None, units_by_name,
                )

            print(
                f"DEBUG: building unit {i + 1}/{len(build_units)} ('{name}')"
                + (" - reviewer named it this round" if mentioned_in_feedback else ""),
                file=sys.stderr,
            )
            unit_stage_info = {
                "slices_total": len(build_units),
                "slices_built": [u.get("name") or f"unit_{j}" for j, u in enumerate(build_units) if j <= i],
                "slices_remaining": [u.get("name") or f"unit_{j}" for j, u in enumerate(build_units) if j > i],
                "partial_build": last_index < len(build_units) - 1,
                "python_units_built": sorted(newly_built | {name}),
            }
            unit_findings = [
                p for p in review_problems
                if _node_belongs_to_unit(_problem_node(p), name, files, code_dir) or _unmatched_route.get(id(p)) == name
            ] if mentioned_in_feedback else []
            result = _build_python_agentic(
                {**state, "total_tokens": total_tokens, "review_problems": unit_findings},
                llm, config, unit_task, code_dir, unit_stage_info, unit_files=files,
                refresh_spec_text=_refresh_unit_task, unit_name=name,
                finding_baseline=_py_snapshot(code_dir) if unit_findings else None,
                contract_check=lambda unit=unit, name=name: (
                    _missing_declared_files(code_dir, unit)
                    + _check_all_units_interface_compliance(code_dir, build_units, newly_built, name)[0]
                ),
            )
            total_tokens = result.get("total_tokens", total_tokens)
            touched_files_all.update({f["path"]: f["action"] for f in (result.get("builder_files_touched") or [])})
            builder_account_last = result.get("builder_account") or builder_account_last
            claimed_all.extend(result.get("claimed_fixed") or [])
            if result.get("builder_status") == "done":
                save_provisional(project_dir, result.get("claimed_fixed") or [])
            if result.get("builder_status") == "done":
                # This unit's verification is the latest view of what's red; earlier units' entries
                # for tests no longer failing are dropped so a fixed test stops being reported.
                tolerated_all = {t["test"]: t for t in (result.get("tolerated_failures") or [])}

            if result.get("builder_status") != "done":
                return {
                    **result,
                    # name - not newly_built as-is: newly_built was seeded from already_built (a
                    # PRIOR round's success for this same unit, if any) before this attempt even
                    # started, and this attempt just failed - its earlier success must not survive a
                    # failed rebuild for the exact same unit (2026-09-29 real incident: a resumed run
                    # skipped a unit known to be broken because its stale prior "built" status was
                    # never cleared here).
                    # Also drop any other unit whose contract is now broken (collateral damage found
                    # during the attempts' contract check), for the same stale-success reason.
                    "python_units_built": sorted(
                        newly_built - {name} - _check_all_units_interface_compliance(code_dir, build_units, newly_built, name)[1]
                    ),
                    "builder_files_touched": [{"path": p, "action": a} for p, a in touched_files_all.items()],
                }

            compliance_problems, affected_units = _check_all_units_interface_compliance(
                code_dir, build_units, newly_built, name,
            )
            if compliance_problems:
                return {
                    **state,
                    "builder_status": "ambiguous",
                    "ambiguity_question": (
                        f"Unit '{name}' was (re)built this round, but a mechanical interface check "
                        "found a real problem (either its own contract, or collateral damage to "
                        "another already-built unit's contract):\n"
                        + "\n".join(f"- {p}" for p in compliance_problems)
                    ),
                    "total_tokens": total_tokens,
                    "python_units_built": sorted(newly_built - affected_units),
                }

            # Written for EVERY unit that passed independent verification, even when the builder
            # never self-reported (ran out of turns): record_journal_entry drops an empty summary,
            # and the "already built" re-derivation above needs this entry, so a verified unit with
            # no account would be rebuilt on every resume (trading-team, 2026-09-30).
            journal_summary = (result.get("builder_account") or "").strip() or (
                f"Unit '{name}' passed independent verification (pytest and smoke); the builder did "
                "not write a self-report. Files touched: "
                + (", ".join(sorted(f["path"] for f in (result.get("builder_files_touched") or []))) or "(none recorded)")
            )
            record_journal_entry(project_dir, name, "builder", journal_summary)
            seam_all[name] = _seam_test_problems(code_dir, unit, units_by_name)
            newly_built.add(name)
        else:
            print(f"DEBUG: skipping unit {i + 1}/{len(build_units)} ('{name}') - already built, not named in reviewer feedback", file=sys.stderr)

        interfaces_by_unit[name] = unit.get("interface", {})

    # 2026-09-29 fix, real incident (video-generator-78, creative-team --until-unit 1 on a real
    # 16-unit approved decomposition): this final return never carried slices_total/slices_built/
    # slices_remaining/partial_build forward - unit_stage_info (above) computed them every round
    # but only ever handed them INTO _build_python_agentic, never back out. So the reviewer's own
    # `if state.get("partial_build")` check (reviewer_node.py) never fired for a python
    # --until-unit run, and it judged the WHOLE spec against only unit 1's code - every problem it
    # raised named a LATER, not-yet-built unit (cli.py, delivery, Telegram, cost caps), burning all
    # 3 rounds and 755K tokens with no code changed at all. The n8n path already does this
    # correctly (its own slice_info, merged via **slice_info into its final "done" return) - this
    # brings the python path to parity. slices_total is ALSO required, not just the three keys
    # named in the report: build_and_review.py's own "_partial" status-suffix logic
    # (`if final_state.get("slices_total"):`) is gated on it, so omitting it would have left
    # partial_build=True correctly stopping the reviewer's over-broad complaint, but the run's own
    # reported status would still never have shown "_partial" or built build_and_review.py's
    # informational "slices" summary.
    unit_slice_info = {
        "slices_total": len(build_units),
        "slices_built": [u.get("name") or f"unit_{j}" for j, u in enumerate(build_units) if j <= last_index],
        "slices_remaining": [u.get("name") or f"unit_{j}" for j, u in enumerate(build_units) if j > last_index],
        "partial_build": last_index < len(build_units) - 1,
    }
    return {
        **state,
        "builder_status": "done",
        "ambiguity_question": None,
        "total_tokens": total_tokens,
        "python_units_built": sorted(newly_built),
        "builder_files_touched": [{"path": p, "action": a} for p, a in touched_files_all.items()],
        "builder_account": builder_account_last,
        "tolerated_failures": [
            {**t, "owner": _owner_unit_name(t["test"], build_units)} for t in tolerated_all.values()
        ],
        "seam_warnings": [p for probs in seam_all.values() for p in probs],
        "claimed_fixed": claimed_all,
        **unit_slice_info,
    }


def builder_node(state: BuildState) -> BuildState:
    spec_path = Path(state["spec_path"])
    code_dir = Path(state["code_dir"])
    project_dir = state.get("project_dir")
    platform = state.get("platform", "python")

    # Loaded BEFORE augmenting the spec (not after, as this used to be ordered) specifically so
    # strip_build_units below can know whether the unit-decomposed path is even in play - see that
    # parameter's own comment for the real incident this reordering exists to support. Harmless,
    # cheap no-op for n8n or a non-decomposed python project (an empty list either way).
    build_units = load_python_build_units(project_dir) if platform != "n8n" else []
    spec_text = _augment_spec_with_scale_adaptive_context(
        spec_path.read_text(encoding="utf-8"), project_dir, workflow=state.get("n8n_workflow"),
        strip_build_units=bool(build_units),
    )
    model_override = state.get("builder_model_override")
    run_id = new_run_id()
    config = run_config(run_id, tags=["builder", platform])

    if platform == "n8n":
        # Separate model role (config/models.yaml: builder_n8n) so this only ever affects n8n
        # builds - the python platform below always uses the "builder" role, untouched.
        llm = resolve_llm_for_role("builder_n8n", project_dir=project_dir, model_override=model_override, platform="n8n")
        return _build_n8n_chunked(state, llm, config, spec_text, code_dir)

    llm = resolve_llm_for_role("builder", project_dir=project_dir, model_override=model_override, platform="python")
    if build_units:
        # Architect-led unit decomposition (2026-09-28, Douglas's direct instruction) - replaces
        # PM-epic/checkpoint stage gating entirely for a project that has it. A project with no
        # Architect decision, or one that named no python build_units, falls through to the plain
        # whole-spec call below exactly as it always has - this machinery is additive, never
        # required.
        return _build_python_units(
            state, llm, config, spec_text, code_dir, build_units, state.get("python_until_unit_index"),
        )
    return _build_python_agentic(state, llm, config, spec_text, code_dir, {})

