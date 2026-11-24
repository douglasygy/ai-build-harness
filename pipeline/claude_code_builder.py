"""Claude Code headless (`claude -p`) as the python builder's turn engine (trading-team trial).

Replaces only the tool-call loop of one builder attempt. Everything after it (independent pytest,
smoke, requirements and contract checks, finding gates, reviewer) is unchanged. Permissions are an
explicit allowlist - NO bypass-permissions flag: file edits plus python/pytest only, mirroring the
OpenRouter builder's run_command allowlist (no pip, no git, no network tools).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .constants import CLAUDE_CODE_MAX_TURNS, CLAUDE_CODE_TIMEOUT_SEC

ALLOWED_TOOLS = ["Read", "Glob", "Grep", "Edit", "Write", "Bash(python:*)", "Bash(pytest:*)"]
DISALLOWED_TOOLS = ["WebFetch", "WebSearch", "Bash(git:*)", "Bash(pip:*)"]

TOOL_NOTE = (
    "TOOLING NOTE: you are running as Claude Code. Wherever these instructions name list_files, "
    "search_files, read_file, write_file, edit_file or run_command, use Glob, Grep, Read, Write, Edit "
    "and Bash (python/pytest only) instead. record_tech_quirk is unavailable. Work only inside the "
    "current directory. Your FINAL message must follow the same rules as stated (e.g. `AMBIGUOUS: ...` "
    "or `ALREADY_FIXED: ...` lines, otherwise a brief account of what you changed)."
)


def build_command(system_prompt: str, model: str | None = None) -> list:
    exe = shutil.which("claude") or "claude"
    cmd = [
        exe, "-p", "--output-format", "json", "--no-session-persistence",
        "--permission-mode", "acceptEdits", "--max-turns", str(CLAUDE_CODE_MAX_TURNS),
        "--append-system-prompt", system_prompt,
        "--allowedTools", *ALLOWED_TOOLS,
        "--disallowedTools", *DISALLOWED_TOOLS,
    ]
    if model:
        cmd += ["--model", model]
    return cmd


def run_claude_code(prompt: str, system_prompt: str, code_dir, model: str | None = None,
                    timeout: int = CLAUDE_CODE_TIMEOUT_SEC, runner=subprocess.run) -> dict:
    """Returns {text, tokens, turns, timed_out, error}. Never raises for a failed session."""
    cmd = build_command(system_prompt + "\n\n" + TOOL_NOTE, model)
    try:
        proc = runner(cmd, input=prompt, cwd=str(Path(code_dir)), capture_output=True, text=True,
                      encoding="utf-8", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"text": "", "tokens": 0, "turns": 0, "timed_out": True, "error": f"timed out after {timeout}s"}
    except OSError as exc:
        return {"text": "", "tokens": 0, "turns": 0, "timed_out": False, "error": f"could not start claude: {exc}"}
    try:
        data = json.loads(proc.stdout or "")
    except ValueError:
        return {"text": "", "tokens": 0, "turns": 0, "timed_out": False,
                "error": f"unparseable claude output (exit {proc.returncode}): {(proc.stdout or proc.stderr or '')[-400:]}"}
    usage = data.get("usage") or {}
    tokens = sum(int(usage.get(k) or 0) for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens"))
    subtype = str(data.get("subtype") or "")
    return {
        "text": str(data.get("result") or "").strip(),
        "tokens": tokens,
        "turns": int(data.get("num_turns") or 0),
        "timed_out": False,
        "error": "" if not data.get("is_error") and subtype in ("", "success") else f"session ended: {subtype or 'error'}",
    }
