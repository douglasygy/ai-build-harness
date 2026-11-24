"""Tools for the agentic python builder (see pipeline/nodes/builder_node.py's _build_python_agentic).

Why this exists: the python builder used to be a single one-shot completion asked to regenerate "the
complete set of files" every attempt, with no way to run anything it wrote - confirmed as the actual
root cause of a real, expensive incident (2026-09-27, creative-team: 5,033,372 tokens spent, and a
crash that was in the code since stage 1 shipped undetected because nothing ever actually ran it).
These six tools let the builder work the way a real developer does - look around, read what's
actually there, make a targeted change, run it, see if it worked - instead of blindly regenerating
everything from a static prompt dump every single retry.

Scoped entirely to one project's code_dir via closures in make_agent_tools() - every tool rejects a
path that would resolve outside it, returning an error string to the model rather than raising (a
tool failure is normal agent feedback to react to, not a crash).

search_files was added after the FIRST real validation attempt against creative-team's actual bug
(a ~40-file, previously-unfamiliar-to-the-agent codebase): confirmed by file mtimes that the agent
spent its entire 581,632-token budget, across all 3 attempts, purely on list_files/read_file calls -
never once reaching write_file/edit_file/run_command. Reading whole files to LOCATE something (a
function, a usage, an import) is exactly the expensive pattern this tool set was built to move away
from - a real coding agent (this session's own Claude Code loop included) greps for what it needs
before reading whole files, it doesn't read a whole unfamiliar codebase file-by-file first.

Read-only access to the installed library source (_LIBRARY_ROOT_PREFIX) was added after a SECOND
real attempt against the same bug, with search_files already in place, still failed to fix it -
diagnosed (by Douglas directly, reading the real traceback and asking the right question) as the
actual bottleneck: the crash's root cause was LangGraph's OWN internal behavior (a node parameter
literally named "runtime" gets silently overridden - confirmed by reading
langgraph/_internal/_runnable.py directly), and the agent had no way to ever read that, because
list_files/read_file/search_files were all scoped to code_dir only. A human engineer debugging an
unexpected third-party library behavior reads the library's source; sandboxing the agent away from
its own dependencies' code made that specific class of bug undiagnosable no matter how many turns
or how strong the model. write_file/edit_file/run_command are NOT extended this way and never will
be - this is read-only, for understanding why an installed dependency behaves a certain way, not a
back door to modify or execute anything outside the project.
"""
from __future__ import annotations

import ast
import re
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from .project_env import ensure_project_python

# Deliberately NOT an open shell - run_command's argv[0] must be one of these, or it's rejected
# without running. This is what keeps a bounded expansion of an already-accepted risk (this pipeline
# already executes LLM-authored code locally via pytest) from becoming an open one: no pip (installing
# a new dependency is a tool-selection decision reserved to Douglas - see AGENTS.md's hard rule, not
# something the coding agent decides mid-build), no git, no arbitrary shell.
_ALLOWED_COMMAND_PREFIXES = {"python", "python3", "pytest", sys.executable}

# Same line-count default as this session's own Read tool - a large file gets a clearly-marked
# truncation rather than silently blowing up the prompt.
_MAX_READ_LINES = 400
_OUTPUT_TRUNCATE = 4000
_SEARCH_MATCH_CAP = 60
_LIST_ENTRY_CAP = 200
_EXCLUDED_DIR_PARTS = {"__pycache__", ".pytest_cache", ".git"}

# A path given to list_files/search_files/read_file starting with this prefix resolves against the
# RUNNING interpreter's own site-packages directory instead of code_dir - e.g.
# "site-packages/langgraph/_internal/_runnable.py". sysconfig.get_paths()["purelib"] (not a
# hardcoded venv path) so this works correctly regardless of where this process's own interpreter
# actually lives. write_file/edit_file/run_command never see this prefix specially at all - they
# keep using _resolve_safe(code_dir, ...) exactly as before, so this is read-only by construction,
# not a permission check that could be bypassed.
_LIBRARY_ROOT_PREFIX = "site-packages/"
_LIBRARY_ROOT = Path(sysconfig.get_paths()["purelib"])

# Substring (never the whole return value - callers append it) that write_file/edit_file include
# when the .py content they just wrote doesn't even parse. Checked via ast.parse immediately, in the
# tool's own return, rather than only surfacing at the end of a whole attempt via pytest/smoke -
# builder_node.py's _is_substantive_write looks for this marker to decide a write was NOT real
# progress, since a file that can't parse can't be what the model meant to leave behind.
_SYNTAX_ERROR_MARKER = "SYNTAX ERROR"


def _check_python_syntax(path: str, content: str) -> str:
    """'' if `path` isn't a .py file or `content` parses cleanly; otherwise a warning string
    containing _SYNTAX_ERROR_MARKER, meant to be appended to a tool's own success message so the
    model sees the problem on the very next turn instead of many turns later via a test failure."""
    if not path.endswith(".py"):
        return ""
    try:
        ast.parse(content)
    except SyntaxError as exc:
        return (
            f"\n\nWARNING - {_SYNTAX_ERROR_MARKER}: '{path}' does not parse: "
            f"{exc.msg} (line {exc.lineno}). Fix this before moving on - a file that doesn't parse "
            "can't run, and this won't count as real progress."
        )
    return ""


def _is_bare_placeholder_content(path: str, content: str) -> bool:
    """True if `path` is a .py file and `content` parses as EXACTLY one top-level statement that is
    a bare, undefined-name expression - e.g. a whole file whose entire content is the single word
    `placeholder`, no quotes, no imports, no real structure. Real incident (2026-09-29, creative-team,
    relayed via video-generator-78): the builder created creative_studio/creative_director_v2.py
    containing only the 12 characters 'placeholder' and nothing else - write_file's existing guard
    (which only protects against OVERWRITING real content, or rejects empty/whitespace content) did
    not stop it, since a bare name is syntactically valid and non-empty. Deliberately narrow: a bare
    name has zero legitimate use as an ENTIRE file's content, unlike a short docstring-only module (a
    real, if minimal, __init__.py pattern - a docstring is a string Constant, not a Name, so it is
    never caught here)."""
    if not path.endswith(".py"):
        return False
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False
    return (
        len(tree.body) == 1
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Name)
    )


# 2026-09-28, Douglas's direct instruction (discussed with him and relayed via video-generator-78):
# real transcripts showed attempt 2 and attempt 3 of _build_python_agentic's OWN internal fix loop
# doing full write_file rewrites of files that already existed and already worked (12 files in one
# batch, then 19 in another) instead of targeted edits against the specific pytest failure - because
# each attempt is a fresh conversation with no memory of what an EARLIER attempt already established,
# a full rewrite can silently invent a different name/shape than what some OTHER file already
# depends on. Concrete real incident this caused: attempt 3 rewrote storage.py from scratch with no
# memory of what attempt 2's conftest.py already expected from it (InMemoryStorage vs.
# InMemoryStorageBackend). write_file's own docstring already said "prefer edit_file" - this makes
# that mechanical instead of advisory, the same "mechanical enforcement, not a polite request"
# principle already proven necessary for the exploration cap (a text-only instruction is already
# known to get ignored). edit_file remains ALWAYS available as an escape hatch, even for a
# deliberate full rewrite - just as a targeted replacement instead of a blind full-file swap.
def _has_substantive_existing_content(target: Path) -> bool:
    if not target.exists() or target.is_dir():
        return False
    try:
        return bool(target.read_text(encoding="utf-8").strip())
    except (UnicodeDecodeError, OSError):
        return False


# Runs pytest's COLLECTION phase only (imports every test file and whatever it imports, runs
# nothing) - much faster than a real test run, and catches an import-time break (a renamed class, a
# missing name) immediately, in the SAME turn it was introduced, instead of only at the very end of
# a whole attempt via the real pytest run that already happens there. Deliberately reuses pytest's
# own import machinery rather than hand-walking the package tree and importing each file directly -
# pytest already handles real package/conftest.py semantics correctly, and doing this in-process
# would leave stale sys.modules entries behind after checking possibly-broken code.
_IMPORT_CHECK_TIMEOUT = 30
_NO_TESTS_COLLECTED_EXIT_CODE = 5


def _check_package_still_imports(code_dir: Path) -> str:
    """'' if collection succeeds OR there's nothing to collect yet (exit code 5 - not a failure,
    just nothing written yet that references anything); otherwise a WARNING string safe to append
    to write_file/edit_file's own return value, same convention as _check_python_syntax above."""
    # Target argument is "." (not str(code_dir)) - cwd is already pinned to code_dir below, and
    # code_dir is not guaranteed to be an ABSOLUTE path (2026-09-28, found while fixing the same
    # class of bug in builder_node.py's _run_pytest): passing the same possibly-relative string as
    # BOTH cwd and the target would double-resolve it against the new cwd.
    python, env_error = ensure_project_python(code_dir)
    if env_error:
        return "\n\nWARNING - PROJECT ENVIRONMENT: " + env_error
    try:
        result = subprocess.run(
            [python, "-m", "pytest", "--collect-only", "-q", "."],
            capture_output=True, text=True, timeout=_IMPORT_CHECK_TIMEOUT, cwd=str(code_dir),
        )
    except subprocess.TimeoutExpired:
        return "\n\nWARNING - IMPORT CHECK TIMED OUT: could not verify the package still imports within 30s."
    if result.returncode in (0, _NO_TESTS_COLLECTED_EXIT_CODE):
        return ""
    output = (result.stdout + "\n" + result.stderr)[-2000:]
    return (
        "\n\nWARNING - IMPORT CHECK FAILED: something no longer imports correctly (this checks the "
        f"WHOLE package, not just the file you just touched):\n{output}"
    )


def _resolve_safe(code_dir: Path, rel_path: str) -> Optional[Path]:
    """None if rel_path would resolve outside code_dir (traversal guard) - the one check every
    write_file/edit_file/run_command call runs before touching the filesystem. Never resolves into
    _LIBRARY_ROOT - that prefix is meaningless here and just becomes a literal subfolder name."""
    try:
        candidate = (code_dir / rel_path).resolve()
        candidate.relative_to(code_dir.resolve())
    except (ValueError, OSError):
        return None
    return candidate


def _resolve_readable(code_dir: Path, rel_path: str) -> Optional[Path]:
    """Like _resolve_safe, but ALSO allows a path prefixed with _LIBRARY_ROOT_PREFIX to resolve
    (read-only) into the running interpreter's own installed packages - see the module docstring
    for why. Used only by list_files/search_files/read_file."""
    if rel_path.startswith(_LIBRARY_ROOT_PREFIX):
        lib_rel = rel_path[len(_LIBRARY_ROOT_PREFIX):]
        try:
            candidate = (_LIBRARY_ROOT / lib_rel).resolve()
            candidate.relative_to(_LIBRARY_ROOT.resolve())
        except (ValueError, OSError):
            return None
        return candidate
    return _resolve_safe(code_dir, rel_path)


def _display_path(code_dir: Path, p: Path) -> str:
    """The path text to show the model for a resolved file - project-relative as always, or
    _LIBRARY_ROOT_PREFIX-relative when it resolved into the library root instead, so a result from
    site-packages never looks like it lives inside the project."""
    try:
        return p.relative_to(code_dir.resolve()).as_posix()
    except ValueError:
        pass
    try:
        return _LIBRARY_ROOT_PREFIX + p.relative_to(_LIBRARY_ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def _match_line_numbers(text: str, needle: str) -> list[int]:
    numbers, start = [], 0
    while (idx := text.find(needle, start)) != -1:
        numbers.append(text.count("\n", 0, idx) + 1)
        start = idx + 1
    return numbers


def _closest_lines_hint(text: str, old_string: str) -> str:
    """When an edit's old_string matches nothing, show the file's real nearest lines to its first
    non-blank line, so the model can correct the text without spending a turn on read_file."""
    import difflib

    wanted = next((ln.strip() for ln in old_string.splitlines() if ln.strip()), "")
    if not wanted:
        return ""
    lines = text.splitlines()
    stripped = [ln.strip() for ln in lines]
    close = difflib.get_close_matches(wanted, stripped, n=3, cutoff=0.5)
    if not close:
        return ""
    shown = []
    for match in close:
        i = stripped.index(match)
        shown.append(f"  line {i + 1}: {lines[i]!r}")
    return "\nClosest real line(s) in the file (exact text, including indentation):\n" + "\n".join(shown)


def make_agent_tools(code_dir: Path) -> list:
    code_dir = Path(code_dir)

    @tool
    def list_files(path: str = ".") -> str:
        """List files and directories under the given path (relative to the project root, default
        the whole project). Use this first to see what actually exists before reading or writing -
        never assume a file's location. Prefix a path with 'site-packages/' (e.g.
        'site-packages/langgraph') to list inside an INSTALLED dependency instead of the project -
        read-only, useful when you need to understand why an installed library behaves a certain
        way, not just what your own code does."""
        target = _resolve_readable(code_dir, path)
        if target is None:
            return f"ERROR: '{path}' resolves outside the allowed directories - not allowed."
        if not target.exists():
            return f"ERROR: '{path}' does not exist."
        if target.is_file():
            return f"'{path}' is a file, not a directory - use read_file instead."
        entries = []
        for p in sorted(target.rglob("*")):
            if any(part in _EXCLUDED_DIR_PARTS for part in p.parts):
                continue
            entries.append(f"{'[dir] ' if p.is_dir() else ''}{_display_path(code_dir, p)}")
        if not entries:
            return f"'{path}' exists but is empty."
        listing = "\n".join(entries[:_LIST_ENTRY_CAP])
        if len(entries) > _LIST_ENTRY_CAP:
            listing += f"\n... and {len(entries) - _LIST_ENTRY_CAP} more entries (narrow `path` to see them)."
        return listing

    @tool
    def search_files(pattern: str, path: str = ".") -> str:
        """Search for a regex pattern across real files under the given path (relative to the
        project root), returning each match as 'file:line: the matching line' - like grep. Use this
        FIRST to locate relevant code (a function definition, a call site, an import) instead of
        reading whole files one by one to find something - much cheaper, and the right way to work
        in a codebase you don't already have memorized. Prefix a path with 'site-packages/' (e.g.
        'site-packages/langgraph') to search inside an INSTALLED dependency instead of the project -
        read-only; use this when an installed library seems to be behaving unexpectedly and you need
        to see why, not just guess."""
        target = _resolve_readable(code_dir, path)
        if target is None:
            return f"ERROR: '{path}' resolves outside the allowed directories - not allowed."
        if not target.exists():
            return f"ERROR: '{path}' does not exist."
        try:
            regex = re.compile(pattern)
        except re.error:
            # A plain-text query like "Config(" is not a valid regex; search it literally.
            regex = re.compile(re.escape(pattern))
        candidates = [target] if target.is_file() else sorted(target.rglob("*"))
        matches = []
        for file_path in candidates:
            if not file_path.is_file() or any(part in _EXCLUDED_DIR_PARTS for part in file_path.parts):
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = _display_path(code_dir, file_path)
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{rel}:{lineno}: {line.strip()}")
                    if len(matches) >= _SEARCH_MATCH_CAP:
                        break
            if len(matches) >= _SEARCH_MATCH_CAP:
                break
        if not matches:
            return f"No matches for pattern '{pattern}' under '{path}'."
        result = "\n".join(matches)
        if len(matches) >= _SEARCH_MATCH_CAP:
            result += f"\n... (stopped at {_SEARCH_MATCH_CAP} matches - narrow the pattern or path for more)"
        return result

    @tool
    def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a file's real current contents (path relative to the project root), with line
        numbers so you can reference exact lines. Always read a file before editing it - never edit
        from memory of what you wrote earlier. At most 400 lines come back per call: for a longer
        file, pass start_line/end_line (1-based, inclusive) to read just the part you need -
        cheaper than reading it all. Prefix a path with 'site-packages/' (e.g.
        'site-packages/langgraph/graph/state.py') to read inside an INSTALLED dependency instead of
        the project - read-only; use this to understand why a library behaves a certain way instead
        of guessing at its internals."""
        target = _resolve_readable(code_dir, path)
        if target is None:
            return f"ERROR: '{path}' resolves outside the allowed directories - not allowed."
        if not target.exists():
            return f"ERROR: '{path}' does not exist. Use list_files to see what's actually there."
        if target.is_dir():
            return f"ERROR: '{path}' is a directory, not a file - use list_files on it instead."
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"ERROR: '{path}' is not a text file (binary content)."
        lines = text.splitlines()
        first = max(1, int(start_line or 1))
        last = int(end_line) if end_line and int(end_line) > 0 else len(lines)
        last = min(last, len(lines), first + _MAX_READ_LINES - 1)
        shown = lines[first - 1:last]
        numbered = "\n".join(f"{first + i}\t{line}" for i, line in enumerate(shown))
        if last < len(lines):
            numbered += (
                f"\n... (lines {last + 1}-{len(lines)} not shown; call read_file again with "
                f"start_line={last + 1} to continue)"
            )
        return numbered

    @tool
    def write_file(path: str, content: str) -> str:
        """Create a new file with the given content (path relative to the project root). REJECTED
        for a file that already exists with real content - use edit_file for a targeted change
        instead; a full rewrite of something that already works risks silently inventing a
        different name/shape than what some OTHER file already depends on, especially across a
        fresh attempt with no memory of what an earlier one already established. Only usable to
        overwrite an existing file when it's currently empty or a trivial stub."""
        target = _resolve_safe(code_dir, path)
        if target is None:
            return f"ERROR: '{path}' resolves outside the project directory - not allowed."
        if _has_substantive_existing_content(target):
            return (
                f"ERROR: '{path}' already exists with real content - write_file only creates new "
                "files or overwrites an empty/trivial one. Use edit_file for a targeted change "
                "instead of a full rewrite, even if the change is large - read the file first if "
                "you're not certain of its exact current content."
            )
        if _is_bare_placeholder_content(path, content):
            return (
                f"ERROR: '{path}' content is just a bare, undefined name with nothing else - this "
                "is not real code and will not be written. Write the actual implementation, or use "
                "run_command/read_file first if you're not ready to write this file yet."
            )
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        result = f"Wrote {'(overwrote existing) ' if existed else '(new file) '}'{path}' - {len(content)} bytes."
        result += _check_python_syntax(path, content)
        if path.endswith(".py") and _SYNTAX_ERROR_MARKER not in result:
            result += _check_package_still_imports(code_dir)
        return result

    @tool
    def edit_file(path: str, old_string: str, new_string: str) -> str:
        """Replace an exact, unique occurrence of old_string with new_string in an existing file
        (path relative to the project root). old_string must match the file's real current content
        exactly, including whitespace, and appear exactly once - include enough surrounding context
        to make it unique. Read the file first if you're not certain of its exact current content."""
        target = _resolve_safe(code_dir, path)
        if target is None:
            return f"ERROR: '{path}' resolves outside the project directory - not allowed."
        if not target.exists():
            return f"ERROR: '{path}' does not exist - use write_file to create it first."
        text = target.read_text(encoding="utf-8")
        count = text.count(old_string)
        if count == 0:
            return (
                f"ERROR: old_string not found in '{path}' - it must match the real file exactly, "
                "including whitespace. Use read_file to see the actual current content."
                + _closest_lines_hint(text, old_string)
            )
        if count > 1:
            line_numbers = _match_line_numbers(text, old_string)
            return (
                f"ERROR: old_string appears {count} times in '{path}' (starting at line(s) "
                f"{', '.join(str(n) for n in line_numbers[:8])}) - it must be unique. Include more "
                "surrounding context."
            )
        new_text = text.replace(old_string, new_string, 1)
        target.write_text(new_text, encoding="utf-8")
        result = f"Edited '{path}'." + _check_python_syntax(path, new_text)
        if path.endswith(".py") and _SYNTAX_ERROR_MARKER not in result:
            result += _check_package_still_imports(code_dir)
        return result

    @tool
    def run_command(command: str) -> str:
        """Run a command to verify your work actually works - e.g. `pytest`, `python -m pytest
        tests/ -q`, `python smoke_run.py`, or a one-off `python -c "..."` diagnostic. Only python/
        pytest commands are allowed (no package installs, no git, no shell) - use forward slashes in
        any paths. ALWAYS run something real before reporting the work as done; a claim that isn't
        backed by a real, successful run here is not trustworthy."""
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return f"ERROR: could not parse command: {exc}"
        if not argv:
            return "ERROR: empty command."
        if argv[0] not in _ALLOWED_COMMAND_PREFIXES:
            return (
                f"ERROR: '{argv[0]}' is not allowed here - only python/pytest commands may run "
                "directly (no package installs, no git, no shell). If the task genuinely needs a "
                "new dependency, say so as your final answer instead of reporting done - that's a "
                "decision reserved to Douglas, not something to install unilaterally."
            )
        # A bare "pytest"/"python3" entry-point name isn't reliably resolvable by CreateProcess on
        # Windows without shell=True (confirmed in practice: it fails with WinError 2 unless the
        # venv's Scripts dir happens to be on PATH) - exactly why _run_pytest elsewhere in this file
        # always spawns [sys.executable, "-m", "pytest", ...] instead of bare "pytest". Same fix
        # here, transparent to the agent: it can type the natural command either way.
        python, env_error = ensure_project_python(code_dir)
        if env_error:
            return f"ERROR: {env_error}"
        if argv[0] == "pytest":
            argv = [python, "-m", "pytest"] + argv[1:]
        elif argv[0] in ("python3", "python"):
            argv = [python] + argv[1:]
        try:
            result = subprocess.run(
                argv, cwd=str(code_dir), capture_output=True, text=True, timeout=120, shell=False,
            )
        except subprocess.TimeoutExpired:
            return f"'{command}' did not finish within 120 seconds - timed out."
        except OSError as exc:
            return f"ERROR: could not run '{command}': {exc}"
        output = (result.stdout + "\n" + result.stderr)[-_OUTPUT_TRUNCATE:]
        return f"exit code {result.returncode}\n{output}"

    return [list_files, search_files, read_file, write_file, edit_file, run_command]
