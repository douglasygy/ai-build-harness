"""Mechanical check: every third-party import in a project's code must be declared in
code/requirements.txt (trading-team, 2026-09-30: tzdata was used via ZoneInfo but never declared, and
only the reviewer noticed). Applies only to projects that have a requirements.txt - the same
condition under which the pipeline builds a per-project venv (pipeline/project_env.py).
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# import name -> distribution name, where the two differ (normalized comparison covers the rest,
# e.g. langchain_openai -> langchain-openai).
_IMPORT_TO_DIST = {
    "yaml": "pyyaml", "pil": "pillow", "sklearn": "scikit-learn", "cv2": "opencv-python",
    "bs4": "beautifulsoup4", "dotenv": "python-dotenv", "dateutil": "python-dateutil",
    "jwt": "pyjwt", "attr": "attrs", "dns": "dnspython", "serial": "pyserial", "magic": "python-magic",
    "google": None, "moomoo": "moomoo-api", "exchange_calendars": "exchange-calendars",
}
_ALWAYS_OK = {"pytest", "_pytest", "pip", "setuptools", "pkg_resources"}
_SKIP_DIRS = {".venv", "venv", "__pycache__", "site-packages", "node_modules", ".git"}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared(requirements: Path) -> set:
    names = set()
    for line in requirements.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "git+", "http")):
            continue
        m = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", line)
        if m:
            names.add(_norm(m.group(0)))
    return names


_DYNAMIC_IMPORT_FUNCS = {"import_module", "__import__", "find_spec"}


def _dynamic_import_name(call: ast.Call) -> str:
    """The module name of importlib.import_module("x") / __import__("x") / importlib.util.find_spec("x")
    when it is a plain string literal (absolute names only); "" otherwise."""
    func = call.func
    fname = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
    if fname not in _DYNAMIC_IMPORT_FUNCS or not call.args:
        return ""
    arg = call.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        name = arg.value.strip()
        if name and not name.startswith(".") and re.match(r"[A-Za-z_][A-Za-z0-9_.]*$", name):
            return name
    return ""


def undeclared_third_party_imports(code_dir) -> list:
    """Problem strings, [] when fine or when there is no requirements.txt to check against."""
    code_dir = Path(code_dir)
    requirements = code_dir / "requirements.txt"
    if not requirements.exists():
        return []
    declared = _declared(requirements)
    py_files = [
        p for p in code_dir.rglob("*.py")
        if not (set(p.relative_to(code_dir).parts) & _SKIP_DIRS)
    ]
    local = {p.stem for p in py_files} | {p.parent.name for p in py_files} | {
        d.name for d in code_dir.rglob("*") if d.is_dir() and not (set(d.relative_to(code_dir).parts) & _SKIP_DIRS)
    }
    stdlib = set(sys.stdlib_module_names)
    problems: dict = {}
    zoneinfo_used = False
    for path in py_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue
        rel = path.relative_to(code_dir).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                tops = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                tops = [node.module.split(".")[0]]
            elif isinstance(node, ast.Call) and _dynamic_import_name(node):
                tops = [_dynamic_import_name(node).split(".")[0]]
            else:
                continue
            for top in tops:
                if top == "zoneinfo":
                    zoneinfo_used = True
                if top in stdlib or top in local or top in _ALWAYS_OK or top == "__future__":
                    continue
                dist = _IMPORT_TO_DIST.get(top.lower(), top) if top.lower() in _IMPORT_TO_DIST else top
                if dist is None:
                    continue
                if _norm(dist) in declared or _norm(top) in declared:
                    continue
                problems.setdefault(top, rel)
    out = [
        f"'{top}' is imported (first seen in {rel}) but not declared in code/requirements.txt - add its "
        "package to requirements.txt (this is what the project's own environment installs)."
        for top, rel in sorted(problems.items())
    ]
    if zoneinfo_used and "tzdata" not in declared:
        out.append(
            "zoneinfo is used but 'tzdata' is not in code/requirements.txt - time zones fail on Windows "
            "without it; add tzdata."
        )
    return out
