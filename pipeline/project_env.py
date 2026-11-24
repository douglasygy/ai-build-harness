"""Per-project Python environment (Douglas, 2026-09-30, via trading-team session).

A project's builds must run in the project's OWN virtualenv (<project>/.venv), not build-agent's, so
packages the builder adds to <project>/code/requirements.txt actually exist when pytest, the smoke
test and the builder's run_command execute. The venv is created if missing and requirements.txt is
(re)installed whenever its content changes. A project with no code/requirements.txt keeps using
build-agent's own interpreter (nothing to install, nothing changes).
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

REQUIREMENTS_FILENAME = "requirements.txt"
_MARKER_NAME = ".build_agent_requirements.sha"
_INSTALL_TIMEOUT = 900
_TOOLING = ["pytest"]


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _fingerprint(requirements: Path) -> str:
    digest = hashlib.sha256(requirements.read_bytes())
    digest.update(",".join(_TOOLING).encode())
    return digest.hexdigest()


def ensure_project_python(code_dir) -> Tuple[str, Optional[str]]:
    """(interpreter, error). error is None on success; otherwise a message the caller must surface
    as a verification failure (never silently fall back to build-agent's interpreter)."""
    code_dir = Path(code_dir)
    requirements = code_dir / REQUIREMENTS_FILENAME
    if not requirements.exists():
        return sys.executable, None

    venv_dir = (code_dir.resolve().parent) / ".venv"
    python = _venv_python(venv_dir)
    try:
        if not python.exists():
            created = subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                capture_output=True, text=True, timeout=300,
            )
            if created.returncode != 0:
                return sys.executable, f"Could not create {venv_dir}:\n{(created.stdout + created.stderr)[-2000:]}"

        marker = venv_dir / _MARKER_NAME
        fingerprint = _fingerprint(requirements)
        if marker.exists() and marker.read_text(encoding="utf-8").strip() == fingerprint:
            return str(python), None

        installed = subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(requirements.resolve()), *_TOOLING],
            capture_output=True, text=True, timeout=_INSTALL_TIMEOUT, cwd=str(code_dir),
        )
        if installed.returncode != 0:
            return str(python), (
                f"pip install of {requirements} into {venv_dir} failed - fix requirements.txt "
                f"(or report the dependency problem):\n{(installed.stdout + installed.stderr)[-3000:]}"
            )
        marker.write_text(fingerprint, encoding="utf-8")
    except subprocess.TimeoutExpired:
        return sys.executable, f"Timed out preparing the project environment at {venv_dir}."
    except OSError as exc:
        return sys.executable, f"Could not prepare the project environment at {venv_dir}: {exc}"
    return str(python), None
