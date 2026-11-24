"""Independent review of one unit of code that Claude Code wrote directly (no pipeline builder).

usage (run from anywhere, with build-agent's venv):
  python -m pipeline.unit_review --project-dir <project> --unit <name> --focus "<what to review>" file1 file2 ...
  add --platform web for a Next.js/TypeScript unit (uses the reviewer_web role and the web checklist);
  --checks-file <path> attaches the saved output of type-check/lint/build/tests; --design-file <path>
  attaches the approved design text the unit must match.
--context-files a.py,b.py adds files that are NOT under review but touch the unit (the code that calls it,
reads what it saves, or produces what it consumes); the reviewer checks the boundary with them. Always add
the real producers and consumers of anything the unit stores, sends or reads: most defects found late sit
at those boundaries, not inside one file.
The standard defect checklist (pipeline/review_checklist.py) is appended to the reviewer prompt unless
--no-checklist is given.
File paths are relative to <project>/code (or to <project> if there is no code/ folder).

Uses the pipeline's own reviewer SYSTEM_PROMPT and the build-agent OpenRouter key. The reviewer model
is openai/gpt-6.1-sol (a different vendor from Claude Code) unless the project's models.yaml sets its own
reviewer. Prints the reviewer's JSON and the cost; writes a file only if --out is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REVIEWER_MODEL = "openai/gpt-6.1-sol"


def _project_sets_role(project_dir: str, role: str) -> bool:
    from .config_loader import _load_project_raw
    return bool(((_load_project_raw(project_dir).get("roles") or {}).get(role) or {}).get("model"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--focus", required=True)
    parser.add_argument("--spec-sections", default="", help="comma-separated '## ' headings to include; default whole SPEC.md")
    parser.add_argument("--platform", choices=["python", "web"], default="python")
    parser.add_argument("--checks-file", default="", help="saved output of type-check/lint/build/tests (web)")
    parser.add_argument("--design-file", default="", help="approved design text to match (web)")
    parser.add_argument("--context-files", default="", help="comma-separated files NOT under review, shown so the reviewer can check this unit's boundaries with them")
    parser.add_argument("--no-checklist", action="store_true", help="omit the standard defect checklist")
    parser.add_argument("--out", default="", help="optional path to write the reviewer JSON to")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args(argv)

    load_dotenv(REPO_ROOT / ".env")
    from .model_factory import get_chat_model
    from .nodes.reviewer_node import SYSTEM_PROMPT
    from .review_checklist import CHECKLIST_ADDENDUM
    from .web_review_prompt import WEB_REVIEW_ADDENDUM

    project = Path(args.project_dir).resolve()
    code_root = project / "code" if (project / "code").is_dir() else project
    spec = (project / "SPEC.md").read_text(encoding="utf-8")
    if args.spec_sections:
        parts = []
        for heading in [h.strip() for h in args.spec_sections.split(",") if h.strip()]:
            start = spec.find(f"## {heading}")
            if start == -1:
                print(f"spec heading not found: {heading}", file=sys.stderr)
                return 2
            end = spec.find("\n## ", start + 3)
            parts.append(spec[start:end if end != -1 else len(spec)])
        spec = "\n\n".join(parts)
    code = "".join(f"\n===== FILE: {f} =====\n" + (code_root / f).read_text(encoding="utf-8") for f in args.files)
    context_names = [c.strip() for c in args.context_files.split(",") if c.strip()]
    context = "".join(f"\n===== CONTEXT FILE (not under review): {f} =====\n" + (code_root / f).read_text(encoding="utf-8")
                      for f in context_names)
    extra = ""
    if args.checks_file:
        extra += f"\n\nVERIFY OUTPUT:\n{Path(args.checks_file).read_text(encoding='utf-8')}"
    if args.design_file:
        extra += f"\n\nAPPROVED DESIGN:\n{Path(args.design_file).read_text(encoding='utf-8')}"
    if context:
        scope = (f"Review this unit: {args.focus}. Report defects inside the unit, and defects at its boundary with "
                 "the CONTEXT FILES (what the unit reads, saves, sends or calls versus what those files really "
                 "produce and do). Do not report defects that lie only inside a context file. Modules referenced "
                 "but not shown were not shown to you: do not assume they are right or wrong, and do not report them.")
    else:
        scope = (f"Review this unit: {args.focus}. Modules referenced but not shown were not shown to you: do not "
                 "assume they are right or wrong, and do not report them.")
    user = f"SPEC:\n{spec}\n\nCODE UNDER REVIEW:\n{code}{context}{extra}\n\n{scope}"

    web = args.platform == "web"
    role = "reviewer_web" if web else "reviewer"
    system_prompt = SYSTEM_PROMPT + WEB_REVIEW_ADDENDUM if web else SYSTEM_PROMPT
    if not args.no_checklist:
        system_prompt += CHECKLIST_ADDENDUM
    override = None if web or _project_sets_role(str(project), role) else DEFAULT_REVIEWER_MODEL
    llm = get_chat_model(role, model_override=override, project_dir=str(project))
    if str(llm.model_name).startswith("anthropic/"):
        print(f"refusing: {role} resolved to {llm.model_name}, but Claude Code builds, so the reviewer "
              "must be a different vendor", file=sys.stderr)
        return 2
    resp = llm.invoke([("system", system_prompt), ("user", user)])
    usage = (resp.response_metadata or {}).get("token_usage", {})
    text = resp.content if isinstance(resp.content, str) else json.dumps(resp.content)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"model {llm.model_name} | cost ${usage.get('cost')} | tokens {usage.get('prompt_tokens')}/{usage.get('completion_tokens')}"
          f" | files {len(args.files)} + {len(context_names)} context | checklist {'off' if args.no_checklist else 'on'}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
