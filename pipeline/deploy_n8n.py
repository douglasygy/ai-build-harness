"""CLI: turn a built n8n project's verified workflow(s) into their deployable production form.

    python -m pipeline.deploy_n8n --project-dir <path> [--workflow <name>]

Reads <project-dir>/code/n8n/[<workflow>/]workflow.json and .production_triggers.json, writes
<project-dir>/code/n8n/[<workflow>/]deploy/workflow.json and bootstrap_manifest.json. Never touches
any live n8n instance - see pipeline/n8n_deploy.py's module docstring for why that stays a
separate, deliberate step.

Multi-workflow projects (see pipeline/n8n_workflow_target.py - an Architect decision naming more
than one n8n workflow): pass --workflow to deploy just that one, or omit it to deploy every
workflow this project's Architect decision names, one after another - each into its own
code/n8n/<name>/deploy/, so a client's whole multi-workflow automation ships together. A workflow
not built yet is skipped with a warning, not treated as a fatal error for the rest of the batch.
Projects with at most one n8n workflow are unaffected either way - always the original flat
code/n8n/deploy/ path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .n8n_deploy import bootstrap_manifest, build_production_workflow
from .n8n_workflow_target import n8n_subdir, n8n_workflows, resolve_workflow_slug, slugify


def _deploy_one(project_dir: Path, workflow: str | None) -> bool:
    """Returns True if a deploy was actually produced, False if there was nothing built yet to
    deploy (not an error in the multi-workflow batch case - just nothing to do for this one yet).
    """
    n8n_dir = n8n_subdir(project_dir / "code", workflow)
    workflow_path = n8n_dir / "workflow.json"
    triggers_path = n8n_dir / ".production_triggers.json"

    if not workflow_path.exists():
        print(f"No built workflow found at {workflow_path} - run build_and_review.py first.", file=sys.stderr)
        return False

    workflow_json = json.loads(workflow_path.read_text(encoding="utf-8"))
    triggers = json.loads(triggers_path.read_text(encoding="utf-8")) if triggers_path.exists() else []
    if not triggers:
        print(
            f"WARNING [{workflow or project_dir.name}]: no recorded triggers - nothing to swap; the "
            "verification workflow will be copied through with its dev-mode webhook trigger(s) unchanged.",
            file=sys.stderr,
        )

    workflow_name = workflow_json.get("name") or workflow or project_dir.name
    produced = build_production_workflow(workflow_json, triggers, workflow_name)
    manifest = bootstrap_manifest(workflow_name, triggers, produced)

    deploy_dir = n8n_dir / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    (deploy_dir / "workflow.json").write_text(json.dumps(produced, indent=2), encoding="utf-8")
    (deploy_dir / "bootstrap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {deploy_dir / 'workflow.json'}")
    print(f"Wrote {deploy_dir / 'bootstrap_manifest.json'}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", required=True, help="Path to the project folder (holds code/n8n/...)")
    parser.add_argument(
        "--workflow",
        help=(
            "Deploy only this n8n workflow (see pipeline/n8n_workflow_target.py). Required only when "
            "this project's Architect decision names more than one and you want just one deployed; "
            "omit to deploy all of them (or the project's single workflow, if it has just one)."
        ),
    )
    args = parser.parse_args()
    project_dir = Path(args.project_dir)

    if args.workflow:
        slug, problem = resolve_workflow_slug(project_dir, args.workflow)
        if problem:
            print(f"ERROR: {problem}", file=sys.stderr)
            return 1
        ok = _deploy_one(project_dir, slug)
        return 0 if ok else 1

    workflows = n8n_workflows(project_dir)
    if len(workflows) <= 1:
        ok = _deploy_one(project_dir, None)
        return 0 if ok else 1

    print(f"Deploying {len(workflows)} n8n workflows for this project...", file=sys.stderr)
    any_ok = False
    for w in workflows:
        any_ok = _deploy_one(project_dir, slugify(w["name"])) or any_ok
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
