"""Print the models a given project will actually run on, and where each came from.

Usage:
    python -m pipeline.show_models [--project-dir projects/acme-corp] [--platform n8n] [--builder-model deepseek/deepseek-v4-pro]

Used at the start of a project to confirm the model line-up with Douglas before anything is built,
and any time it's worth double-checking that a per-project override is (or isn't) in effect.
Prints JSON: {"<role>": {"model": "...", "source": "global" | "project" | "run-override"}}
"""
from __future__ import annotations

import argparse
import json
import sys

from .config_loader import effective_models, max_rounds, max_run_tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", help="Project folder whose models.yaml (if any) should apply")
    parser.add_argument("--platform", choices=["python", "n8n"], default="python")
    parser.add_argument("--builder-model", help="Preview the effect of a --builder-model override before actually running one")
    args = parser.parse_args()

    out = {
        "platform": args.platform,
        "project_dir": args.project_dir,
        "max_run_tokens": max_run_tokens(args.project_dir),
        "max_rounds": max_rounds(args.project_dir),
        "roles": effective_models(args.project_dir, platform=args.platform, builder_model=args.builder_model),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
