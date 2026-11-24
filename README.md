# AI Build Harness

A multi-model pipeline that turns a rough product idea into reviewed, tested, working code. The design
rule behind it: **no single AI model is trusted alone.** A human writes and approves the spec, one AI
builds, an AI from a *different company* reviews, and nothing ships without live tests and human approval.

Built by Douglas Yap. Python 3.12, LangGraph, OpenRouter (one API for models from several providers),
Langfuse (run tracing), n8n (workflow automation target), Claude Code (orchestrator).

## How it works

```
  idea
    |
    v
 Concept intake (human + Claude Code) --> Business analyst AI (a different vendor) challenges the concept
    |                                       Concept advocate AI answers; only product decisions go to the human
    v
 HUMAN GATE 1: approve the concept
    |
    v
 Spec locked (PM and Architect AIs join on large projects) --> HUMAN GATE 2: approve the spec
    |
    v
 Builder AI: writes code, writes its own tests, runs the real system, repairs failures
    |        (capped retries, capped tool turns, per-run token ceiling)
    v
 Reviewer AI from a different vendor than the builder (enforced in code) reads only the spec
 and the finished code from disk, never the builder's reasoning
    |
    v
 Live verification (n8n workflows are deployed to a test instance and executed against fixtures)
    |
    v
 HUMAN GATE 3: approve deployment. Nothing auto-deploys.
```

## What is in here

| Path | What it does |
|---|---|
| `pipeline/graph.py`, `pipeline/build_and_review.py` | The LangGraph build, review and repair loop |
| `pipeline/nodes/` | The builder and reviewer nodes |
| `pipeline/config_loader.py` | Loads each role's model and refuses to run if builder and reviewer share a vendor |
| `pipeline/unit_review.py` | Independent cross-vendor review of one unit of code, with a standard defect checklist |
| `pipeline/review_*.py` | Concept, PM, Architect and team-composition review stages |
| `pipeline/n8n_*.py`, `pipeline/deploy_n8n.py` | Build, deploy and verify n8n workflows on a test instance |
| `pipeline/token_guard.py`, `pipeline/retry.py` | Cost ceilings and retry caps |
| `pipeline/decision_log.py`, `pipeline/build_scribe.py` | Decision memory: logs every gate decision and distils patterns |
| `config/models.yaml` | Which model fills each role, per role token limits |
| `AGENTS.md` | The working rules the AI team follows (the best place to see how the system is governed) |
| `tests/` | The automated test suite (about 460 tests) |

## Design decisions worth knowing

- **Cross-vendor review.** The reviewer must come from a different provider than the builder, and the
  code refuses to start otherwise.
- **Human approval gates** at concept, spec lock and deployment. AI proposes; the human decides.
- **Bounded by design.** Retry caps, tool-turn caps, per-run token ceilings and model output limits.
- **Decision memory.** Every gate decision is logged, and patterns are regenerated from the logs, never
  edited by hand. The goal is that proposals increasingly match how the owner actually decides.
- **Honest verification.** The builder must run the real system, not just write code that looks right.

## Setup

```
python -m venv .venv
pip install -r requirements.txt
cp .env.example .env     # add your own OpenRouter and Langfuse keys
python -m pytest tests -q
```

`memory/` and `projects/` are empty here on purpose: they hold a user's own decision history and client
projects. The `.claude/skills` the working rules mention are not included in this public copy.

## Status and licence

A personal working system, shared to show how it is built. All rights reserved: you are welcome to read
it, but it is not licensed for reuse or redistribution.
