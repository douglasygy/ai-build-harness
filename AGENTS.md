# AI Build Harness — Working Instructions

This file is the canonical, tool-agnostic instruction set for this repo. Any AI coding tool working
here (Claude Code, Cursor, etc.) should read this first. `CLAUDE.md` just points here.

## What this repo is
A reusable framework for going from a rough product idea to reviewed, working code, one client
project at a time. Each client's own spec and generated code live under `projects/<client-name>/`.
The framework code (this pipeline) stays at the repo root and is shared across all projects.

## The four roles
1. **Concept intake** — whichever AI tool is reading this (e.g. Claude Code), talking directly to
   Douglas. Always read the target project's existing `SPEC.md` and `code/` folder first before
   proposing any spec change — never assume, never start from a blank page on an amend. Drafts the
   spec from conversation, then hands it to the concept-stage bounce below before ever presenting
   decisions to Douglas.
2. **Business analyst**, bounced against **concept_advocate** (`pipeline/review_concept.py`,
   configured in `config/models.yaml`) — these two go back and forth for up to
   `MAX_CONCEPT_ROUNDS` (2) rounds: business_analyst (non-Anthropic, independent) raises gaps/edge
   cases; concept_advocate (Anthropic, represents Agent 1's drafting perspective) resolves what's
   safely a coding/implementation default and refuses to resolve anything that's actually a
   product-intent decision. Neither ever decides what the product should do. The script prints ONE
   final JSON object: `{"summary", "resolved_assumptions", "decisions_needed"}`. Claude Code (the
   real Agent 1) reads this, folds `resolved_assumptions` into the draft, and presents each entry
   in `decisions_needed` to Douglas as an actual decision — e.g. via AskUserQuestion, with the
   given options plus room for him to type his own. This is still a hard gate: nothing is locked in
   until Douglas has answered every decision point and explicitly approves.
3. **Builder** (`pipeline/build_and_review.py` → `pipeline/nodes/builder_node.py`) — for python
   projects, a real tool-using agent (explore, read, edit, run real commands - see
   `pipeline/agent_tools.py`), not a one-shot completion: it writes the code, writes and runs its own
   tests, verifies the real system actually runs, and fixes failures itself (capped at 3 fresh
   attempts, each up to 20 tool-call turns per round (`MAX_AGENT_TURNS`; lowered to 3 on 2026-09-30, restored to 20 by Douglas 2026-10-02) and up to 3 rounds (`MAX_ROUNDS_PER_ATTEMPT`) - a further round runs in the same conversation only if the previous round wrote/edited something; the Architect therefore splits work into sprint-sized units, see `build_limits_sheet`). n8n projects use a separate, per-node build+live-verify
   path (`_build_n8n_chunked`), untouched by this.
4. **Reviewer** (`pipeline/nodes/reviewer_node.py`) — a different vendor from whichever builder model
   is used that run (enforced in `pipeline/config_loader.py` — the script refuses to run otherwise).
   Only ever reads the spec and the finished code fresh from disk, never the builder's reasoning.

## Scale-adaptive roles: PM and Architect (optional)
Two additional roles — PM and Architect — exist for projects big enough to need them, added to or
removed from a project's roster per-project, never baked into every run. The core four above are
never removable, regardless of scale. **Full detail — roster rules, the team-composition gate,
approval mechanics, and how their output actually gets wired into a build — lives in the
`build-agent-scale-adaptive-roles` skill.** Load it whenever a project is big enough that PM or
Architect might join the roster, or when actually running `review_team_composition.py`,
`review_pm.py`, or `review_architect.py`. Most small projects never need this.

## Decision log & preference learning
The long-term goal, in Douglas's own framing: after enough projects, this pipeline should be able to
build and resolve routine decisions on its own, with only a few hard rules requiring him directly —
not by guessing, but by having actually learned how he decides things. A two-tier memory system
does this — `build_scribe` (Descriptive, never decides anything) and the Gatekeeper (Evaluative,
`preference_analyst`) — kept strictly separate, the same principle already behind the reviewer only
ever reading code fresh from disk rather than the builder's reasoning.

**Standing obligation, always active regardless of what else is loaded:** after any gated decision
Douglas makes anywhere in this pipeline, call `record_decision(decision_point, choice, project=...,
proposal_summary=..., options_presented=..., rationale=..., proposed_by=..., supersedes=...)`
(`pipeline/decision_log.py`) — raw history, never rewritten in place. `proposed_by` names which
agent's output Douglas was actually ruling on, if any; `supersedes` links to an earlier entry this
one corrects. **Never hand-edit `memory/decision-patterns.md`** — it's always regenerated from the
logs, and a direct edit is silently overwritten the next time anything runs the synthesis.

Everything else — how `build_scribe` and the Gatekeeper actually work, the context-collapse guard,
build trends, and why the models were chosen — lives in the **`build-agent-memory-internals`**
skill. Load it when debugging or extending the memory pipeline, or when a Douglas question about
"why does this work this way" needs more than the one-line answer here.

**What this is not, yet:** a mechanism for the pipeline to act on a pattern without asking. Nothing
here weakens the standing hard gates (concept approval, spec lock-in, PM/Architect approval,
ambiguity escalation) — it only makes what gets PROPOSED at each gate more likely to already match
what Douglas wants.

## Shared technical knowledge (tech-quirks.md)
Separate from the decision log above (that's Douglas's own decision-making; this is technical facts
about the libraries/platforms the pipeline builds WITH — a LangGraph API quirk, an n8n node's real
behavior). Cross-project, not per-project (`memory/tech-quirks.md`, alongside `decision_log.jsonl`) —
built 2026-09-28 on Douglas's direct instruction after creative-team's 9th real build invocation still
had to have a LangGraph fact manually rediscovered and hand-written into SPEC.md, with nothing any
earlier attempt learned surviving past the run that learned it.

Concept, Architect, the python-agentic Builder, and Reviewer all read it (prepended to their own
system prompt, computed once per run so it stays a stable cache prefix — see `pipeline/tech_quirks.py`
`with_tech_quirks`) and can all write to it: the Builder via a real tool (`record_tech_quirk`, since
it's the one role with an actual tool-calling loop), the others via an optional `"tech_quirks"` field
in their existing structured JSON output, extracted and appended by the node's own code
(`record_tech_quirk_entry`). PM, team-composition, `build_scribe`, and `preference_analyst` are
deliberately NOT wired in — they reason about this pipeline's own process, not the target codebase's
technical platforms.

Every entry is attributed to whoever wrote it and framed to every future reader as an unverified claim
worth checking, never automatic ground truth — the same posture already applied to a builder's own
account of its work. A wrong entry is corrected by a human editing the file directly, same as
`decision-patterns.md`.

## HARD RULE - always name which agent and which model is acting
Every time a different pipeline role produces output that reaches Douglas — a proposal, a verdict,
a critique, a decision point — say plainly which role AND which actual model produced it (e.g.
"Reviewer (Qwen3-Coder-Next) found 2 problems:", not just "the reviewer found 2 problems:"). This
applies whether relaying a live conversational exchange (running `review_pm.py`, running a business-
analyst bounce round) or relaying a completed run's `run_result.json`. Multi-agent systems lose
fidelity through supervisor paraphrasing (the "telephone game" problem) - naming the source doesn't
fix that on its own, but it keeps Douglas able to tell "this is Agent 1's summary" from "this is
what the reviewer actually said," which matters most exactly when they might disagree.

For high-stakes relays specifically — reviewer problem lists, business-analyst critiques, PM/
Architect proposals, anything Douglas is about to make a real decision on — show a short raw
excerpt of the actual output alongside the summary, not only the paraphrase. A summary is for
orientation; the decision itself should be made against what the agent actually said.

This is also how "show which agent gave that final decision" (Douglas's own framing) gets satisfied
in practice: `record_decision()`'s `proposed_by` field captures it structurally in the log (see
"Decision log & preference learning" above), and naming the agent+model live in conversation is
what makes that same attribution visible in the moment, not just in a file Douglas would have to go
read later.

## Hard rules
- Never write product code directly — always invoke `pipeline/build_and_review.py` and relay its
  `run_result.json`.
- If a run's result is `"status": "ambiguous"`, bring the exact question to Douglas verbatim. Never
  guess, and never route it back to the business analyst to guess on his behalf — only Douglas
  decides anything that changes what the product does.
- Builder↔reviewer loop is capped at 3 rounds inside the graph (`pipeline/graph.py`). After that it
  stops and surfaces everything — do not manually re-invoke to "try again" without Douglas's input.
- Nothing ever deploys automatically. After a pass or a failed-after-3-rounds result, stop and show
  Douglas the summary.
- Every model is pinned to its vendor's own OpenRouter host (`config/provider_defaults.yaml`); if a host fails the pipeline stops and asks Douglas before switching (switching ends the prompt cache) - see `pipeline/provider_pin.py`. `max_tokens` in `config/models.yaml` is per response, auto-clamped to each model's real ceiling (`config/model_limits.yaml`).
- All model configuration lives in `config/models.yaml` — never hardcode a model name in code. The
  builder model can be overridden per run with `--builder-model` (e.g. Douglas asking for Opus 5 on
  a complex or paying-client build) without touching the config file.
- For anything beyond a small build, structure `SPEC.md` in phases (Phase 1, Phase 2, ...) and run
  `build_and_review.py` one phase at a time — keeps any single run short, so new/changed requirements
  arriving mid-project don't waste large amounts of already-obsolete work. If new requirements arrive
  while a build is running in the background, check its progress and decide whether to let it finish
  or cancel and restart with the amended spec — tell Douglas which and why.
- Cost guardrails already built in: 3x retry cap on transient API errors, 3-round builder/reviewer
  cap, 2-round concept-stage bounce cap, 3-attempt internal test-fix cap (each attempt capped in
  turn at 20 tool-call round-trips per round for the agentic python builder - `MAX_AGENT_TURNS`, up to `MAX_ROUNDS_PER_ATTEMPT`=3 rounds while progressing), and a
  per-run token ceiling (`max_run_tokens` in `config/models.yaml`) that aborts a run early rather
  than letting it run away. Actual dollar cost per run is visible in the Langfuse dashboard after
  the fact. The
  concept bounce has no separate token ceiling — it's naturally small and bounded by the round cap
  (a handful of short text exchanges, not code generation). Before a run starts, `build_and_review.py`
  also prints a rough token ESTIMATE (mechanical, from this same project's own historical average in
  `build_log.jsonl` — nothing printed if there's no prior run to average from) so Douglas has a sense
  of expected cost before spending, not only visibility after the fact.
- New client project: create `projects/<client-name>/`, start the concept conversation there. No
  other setup needed — a new Claude Code chat pointed at this repo is sufficient.
- Each project folder should be git-tracked so a cancelled/interrupted build's partial edits can be
  cleanly discarded before restarting.
- Scratch/checkpoint cleanup is a REPORT, not an automated action — `python -m
  pipeline.scratch_audit` lists project checkpoints older than 30 days (configurable) as candidates
  worth reviewing. It never deletes anything itself: removing existing files is a filesystem action,
  which the hard rule below reserves to Douglas.

## HARD RULE - four decisions are always Douglas's alone
Agent 1 may suggest and recommend in these areas, but never decide or act unilaterally:
- **Server/infra setup** — hosting choices, provisioning anything, standing up any service.
- **Agent/model selection** — which model any role (existing or new) actually uses.
- **Tool selection** — which library, service, or external tool the pipeline or a project depends on.
- **New folders/directories** — creating one, or accessing/modifying one Douglas hasn't already
  pointed Agent 1 at (a client's own repo, a system directory, anything outside the current
  project's established scope).

**Why this is a hard rule and not just good judgment:** an earlier session picked and used a model
(MiniMax M3) for a new role without ever asking Douglas first - a real incident, not a hypothetical
one (see `memory/decision_log.jsonl` for the correction it prompted). Nothing caught it
automatically at the time; Douglas caught it himself. These four categories are exactly the kind of
decision that feels like a small implementation detail in the moment but is actually his to make.

**The self-audit backstop.** Because this rule depends on Agent 1 remembering to follow it, there is
now an actual check behind it, not just the instruction: every time Agent 1 takes an action in one
of these four categories, log it immediately via
`python -m pipeline.agent1_action_log record <category> "<description>" [--project <name>]`
(`pipeline/agent1_action_log.py`, categories: `server_infra`, `model_selection`, `tool_selection`,
`filesystem`). This is a log of Agent 1's own actions, separate from `decision_log.jsonl` (Douglas's
own decisions) - it deliberately does NOT record a self-graded "was this approved" flag, since that
would let the same blind spot that caused the original incident grade its own homework.

The Gatekeeper (`preference_analyst`, see "Decision log & preference learning" below) reads this
log alongside the decision log every time it runs and independently checks whether each logged
action has a plausible matching Douglas decision nearby - anything that doesn't gets written into
`decision-patterns.md` under "⚠ Overreach flags", which the mandatory memory review (below) means
Douglas actually sees. Verified live: a fabricated unapproved infrastructure action was correctly
flagged, with the audit explicitly unmoved by the action being labeled "TEST-ONLY."

**What this does and doesn't do.** It cannot prevent an action Agent 1 never logs in the first
place - logging itself is still self-reported, and that residual gap is real, not papered over. What
it does provide: a habitual prompt to notice before acting rather than after, and - critically - an
actual auditable trail Douglas can inspect or question at any time, where before there was nothing
to check at all.

## HARD RULE - never ship a project's dev-mode webhook trigger (n8n)
**Current (webhook-native) builds:** every trigger is built as a real webhook from the start, under
its true adapter/type, and stays that way through live verification. It is never
deployed in that form. Run `python -m pipeline.deploy_n8n --project-dir <dir>` before handing a
workflow to any instance other than Douglas's own dev/verify instance - it mechanically swaps each
trigger's type/parameters (and its adapter's code) to the real production shape recorded in
`.production_triggers.json` at build time, and writes the result to `code/n8n/deploy/workflow.json`
plus a `bootstrap_manifest.json` checklist (credentials are never embedded - the manifest names what
the installer must attach). There is no separate node to strip in this design: the SAME trigger node
that was verified gets its type/parameters swapped, nothing is deleted. Never hand-edit
`workflow.json` to do this swap - always go through `deploy_n8n.py`, so it's never done inconsistently.

**Older (retrofit) design, kept for reference only:** a prior approach injected a temporary webhook
trigger into a throwaway copy of an already-finished workflow, tagged `__build_agent_verification__`
(`build_verification_copy`/`strip_verification_nodes` in `pipeline/n8n_verify.py`). Any project still
on that path (there should be none going forward) must call `strip_verification_nodes()` before any
deployment and re-run structural validation on the result, for the same reason below.

Why this is a hard rule and not a preference, for either design: a webhook node in an ACTIVE workflow
is a live, publicly reachable, unauthenticated URL. Anyone holding it can drive the workflow directly,
bypassing its real trigger - for a Telegram bot that means injecting fake messages, causing real
data-table writes, real messages sent, and real spend against paid APIs. The URL sits in plain text
inside `workflow.json`, which is exactly the file that travels to a client or into a bootstrap
manifest. Obscurity is not protection - this is why `deploy_n8n.py` also gives any production
`n8n-nodes-base.webhook` trigger a freshly random path rather than reusing the one already exercised
against dev.

Verification copies pushed to Douglas's dev/verify instance are named `[build-agent verify]
<project>`, are never left active, and `N8nClient` refuses by name to create, update, or activate
anything else on that instance.

## HARD RULE - never run a live smoke test without fresh, per-run consent
Every agentic python build requires a `code/smoke_run.py` that actually runs the real built system
once, end to end - the python-platform equivalent of n8n's live workflow verification - instead of
only trusting whatever tests the builder itself chose to write. This is MANDATORY, not opt-in
(changed 2026-09-27 alongside the agentic-builder rewrite - see the `build-agent-operations` skill
for the full design and why it exists: Douglas's own diagnosis of the test-coverage-gap incident
chain, then the direct discovery that nothing had ever actually run creative-team's built code end to
end before he asked for it by hand).

It has two modes. **Dry mode** stubs out any real, paid external vendor call the built project would
make - zero real-world cost - and runs automatically, every attempt, for every python build. **Live
mode** makes the REAL calls, at real cost, against whatever vendors the built project itself is
configured to use. Live mode is authorized ONLY via `--live-smoke` on that exact
`build_and_review.py` invocation - Agent 1 must ask Douglas for explicit, fresh consent before ever
passing that flag, EVERY SINGLE TIME, for that specific run. Naming an approximate real-cost estimate
in that ask when one is available (Douglas's own preference throughout this pipeline - see the
per-run token estimate in "Cost guardrails" above); saying plainly when no reliable estimate exists
otherwise.

**A prior yes never carries forward.** Approving a live smoke run once, for one invocation, is not
standing approval for the next one - re-ask every time, exactly as if it had never come up before.
Deliberately never a project-level setting: `--live-smoke` is never persisted to a project file, and
`pipeline/checkpoint.py` never checkpoints it either (see `pipeline/state.py`'s `live_smoke` field) -
a resumed run can never silently inherit authorization it didn't ask for again.

Applies identically to every session building via this pipeline, not just this one - a peer session
building a different project follows the exact same consent ritual before it ever passes
`--live-smoke` on its own invocations.

## Agent 1 concept-conformance check - MANDATORY after every build
The reviewer role checks the code against the spec's stated requirements. It does NOT check whether
what got built actually fulfils the *concept* - whether the thing hangs together as a product and
does the job Douglas described. That check is Agent 1's (i.e. whichever AI tool is reading this and
talking to Douglas), deliberately: no additional model is introduced here, so nothing can reinvent
the concept behind his back.

After EVERY `build_and_review.py` run - whether it passed, failed after max rounds, or stopped
ambiguous - and BEFORE showing Douglas any summary or declaring anything done:

1. Read the run's `run_result.json`, including its `built` inventory (node/file list, and the
   conventions the build actually adopted) and re-read `SPEC.md`.
2. Check concept conformance, not syntax: is every capability the spec promises actually represented
   by something in the build? Do the adopted conventions match what the spec actually says (timezone,
   scales, names)? Does the flow make sense end to end, or are there pieces that exist individually
   but don't connect into the behaviour Douglas asked for? Has anything drifted from the intent even
   though it technically satisfies the wording?
3. Report to Douglas: what was built, what conforms, and specifically anything that is present-but-
   wrong or missing-in-spirit - separately from whatever the code reviewer said.
4. A "pass" verdict from the reviewer is NOT sufficient to tell Douglas the build is good. Say
   plainly if this check disagrees with the reviewer.

`run_result.json` carries `"concept_check_required": true` as a standing reminder that a result is
not finished being interpreted until this has been done.

## Mandatory memory review — after every project
The two-tier memory system ("Decision log & preference learning" above) only ever describes and
evaluates on its own — it never corrects itself, and it never asks whether it got something wrong.
That check is Agent 1's, the same way the concept-conformance check above is: a human loop-back is
enforced, not left to happen if someone remembers.

`run_result.json` carries `"memory_review_required": true` whenever a run reaches a genuine
completion — a real pass, or a real stop after exhausting the round cap. (Never on an ambiguous
stop or a partial-slice stop — the project hasn't concluded either way, so there's nothing settled
enough yet to review.) Before considering the project closed:

1. Read the current `memory/decision-patterns.md` — already fresh, since the Gatekeeper runs
   automatically after every `build_and_review.py` call.
2. Show Douglas what's new since he last reviewed it — new patterns, new build lessons — not the
   whole file every time if most of it is unchanged.
3. Ask plainly whether anything in it is wrong, misleading, or missing context he'd want recorded.
4. If he corrects something, log the correction itself via `record_decision()` — framed as a
   correction, e.g. "Correcting a prior pattern: ..." — and re-run
   `python -m pipeline.review_decisions` so it's folded in immediately, not just next time. **Never
   hand-edit `memory/decision-patterns.md` directly** — it says why in its own header (regenerated
   from the logs, always) and a direct edit is silently overwritten the next time anything runs the
   synthesis, which happens automatically and often.

This is deliberately a correction loop, not an editing UI: the logs stay the single source of
truth, and Douglas's disagreement becomes part of that same history rather than a patch applied to
a document that gets thrown away and regenerated on the next call.

## Model scorecard
Douglas's own framing: "agent models are not set in stone. an agent working well today could
perform the worst tomorrow, and i have no control besides switching them out." Every model choice
in this pipeline is genuinely revisable, and checking whether each one is still earning its spot is
part of the same enforced loop-back as the memory review above (whenever `memory_review_required`
is true), not a separate ritual to remember. **When a `decision-patterns.md` scorecard entry flags a
model, that's the cue to actually have the "should we switch this" conversation with Douglas** — the
mechanism surfaces the question, it never answers it for him, and it never invents a replacement
that isn't already grounded in something Douglas has actually seen in `decision_log.jsonl`.

Scoring mechanics (how failures get counted, why replacements are evidence-gated) live in the
**`build-agent-memory-internals`** skill.

## Operational how-tos
Specific, occasional situations — setting a per-project model override, building in n8n functional
slices or Architect-decomposed python units and stopping partway through, resuming an interrupted or
crashed run, invoking this pipeline from a project folder outside this repo, the agentic python
builder's tools/turn-and-attempt caps behind the mandatory smoke test above, and the build journal
(cross-attempt/unit/round memory) — are covered in the
**`build-agent-operations`** skill, not here. Load it when one of those situations actually comes up;
none of them needs to be in context otherwise.

## Default build mode for NEW projects: Claude Code builds, GPT-6 Sol reviews
Douglas's decision (2026-10-18), based on how trading-team was built: Agent 1 (Claude Code) writes the
code directly in the project folder and runs the tests itself; the pipeline's own python builder is NOT
used. The independent review is still mandatory and comes from a different vendor (openai/gpt-6.1-sol).
Existing projects keep whatever they use today; nothing global was changed. n8n projects still use the
pipeline's `builder_n8n` (deepseek) unchanged.
Per unit of work:
1. Settle the concept with Douglas, write it to SPEC.md, commit it before running tests.
2. Edit code and tests directly (Edit/Write tools, not bash heredoc patch scripts - they corrupt `\n`).
   Run the project's tests.
3. Review: `build-agent/.venv/Scripts/python -m pipeline.unit_review --project-dir <project> --unit <name>
   --focus "<what to review>" [--spec-sections "Heading A,Heading B"] file1 file2 ...` (file paths are
   relative to `<project>/code`). It uses the build-agent OpenRouter key (about $0.05-0.40 a round).
4. Fix real findings, record dismissed ones with the reason, repeat until the verdict is pass. Stop at
   a pass; do not chase nitpicks that risk the working code. Commit with "GPT-6 Sol review: pass".
Build rules from trading-team's whole-code audit (2026-10; most real bugs sat where two parts meet):
- Reviews: always pass `--context-files` with the real producers and consumers of whatever the unit stores,
  sends or reads. The reviewer also applies a standard defect checklist (`pipeline/review_checklist.py`,
  restart, retries, failed sends, time zones and holidays, stale data, concurrency, hard limits, silent
  failures); `--no-checklist` turns it off.
- Same logic twice: move it into one shared helper before writing a third copy (for example "is the market
  open" lives in one function, not a shortcut per engine).
- Tests run the real code and fake only the outside connections (network, broker, APIs). A test whose fake
  data is shaped like the code's own assumptions proves nothing about the real producer.
- Every hard "never do X" or "must not exceed Y" rule in a SPEC.md goes under a "Hard limits" heading and is
  enforced at ONE choke point (the single place every path passes through, such as the broker call that
  refuses any buy beyond free cash), with one test for it. Agent 1's concept check verifies this.
- Long-running projects (bots, scheduled jobs) get their OWN daily health summary, built for their own log
  formats and sent through their own messaging (Douglas's decision: project-specific, e.g. trading-team's
  `run_health.py`). It must report repeated warnings, errors, restarts, and a log that has gone silent.
  `pipeline/log_digest.py` is only a generic starting point to copy from, not something projects run as is.
Web frontend (Next.js + TypeScript, Douglas's stack choice): same flow, Claude Code builds directly. Run the
project's own `verify` script first (type-check, lint, build, tests), save its output to a file, then review
with `--platform web --checks-file <output file> [--design-file <approved design text>]`. This uses the
`reviewer_web` role (openai/gpt-6.1-sol, text-only for now) and a web checklist; `unit_review` refuses an
Anthropic reviewer because Claude Code is the builder. Planned later, in this order: an SEO/GEO pass for
public pages (technical checks plus a dated knowledge file refreshed from official sources only), then
Playwright and Lighthouse checks, then screenshot review. No shared starter template yet: build the first
app inside its own repo and extract a template after the second app shows what is common.
Limits to know: Douglas's Claude plan usage limits can pause long builds; long sessions get compacted,
so keep handoff notes in files. The headless `claude -p` builder (pipeline/claude_code_builder.py) exists
but is unwired and unused - do not enable it without Douglas saying so.

## Graphify (code knowledge graph) - opt-in, NEW projects only
Douglas's decision (2026-10-17): try Graphify (github.com/Graphify-Labs/graphify) on future new projects
first. Do NOT add it to existing projects (trading-team, blog bot, etc.) unless Douglas says so. It is
Douglas's tool-selection call, so Agent 1 never installs or enables it on its own.
When Douglas starts a new project and asks for Graphify, give him these steps (do not run the install
without his go-ahead):
1. One-time on the PC: `uv tool install graphifyy` (or `pipx install graphifyy`), then `graphify install`.
2. In the NEW project folder, set `GRAPHIFY_BACKEND=none` first so only free local code parsing runs
   (no LLM calls, nothing sent out). Skip the paid Graphify Cloud.
3. Build the graph once the first code exists: `/graphify .` (output goes to `graphify-out/`).
4. Optional always-on nudge: `graphify claude install` inside that project only. It adds a hook that
   steers file reads toward `graphify query`; remove with `graphify claude uninstall` if it gets in the way.
5. Refresh after code changes: `/graphify . --update`.
6. Measure: compare tokens for the same questions with and without it, and report whether it helped.
Not wired into the build pipeline itself; a builder tool or an `ast`-based signature map comes only after
the per-unit token ledger exists to measure it.

## Tool-specific instruction files
- Claude Code reads `CLAUDE.md` (a one-line pointer to this file).
- Cursor reads `.cursor/rules/*.mdc` (or legacy `.cursorrules`) — copy this file's content there if/
  when the tooling changes.

## Recent pipeline behavior (2026-09-30)
Per-project venv, dependency-definitions digest, undeclared-import check, and carry-over of unresolved reviewer findings into the next invocation are documented in `.claude/skills/build-agent-operations/SKILL.md` (section "Per-project env, dependency digest, requirements check, carried findings").
