MAX_ROUNDS = 3
MAX_INTERNAL_FIX_ATTEMPTS = 3
MAX_RETRY_ATTEMPTS = 3
MAX_CONCEPT_ROUNDS = 2
MAX_TEAM_COMPOSITION_ROUNDS = 2
# Tool-call round-trips within ONE agentic builder attempt (see pipeline/nodes/builder_node.py's
# _build_python_agentic) - distinct from MAX_INTERNAL_FIX_ATTEMPTS above, which caps how many FRESH
# attempts get made.
#
# Lowered from 20 (2026-09-27, a real transcript obtained via this same incident's own DEBUG
# logging): 20 turns let ONE attempt consume the ENTIRE run's token budget by itself (14 turns hit
# 421,547 tokens, over_budget fired mid-turn-15) - because every turn resends the whole accumulated
# conversation, cost per turn compounds fast, not linearly. Attempts 2 and 3 never got a chance to
# even start. 10 keeps a single attempt from being able to exhaust the whole run on its own, and
# forces a genuinely stuck attempt into a FRESH, short-context retry sooner rather than later -
# the real throwaway-project success used only 3 turns, so this is not a tight ceiling for a task
# that's actually converging.
# 2026-09-30 (Douglas): lowered 10 -> 3.
# 2026-10-02 (Douglas): "update max turns back to before. 20. For all projects in the pipeline." The 3-turn cap
# caused repeated out-of-turns restarts and re-reads (units cost 1-5M tokens each). Rounds-while-progressing is kept.
MAX_AGENT_TURNS = 20
# 2026-09-30 (Douglas): MAX_AGENT_TURNS is one ROUND. An attempt continues in the SAME conversation
# (append-only, so the prompt cache holds) for another round only if the previous round made a
# substantive write/edit; a round with no progress ends the attempt. At most this many rounds.
MAX_ROUNDS_PER_ATTEMPT = 3

# Replaces the single cumulative max_run_tokens ceiling ENTIRELY, for _build_python_agentic only
# (2026-09-28, Douglas's exact numbers, discussed and relayed via video-generator-78 - n8n and every
# other role keep using config/models.yaml's max_run_tokens/over_budget() exactly as before, this is
# a python-agentic-builder-specific change). Old model: one total_tokens counter, shared across all
# 3 internal attempts, all builder-reviewer rounds, AND every resumed CLI invocation of the same
# checkpoint - it only grows, forcing Douglas to keep manually raising it as real, legitimate work
# added up (500K -> 1M in one day). Douglas's own framing: "its damn dumb that attempt 2 doesn't
# auto learn attempt 1 lesson... imagine a human coded once, redo but wiped his brain entirely" - a
# fresh attempt SHOULD get a fresh budget, not inherit exhaustion from a fresh conversation it never
# even saw.
#
# MAX_TOKENS_PER_ATTEMPT resets to 0 at the START of every individual attempt (not inherited from
# the previous one) - checked against attempt_tokens, ending JUST that attempt (verification still
# runs on whatever's on disk, same as running out of turns) rather than the whole invocation.
#
# MAX_TOKENS_PER_INVOCATION resets to 0 at the START of every invocation of _build_python_agentic -
# whether that's the very first loop-1 pass on a fresh checkpoint, or a loop-2 round's builder
# response after reviewer feedback - checked against invocation_tokens, ending the WHOLE invocation
# (this run's real backstop is now loop 2's own existing MAX_ROUNDS above, not a token count).
# Deliberately COMPUTED, not an independent literal, so it can never drift out of sync with
# MAX_INTERNAL_FIX_ATTEMPTS if that's ever tuned later - checked against real evidence before
# agreeing these numbers are sane: the one run that actually succeeded the same day spent
# 269K/288K/281K across its 3 attempts, all comfortably under 500K each.
#
# The existing lifetime-cumulative total_tokens (state["total_tokens"], persisted across checkpoint
# resumes) is NOT retired - it keeps accumulating exactly as before, for cost-visibility/reporting
# (Langfuse, build_log.jsonl, run_result.json) only. It is no longer what any stop-check compares
# against for this role.
MAX_TOKENS_PER_ATTEMPT = 500_000
MAX_TOKENS_PER_INVOCATION = MAX_TOKENS_PER_ATTEMPT * MAX_INTERNAL_FIX_ATTEMPTS

# Reviewer-only reset-based ceiling (2026-09-29, Douglas's direction relayed via
# video-generator-78, extending the same reset-based principle above to the reviewer role). The
# NUMBERS below are Douglas's own confirmed figures (relayed via video-generator-78), replacing
# Agent 1's initial proposal of 1.5M/4.5M - same derivation (PER_ROUND * MAX_ROUNDS), his own values.
#
# Real incident this fixes: reviewer_node.py's own stop-check compared the SAME lifetime-cumulative
# total_tokens the builder also writes into against max_run_tokens (config/models.yaml, 1,000,000
# for creative-team) - so once a real 6-unit build legitimately cost ~2.24M tokens BEFORE the
# reviewer's first call, the reviewer's own designed-in MAX_ROUNDS=3 was dead on arrival: it always
# got exactly 1 round no matter what it found, because the ceiling was already blown by cost that
# had nothing to do with reviewing. The old lifetime-cumulative total_tokens field is NOT retired -
# see MAX_TOKENS_PER_INVOCATION's own comment above for why (cost-visibility/reporting only, never a
# stop-check input for a role with its own dedicated ceiling).
#
# The reviewer's actual call pattern is a single LLM call per round (plus up to
# MAX_INTERNAL_FIX_ATTEMPTS on-the-spot retries for a malformed/unverifiable response within that
# SAME round, re-reading the same codebase text each time - no multi-turn tool loop like the
# builder's own per-attempt problem). That shape doesn't need the builder's two-tier per-attempt/
# per-invocation split: one dedicated, reviewer-only CUMULATIVE ceiling across all of this run's
# rounds is enough - reset to 0 at the start of a fresh run exactly like total_tokens itself is
# (never reset just by a checkpoint resume, for the same reason a resume can't dodge either ceiling
# for free).
#
# MAX_REVIEWER_TOKENS_PER_ROUND is a generous single-round baseline, used only to DERIVE the run
# figure below - not separately enforced mid-round (no real incident calls for that, and the
# reviewer's own MAX_INTERNAL_FIX_ATTEMPTS retry count already bounds a single round's blast radius
# by count). Real evidence Agent 1's initial proposal was grounded in: creative-team's real round 1
# on a genuine 6-unit build cost roughly 450K tokens for that one call - almost entirely the resent
# codebase text, which grows with the project (inherent to "the reviewer reads the whole codebase
# fresh every round," unchanged and not part of this fix).
MAX_REVIEWER_TOKENS_PER_ROUND = 1_000_000
MAX_REVIEWER_TOKENS_PER_RUN = MAX_REVIEWER_TOKENS_PER_ROUND * MAX_ROUNDS

# Sprint-style capacity for one python build unit (2026-09-30, Douglas: "bite sized small units,
# think Jira sprints - per sprint only X effort and tasks"). These are Agent 1's starting heuristics
# originally derived from MAX_AGENT_TURNS=3 (3 tool round trips per builder attempt; a turn may batch several
# writes) - Douglas may retune them. Enforced mechanically in review_architect.validate_build_units.
UNIT_MAX_EFFORT_POINTS = 3  # effort scale 1 (one small file) .. 3 (the most one unit may hold)
UNIT_MAX_FILES = 3
UNIT_MAX_EXPORTS = 5

# 2026-09-30 (Douglas): rewriting an earlier message mid-attempt changes the start of the prompt, so
# the provider's prompt cache (cheap re-reads of everything already sent) misses on every later turn.
# False = leave earlier tool results untouched within an attempt (each new attempt starts fresh anyway).
PRUNE_STALE_EXPLORATION_MID_ATTEMPT = False

# Claude Code headless builder backend (trading-team trial, Douglas 2026-10-03). No OpenRouter cost
# (Douglas's Claude subscription), so these bound runtime and turns, not dollars. One attempt = one
# `claude -p` session; the pipeline's independent verification runs after it exactly as for the
# OpenRouter builder.
CLAUDE_CODE_MAX_TURNS = 40
CLAUDE_CODE_TIMEOUT_SEC = 1800
