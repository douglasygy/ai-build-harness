"""Standard defect checklist appended to the shared reviewer SYSTEM_PROMPT by `unit_review` (on by default).

Origin: trading-team's whole-code audit (2026-10) found most real bugs in the same few groups, none of which a
single-file review catches: two parts disagreeing about what is saved, the same rule copied with different
shortcuts, tests built from fake data shaped like the code's own assumptions, and failures that were logged but
never acted on. Each item below is worded as a demonstrable defect, not a best-practice opinion, so it stays
inside the shared prompt's rule that a finding must be real. An item that does not apply to the unit produces
no finding.
"""

CHECKLIST_ADDENDUM = """

STANDARD DEFECT CHECKLIST (applies whatever this SPEC says; report an item only when you can point to the
exact code that has the defect, never because the topic merely exists). For each item, look at this unit and at
any CONTEXT FILES it touches:
- Boundaries: the unit saves, sends or returns something that another file reads (a stored field, a column, a
  dict key, a unit of measure, a status string), or reads something another file produces, and the two disagree
  about its name, shape, meaning or units. Check the reader against the real writer, not against the tests.
- Duplicated rules: the same decision (is the market open, is this a business day, is this amount allowed)
  implemented in more than one place with different shortcuts, so one copy handles a case the others get wrong.
- Restart: if the process stops halfway through this work and starts again, state is left half-written, a step
  runs twice, or a stored position is read as if the work had finished.
- Retries and duplicates: a retried, repeated or replayed call can act twice (a second order, a second message,
  a second row) because nothing identifies the repeat.
- Failed sends: a message, request or write can fail and the failure is swallowed, only logged, or leaves the
  caller believing it succeeded.
- Time: a time zone, daylight-saving shift, weekend, public holiday or half-day is handled by a fixed constant
  (always 16:00, weekdays only) where the unit's behaviour depends on it.
- Stale data: a value read earlier is used after it can have changed (a cached price or balance, a list read
  before a loop that modifies it) or is used with no check of how old it is where age matters.
- Concurrency: two jobs, threads or processes can touch the same file, row or counter with no lock, transaction
  or single owner.
- Hard limits: a "never do X" or "must not exceed Y" rule in the SPEC is enforced in only some of the paths that
  can do X, instead of at the one place all of them pass through. Quote the SPEC line and name the path that
  gets round it.
- Silent failures: a warning or error condition is logged on every cycle but nothing changes (no counter, no
  alert, no stop), so a persistent problem would go unnoticed.
Never report an item as "not tested" unless the SPEC demands that behaviour; report the defect itself.
"""
