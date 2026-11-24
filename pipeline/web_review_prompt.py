"""Web-frontend addendum appended to the shared reviewer SYSTEM_PROMPT for `unit_review --platform web`.

The shared prompt supplies the JSON format, severity rules and the mechanical anti-invention checks.
This addendum only adds what to look for in a Next.js + TypeScript unit, and keeps the same rule that
a finding must be a real, demonstrable defect, not a taste or best-practice opinion. SEO/GEO scoring is a
separate later pass and is deliberately not covered here.
"""

WEB_REVIEW_ADDENDUM = """

WEB FRONTEND UNIT (Next.js + TypeScript). The code you were given is web UI or web-app code, not python.
Ignore the python-specific wording above about pytest, stage stubs and smoke_run.py. Everything else above
still applies: report only real, demonstrable defects you can point to in the text you were given, with
the same JSON format, severities and verbatim-quote rules. Demonstrable web defects to look for:
- Security (must_fix): a secret, API key or server-only value exposed to the browser (a NEXT_PUBLIC_ variable
  holding a key, a server-only module imported into a "use client" file); unsanitised user input rendered
  through dangerouslySetInnerHTML or built into a URL, query or HTML string; an API route or server action that
  acts on user data without any authentication or authorisation check the SPEC requires.
- Correctness (must_fix when the feature would not work): a server component using client-only hooks or
  browser APIs, or a client component doing server-only work; a data fetch whose result shape the UI reads
  differently from what the real endpoint or type produces; unhandled loading, error or empty states that the
  SPEC names; a type escape (`any`, `as unknown as`, @ts-ignore) hiding a real mismatch.
- Accessibility (should_fix, must_fix when the SPEC demands it): an <img> without alt text, a form control
  without a label, a clickable non-button element without keyboard handling, an icon-only control without an
  accessible name, colour used as the only carrier of meaning where the code makes that visible.
- Responsive layout (should_fix): a fixed pixel width or height that provably breaks a narrow mobile viewport.
- Design match: only when an APPROVED DESIGN section is given. Report a mismatch with it as a
  "spec_requirement" problem quoting the design text verbatim; never invent design requirements.
- Tests: user-facing logic with no test at all is a "missing_file" problem only when it names a real anchor
  file, exactly as for python.
If a "VERIFY OUTPUT" section is given (type-check, lint, build and test results from the builder's machine), it
is mechanical evidence of what those commands printed, not proof the code is correct. Any failing command in it
is a must_fix problem naming the file the error points at. Never report visual taste, naming, folder layout or
framework best-practice opinions.
"""
