r"""Single, correct implementation of "pull the JSON object out of a model's raw text response" -
every pipeline role that asks a model for ONLY a JSON object needs this, and it used to be
duplicated (with an identical bug) across eight separate files: build_scribe.py,
review_decisions.py, review_architect.py, review_pm.py, review_team_composition.py,
review_concept.py, and nodes/builder_node.py + nodes/reviewer_node.py.

Real incident this fixes: every one of those eight used ``re.search(r"\{.*\}", text, re.DOTALL)`` -
greedy, so it matches from the FIRST "{" all the way to the LAST "}" anywhere in the response, with
no understanding of nested braces or string content. build_scribe.py is handed a whole
run_result.json as input and asked to summarize it - if its response ever echoes or quotes any part
of that back (a stray "{" or "}" appearing anywhere after the real JSON object), the regex silently
swallowed everything through the last such brace in the response, producing a malformed blob instead
of the small intended object. Confirmed in practice: "Expecting ',' delimiter: line 259 column 6"
deep inside what should have been a ~5-field JSON object.

The fix: a real JSON parser (json.JSONDecoder.raw_decode), not a regex - it parses exactly the
valid object starting at the first "{" and stops there, correctly ignoring whatever text follows,
because it actually tracks nesting/string-escaping instead of guessing from brace positions.
"""
from __future__ import annotations

import json


def extract_json_object(text: str, context: str = "response") -> dict:
    """Parses the first valid JSON object found in `text`, ignoring any leading/trailing text
    around it. Raises ValueError (never json.JSONDecodeError directly) if no `{` exists at all, or
    if what follows the first `{` isn't valid JSON - callers already handle ValueError uniformly
    (typically: feed the error back to the model and ask it to retry).
    """
    start = text.find("{")
    if start == -1:
        raise ValueError(f"Expected a JSON object in the {context}, got: {text[:500]}")
    try:
        obj, _end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Expected a JSON object in the {context} ({exc}; response is {len(text)} chars, "
            f"likely truncated if it stops mid-structure). Start: {text[:300]!r} ... End: {text[-300:]!r}"
        ) from exc
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object in the {context}, got a {type(obj).__name__}: {text[:500]}")
    return obj
