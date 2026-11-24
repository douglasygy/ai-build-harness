from __future__ import annotations

import time
from typing import Callable, Optional, TypeVar

from .constants import MAX_RETRY_ATTEMPTS

T = TypeVar("T")


def with_retries(fn: Callable[[], T], max_attempts: int = MAX_RETRY_ATTEMPTS, base_delay: float = 2.0) -> T:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # deliberately broad: any transient API failure should retry, then fail loud
            last_exc = exc
            if attempt == max_attempts:
                break
            time.sleep(base_delay * attempt)
    raise RuntimeError(f"Failed after {max_attempts} attempts: {last_exc}") from last_exc


class EmptyResponseError(Exception):
    """Raised when a model call returns HTTP success with empty/whitespace content - not an
    exception `with_retries` alone would ever see or retry against.

    Real incident (reported by a peer session, confirmed against this repo's own code
    2026-09-26): review_concept.py's business_analyst call returned HTTP 200 with empty content
    on a round-2 call, mid-bounce. with_retries only catches thrown exceptions, so the empty
    response sailed straight through to extract_json_object, which raised on nothing to parse -
    caught only at the OUTER level of the whole multi-round bounce, discarding every round
    already completed and paid for (15,613 tokens, in the reported case) instead of just retrying
    the one bad call. A live diagnostic call with the identical prompt immediately afterward
    returned a clean 2,787-char JSON response - confirming an intermittent provider-side flake,
    not a ceiling or prompt problem (max_tokens was nowhere near hit either time).
    """


def with_retries_nonempty(fn: Callable[[], T], max_attempts: int = MAX_RETRY_ATTEMPTS, base_delay: float = 2.0) -> T:
    """Same as with_retries, but ALSO treats an empty/whitespace `.content` result as a failure
    worth retrying - not just a thrown exception. Every current caller of with_retries in this
    repo is an LLM invocation expecting real text back, so this is the version every LLM call
    site should use instead of bare with_retries; see EmptyResponseError for why.
    """

    def checked() -> T:
        response = fn()
        content = getattr(response, "content", None)
        if not (content or "").strip():
            raise EmptyResponseError(
                "Model call returned HTTP success but empty/whitespace content "
                f"(response_metadata={getattr(response, 'response_metadata', None)!r})"
            )
        return response

    return with_retries(checked, max_attempts=max_attempts, base_delay=base_delay)


def with_retries_tool_call(fn: Callable[[], T], max_attempts: int = MAX_RETRY_ATTEMPTS, base_delay: float = 2.0) -> T:
    """Same problem as with_retries_nonempty, different fix - for a tool-bound LLM call (see
    pipeline/agent_tools.py, pipeline/nodes/builder_node.py's _build_python_agentic), where a
    genuinely valid response often has EMPTY `.content` with the real payload in `.tool_calls`
    instead. with_retries_nonempty's plain "is content empty" check would misread every normal tool
    call as a failed/empty response and retry it 3x before giving up for real - this checks BOTH:
    empty content is only a real failure when there are also no tool_calls.

    Deliberately a separate function, not a change to with_retries_nonempty itself - every existing
    non-tool-calling call site (9 of them, as of this writing) must keep working exactly as before.
    """

    def checked() -> T:
        response = fn()
        content = getattr(response, "content", None)
        tool_calls = getattr(response, "tool_calls", None)
        if not (content or "").strip() and not tool_calls:
            raise EmptyResponseError(
                "Model call returned HTTP success but empty/whitespace content and no tool_calls "
                f"(response_metadata={getattr(response, 'response_metadata', None)!r})"
            )
        return response

    return with_retries(checked, max_attempts=max_attempts, base_delay=base_delay)
