"""Daily health digest for long-running projects (bots, scheduled jobs): turns log files into a short summary.

Origin: trading-team's audit found warnings logged every cycle that nobody read. This module only READS logs and
prints text; sending it (Telegram, email) is the project's own job, so it stays a template with no network and no
secrets. Douglas decided health summaries are project-specific (trading-team has its own run_health.py): copy
this as a starting point and adapt it to the project's log formats instead of running it unchanged.

usage:  python -m pipeline.log_digest --logs bot.log other.log [--hours 24] [--min-repeats 3]
                                       [--restart-marker "bot starting"] [--retry-marker "retry"]

Log lines are expected to start with a "YYYY-MM-DD HH:MM:SS" (or "T") timestamp and contain a level word
(WARNING, ERROR, CRITICAL), as python's logging does by default. Timestamps are compared with this machine's
local time, so run it on the same machine and time zone the project logs in. Lines without a timestamp are
ignored. A log with NO lines in the window is itself reported: silence usually means the job stopped.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})")
_LEVEL_RE = re.compile(r"\b(WARNING|ERROR|CRITICAL)\b")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_SYMBOL_RE = re.compile(r"\b[A-Z]{2,6}\b")


def _normalize(message: str) -> str:
    """Group repeats of one message that differ only in numbers, quoted values or ALL-CAPS symbols (tickers)."""
    return _NUMBER_RE.sub("N", _SYMBOL_RE.sub("S", _QUOTED_RE.sub("'X'", message))).strip()[:160]


def build_digest(log_paths, hours: float = 24, min_repeats: int = 3, restart_marker: str = "",
                 retry_marker: str = "retry", now: datetime | None = None) -> str:
    now = now or datetime.now()
    since = now - timedelta(hours=hours)
    lines_out = [f"Health digest, last {hours:g}h up to {now:%Y-%m-%d %H:%M}"]
    problems = 0
    for path in map(Path, log_paths):
        if not path.is_file():
            lines_out.append(f"- {path.name}: LOG FILE MISSING")
            problems += 1
            continue
        in_window = 0
        issues: Counter = Counter()
        levels: Counter = Counter()
        restarts = retries = 0
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _TS_RE.match(raw)
            if not m:
                continue
            ts = datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}")
            if ts < since or ts > now:
                continue
            in_window += 1
            low = raw.lower()
            if restart_marker and restart_marker.lower() in low:
                restarts += 1
            level = _LEVEL_RE.search(raw)
            if level:
                levels[level.group(1)] += 1
                issues[_normalize(raw[level.end():])] += 1
                if retry_marker and retry_marker.lower() in low:
                    retries += 1
        if in_window == 0:
            lines_out.append(f"- {path.name}: NO LOG LINES in the window - the job may have stopped")
            problems += 1
            continue
        summary = ", ".join(f"{levels[k]} {k.lower()}" for k in ("CRITICAL", "ERROR", "WARNING") if levels[k]) or "no warnings"
        extra = (f", {restarts} restarts" if restart_marker else "") + (f", {retries} retry problems" if retries else "")
        lines_out.append(f"- {path.name}: {in_window} lines, {summary}{extra}")
        repeated = [(msg, n) for msg, n in issues.most_common() if n >= min_repeats]
        for msg, n in repeated[:10]:
            lines_out.append(f"    repeated {n}x: {msg}")
        problems += len(repeated) + (1 if levels["ERROR"] or levels["CRITICAL"] else 0) + (1 if restarts > 1 else 0)
    lines_out.append("Needs a look." if problems else "Nothing needs attention.")
    return "\n".join(lines_out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--restart-marker", default="", help="text in the log line the process writes at start-up")
    parser.add_argument("--retry-marker", default="retry")
    args = parser.parse_args(argv)
    print(build_digest(args.logs, args.hours, args.min_repeats, args.restart_marker, args.retry_marker))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
