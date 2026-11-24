from datetime import datetime

from pipeline.log_digest import build_digest

NOW = datetime(2026, 10, 27, 12, 0, 0)


def _log(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_repeated_warnings_are_grouped_and_old_lines_ignored(tmp_path):
    p = _log(tmp_path, "bot.log", [
        "2026-10-27 09:00:01,100 WARNING feed: stale price for AAPL age 31s",
        "2026-10-27 09:05:01,100 WARNING feed: stale price for MSFT age 45s",
        "2026-10-27 09:10:01,100 WARNING feed: stale price for NVDA age 52s",
        "2026-10-20 09:00:01,100 ERROR old: ignored, outside the window",
        "2026-10-27 09:11:00,000 INFO fine",
    ])
    out = build_digest([p], now=NOW)
    assert "3 warning" in out and "repeated 3x: feed: stale price for S age Ns" in out
    assert "ignored" not in out and "Needs a look." in out


def test_silent_and_missing_logs_are_reported(tmp_path):
    quiet = _log(tmp_path, "quiet.log", ["2026-10-01 09:00:00 INFO long ago"])
    out = build_digest([quiet, tmp_path / "gone.log"], now=NOW)
    assert "NO LOG LINES" in out and "LOG FILE MISSING" in out


def test_clean_log_and_restart_count(tmp_path):
    p = _log(tmp_path, "ok.log", ["2026-10-27 08:00:00 INFO bot starting", "2026-10-27 08:01:00 INFO working"])
    out = build_digest([p], restart_marker="bot starting", now=NOW)
    assert "1 restarts" in out and "Nothing needs attention." in out
