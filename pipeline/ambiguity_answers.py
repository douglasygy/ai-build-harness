"""Douglas's answers to AMBIGUOUS questions, fed back into later runs (trading-team, 2026-09-30).

Stored append-only in <project>/.ambiguity_answers.jsonl so a ruling (for example "FALSE POSITIVE,
only fix the error text") keeps applying to every later run instead of being re-raised. Each record:
{timestamp, question, answer, dismiss: [substrings]}. `dismiss` substrings drop carried findings
containing them (case-insensitive); the answer text itself is shown to the builder and reviewer as an
authoritative ruling.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

FILENAME = ".ambiguity_answers.jsonl"


def record_answer(project_dir, question: str, answer: str, dismiss: list | None = None) -> dict:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": (question or "").strip(),
        "answer": answer.strip(),
        "dismiss": [d.strip() for d in (dismiss or []) if d and d.strip()],
    }
    with open(Path(project_dir) / FILENAME, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def load_answers(project_dir) -> list:
    path = Path(project_dir) / FILENAME
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict) and str(e.get("answer", "")).strip():
            out.append(e)
    return out


def format_rulings(answers: list) -> list:
    return [
        (f"Question: {a['question']}\n  " if a.get("question") else "") + f"Ruling: {a['answer']}"
        for a in answers
    ]


def rulings_block(rulings: list) -> str:
    if not rulings:
        return ""
    return (
        "HUMAN RULINGS (Douglas's answers to earlier AMBIGUOUS questions on this project - "
        "authoritative; follow them, and do not re-raise anything they rule out):\n"
        + "\n".join(f"- {r}" for r in rulings)
        + "\nA ruling IS the answer to the question it followed. NEVER reply AMBIGUOUS asking about a "
        "finding or question a ruling already covers: do exactly what the ruling says, and if that "
        "action is already done in the code, treat the finding as resolved and finish. If a ruling "
        "retires or removes a feature/module, tests that cover only that retired feature are retired "
        "with it: delete them or rewrite them to the replacement path the ruling names - the 'do not "
        "weaken or delete tests' rule does not apply to them."
    )


def drop_dismissed(findings: list, answers: list) -> list:
    needles = [d.lower() for a in answers for d in (a.get("dismiss") or [])]
    if not needles:
        return list(findings)
    return [f for f in findings if not any(n in f.lower() for n in needles)]
