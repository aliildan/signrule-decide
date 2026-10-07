"""Build Kev/Jev training requests: `state` + typed `questions` with `label` (design-02 §3, §7).

Masked questions are simply absent. Kev's `load_records` accepts exactly this shape plus `_meta`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from signrule.common.paths import CONFIG_DIR

STATE_KEYS = (
    "jurisdiction",
    "legal_form",
    "signature_rule",
    "procuration_rule",
    "roles",
    "role_notes",
)


@dataclass(frozen=True)
class QuestionsConfig:
    questions: dict[str, dict[str, Any]]

    def keys(self, qid: str) -> list[str]:
        q = self.questions[qid]
        if q["type"] == "choice":
            return list(q["criteria"])
        if q["type"] == "score":
            return [str(i) for i in range(len(q["criteria"]))]
        return ["false", "true"]


@cache
def load_questions(path: Path = CONFIG_DIR / "questions.yaml") -> QuestionsConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for qid, q in raw.items():
        if q["type"] == "choice":
            q["criteria"] = {str(k): v for k, v in q["criteria"].items()}
        if q["type"] not in ("choice", "noul", "score"):
            raise ValueError(f"questions.yaml {qid}: bad type {q['type']}")
    return QuestionsConfig(raw)


def build_state(
    jurisdiction: str,
    legal_form: str | None,
    signature_rule: str | None,
    procuration_rule: str | None,
    roles: list[tuple[str, int]],
    role_notes: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """State dict in fixed key order; empty fields are omitted (Kev renders `key: value` lines)."""
    values: dict[str, Any] = {
        "jurisdiction": jurisdiction,
        "legal_form": legal_form,
        "signature_rule": signature_rule,
        "procuration_rule": procuration_rule,
        "roles": [{"role": r, "count": n} for r, n in roles] or None,
        "role_notes": [{"role": r, "note": t} for r, t in (role_notes or [])] or None,
    }
    return {k: values[k] for k in STATE_KEYS if values[k] not in (None, "")}


def to_request(
    state: dict[str, Any],
    answers: dict[str, Any],
    questions: QuestionsConfig,
    *,
    source: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Labelled request. `answers` holds only determined questions (others are masked)."""
    qs: dict[str, Any] = {}
    for qid, label in answers.items():
        if qid not in questions.questions:
            raise KeyError(f"answer for unknown question {qid!r}")
        q = questions.questions[qid]
        if q["type"] == "choice" and label not in q["criteria"]:
            raise ValueError(f"{qid}: label {label!r} not among criteria")
        if q["type"] == "noul" and not isinstance(label, bool):
            raise TypeError(f"{qid}: noul label must be bool")
        if q["type"] == "score" and not (
            isinstance(label, int) and 0 <= label < len(q["criteria"])
        ):
            raise ValueError(f"{qid}: score label must be a level index")
        entry = {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
        qs[qid] = {**entry, "label": label, "src": f"{source}_{qid}"}
    out: dict[str, Any] = {"state": state, "questions": qs}
    if meta:
        out["_meta"] = meta
    return out
