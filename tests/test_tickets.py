"""Tests for scripts/tickets.py (plan markdown -> GitHub issue specs)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("tickets", ROOT / "scripts/tickets.py")
assert _spec and _spec.loader
tickets = importlib.util.module_from_spec(_spec)
sys.modules["tickets"] = tickets
_spec.loader.exec_module(tickets)

PLAN = """# Plan 02 — Norway ingest (Enhetsregisteret + Fullmakttjenesten)

**Goal:** Every in-scope entity cached.

**Area:** ingest

### Task 0: Client and stripping (done in bootstrap, see implement-00)

**Labels:** area:ingest, type:task
**Acceptance:** tests green.

### Task 3: Entities snapshot (first live fetch)

**Labels:** area:ingest, type:task, user-action
**Acceptance:** manifest exists.

- [ ] **Step 1:** secret step detail that must not reach GitHub
"""

ROADMAP = """## Decisions waiting for you

| ID | Question | Recommendation |
|---|---|---|
| D1 | Build on Kev? | yes |
| O3 | `ambiguity` labels: (a) or (b)? | a |

## User actions
"""


def parse(text: str = PLAN):
    return tickets.parse_plan(text, Path("docs/plans/plan-02-norway-ingest.md"))


def test_parse_plan_header():
    plan = parse()
    assert plan.number == 2
    assert plan.milestone == "P02 Norway ingest"
    assert plan.goal == "Every in-scope entity cached."


def test_parse_tasks_with_labels_and_acceptance():
    plan = parse()
    assert [t.key for t in plan.tasks] == ["P02.T0", "P02.T3"]
    t3 = plan.tasks[1]
    assert t3.title == "P02.T3 Entities snapshot (first live fetch)"
    assert t3.labels == ["area:ingest", "type:task", "user-action", "plan:P02"]
    assert t3.acceptance == "manifest exists."
    assert not t3.done


def test_bootstrap_tasks_are_marked_done():
    t0 = parse().tasks[0]
    assert t0.done and t0.title == "P02.T0 Client and stripping"


def test_issue_body_has_acceptance_and_plan_path_but_no_steps():
    body = tickets.issue_body(parse(), parse().tasks[1])
    assert "manifest exists." in body
    assert "docs/plans/plan-02-norway-ingest.md" in body
    assert "secret step detail" not in body


def test_parse_decisions_table():
    ds = tickets.parse_decisions(ROADMAP)
    assert [d.key for d in ds] == ["D1", "O3"]
    assert ds[0].title == "Decision D1: Build on Kev?"
    assert ds[1].recommendation == "a"
