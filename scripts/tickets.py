"""Mirror plan tasks and pending decisions into GitHub issues.

    uv run python scripts/tickets.py sync [--dry-run] [--repo OWNER/NAME] [--project TITLE]

Plans (`docs/plans/plan-NN-*.md`) stay the source of truth: one issue per `### Task N:` heading,
one milestone per plan, labels from the task's `**Labels:**` line plus `plan:PNN`. Decisions come
from the "Decisions waiting for you" table in `docs/ROADMAP.md`. Issues carry the goal, the
acceptance line and the local plan path, never the plan's step text. Re-running is idempotent:
issues are matched by their key prefix (`P02.T3`, `D1`).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLANS_DIR = ROOT / "docs" / "plans"
ROADMAP = ROOT / "docs" / "ROADMAP.md"
FOOTER = "_Mirrored by `scripts/tickets.py`. Edit the plan, then re-sync._"

LABEL_COLORS = {
    "area:": "1d76db",
    "type:": "5319e7",
    "plan:": "c5def5",
    "user-action": "d93f0b",
    "decision": "fbca04",
}

H1 = re.compile(r"^# Plan (\d+) — (.+)$", re.M)
TASK = re.compile(r"^### Task (\d+): (.+)$", re.M)
BOOTSTRAP = re.compile(r"\s*\(done in bootstrap[^)]*\)\s*$")


@dataclass
class Task:
    key: str
    title: str
    labels: list[str]
    acceptance: str
    number: int
    done: bool = False


@dataclass
class Plan:
    number: int
    title: str
    milestone: str
    goal: str
    path: Path
    tasks: list[Task] = field(default_factory=list)


@dataclass
class Decision:
    key: str
    title: str
    question: str
    recommendation: str


def _field(block: str, name: str) -> str:
    m = re.search(rf"^\*\*{name}:\*\*\s*(.+)$", block, re.M)
    return m.group(1).strip() if m else ""


def parse_plan(text: str, path: Path) -> Plan:
    h1 = H1.search(text)
    if not h1:
        raise ValueError(f"{path}: missing '# Plan NN — Title' heading")
    number = int(h1.group(1))
    title = h1.group(2).strip()
    short = re.split(r"[:(]", title, maxsplit=1)[0].strip()
    plan = Plan(number, title, f"P{number:02d} {short}", _field(text, "Goal"), path)
    heads = list(TASK.finditer(text))
    for i, m in enumerate(heads):
        block = text[m.end() : heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        n = int(m.group(1))
        raw_title = m.group(2).strip()
        done = bool(BOOTSTRAP.search(raw_title))
        key = f"P{number:02d}.T{n}"
        labels = [s.strip() for s in _field(block, "Labels").split(",") if s.strip()]
        plan.tasks.append(
            Task(
                key=key,
                title=f"{key} {BOOTSTRAP.sub('', raw_title)}",
                labels=[*labels, f"plan:P{number:02d}"],
                acceptance=_field(block, "Acceptance"),
                number=n,
                done=done,
            )
        )
    return plan


def parse_decisions(text: str) -> list[Decision]:
    m = re.search(r"^## Decisions waiting for you\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not m:
        return []
    out = []
    for line in m.group(1).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] in ("ID", "") or set(cells[0]) <= {"-"}:
            continue
        key, question, rec = cells
        out.append(Decision(key, f"Decision {key}: {question}", question, rec))
    return out


def issue_body(plan: Plan, task: Task) -> str:
    rel = plan.path.relative_to(ROOT) if plan.path.is_absolute() else plan.path
    return (
        f"**Plan:** `{rel}` → Task {task.number} (private, local)\n"
        f"**Plan goal:** {plan.goal}\n\n"
        f"**Acceptance:** {task.acceptance or '(see plan)'}\n\n---\n{FOOTER}\n"
    )


def decision_body(d: Decision) -> str:
    return (
        f"**Question:** {d.question}\n\n**Recommendation:** {d.recommendation}\n\n"
        "Context: `docs/design/` (private, local) and `docs/ROADMAP.md`. Reply here with your "
        "decision; it is then logged in `docs/decisions.md` (plan-01 Task 8).\n\n---\n"
        f"{FOOTER}\n"
    )


class Gh:
    def __init__(self, repo: str, dry_run: bool) -> None:
        self.repo = repo
        self.dry_run = dry_run

    def run(self, *args: str, mutate: bool = True) -> str:
        if mutate and self.dry_run:
            print("DRY", "gh", *args)
            return ""
        res = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
        return res.stdout

    def ensure_label(self, name: str) -> None:
        color = next((c for p, c in LABEL_COLORS.items() if name.startswith(p)), "ededed")
        self.run("label", "create", name, "--repo", self.repo, "--color", color, "--force")

    def milestones(self) -> dict[str, int]:
        out = self.run("api", f"repos/{self.repo}/milestones?state=all&per_page=100", mutate=False)
        return {m["title"]: m["number"] for m in json.loads(out or "[]")}

    def ensure_milestone(self, title: str, description: str, existing: dict[str, int]) -> None:
        if title in existing:
            return
        self.run(
            "api",
            f"repos/{self.repo}/milestones",
            "-f",
            f"title={title}",
            "-f",
            f"description={description}",
        )
        existing[title] = -1

    def issues(self) -> dict[str, dict]:
        out = self.run(
            "issue",
            "list",
            "--repo",
            self.repo,
            "--state",
            "all",
            "--limit",
            "1000",
            "--json",
            "number,title,state,body,url",
            mutate=False,
        )
        found = {}
        for i in json.loads(out or "[]"):
            key = i["title"].split(" ", 1)[0]
            if i["title"].startswith("Decision "):
                key = i["title"].split(":", 1)[0].removeprefix("Decision ")
            found[key] = i
        return found


def sync(repo: str, dry_run: bool, project: str | None) -> int:
    gh = Gh(repo, dry_run)
    plans = [
        parse_plan(p.read_text(encoding="utf-8"), p) for p in sorted(PLANS_DIR.glob("plan-*.md"))
    ]
    decisions = parse_decisions(ROADMAP.read_text(encoding="utf-8")) if ROADMAP.exists() else []

    labels = {lab for p in plans for t in p.tasks for lab in t.labels} | {"decision"}
    for lab in sorted(labels):
        gh.ensure_label(lab)
    existing_ms = gh.milestones()
    for p in plans:
        gh.ensure_milestone(p.milestone, p.goal[:250], existing_ms)
    existing = gh.issues()

    created: list[str] = []
    for p in plans:
        for t in p.tasks:
            body = issue_body(p, t)
            if t.key in existing:
                if existing[t.key]["body"].strip() != body.strip():
                    gh.run(
                        "issue",
                        "edit",
                        str(existing[t.key]["number"]),
                        "--repo",
                        repo,
                        "--body",
                        body,
                        "--title",
                        t.title,
                    )
                continue
            args = [
                "issue",
                "create",
                "--repo",
                repo,
                "--title",
                t.title,
                "--body",
                body,
                "--milestone",
                p.milestone,
            ]
            for lab in t.labels:
                args += ["--label", lab]
            url = gh.run(*args).strip()
            created.append(url)
            if t.done and url:
                gh.run(
                    "issue",
                    "close",
                    url,
                    "--repo",
                    repo,
                    "--comment",
                    "Done during bootstrap; see docs/implement/implement-00-bootstrap.md.",
                )
    for d in decisions:
        if d.key in existing:
            continue
        url = gh.run(
            "issue",
            "create",
            "--repo",
            repo,
            "--title",
            d.title,
            "--body",
            decision_body(d),
            "--label",
            "decision",
            "--milestone",
            plans[0].milestone,
        ).strip()
        created.append(url)

    if project and created and not dry_run:
        owner = repo.split("/")[0]
        listing = json.loads(
            gh.run("project", "list", "--owner", owner, "--format", "json", mutate=False)
        )
        proj = next((x for x in listing.get("projects", []) if x["title"] == project), None)
        if proj is None:
            proj = json.loads(
                gh.run(
                    "project", "create", "--owner", owner, "--title", project, "--format", "json"
                )
            )
        for url in created:
            gh.run("project", "item-add", str(proj["number"]), "--owner", owner, "--url", url)

    n_tasks = sum(len(p.tasks) for p in plans)
    print(
        f"tickets: {len(plans)} plans, {n_tasks} tasks, {len(decisions)} decisions; "
        f"created {len(created)}{' (dry run)' if dry_run else ''}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--repo", default="aliildan/signrule-decide-dev")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--project", default="SignRule-Decide")
    args = ap.parse_args(argv)
    return sync(args.repo, args.dry_run, args.project)


if __name__ == "__main__":
    sys.exit(main())
