"""Required analyses (CLAUDE.md §8) from a scored model's item sidecar.

    uv run python eval/analysis.py --split random --part test --model no-9b-base

Aggregate outputs (public): results/<jur>-<split>/analysis/<model>.<part>.json with the
`rule_type` confusion matrix and accuracy by text-length bucket and legal form.
Private output: runs/eval/<jur>-<split>/<model>.<part>.top_errors.md, the 50 highest-confidence
errors with their **masked** state, for reading by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

from signrule.common.paths import REPO_ROOT
from signrule.evaluation.harness import load_requests
from signrule.format.request import load_questions


def length_bucket(n: int) -> str:
    for edge in (1, 20, 40, 80, 160):
        if n < edge:
            return "none" if edge == 1 else f"<{edge}"
    return ">=160"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jurisdiction", default="no")
    ap.add_argument("--split", default="random")
    ap.add_argument("--part", default="test")
    ap.add_argument("--model", required=True)
    ap.add_argument("--top", type=int, default=50)
    a = ap.parse_args(argv)
    qc = load_questions()
    tag = f"{a.jurisdiction}-{a.split}"
    side = REPO_ROOT / "runs" / "eval" / tag / f"{a.model}.{a.part}.items.jsonl"
    items = [json.loads(line) for line in side.read_text().splitlines()]
    reqs = {r["_meta"]["id"]: r for r in load_requests(a.jurisdiction, a.split, a.part)}

    keys = qc.keys("rule_type")
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    by_len: dict[str, list[bool]] = defaultdict(list)
    by_form: dict[str, list[bool]] = defaultdict(list)
    errors = []
    for it in items:
        st = reqs[it["rid"]]["state"]
        ok = it["pred"] == it["y"]
        text = (st.get("signature_rule") or "") + " " + (st.get("procuration_rule") or "")
        by_len[length_bucket(len(text.strip()))].append(ok)
        by_form[str(st.get("legal_form"))].append(ok)
        if it["qid"] == "rule_type":
            confusion[keys[it["y"]]][keys[it["pred"]]] += 1
        if not ok:
            errors.append(it)

    def acc(v: list[bool]) -> dict[str, float]:
        return {"n": len(v), "accuracy": sum(v) / len(v)}

    out = {
        "model": a.model,
        "split": a.split,
        "part": a.part,
        "n_items": len(items),
        "rule_type_confusion": {g: dict(c) for g, c in sorted(confusion.items())},
        "accuracy_by_text_length": {k: acc(v) for k, v in sorted(by_len.items())},
        "accuracy_by_legal_form": {k: acc(v) for k, v in sorted(by_form.items())},
        "errors_by_question": dict(Counter(e["qid"] for e in errors)),
    }
    res = REPO_ROOT / "results" / tag / "analysis"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{a.model}.{a.part}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    errors.sort(key=lambda e: -e["conf"])
    lines = [f"# {a.model} · {tag} · {a.part} — top {a.top} confident errors (masked, private)", ""]
    for e in errors[: a.top]:
        st = reqs[e["rid"]]["state"]
        k = qc.keys(e["qid"])
        lines += [
            f"- **{e['qid']}** gold `{k[e['y']]}` pred `{k[e['pred']]}` conf {e['conf']:.3f} "
            f"({st.get('legal_form')})",
            f"  - S: {st.get('signature_rule') or '–'}",
            f"  - P: {st.get('procuration_rule') or '–'}",
            f"  - roles: {', '.join(f'{r["role"]}={r["count"]}' for r in st.get('roles') or [])}",
        ]
    priv = REPO_ROOT / "runs" / "eval" / tag / f"{a.model}.{a.part}.top_errors.md"
    priv.write_text("\n".join(lines) + "\n")
    print(f"analysis -> results/{tag}/analysis/{a.model}.{a.part}.json")
    print(f"errors   -> {priv.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
