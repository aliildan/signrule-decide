"""Masked review window: a small, scrubbed, logged sample of model inputs (owner, 2026-10-05).

    uv run python eval/review.py --jurisdiction at --split pilot --part test --stratum at-text
    uv run python eval/review.py --jurisdiction at --split pilot --part test --n 20 \
        --model no-4b-v01 --tag no-random-to-at-pilot --qid ceo_alone --errors

This is the only sanctioned way for Claude to see record content. It reads already-masked
requests from data/processed (names are [PERSON_n]/[FIRMA_n], IDs stripped at ingest), then scrubs
again: personal IDs, birth dates and e-mails -> [ID]/[DATE]/[EMAIL], and every capitalised word
that is not legal vocabulary or a role label of that jurisdiction -> [X]. At most 30 records per
call. Each call is appended to runs/review/log.jsonl (time, arguments, record ids, scrub counts)
and the output is also written to runs/review/<time>.txt for the owner.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import UTC, datetime
from typing import Any

from signrule.common.paths import REPO_ROOT
from signrule.evaluation.harness import load_requests
from signrule.format.request import load_questions
from signrule.ingest.pii import redact_text
from signrule.ontology.mapping import load_legal_vocab, load_roles

MAX_RECORDS = 30
REVIEW_DIR = REPO_ROOT / "runs" / "review"
_TOKEN = re.compile(r"\[(?:PERSON|FIRMA)_\d+\]|\[(?:ID|DATE|EMAIL|X)\]")
_WORD = re.compile(r"[^\W\d_]+(?:['\-][^\W\d_]+)*")
LEGAL_FORMS = frozenset(
    {"as", "asa", "ans", "da", "ks", "sa", "ba", "se", "nuf", "iks", "brl", "gmbh", "ag", "kg"}
    | {"og", "flexkapg", "oü", "aps", "a/s", "i/s", "k/s", "co"}
)
TEXT_KEYS = ("signature_rule", "procuration_rule")


class Scrubber:
    def __init__(self, jur: str) -> None:
        roles = load_roles()
        labels = {i.label for i in roles.codes.get(jur, {}).values()}
        self.allowed = (
            load_legal_vocab(jur)
            | LEGAL_FORMS
            | {w.lower() for lab in labels for w in _WORD.findall(lab)}
        )
        self.replaced = 0
        self.redacted = 0

    def _word(self, m: re.Match[str]) -> str:
        w = m.group(0)
        if not w[:1].isupper():
            return w
        if all(part.lower() in self.allowed for part in re.split(r"['\-]", w) if part):
            return w
        self.replaced += 1
        return "[X]"

    def text(self, s: str) -> str:
        s, n = redact_text(s)
        self.redacted += n
        parts = _TOKEN.split(s)
        tokens = _TOKEN.findall(s)
        out = [_WORD.sub(self._word, p) for p in parts]
        return "".join(x for pair in zip(out, [*tokens, ""], strict=True) for x in pair)


def render_record(
    req: dict[str, Any], scrub: Scrubber, preds: dict[str, tuple[str, float]] | None
) -> str:
    st, meta = req["state"], req.get("_meta", {})
    head = f"[{st.get('jurisdiction', '?')} · {meta.get('id', '?')}"
    head += f" · {meta['stratum']}]" if meta.get("stratum") else "]"
    lines = [head]
    for k in ("jurisdiction", "legal_form"):
        if st.get(k):
            lines.append(f"{k}: {st[k]}")
    for k in TEXT_KEYS:
        if st.get(k):
            lines.append(f"{k}: {scrub.text(str(st[k]))}")
    if st.get("roles"):
        lines.append("roles: " + ", ".join(f"{r['role']} ×{r['count']}" for r in st["roles"]))
    for note in st.get("role_notes") or []:
        lines.append(f"role_note: {note['role']}: {scrub.text(str(note['note']))}")
    labels = " · ".join(f"{q}={v['label']}" for q, v in req["questions"].items())
    lines.append(f"labels: {labels}")
    if preds:
        marks = []
        for q, v in req["questions"].items():
            if q in preds:
                key, conf = preds[q]
                ok = "✓" if str(key) == str(v["label"]).lower() else "✗"
                marks.append(f"{q}={key} p={conf:.2f} {ok}")
        lines.append("model: " + " · ".join(marks))
    return "\n".join(lines)


def render_strands(req: dict[str, Any], scrub: Scrubber, qid: str) -> str:
    """The scrubbed record's Strands Decider prompt for one question, as the trainer renders it
    (plan-17: review the new input format before training). Needs the `strands` group."""
    from strands_decider.data.format import Example
    from strands_decider.prompting import build_prompt

    from signrule.train.strands_data import to_examples

    st = dict(req["state"])
    for k in TEXT_KEYS:
        if st.get(k):
            st[k] = scrub.text(str(st[k]))
    if st.get("role_notes"):
        st["role_notes"] = [{**n, "note": scrub.text(str(n["note"]))} for n in st["role_notes"]]
    (ex,) = to_examples({"state": st, "questions": {qid: req["questions"][qid]}})
    text, _ = build_prompt(ex["state"], Example.from_dict(ex).to_question())
    return f"--- strands prompt ({qid}, label {ex['options'][ex['label']][0]}) ---\n{text}"


def load_predictions(tag: str, model: str, part: str) -> dict[str, dict[str, tuple[str, float]]]:
    qc = load_questions()
    side = REPO_ROOT / "runs" / "eval" / tag / f"{model}.{part}.items.jsonl"
    out: dict[str, dict[str, tuple[str, float]]] = {}
    for line in side.read_text().splitlines():
        it = json.loads(line)
        out.setdefault(it["rid"], {})[it["qid"]] = (qc.keys(it["qid"])[it["pred"]], it["conf"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jurisdiction", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--part", default="val")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stratum")
    ap.add_argument("--model", help="show this model's predictions (needs --tag)")
    ap.add_argument("--tag", help="results tag of the sidecar, e.g. no-random-to-at-pilot")
    ap.add_argument("--qid", help="only records that carry this question")
    ap.add_argument("--errors", action="store_true", help="only records the model got wrong")
    ap.add_argument("--ids", help="comma-separated record ids (e.g. flagged by local_audit)")
    ap.add_argument("--render-strands", metavar="QID", help="also show the Strands prompt")
    a = ap.parse_args(argv)
    if a.n > MAX_RECORDS:
        raise SystemExit(f"at most {MAX_RECORDS} records per review")
    if a.errors and not (a.model and a.tag):
        raise SystemExit("--errors needs --model and --tag")

    reqs = load_requests(a.jurisdiction, a.split, a.part)
    preds = load_predictions(a.tag, a.model, a.part) if a.model and a.tag else None

    wanted = set(a.ids.split(",")) if a.ids else None

    def keep(r: dict[str, Any]) -> bool:
        if wanted is not None and r["_meta"]["id"] not in wanted:
            return False
        if a.stratum and r["_meta"].get("stratum") != a.stratum:
            return False
        if a.qid and a.qid not in r["questions"]:
            return False
        if a.errors and preds is not None:
            p = preds.get(r["_meta"]["id"], {})
            qids = [a.qid] if a.qid else list(r["questions"])
            return any(
                q in p and str(p[q][0]) != str(r["questions"][q]["label"]).lower() for q in qids
            )
        return True

    pool = [r for r in reqs if keep(r)]
    sample = random.Random(a.seed).sample(pool, min(a.n, len(pool)))
    jur = a.jurisdiction.upper()
    scrub = Scrubber(jur)
    blocks = [
        render_record(r, scrub, (preds or {}).get(r["_meta"]["id"]) if preds else None)
        for r in sample
    ]
    if a.render_strands:
        blocks = [
            b + "\n" + render_strands(r, scrub, a.render_strands)
            if a.render_strands in r["questions"]
            else b
            for b, r in zip(blocks, sample, strict=True)
        ]
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    header = (
        f"review {a.jurisdiction}/{a.split}/{a.part} · {len(sample)} of {len(pool)} matching · "
        f"scrub: {scrub.replaced} words -> [X], {scrub.redacted} IDs/dates/e-mails redacted"
    )
    text = header + "\n\n" + "\n\n".join(blocks) + "\n"
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    (REVIEW_DIR / f"{ts}-{a.jurisdiction}-{a.split}.txt").write_text(text, encoding="utf-8")
    with (REVIEW_DIR / "log.jsonl").open("a", encoding="utf-8") as f:
        log = {
            "at": ts,
            "args": {k: v for k, v in vars(a).items()},
            "ids": [r["_meta"]["id"] for r in sample],
            "scrub_replaced": scrub.replaced,
            "scrub_redacted": scrub.redacted,
        }
        f.write(json.dumps(log) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
