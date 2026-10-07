"""Local-LLM audit of the Austrian rendering (runs on this machine; prints counts and ids only).

    uv run python eval/local_audit.py faithful --split pilot --n 500
    uv run python eval/local_audit.py names --split pilot --n 500

`faithful`: the local model sees the raw register entry (vault) next to our rendered, masked line
and flags duplicated, missing or garbled wording. `names`: it checks the rendered text for person
names that survived masking. Notes from the model stay in the vault (interim/at/qa/); stdout
carries only counts and company numbers (FNR), which go to the masked review window
(eval/review.py) for a human-readable look. QA flags only — never labels (CLAUDE.md §2.1).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from typing import Any

from signrule.common.paths import INTERIM_DIR, require_vault
from signrule.evaluation.harness import load_requests
from signrule.ingest.ingest_at import RAW, _current, _first, _texts
from signrule.qa.local_llm import DEFAULT_MODEL, LocalLLM

ISSUES = ["none", "duplicated_wording", "missing_wording", "garbled_wording", "wrong_role", "other"]
FAITHFUL_SCHEMA = {
    "type": "object",
    "properties": {
        "issue": {"type": "string", "enum": ISSUES},
        "note": {"type": "string", "maxLength": 200},
    },
    "required": ["issue", "note"],
}
NAMES_SCHEMA = {
    "type": "object",
    "properties": {"has_person_name": {"type": "boolean"}, "note": {"type": "string"}},
    "required": ["has_person_name", "note"],
}
FAITHFUL_SYSTEM = (
    "You check a data pipeline. You get the raw entries of an Austrian company-register extract "
    "(per function: code, representation code, and the register's text fragments, which are "
    "wrapped lines of one text; TXTVERTR repeats part of TEXT) and the pipeline's rendering of the "
    "same company. In the rendering, person names are replaced by [PERSON_n] or [FIRMA_n], role "
    "names are normalised (e.g. GESCHÄFTSFÜHRER/IN -> Geschäftsführer), dates are kept. These "
    "changes are intended. Report a problem only if the rendering repeats wording that appears "
    "once in the register (duplicated_wording), drops register wording (missing_wording), breaks "
    "or merges words so the meaning changes (garbled_wording), or assigns a text to the wrong role "
    "(wrong_role). Otherwise answer issue=none."
)
NAMES_SYSTEM = (
    "You check masked company-register texts. Tokens like [PERSON_1] or [FIRMA_2] are masked names "
    "and are fine. Role words (Geschäftsführer, Prokurist, Vorstandsmitglied, styreleder, daglig "
    "leder, …), legal terms and company legal forms are fine. Answer has_person_name=true only if "
    "the text still contains a natural person's first or last name."
)


def raw_view(fnr: str) -> list[dict[str, Any]] | None:
    p = RAW / "auszug" / f"{fnr}.json"
    if not p.exists():
        return None
    resp = _first(
        json.loads(p.read_text(encoding="utf-8")).get("body", {}).get("AUSZUG_V2_RESPONSE")
    )
    out = []
    for fun in (resp or {}).get("FUN") or []:
        for e in _current(fun.get("FU_DKZ10")):
            vart = _first(e.get("VART")) or {}
            out.append(
                {
                    "function": fun.get("@FKENTEXT"),
                    "person": fun.get("@PNR"),
                    "representation_code": _first(vart.get("CODE"))
                    if isinstance(vart, dict)
                    else None,
                    "TEXT": _texts(e.get("TEXT")),
                    "TXTVERTR": _texts(e.get("TXTVERTR")),
                }
            )
    return out


def rendered(req: dict[str, Any]) -> str:
    st = req["state"]
    return "\n".join(f"{k}: {st[k]}" for k in ("signature_rule", "procuration_rule") if st.get(k))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("task", choices=["faithful", "names"])
    ap.add_argument("--split", default="pilot")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stratum")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args(argv)
    require_vault(INTERIM_DIR)
    reqs = [
        r
        for r in load_requests("at", a.split, "test")
        if not a.stratum or r["_meta"].get("stratum") == a.stratum
    ]
    sample = random.Random(a.seed).sample(reqs, min(a.n, len(reqs)))
    llm = LocalLLM(model=a.model)
    qa_dir = INTERIM_DIR / "at" / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    flagged: list[str] = []
    with (qa_dir / f"{a.task}-{a.split}.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(sample, 1):
            fnr = r["_meta"]["id"]
            if a.task == "faithful":
                raw = raw_view(fnr)
                if raw is None:
                    counts["no raw extract"] += 1
                    continue
                user = (
                    "RAW REGISTER ENTRIES:\n"
                    + json.dumps(raw, ensure_ascii=False, indent=1)
                    + "\n\nRENDERING:\n"
                    + rendered(r)
                )
                ans = llm.ask_json(FAITHFUL_SYSTEM, user, FAITHFUL_SCHEMA)
                key = ans["issue"]
            else:
                ans = llm.ask_json(NAMES_SYSTEM, rendered(r), NAMES_SCHEMA)
                key = "person name" if ans["has_person_name"] else "none"
            counts[key] += 1
            if key != "none":
                flagged.append(fnr)
            f.write(json.dumps({"id": fnr, **ans}, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"  {i}/{len(sample)} {dict(counts)}", flush=True)
    llm.close()
    print(f"{a.task} on at/{a.split} ({len(sample)} sampled, model {a.model}): {dict(counts)}")
    print(f"flagged ids (first 30 of {len(flagged)}): {' '.join(flagged[:30])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
