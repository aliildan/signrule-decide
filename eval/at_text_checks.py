"""Aggregate checks of Austrian representation-text phenomena (counts only, no text printed).

    uv run python eval/at_text_checks.py

Checks the cached extracts for: several TXTVERTR/TEXT fragments per entry (line-wrapped text),
fragments ending in a hyphen (broken words), function entries with an end date in the future,
time-limited wording ("von … bis"), "nicht vertretungsbefugt", gender slash forms, branch
restrictions (Zweigniederlassung), and missing spaces between words after joining.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import date

from signrule.ingest.ingest_at import RAW, _current, _first, _texts


def main() -> int:
    c: Counter[str] = Counter()
    today = date.today().isoformat()
    glued = re.compile(
        r"\b(mit|oder|und|einem|einer|dem|der|den)(einem|einer|dem|der|den|weiteren)\b"
    )
    for p in sorted((RAW / "auszug").glob("*.json")):
        body = json.loads(p.read_text(encoding="utf-8")).get("body")
        resp = _first((body or {}).get("AUSZUG_V2_RESPONSE")) or {}
        if not resp or resp.get("@SKIPPED"):
            continue
        c["companies"] += 1
        for fun in resp.get("FUN") or []:
            for e in fun.get("FU_DKZ10") or []:
                if e.get("@AUFRECHT") == "false":
                    continue
                datbis = (
                    (e.get("DATBIS") or [None])[0]
                    if isinstance(e.get("DATBIS"), list)
                    else e.get("DATBIS")
                )
                if datbis:
                    c["entries with DATBIS"] += 1
                    if str(datbis) > today:
                        c["entries with DATBIS in the future"] += 1
            for e in _current(fun.get("FU_DKZ10")):
                c["current entries"] += 1
                txt, tv = _texts(e.get("TEXT")), _texts(e.get("TXTVERTR"))
                c[f"TEXT fragments={min(len(txt), 3)}"] += 1
                if tv:
                    c[f"TXTVERTR fragments={min(len(tv), 3)}"] += 1
                frags = txt + tv
                c["fragment ends with '-'"] += sum(f.rstrip().endswith("-") for f in frags[:-1])
                joined = " ".join(frags)
                low = joined.lower()
                c["'von … bis' wording"] += bool(
                    re.search(r"\bvon \d{1,2}\.\d{1,2}\.\d{4} bis\b", low)
                )
                c["'nicht vertretungsbefugt'"] += "nicht vertretungsbefugt" in low
                c["gender slash form (/in, /einer …)"] += bool(
                    re.search(r"/(in|innen|einer|einem|en)\b", low)
                )
                c["Zweigniederlassung (branch restriction)"] += "zweigniederlassung" in low
                c["glued words (e.g. miteinem)"] += bool(glued.search(low))
                c["role word before 'vertritt'"] += bool(re.match(r"^\S+ vertritt\b", joined))
                c[f"VART={e.get('VART') and (_first(e.get('VART')) or {}).get('CODE')}"] += 1
    print(json.dumps(dict(sorted(c.items())), indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
