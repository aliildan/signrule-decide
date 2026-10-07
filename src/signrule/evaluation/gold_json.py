"""Offline annotation in a JSON file (alternative to the web tool, same records and guidelines).

    python -m signrule.evaluation.gold_json export --set at-text --annotator ali
    python -m signrule.evaluation.gold_json import --set at-text --annotator ali

`export` writes data/gold/<set>/to_annotate_<annotator>.json (one file per annotator, blind: no
other annotator's answers, no model predictions) and copies the guidelines next to it. The
annotator fills each item's `answer`; groups are typed as "CEO:1 + PROKURIST:1" or "ALL_BOARD".
`import` validates every item and appends valid answers to data/gold/<set>/<annotator>.jsonl in
the web tool's record format. Output: counts and errors as item number + field, never content.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from signrule.common.paths import GOLD_DIR, REPO_ROOT, require_vault
from signrule.ontology.cir import ROLES

COLLECTIVES = ("ALL_BOARD", "ALL_PARTNERS", "ALL_EXECUTIVES")
STATUSES = ("rule", "uninterpretable", "none")
PRESENT = ("yes", "no", "unknown")
MODES = ("sole", "joint", "mixed", "unknown")
_ROLE = re.compile(r"^([A-Z_]+)\s*:\s*(\d+)$")
# Guides copied next to each exported file (docs/ is private; the annotator gets a copy).
GUIDES = {
    "no-rt": ("annotation_guidelines.md", "annotation_guide_no.md"),
    "dk-text": ("annotation_guidelines.md", "annotation_guide_dk.md"),
}
DEFAULT_GUIDES = ("annotation_guidelines.md",)
# Roles a set's supplement rules out (the harmonisation EXECUTIVE_MEMBER -> BOARD_MEMBER is for
# the Austrian Vorstand; a Danish direktør is a CEO).
REFUSED_ROLES = {"dk-text": {"EXECUTIVE_MEMBER": "use CEO for the Danish direktion"}}

BLANK_ANSWER: dict[str, Any] = {
    "status": "",
    "alternatives": [],
    "person_specific": False,
    "procuration_present": "",
    "procuration_mode": "",
    "ambiguity": None,
    "note": "",
}
EXAMPLE = {
    "item": "example (fictitious, not part of the batch)",
    "show": {
        "legal_form": "GmbH",
        "signing_lines": [
            "Geschäftsführer [PERSON_1]: vertritt seit 01.01.2020 gemeinsam mit einem weiteren "
            "Geschäftsführer oder einem Prokuristen",
            "Geschäftsführer [PERSON_2]: vertritt seit 01.01.2021 gemeinsam mit einem weiteren "
            "Geschäftsführer oder einem Prokuristen",
        ],
        "procuration_lines": [
            "Prokurist [PERSON_3]: vertritt seit 01.01.2022 gemeinsam mit einem Geschäftsführer"
        ],
        "roles": "Geschäftsführer ×2, Prokurist ×1",
    },
    "answer": {
        "status": "rule",
        "alternatives": ["CEO:2", "CEO:1 + PROKURIST:1"],
        "person_specific": False,
        "procuration_present": "yes",
        "procuration_mode": "joint",
        "ambiguity": 0,
        "note": "",
    },
}


def parse_group(text: str) -> dict[str, Any]:
    """'CEO:1 + PROKURIST:1' -> {"roles": [["CEO", 1], ["PROKURIST", 1]], "collective": None}."""
    t = text.strip()
    if t in COLLECTIVES:
        return {"roles": [], "collective": t}
    roles = []
    for part in t.split("+"):
        m = _ROLE.match(part.strip())
        if not m or m.group(1) not in ROLES or int(m.group(2)) < 1:
            raise ValueError(f"bad group part {part.strip()!r}")
        roles.append([m.group(1), int(m.group(2))])
    return {"roles": roles, "collective": None}


def to_record(
    answer: dict[str, Any], item_id: str, annotator: str, set_name: str = ""
) -> dict[str, Any] | None:
    """Validated web-tool record, or None if the answer is still blank. Raises ValueError."""
    status = str(answer.get("status") or "").strip()
    if not status and answer.get("ambiguity") is None and not answer.get("alternatives"):
        return None
    errors = []
    if status not in STATUSES:
        errors.append("status")
    groups = []
    for g in answer.get("alternatives") or []:
        try:
            groups.append(parse_group(str(g)))
        except ValueError:
            errors.append("alternatives")
    refused = REFUSED_ROLES.get(set_name, {})
    for g in groups:
        for role, _ in g["roles"]:
            if role in refused:
                errors.append(f"alternatives ({refused[role]})")
    if status == "rule" and not groups:
        errors.append("alternatives (status rule needs at least one group)")
    amb = answer.get("ambiguity")
    if amb not in (0, 1, 2, 3):
        errors.append("ambiguity")
    present = str(answer.get("procuration_present") or "unknown")
    mode = str(answer.get("procuration_mode") or "unknown")
    if present not in PRESENT:
        errors.append("procuration_present")
    if mode not in MODES:
        errors.append("procuration_mode")
    if not isinstance(answer.get("person_specific", False), bool):
        errors.append("person_specific")
    if errors:
        raise ValueError(", ".join(dict.fromkeys(errors)))
    return {
        "item_id": item_id,
        "annotator": annotator,
        "skipped": False,
        "status": status,
        "alternatives": groups if status == "rule" else [],
        "person_specific": bool(answer.get("person_specific", False)),
        "procuration": {"present": present, "mode": mode},
        "ambiguity": int(amb),
        "note": str(answer.get("note") or "").strip(),
        "source": "json",
        "imported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def show(state: dict[str, Any]) -> dict[str, Any]:
    def lines(t: Any) -> list[str]:
        return [p for p in str(t or "").split("; ") if p.strip()]

    return {
        "legal_form": state.get("legal_form"),
        "signing_lines": lines(state.get("signature_rule")),
        "procuration_lines": lines(state.get("procuration_rule")),
        "roles": ", ".join(f"{r['role']} ×{r['count']}" for r in state.get("roles") or []),
    }


def export_file(
    set_name: str, annotator: str, root: Path = GOLD_DIR, docs_dir: Path | None = None
) -> Path:
    """Write the annotator's file and copy the set's guides from `docs_dir` (default: the private
    docs/ folder, which exists only on the owner's machine)."""
    docs_dir = docs_dir or REPO_ROOT / "docs"
    missing = [g for g in GUIDES.get(set_name, DEFAULT_GUIDES) if not (docs_dir / g).exists()]
    if missing:
        raise FileNotFoundError(f"guides missing in {docs_dir}: {missing}")
    batch = [
        json.loads(x)
        for x in (root / set_name / "batch.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    doc = {
        "set": set_name,
        "annotator": annotator,
        "how_to": [
            "Read annotation_guidelines.md first (§3 groups, §4 procuration, §5 ambiguity, §6)"
            + "".join(f" and {g} (this set)" for g in GUIDES.get(set_name, DEFAULT_GUIDES)[1:])
            + ".",
            "Fill only the 'answer' of each item; never change 'id' or 'show'.",
            "status: rule | uninterpretable | none",
            "alternatives: list of groups, one string per group, e.g. 'CEO:2', "
            "'CEO:1 + PROKURIST:1', 'CHAIR:1', or a collective 'ALL_BOARD'.",
            f"roles: {', '.join(sorted(ROLES))}",
            "person_specific: true if some powers belong to individual persons only (§6).",
            "procuration_present: yes | no | unknown; procuration_mode: sole | joint | mixed | "
            "unknown",
            "ambiguity: 0, 1, 2 or 3 (a number, no quotes); note: free text, optional.",
            "Leave an item completely blank to skip it for now; save often; keep the file valid "
            "JSON.",
        ],
        "example": EXAMPLE,
        "items": [
            {
                "n": i + 1,
                "id": it["item_id"],
                "show": show(it["state"]),
                "answer": dict(BLANK_ANSWER),
            }
            for i, it in enumerate(batch)
        ],
    }
    out = root / set_name / f"to_annotate_{annotator}.json"
    if out.exists():
        raise FileExistsError(f"{out.name} exists; it may already contain answers")
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    for guide in GUIDES.get(set_name, DEFAULT_GUIDES):
        shutil.copyfile(docs_dir / guide, root / set_name / guide)
    return out


def import_file(set_name: str, annotator: str, root: Path = GOLD_DIR) -> dict[str, Any]:
    path = root / set_name / f"to_annotate_{annotator}.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"fatal": f"not valid JSON at line {e.lineno}, column {e.colno}"}
    if doc.get("annotator") != annotator or doc.get("set") != set_name:
        return {"fatal": "file belongs to another annotator or set"}
    batch_ids = {
        json.loads(x)["item_id"]
        for x in (root / set_name / "batch.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    }
    records, errors, blank = [], [], 0
    for it in doc.get("items") or []:
        if it.get("id") not in batch_ids:
            errors.append(f"item {it.get('n')}: unknown id")
            continue
        try:
            rec = to_record(it.get("answer") or {}, it["id"], annotator, set_name)
        except ValueError as e:
            errors.append(f"item {it.get('n')}: {e}")
            continue
        if rec is None:
            blank += 1
        else:
            records.append(rec)
    if not errors and records:
        with (root / set_name / f"{annotator}.jsonl").open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"valid": len(records), "blank": blank, "errors": errors, "written": not errors}


def merge_file(
    set_name: str, annotator: str, filled: Path, root: Path = GOLD_DIR
) -> dict[str, Any]:
    """Copy the answers of a file the annotator rebuilt (e.g. retyped from pasted text) into their
    exported file, item by item number, only where the item id matches exactly. A mechanical
    transfer of the annotator's own answers; ids and items of the exported file stay unchanged."""
    path = root / set_name / f"to_annotate_{annotator}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    filled_doc = json.loads(filled.read_text(encoding="utf-8"))
    # the exported layout ("items"), an answers-only file ("answers": [{n, id, answer}, …]) or a
    # bare list of such entries
    if isinstance(filled_doc, list):
        entries = filled_doc
    else:
        entries = filled_doc.get("items") or filled_doc.get("answers") or []
    src = {it.get("n"): it for it in entries}
    copied, mismatch, missing = 0, [], []
    for it in doc["items"]:
        other = src.get(it["n"])
        if other is None:
            missing.append(it["n"])
        elif str(other.get("id")) != it["id"]:
            mismatch.append(it["n"])
        else:
            it["answer"] = {k: other.get("answer", {}).get(k, v) for k, v in it["answer"].items()}
            copied += 1
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"copied": copied, "id_mismatch": mismatch, "missing": missing}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gold_json", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["export", "import", "merge"])
    ap.add_argument("--set", required=True)
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--from", dest="filled", help="merge: the rebuilt file, in the set folder")
    a = ap.parse_args(argv)
    require_vault(GOLD_DIR)
    if a.cmd == "export":
        out = export_file(a.set, a.annotator)
        guides = ", ".join(GUIDES.get(a.set, DEFAULT_GUIDES))
        print(f"wrote {out.relative_to(GOLD_DIR.parent.parent)} (+ {guides})")
        return 0
    if a.cmd == "merge":
        if not a.filled:
            raise SystemExit("merge needs --from <file name in the set folder>")
        m = merge_file(a.set, a.annotator, GOLD_DIR / a.set / Path(a.filled).name)
        print(
            f"copied {m['copied']}; id mismatch at items {m['id_mismatch'] or 'none'}; "
            f"missing items {m['missing'] or 'none'}"
        )
        return 0 if not m["id_mismatch"] and not m["missing"] else 1
    res = import_file(a.set, a.annotator)
    if "fatal" in res:
        print(f"import failed: {res['fatal']}")
        return 1
    print(f"valid {res['valid']}, blank {res['blank']}, errors {len(res['errors'])}")
    for e in res["errors"][:50]:
        print("  " + e)
    print("written" if res["written"] else "nothing written: fix the errors and import again")
    return 0 if res["written"] else 1


if __name__ == "__main__":
    sys.exit(main())
