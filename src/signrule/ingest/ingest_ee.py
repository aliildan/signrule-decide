"""Estonia ingest: e-Äriregister open data (CC BY 4.0, terms accepted 2026-10-05; evaluation only).

    python -m signrule.ingest.ingest_ee schema --file persons   # key structure (nothing stored)
    python -m signrule.ingest.ingest_ee fetch                   # whitelisted projection -> vault
    python -m signrule.ingest.ingest_ee stats                   # aggregate counts only

The files carry more personal data than we need (birth dates, hashed ID codes, e-mail, addresses,
foreign ID codes, building-association members, shareholdings). `fetch` therefore writes a
whitelisted projection per company: representation texts and their register type codes, roles
with dates, and the natural persons' names as a separate list used only to mask the texts later.
Sole traders (FIE) are dropped (CLAUDE.md §6.3). IDs, birth dates and e-mails inside free texts are
replaced by [ID]/[DATE]/[EMAIL]; the value scanner runs before anything is written.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from signrule.common.paths import CONFIG_DIR, raw_dir, require_vault
from signrule.ingest.http import RegisterClient
from signrule.ingest.ingest_no import key_paths
from signrule.ingest.pii import redact_text, scan_for_personal_ids, strip_personal_ids

BASE = "https://avaandmed.ariregister.rik.ee/sites/default/files/avaandmed"
FILES = {
    "persons": "ettevotja_rekvisiidid__kaardile_kantud_isikud.json.zip",
    "shareholders": "ettevotja_rekvisiidid__osanikud.json.zip",
    "basic": "ettevotja_rekvisiidid__lihtandmed.csv.zip",
}
RAW = raw_dir("ee")
ENUM_KEY = re.compile(r"(?i)(roll|liik|tyyp|tüüp|kood|staatus|vorm)")

SOLE_TRADER_ROLES = frozenset({"FIE"})
REP_KEEP = ("esinduse_tyyp", "esinduse_tyyp_tekstina", "algus_kpv", "lopp_kpv", "kaardi_tyyp")
PERSON_KEEP = (
    "isiku_roll",
    "isiku_roll_tekstina",
    "isiku_tyyp",
    "algus_kpv",
    "lopp_kpv",
    "volituste_loppemise_kpv",
)
NAME_KEYS = ("eesnimi", "nimi_arinimi")
BASIC_KEEP = re.compile(
    r"(?i)^(ariregistri_kood|nimi|.*oiguslik_vorm.*|.*staatus.*|.*esmakande.*)$"
)


def make_client() -> RegisterClient:
    cfg = yaml.safe_load((CONFIG_DIR / "ingest.yaml").read_text(encoding="utf-8"))
    return RegisterClient(user_agent=cfg["user_agent"], max_rps=1.0)


def download(client: RegisterClient, name: str) -> tuple[bytes, str]:
    resp = client.get_bytes(f"{BASE}/{name}")
    data = resp.content
    return data, hashlib.sha256(data).hexdigest()


def unzip_one(data: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        (member,) = [n for n in z.namelist() if not n.endswith("/")][:1]
        return z.read(member)


def enum_values(
    obj: Any, prefix: str = "", out: dict[str, Counter[str]] | None = None
) -> dict[str, Counter[str]]:
    """Value counts for enumeration-like keys (role/type/code/status), never free text or names."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, str) and ENUM_KEY.search(k) and len(v) <= 40:
                out.setdefault(p, Counter())[v] += 1
            else:
                enum_values(v, p, out)
    elif isinstance(obj, list):
        for v in obj:
            enum_values(v, f"{prefix}[]", out)
    return out


def _text(v: Any) -> str | None:
    s = str(v).strip() if v is not None else ""
    return s or None


def project_company(rec: dict[str, Any]) -> tuple[dict[str, Any] | None, int]:
    """Whitelisted projection of one persons-file record; None for sole traders.

    Returns (projection, number of ID/date/e-mail replacements made in its texts).
    """
    persons = rec.get("kaardile_kantud_isikud") or []
    if any(p.get("isiku_roll") in SOLE_TRADER_ROLES for p in persons):
        return None, 0
    redacted = 0

    def clean(s: Any) -> str | None:
        nonlocal redacted
        t = _text(s)
        if t is None:
            return None
        t, n = redact_text(t)
        redacted += n
        return t

    names: list[str] = []
    out_persons = []
    for p in persons:
        if p.get("isiku_tyyp") == "F":
            name = " ".join(x for x in (_text(p.get(k)) for k in NAME_KEYS) if x)
            if name and name not in names:
                names.append(name)
        out_persons.append({k: p[k] for k in PERSON_KEEP if _text(p.get(k))})
    special = []
    for e in rec.get("esindusoiguse_eritingimused") or []:
        row = {k: e[k] for k in REP_KEEP if _text(e.get(k))}
        row["text"] = clean(e.get("esinduse_sisu"))
        special.append(row)
    normal = [
        {"roll": n.get("roll"), "text": clean(n.get("sisu"))}
        for n in rec.get("esindusoiguse_normaalregulatsioonid") or []
    ]
    out = {
        "ariregistri_kood": _text(rec.get("ariregistri_kood")),
        "nimi": _text(rec.get("nimi")),
        "representation_special": special,
        "representation_normal": normal,
        "persons": out_persons,
        "mask_names": names,
    }
    return out, redacted


def _write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def latest(prefix: str) -> Path:
    paths = sorted(RAW.glob(f"{prefix}_*.jsonl.gz"))
    if not paths:
        raise SystemExit(f"no {prefix} projection under the EE raw dir; run `fetch` first")
    return paths[-1]


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def cmd_schema(args: argparse.Namespace) -> int:
    name = FILES[args.file]
    with make_client() as client:
        data, _ = download(client, name)
    raw = unzip_one(data)
    if name.endswith(".csv.zip"):
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter=";"))
        print(f"{name}: {len(data) >> 20} MiB zip, {len(rows)} rows; columns:")
        for col in rows[0] if rows else []:
            ctr = Counter(r[col] for r in rows[: args.sample])
            shown = dict(ctr.most_common(25)) if ENUM_KEY.search(col) and len(ctr) <= 60 else ""
            print(f"  {col}: {len(ctr)} distinct {shown}")
        return 0
    obj = json.loads(raw)
    sample = obj[: args.sample] if isinstance(obj, list) else obj
    n = len(obj) if isinstance(obj, list) else 1
    print(
        f"{name}: {len(data) >> 20} MiB zip, {n} top-level records; key paths over {len(sample)}:"
    )
    for p, c in sorted(key_paths(sample).items()):
        print(f"  {c:>9d}  {p}")
    print("enumeration-like values (only short values of role/type/code/status keys):")
    for p, ctr in sorted(enum_values(sample).items()):
        if len(ctr) <= 60:
            print(f"  {p}: {dict(ctr.most_common(25))}")
        else:
            print(f"  {p}: {len(ctr)} distinct values (not shown)")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    require_vault()
    RAW.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    stamp = now.strftime("%Y-%m-%d")
    manifest: dict[str, Any] = {
        "fetched_at": now.isoformat(timespec="seconds"),
        "licence": "CC BY 4.0",
        "files": {},
    }
    with make_client() as client:
        data, digest = download(client, FILES["persons"])
        records = json.loads(unzip_one(data))
        rows, sole, redacted = [], 0, 0
        for rec in records:
            proj, n = project_company(rec)
            if proj is None:
                sole += 1
                continue
            rows.append(proj)
            redacted += n
        rows, removed = strip_personal_ids(rows)
        hits = scan_for_personal_ids(rows)
        if hits:
            raise SystemExit(f"persons: {len(hits)} personal-ID-like values at e.g. {hits[0]}")
        out = RAW / f"persons_{stamp}.jsonl.gz"
        _write_jsonl_gz(out, rows)
        manifest["files"][FILES["persons"]] = {
            "sha256_download": digest,
            "records": len(records),
            "stored": out.name,
            "stored_records": len(rows),
            "dropped_sole_traders": sole,
            "redacted_in_text": redacted,
            "removed_fields": removed,
        }

        data, digest = download(client, FILES["basic"])
        csv_rows = list(
            csv.DictReader(io.StringIO(unzip_one(data).decode("utf-8-sig")), delimiter=";")
        )
        cols = [c for c in (csv_rows[0] if csv_rows else {}) if BASIC_KEEP.match(c or "")]
        form_col = next((c for c in cols if "oiguslik_vorm" in c), None)
        keep = [
            {c: r.get(c) for c in cols}
            for r in csv_rows
            if not (form_col and "füüsilisest isikust" in (r.get(form_col) or "").lower())
        ]
        out = RAW / f"basic_{stamp}.jsonl.gz"
        _write_jsonl_gz(out, keep)
        manifest["files"][FILES["basic"]] = {
            "sha256_download": digest,
            "records": len(csv_rows),
            "stored": out.name,
            "stored_records": len(keep),
            "columns_kept": cols,
        }
    (RAW / f"manifest_{stamp}.json").write_text(json.dumps(manifest, indent=1))
    for name, m in manifest["files"].items():
        print(f"{name}: {m['records']} records -> {m['stored_records']} stored ({m['stored']})")
    p = manifest["files"][FILES["persons"]]
    print(f"  sole traders dropped {p['dropped_sole_traders']}, text redactions {redacted}")
    return 0


def _active(row: dict[str, Any]) -> bool:
    return not row.get("lopp_kpv")


def _length_bucket(n: int) -> str:
    for hi in (50, 100, 200, 400, 800):
        if n <= hi:
            return f"<= {hi}"
    return "> 800"


TEMPLATE_MIN = 50  # a text shared by >= 50 companies is a register template, safe to print


def stats(companies: list[dict[str, Any]], basic: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate counts only: no names, no registry codes, and only texts that at least
    TEMPLATE_MIN companies share (register templates, not individual wording)."""
    form_col = next((c for c in (basic[0] if basic else {}) if "oiguslik_vorm" in c), None)
    status_col = next((c for c in (basic[0] if basic else {}) if "staatus" in c), None)
    by_code = {r.get("ariregistri_kood"): r for r in basic}
    out: dict[str, Any] = {"companies": len(companies)}
    forms: Counter[str] = Counter()
    forms_special: Counter[str] = Counter()
    types: Counter[str] = Counter()
    type_label: dict[str, Counter[str]] = {}
    lengths: Counter[str] = Counter()
    normal_roles: Counter[str] = Counter()
    normal_distinct: dict[str, set[str]] = {}
    roles: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    special_texts: Counter[str] = Counter()
    normal_texts: Counter[str] = Counter()
    n_special = n_special_texts = n_name_hits = 0
    for c in companies:
        b = by_code.get(c.get("ariregistri_kood")) or {}
        form = (b.get(form_col) if form_col else None) or "unknown"
        status = (b.get(status_col) if status_col else None) or "unknown"
        forms[form] += 1
        statuses[status] += 1
        act = [e for e in c["representation_special"] if _active(e)]
        if act:
            n_special += 1
            forms_special[form] += 1
        for e in act:
            t = e.get("esinduse_tyyp") or "none"
            types[t] += 1
            if e.get("esinduse_tyyp_tekstina"):
                type_label.setdefault(t, Counter())[e["esinduse_tyyp_tekstina"]] += 1
            if e.get("text"):
                n_special_texts += 1
                special_texts[e["text"]] += 1
                lengths[_length_bucket(len(e["text"]))] += 1
                n_name_hits += any(nm in e["text"] for nm in c["mask_names"])
        for n in c["representation_normal"]:
            normal_roles[n.get("roll") or "none"] += 1
            if n.get("text"):
                normal_distinct.setdefault(n.get("roll") or "none", set()).add(n["text"])
                normal_texts[n["text"]] += 1
        for p in c["persons"]:
            if _active(p):
                roles[p.get("isiku_roll") or "none"] += 1
    out.update(
        {
            "companies_with_active_special_rule": n_special,
            "share_with_special_rule": n_special / max(len(companies), 1),
            "by_legal_form": {
                f: {"companies": n, "with_special_rule": forms_special[f]}
                for f, n in forms.most_common(20)
            },
            "by_status": dict(statuses.most_common(10)),
            "special_rule_types": {
                t: {
                    "n": n,
                    "label": (type_label.get(t) or Counter()).most_common(1)[0][0]
                    if type_label.get(t)
                    else None,
                }
                for t, n in types.most_common()
            },
            "special_texts": n_special_texts,
            "special_texts_distinct": len(special_texts),
            "special_texts_naming_a_card_person": n_name_hits,
            "special_text_templates": {
                t: n for t, n in special_texts.most_common(10) if n >= TEMPLATE_MIN
            },
            "normal_regulation_templates": {
                t: n for t, n in normal_texts.most_common(12) if n >= TEMPLATE_MIN
            },
            "special_text_length": dict(sorted(lengths.items())),
            "normal_regulation_roles": dict(normal_roles.most_common()),
            "normal_regulation_distinct_texts": {k: len(v) for k, v in normal_distinct.items()},
            "active_roles": dict(roles.most_common(30)),
        }
    )
    return out


def cmd_stats(args: argparse.Namespace) -> int:
    require_vault()
    companies = read_jsonl_gz(latest("persons"))
    basic = read_jsonl_gz(latest("basic"))
    print(json.dumps(stats(companies, basic), indent=1, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ingest_ee", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("schema")
    p.add_argument("--file", choices=sorted(FILES), default="persons")
    p.add_argument("--sample", type=int, default=5000)
    p.set_defaults(func=cmd_schema)
    sub.add_parser("fetch").set_defaults(func=cmd_fetch)
    sub.add_parser("stats").set_defaults(func=cmd_stats)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
