"""Austria ingest: Firmenbuch High Value Datasets via JustizOnline (SOAP 1.2, plan-05).

    python -m signrule.ingest.ingest_at frame --von 2025-01-01 --bis 2025-03-31
    python -m signrule.ingest.ingest_at fetch --sample 2000
    python -m signrule.ingest.ingest_at stats

`frame` collects company numbers (FNR) from the daily change feed (VERAENDERUNGENFIRMA);
`fetch` requests `AUSZUG_V2` with UMFANG=Kurzinformation (the only scope HVD serves) for a seeded
sample; `stats` prints aggregates only (legal forms, function codes, VART codes, free-text
presence). The API key is read from the secrets folder at runtime (name `justiz`) and sent as the
`X-API-KEY` header; it is never printed or stored. Birth dates are stripped and personal addresses
(PER/PE_DKZ03) dropped before anything is written; names stay in the vault for masking.
Licence: CC BY 4.0 — "Firmenbuch – Bundesministerium für Justiz / JustizOnline (HVD)".
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import yaml

from signrule.common.paths import CONFIG_DIR, raw_dir, require_vault
from signrule.common.secrets import load_secret
from signrule.ingest.http import CachedResponse, IngestError, RegisterClient
from signrule.ingest.pii import redact_text

SOAP_URL = "https://justizonline.gv.at/jop/api/at.gv.justiz.fbw/ws"
SOAP_NS = "http://www.w3.org/2003/05/soap-envelope"
NS_AUSZUG = "ns://firmenbuch.justiz.gv.at/Abfrage/v2/AuszugRequest"
NS_VERAEND = "ns://firmenbuch.justiz.gv.at/Abfrage/VeraenderungenFirmaRequest"
RAW = raw_dir("at")


# ---- SOAP -----------------------------------------------------------------------------------


def _envelope(prefix: str, ns: str, op: str, fields: list[tuple[str, str | None]]) -> bytes:
    inner = "".join(f"<{prefix}:{k}>{escape(v)}</{prefix}:{k}>" for k, v in fields if v is not None)
    return (
        f'<soap:Envelope xmlns:soap="{SOAP_NS}" xmlns:{prefix}="{ns}">'
        f"<soap:Header/><soap:Body><{prefix}:{op}>{inner}</{prefix}:{op}></soap:Body>"
        "</soap:Envelope>"
    ).encode()


def auszug_request(fnr: str, stichtag: str) -> bytes:
    return _envelope(
        "aus",
        NS_AUSZUG,
        "AUSZUG_V2_REQUEST",
        [("FNR", fnr), ("STICHTAG", stichtag), ("UMFANG", "Kurzinformation")],
    )


def veraenderungen_request(von: str, bis: str, rechtsform: str | None = None) -> bytes:
    return _envelope(
        "ver",
        NS_VERAEND,
        "VERAENDERUNGENFIRMAREQUEST",
        [("VON", von), ("BIS", bis), ("RECHTSFORM", rechtsform)],
    )


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def xml_to_obj(elem: ET.Element) -> Any:
    """Element -> JSON-like: attributes as '@NAME', children always as lists, text as '#text'
    (or the plain string for leaf elements without attributes)."""
    obj: dict[str, Any] = {f"@{_local(k)}": v for k, v in elem.attrib.items()}
    children = list(elem)
    text = (elem.text or "").strip()
    if not children:
        if not obj:
            return text
        if text:
            obj["#text"] = text
        return obj
    for child in children:
        obj.setdefault(_local(child.tag), []).append(xml_to_obj(child))
    return obj


def parse_soap_bytes(content: bytes) -> dict[str, Any]:
    root = ET.fromstring(content)
    body = next((c for c in root if _local(c.tag) == "Body"), None)
    if body is None or not list(body):
        raise ValueError("SOAP response without body")
    payload = list(body)[0]
    if _local(payload.tag) == "Fault":
        raise ValueError("SOAP fault")
    return {_local(payload.tag): xml_to_obj(payload)}


def drop_person_addresses(obj: Any) -> Any:
    """Remove personal addresses (PER/PE_DKZ03) anywhere in the tree (CLAUDE.md §6 step 3)."""
    if isinstance(obj, dict):
        return {k: drop_person_addresses(v) for k, v in obj.items() if k != "PE_DKZ03"}
    if isinstance(obj, list):
        return [drop_person_addresses(v) for v in obj]
    return obj


def norm_fnr(fnr: str) -> str:
    return "".join(fnr.split()).lower()


SOLE_TRADER_FORMS = frozenset({"EU"})  # Einzelunternehmer/in: the firm is a natural person


def _rechtsform_code(resp: dict[str, Any]) -> str | None:
    firma = _first(resp.get("FIRMA")) or {}
    code = None
    for e in _current(firma.get("FI_DKZ07")):
        code, _ = _code_text(e.get("RECHTSFORM"))
    return code


def redact_texts(obj: Any) -> Any:
    """Birth dates/years, personal IDs and e-mails inside free-text fields (TEXT, TXTVERTR, …)
    -> [DATE]/[ID]/[EMAIL] (e.g. "dem Prokuristen …, geb. 1971")."""
    if isinstance(obj, dict):
        return {k: redact_texts(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_texts(v) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj)[0]
    return obj


def sanitize(obj: Any) -> Any:
    """Cache transform: drop personal addresses, redact birth data inside texts; replace a sole
    trader's extract by a stub (CLAUDE.md §6.3: sole-trader records are dropped, not stored)."""
    obj = redact_texts(drop_person_addresses(obj))
    resp = _first(obj.get("AUSZUG_V2_RESPONSE")) if isinstance(obj, dict) else None
    if isinstance(resp, dict) and _rechtsform_code(resp) in SOLE_TRADER_FORMS:
        return {"AUSZUG_V2_RESPONSE": {"@FNR": resp.get("@FNR", ""), "@SKIPPED": "sole_trader"}}
    return obj


# ---- extraction -----------------------------------------------------------------------------


def _first(node: Any) -> Any:
    return node[0] if isinstance(node, list) and node else node


def _text(node: Any) -> str | None:
    node = _first(node)
    if node is None:
        return None
    if isinstance(node, dict):
        return node.get("#text")
    return str(node) or None


def _texts(nodes: list[Any] | None) -> list[str]:
    return [t for t in (_text([x]) for x in nodes or []) if t]


def _code_text(node: Any) -> tuple[str | None, str | None]:
    node = _first(node)
    if not isinstance(node, dict):
        return None, None
    return _text(node.get("CODE")), _text(node.get("TEXT"))


def _current(entries: list[Any] | None, today: str | None = None) -> list[dict[str, Any]]:
    """Entries in force: not superseded (AUFRECHT) and without an end date in the past."""
    today = today or date.today().isoformat()
    out = []
    for e in entries or []:
        if not isinstance(e, dict) or e.get("@AUFRECHT", "true") == "false":
            continue
        end = _text(e.get("DATBIS"))
        if end and end[:10] <= today:
            continue
        out.append(e)
    return out


@dataclass
class AtFunction:
    pnr: str | None
    fken: str | None
    fkentext: str | None
    vart_code: str | None
    vart_text: str | None
    vsbeide_code: str | None
    vsbeide_text: str | None
    txtvertr: list[str] = field(default_factory=list)
    nur_fuer: list[str] = field(default_factory=list)  # VERTRETUNGSBEFUGTNURFUER (e.g. a branch)
    text: list[str] = field(default_factory=list)  # TEXT remarks on the function entry
    has_bezugsperson: bool = False


@dataclass
class AtPerson:
    legal: bool  # a company (BEZEICHNUNG) rather than a natural person
    names: list[str]  # all spellings, longest first: formatted with titles, full, surname


@dataclass
class AtRecord:
    fnr: str
    rechtsform_code: str | None
    rechtsform_text: str | None
    functions: list[AtFunction]
    person_names: dict[str, str]  # PNR -> full name, masking only (never leaves the vault)
    persons: dict[str, AtPerson] = field(default_factory=dict)


def extract_at(obj: dict[str, Any]) -> AtRecord:
    resp = _first(obj.get("AUSZUG_V2_RESPONSE")) or {}
    if resp.get("@SKIPPED") == "sole_trader":
        return AtRecord(norm_fnr(resp.get("@FNR", "")), "EU", None, [], {})
    firma = _first(resp.get("FIRMA")) or {}
    rf_code = rf_text = None
    for e in _current(firma.get("FI_DKZ07")):
        rf_code, rf_text = _code_text(e.get("RECHTSFORM"))
    functions = []
    for fun in resp.get("FUN") or []:
        for e in _current(fun.get("FU_DKZ10")):
            vc, vt = _code_text(e.get("VART"))
            bc, bt = _code_text(e.get("VSBEIDE"))
            functions.append(
                AtFunction(
                    pnr=fun.get("@PNR"),
                    fken=fun.get("@FKEN"),
                    fkentext=fun.get("@FKENTEXT"),
                    vart_code=vc,
                    vart_text=vt,
                    vsbeide_code=bc,
                    vsbeide_text=bt,
                    txtvertr=_texts(e.get("TXTVERTR")),
                    nur_fuer=_texts(e.get("VERTRETUNGSBEFUGTNURFUER")),
                    text=_texts(e.get("TEXT")),
                    has_bezugsperson=bool(_text(e.get("BEZUGSPERSON"))),
                )
            )
    names: dict[str, str] = {}
    persons: dict[str, AtPerson] = {}
    for per in resp.get("PER") or []:
        pnr = per.get("@PNR")
        for e in _current(per.get("PE_DKZ02")):
            company = _text(e.get("BEZEICHNUNG"))
            full = " ".join(filter(None, (_text(e.get("VORNAME")), _text(e.get("NACHNAME")))))
            variants = [company, _text(e.get("NAME_FORMATIERT")), full, _text(e.get("NACHNAME"))]
            uniq = sorted({v.strip() for v in variants if v and v.strip()}, key=len, reverse=True)
            if pnr and uniq:
                names[pnr] = company or full or uniq[0]
                persons[pnr] = AtPerson(legal=bool(company), names=uniq)
    return AtRecord(norm_fnr(resp.get("@FNR", "")), rf_code, rf_text, functions, names, persons)


def frame_fnrs(obj: dict[str, Any]) -> list[str]:
    resp = _first(obj.get("VERAENDERUNGENFIRMARESPONSE")) or {}
    return sorted(
        {norm_fnr(_text(v.get("FNR")) or "") for v in resp.get("VERAENDERUNG") or []} - {""}
    )


# ---- fetching -------------------------------------------------------------------------------


def _headers(key: str) -> dict[str, str]:
    return {"X-API-KEY": key, "Content-Type": "application/soap+xml;charset=UTF-8"}


def _post(
    client: RegisterClient, key: str, body: bytes, cache: Path, cache_key: dict[str, str]
) -> CachedResponse:
    return client.fetch_cached(
        "POST",
        SOAP_URL,
        cache,
        content=body,
        headers=_headers(key),
        parse=lambda r: parse_soap_bytes(r.content),
        transform=sanitize,
        cache_key=cache_key,
    )


def fetch_auszug(
    client: RegisterClient, key: str, fnr: str, stichtag: str, cache: Path
) -> CachedResponse:
    return _post(
        client,
        key,
        auszug_request(fnr, stichtag),
        cache,
        {"operation": "AUSZUG_V2", "fnr": norm_fnr(fnr)},
    )


def make_client(max_rps: float) -> RegisterClient:
    cfg = yaml.safe_load((CONFIG_DIR / "ingest.yaml").read_text(encoding="utf-8"))
    return RegisterClient(
        user_agent=cfg["user_agent"], max_rps=max_rps, max_retries=int(cfg.get("max_retries", 6))
    )


def cmd_frame(args: argparse.Namespace) -> int:
    require_vault()
    key = load_secret("justiz", "JUSTIZ_API_KEY")
    day, end = date.fromisoformat(args.von), date.fromisoformat(args.bis)
    total: set[str] = set()
    with make_client(args.rps) as client:
        while day <= end:
            d = day.isoformat()
            name = f"{d}_{args.rechtsform}.json" if args.rechtsform else f"{d}.json"
            try:
                r = _post(
                    client,
                    key,
                    veraenderungen_request(d, d, args.rechtsform),
                    RAW / "frame" / name,
                    {"operation": "VERAENDERUNGENFIRMA", "day": d},
                )
                if r.body:
                    total.update(frame_fnrs(r.body))
            except IngestError as e:
                print(f"  {d}: {e}")
            day += timedelta(days=1)
    print(f"frame: {len(total)} distinct FNRs from {args.von}..{args.bis}")
    return 0


def cached_frame() -> list[str]:
    fnrs: set[str] = set()
    for p in sorted((RAW / "frame").glob("*.json")):
        env = json.loads(p.read_text(encoding="utf-8"))
        if env.get("body"):
            fnrs.update(frame_fnrs(env["body"]))
    return sorted(fnrs)


def cmd_fetch(args: argparse.Namespace) -> int:
    require_vault()
    key = load_secret("justiz", "JUSTIZ_API_KEY")
    fnrs = cached_frame()
    if not fnrs:
        raise SystemExit("empty frame; run `ingest_at frame` first")
    sample = random.Random(args.seed).sample(fnrs, min(args.sample, len(fnrs)))
    stichtag = args.stichtag or (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    ok = err = 0
    with make_client(args.rps) as client:
        for i, fnr in enumerate(sample, 1):
            try:
                fetch_auszug(client, key, fnr, stichtag, RAW / "auszug" / f"{norm_fnr(fnr)}.json")
                ok += 1
            except IngestError as e:
                err += 1
                print(f"  {norm_fnr(fnr)}: {e}")
            if i % 100 == 0:
                print(
                    f"  {i}/{len(sample)} ok={ok} errors={err} status={client.stats.status_counts}",
                    flush=True,
                )
    print(
        f"fetch: {ok} ok, {err} errors "
        f"(frame {len(fnrs)}, sample {len(sample)}, Stichtag {stichtag})"
    )
    return 0


def scrub_cached(path: Path) -> bool:
    """Re-apply `sanitize` to one cached envelope; True if it was rewritten."""
    env = json.loads(path.read_text(encoding="utf-8"))
    body = env.get("body")
    if not body:
        return False
    clean = sanitize(body)
    if clean == body:
        return False
    env["body"] = clean
    env["scrubbed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(env, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return True


def cmd_scrub(args: argparse.Namespace) -> int:
    require_vault()
    n = sum(scrub_cached(p) for p in sorted((RAW / "auszug").glob("*.json")))
    print(f"scrub: {n} cached extracts rewritten (sole traders -> stub, addresses dropped)")
    return 0


TEMPLATE_MIN = 20  # a text shared by >= 20 companies is a register template, safe to print


_DATE_RE = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b|\b\d{4}-\d{2}-\d{2}\b")


def template_key(text: str) -> str:
    """Text with dates replaced by <DATUM>, so register boilerplate collapses to templates."""
    return _DATE_RE.sub("<DATUM>", text)


def _length_bucket(n: int) -> str:
    for hi in (50, 100, 200, 400):
        if n <= hi:
            return f"<= {hi}"
    return "> 400"


def cmd_stats(args: argparse.Namespace) -> int:
    """Aggregates only: codes, counts, and texts that at least TEMPLATE_MIN companies share."""
    keys = ("rechtsform", "fken", "vart", "vsbeide", "fken_vart", "fields", "txt_len")
    c: dict[str, Counter[str]] = {k: Counter() for k in keys}
    txt_companies: dict[str, set[str]] = {}
    text_companies: dict[str, set[str]] = {}
    n = with_txt = skipped = 0
    for p in sorted((RAW / "auszug").glob("*.json")):
        env = json.loads(p.read_text(encoding="utf-8"))
        if not env.get("body"):
            continue
        rec = extract_at(env["body"])
        if rec.rechtsform_code in SOLE_TRADER_FORMS:
            skipped += 1
            continue
        n += 1
        c["rechtsform"][f"{rec.rechtsform_code} {rec.rechtsform_text}"] += 1
        for f in rec.functions:
            c["fken"][f"{f.fken} {f.fkentext}"] += 1
            c["vart"][f"{f.vart_code} | {f.vart_text}"] += 1
            if f.vsbeide_code:
                c["vsbeide"][f"{f.vsbeide_code} | {f.vsbeide_text}"] += 1
            c["fken_vart"][f"{f.fken} / {f.vart_code} / {f.vsbeide_code}"] += 1
            for name, present in (
                ("TXTVERTR", f.txtvertr),
                ("VERTRETUNGSBEFUGTNURFUER", f.nur_fuer),
                ("TEXT", f.text),
                ("BEZUGSPERSON", f.has_bezugsperson),
            ):
                c["fields"][name] += bool(present)
            for t in f.txtvertr:
                c["txt_len"][_length_bucket(len(t))] += 1
                txt_companies.setdefault(template_key(t), set()).add(rec.fnr)
            for t in f.text:
                text_companies.setdefault(f"{f.fken}: {template_key(t)}", set()).add(rec.fnr)
        with_txt += any(f.txtvertr for f in rec.functions)
    print(
        f"auszug: {n} companies (+{skipped} sole traders skipped); "
        f"{with_txt} with free-text representation (TXTVERTR)"
    )
    for k, ctr in c.items():
        print(f"  {k}: {json.dumps(dict(ctr.most_common(40)), ensure_ascii=False)}")
    for label, groups in (("TXTVERTR", txt_companies), ("TEXT", text_companies)):
        templates = sorted(
            ((len(v), t) for t, v in groups.items() if len(v) >= TEMPLATE_MIN), reverse=True
        )
        covered = sum(k for k, _ in templates)
        print(
            f"  {label}: {len(groups)} distinct (dates normalised); "
            f"shared by >= {TEMPLATE_MIN} companies ({covered} uses):"
        )
        for k, t in templates[:30]:
            print(f"    {k:>6d}  {t}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ingest_at", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("frame", help="company numbers from the daily change feed")
    p.add_argument("--von", required=True)
    p.add_argument("--bis", required=True)
    p.add_argument("--rechtsform")
    p.add_argument("--rps", type=float, default=2.0)
    p.set_defaults(func=cmd_frame)
    p = sub.add_parser("fetch", help="Kurzinformation extracts for a seeded sample of the frame")
    p.add_argument("--sample", type=int, default=2000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--stichtag")
    p.add_argument("--rps", type=float, default=2.0)
    p.set_defaults(func=cmd_fetch)
    sub.add_parser("scrub", help="re-apply sanitize to cached extracts").set_defaults(
        func=cmd_scrub
    )
    sub.add_parser("stats", help="aggregate counts over cached extracts").set_defaults(
        func=cmd_stats
    )
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
