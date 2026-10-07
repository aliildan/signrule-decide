"""Denmark ingest (plan-04): CVR system-til-system (Elasticsearch 6.8) for rule texts and roles;
Datafordeleren for metadata. Denmark is an evaluation jurisdiction (no structured labels for us).

    python -m signrule.ingest.ingest_dk schema     # Datafordeler GraphQL schema (metadata only)
    python -m signrule.ingest.ingest_dk es-count   # CVR ES: document counts for the scope queries
    python -m signrule.ingest.ingest_dk es-fetch   # CVR ES: whitelisted projection of the scope
    python -m signrule.ingest.ingest_dk es-stats   # aggregate statistics over the projection

Credentials come from secrets/*.env at runtime and are never printed or logged; Datafordeler puts
the API key in the URL, so URLs and error bodies are never echoed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import httpx
import yaml

from signrule.common.paths import CONFIG_DIR, REPO_ROOT, raw_dir, require_vault
from signrule.common.secrets import load_secret
from signrule.ingest.pii import SANITIZER_VERSION, redact_text, scan_for_personal_ids

GRAPHQL = "https://graphql.datafordeler.dk/CVR/v2"
# The service is HTTP only (no TLS endpoint); basic auth with the free system-til-system user.
CVR_ES = "http://distribution.virk.dk/cvr-permanent"
ES_RPS = 2.0  # well under the project limit (§2.7); one request at a time
EXTERNAL = REPO_ROOT / "docs" / "external" / "datafordeler-cvr"
RAW = raw_dir("dk")
INTERESTING = re.compile(
    r"tegning|virksomhed|deltag|relation|funktion|reklame|ledelse|organisation", re.I
)


def _user_agent() -> str:
    return yaml.safe_load((CONFIG_DIR / "ingest.yaml").read_text())["user_agent"]


def _get(url: str, params: dict[str, str]) -> httpx.Response:
    try:
        return httpx.get(url, params=params, headers={"User-Agent": _user_agent()}, timeout=120)
    except httpx.HTTPError as e:  # message would contain the URL with the key
        raise SystemExit(f"request failed: {type(e).__name__}") from None


def parse_sdl_types(sdl: str) -> dict[str, list[str]]:
    """type/input/enum name -> field names, from GraphQL SDL (no values involved)."""
    types: dict[str, list[str]] = {}
    for m in re.finditer(
        r"^(?:type|input|enum|interface)\s+(\w+)[^{]*\{(.*?)^\}", sdl, re.M | re.S
    ):
        fields = re.findall(r"^\s+(\w+)\s*[:(]", m.group(2), re.M) or re.findall(
            r"^\s+(\w+)\s*$", m.group(2), re.M
        )
        types[m.group(1)] = fields
    return types


def cmd_schema(args: argparse.Namespace) -> int:
    key = load_secret("datafordeler", "DATAFORDELER_API_KEY")
    base = GRAPHQL.rsplit("/", 1)[0]
    resp = _get(f"{base}/{args.version}/schema", {"apikey": key})
    if resp.status_code != 200:
        print(f"schema request CVR/{args.version}: HTTP {resp.status_code}")
        return 1
    sdl = resp.text
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    (EXTERNAL / f"cvr_{args.version}_schema.graphql").write_text(sdl, encoding="utf-8")
    types = parse_sdl_types(sdl)
    print(f"schema: {len(sdl):,} chars, {len(types)} types -> docs/external/datafordeler-cvr/")
    hits = {t: f for t, f in types.items() if INTERESTING.search(t)}
    for t, fields in sorted(hits.items()):
        print(f"  {t}: {', '.join(fields[:40])}{' …' if len(fields) > 40 else ''}")
    tegn = sorted(t for t in types if re.search("tegning", t, re.I))
    tegn_fields = sorted(
        {f"{t}.{f}" for t, fs in types.items() for f in fs if re.search("tegning", f, re.I)}
    )
    print(f"Tegningsregel types: {tegn or 'none'}; fields: {tegn_fields or 'none'}")
    return 0


class CvrEs:
    """Minimal CVR Elasticsearch client: basic auth from the "cvr" secrets file, descriptive
    User-Agent, ES_RPS, backoff on 429/5xx. Never prints credentials, queries or response bodies."""

    def __init__(self) -> None:
        auth = (load_secret("cvr", "CVR_ES_USER"), load_secret("cvr", "CVR_ES_PASSWORD"))
        self._http = httpx.Client(
            auth=auth, headers={"User-Agent": _user_agent()}, timeout=180, base_url=CVR_ES
        )
        self._last = 0.0
        self.requests = 0

    def post(self, path: str, body: dict[str, Any], params: dict[str, str] | None = None) -> Any:
        for attempt in range(7):
            wait = self._last + 1 / ES_RPS - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self._http.post(path, json=body, params=params)
            except httpx.HTTPError as e:
                status, err = None, type(e).__name__
            else:
                status, err = resp.status_code, None
                self.requests += 1
                if status == 200:
                    return resp.json()
                if status not in (429, 500, 502, 503, 504):
                    raise SystemExit(f"CVR ES {path}: HTTP {status}")
            time.sleep(min(120, 2**attempt + random.random()))
            print(f"CVR ES {path}: retry {attempt + 1} after {err or status}", file=sys.stderr)
        raise SystemExit(f"CVR ES {path}: giving up")

    def close(self) -> None:
        self._http.close()


TEGNINGSREGEL = {"match": {"Vrvirksomhed.attributter.type": "TEGNINGSREGEL"}}


def _with(*clauses: dict[str, Any]) -> dict[str, Any]:
    return {"bool": {"filter": [TEGNINGSREGEL, *clauses]}}


SCOPE_QUERIES: dict[str, dict[str, Any]] = {
    "all companies": {"match_all": {}},
    "with a tegningsregel attribute": TEGNINGSREGEL,
    "... status Normal": _with(
        {"match": {"Vrvirksomhed.virksomhedMetadata.sammensatStatus": "Normal"}}
    ),
    "... status Aktiv": _with(
        {"match": {"Vrvirksomhed.virksomhedMetadata.sammensatStatus": "Aktiv"}}
    ),
    "... reklamebeskyttet": _with({"term": {"Vrvirksomhed.reklamebeskyttet": True}}),
}


def cmd_es_count(args: argparse.Namespace) -> int:
    es = CvrEs()
    try:
        for name, query in SCOPE_QUERIES.items():
            n = es.post("/virksomhed/_count", {"query": query})["count"]
            print(f"{name}: {n:,}")
    finally:
        es.close()
    return 0


# Only the bodies that can sign: management (Direktion, Bestyrelse, Likvidator, Filialbestyrere, …)
# and fully liable partners (Interessenter, K-repræsentant). Owners (EJERREGISTER), founders,
# auditors and every other attribute are never stored.
MANAGEMENT = frozenset({"LEDELSESORGAN", "FULDT_ANSVARLIG_DELTAGERE"})
ES_SOURCE = [
    "Vrvirksomhed.cvrNummer",
    "Vrvirksomhed.reklamebeskyttet",
    "Vrvirksomhed.virksomhedMetadata.sammensatStatus",
    "Vrvirksomhed.virksomhedMetadata.nyesteVirksomhedsform.kortBeskrivelse",
    "Vrvirksomhed.virksomhedMetadata.nyesteVirksomhedsform.virksomhedsformkode",
    "Vrvirksomhed.attributter.type",
    "Vrvirksomhed.attributter.vaerdier.vaerdi",
    "Vrvirksomhed.attributter.vaerdier.periode",
    "Vrvirksomhed.deltagerRelation.deltager.enhedstype",
    "Vrvirksomhed.deltagerRelation.deltager.navne",
    "Vrvirksomhed.deltagerRelation.organisationer.hovedtype",
    "Vrvirksomhed.deltagerRelation.organisationer.organisationsNavn",
    "Vrvirksomhed.deltagerRelation.organisationer.medlemsData.attributter.type",
    "Vrvirksomhed.deltagerRelation.organisationer.medlemsData.attributter.vaerdier.vaerdi",
    "Vrvirksomhed.deltagerRelation.organisationer.medlemsData.attributter.vaerdier.periode",
]
SCOPE = {
    "bool": {
        "filter": [
            TEGNINGSREGEL,
            {"match": {"Vrvirksomhed.virksomhedMetadata.sammensatStatus": "Normal"}},
        ],
        "must_not": [{"term": {"Vrvirksomhed.reklamebeskyttet": True}}],
    }
}
PAGE = 1000


def _current(values: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [v for v in values or [] if not (v.get("periode") or {}).get("gyldigTil")]


def _name(deltager: dict[str, Any]) -> str | None:
    names = deltager.get("navne") or []
    cur = _current(names) or names[-1:]
    n = (cur[0].get("navn") or "").strip() if cur else ""
    return n or None


def project_dk_company(doc: dict[str, Any]) -> tuple[dict[str, Any] | None, int]:
    """Whitelisted projection of one CVR ES document; None for advertising-protected companies.

    Keeps every tegningsregel version with its period, the current management/partner functions
    and the names needed for masking. Returns (projection, ID/date/e-mail redactions in the texts).
    """
    v = doc.get("Vrvirksomhed") or {}
    if v.get("reklamebeskyttet"):
        return None, 0
    redacted = 0
    rule_texts = []
    for attr in v.get("attributter") or []:
        if attr.get("type") != "TEGNINGSREGEL":
            continue
        for val in attr.get("vaerdier") or []:
            text = (val.get("vaerdi") or "").strip()
            if not text:
                continue
            text, n = redact_text(text)
            redacted += n
            per = val.get("periode") or {}
            rule_texts.append(
                {"text": text, "from": per.get("gyldigFra"), "to": per.get("gyldigTil")}
            )
    rule_texts.sort(key=lambda r: r["from"] or "")
    mask_names: list[str] = []
    firm_names: list[str] = []
    functions = []
    for rel in v.get("deltagerRelation") or []:
        deltager = rel.get("deltager") or {}
        etype = deltager.get("enhedstype")
        for org in rel.get("organisationer") or []:
            if org.get("hovedtype") not in MANAGEMENT:
                continue
            body = next((n.get("navn") for n in _current(org.get("organisationsNavn"))), None)
            for md in org.get("medlemsData") or []:
                for attr in md.get("attributter") or []:
                    if attr.get("type") != "FUNKTION":
                        continue
                    for val in _current(attr.get("vaerdier")):
                        row: dict[str, Any] = {
                            "body": body,
                            "hovedtype": org.get("hovedtype"),
                            "function": val.get("vaerdi"),
                            "enhedstype": etype,
                            "from": (val.get("periode") or {}).get("gyldigFra"),
                        }
                        name = _name(deltager)
                        person = etype != "VIRKSOMHED"  # PERSON and ANDEN_DELTAGER (no CPR)
                        names = mask_names if person else firm_names
                        if name:
                            if name not in names:
                                names.append(name)
                            row["person" if person else "firm"] = names.index(name)
                        functions.append(row)
    meta = v.get("virksomhedMetadata") or {}
    form = meta.get("nyesteVirksomhedsform") or {}
    out = {
        "cvr": v.get("cvrNummer"),
        "legal_form": form.get("kortBeskrivelse"),
        "formkode": form.get("virksomhedsformkode"),
        "status": meta.get("sammensatStatus"),
        "rule_texts": rule_texts,
        "functions": functions,
        "mask_names": mask_names,
        "firm_names": firm_names,
    }
    return out, redacted


def _write_jsonl_gz(path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def latest_projection():
    paths = sorted(RAW.glob("cvr_es_*.jsonl.gz"))
    if not paths:
        raise SystemExit("no CVR ES projection under the DK raw dir; run `es-fetch` first")
    return paths[-1]


def cmd_es_fetch(args: argparse.Namespace) -> int:
    require_vault()
    RAW.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    es = CvrEs()
    rows, seen, protected, redacted, after = [], 0, 0, 0, None
    try:
        total = es.post("/virksomhed/_count", {"query": SCOPE})["count"]
        print(f"scope: {total:,} companies")
        while True:
            body: dict[str, Any] = {
                "size": PAGE,
                "query": SCOPE,
                "_source": ES_SOURCE,
                "sort": [{"Vrvirksomhed.cvrNummer": "asc"}],
            }
            if after is not None:
                body["search_after"] = after
            hits = es.post("/virksomhed/_search", body)["hits"]["hits"]
            if not hits:
                break
            for hit in hits:
                seen += 1
                proj, n = project_dk_company(hit.get("_source") or {})
                if proj is None:
                    protected += 1
                    continue
                rows.append(proj)
                redacted += n
            after = hits[-1]["sort"]
            if args.limit and seen >= args.limit:
                break
            if seen % 20000 < PAGE:
                print(f"  {seen:,}/{total:,}", flush=True)
    finally:
        es.close()
    hits_ids = scan_for_personal_ids(rows)
    if hits_ids:
        raise SystemExit(f"{len(hits_ids)} personal-ID-like values at e.g. {hits_ids[0]}")
    stamp = now.strftime("%Y-%m-%d")
    out = RAW / f"cvr_es_{stamp}.jsonl.gz"
    _write_jsonl_gz(out, rows)
    manifest = {
        "fetched_at": now.isoformat(timespec="seconds"),
        "source": CVR_ES,
        "query": SCOPE,
        "source_fields": ES_SOURCE,
        "licence": "Vilkår for brug af danske offentlige data (v4)",
        "scope_count": total,
        "documents_seen": seen,
        "dropped_reklamebeskyttet": protected,
        "stored": out.name,
        "stored_records": len(rows),
        "redacted_in_text": redacted,
        "sanitizer": SANITIZER_VERSION,
        "requests": es.requests,
        "limit": args.limit,
    }
    (RAW / f"cvr_es_manifest_{stamp}.json").write_text(json.dumps(manifest, indent=1))
    print(
        f"{seen:,} documents -> {len(rows):,} stored ({out.name}); "
        f"reklamebeskyttet dropped {protected}, text redactions {redacted}, requests {es.requests}"
    )
    return 0


def _length_bucket(n: int) -> str:
    for lim in (40, 80, 160, 320, 640):
        if n < lim:
            return f"<{lim}"
    return ">=640"


def stats(rows: list[dict[str, Any]], template_min: int = 20) -> dict[str, Any]:
    """Aggregate statistics: counts only, plus rule texts shared by >= template_min companies
    after their own names are masked (standard wording, never one company's text)."""
    c: dict[str, Counter[Any]] = {
        k: Counter()
        for k in ("legal_form", "current_texts", "text_len", "function", "body", "n_signers_listed")
    }
    templates: Counter[str] = Counter()
    for r in rows:
        c["legal_form"][r["legal_form"]] += 1
        cur = [t for t in r["rule_texts"] if not t["to"]]
        c["current_texts"][len(cur)] += 1
        for f in r["functions"]:
            c["function"][f["function"]] += 1
            c["body"][f["body"]] += 1
        c["n_signers_listed"][min(len(r["functions"]), 6)] += 1
        for t in cur:
            c["text_len"][_length_bucket(len(t["text"]))] += 1
            text = t["text"]
            for i, name in enumerate(r["mask_names"]):
                text = text.replace(name, f"[PERSON_{i + 1}]")
            for i, name in enumerate(r["firm_names"]):
                text = text.replace(name, f"[FIRMA_{i + 1}]")
            templates[" ".join(text.split())] += 1
    shared = {t: n for t, n in templates.most_common() if n >= template_min}
    n_texts = sum(templates.values())
    return {
        "companies": len(rows),
        **{k: dict(v.most_common(30)) for k, v in c.items()},
        "unique_texts": len(templates),
        "templates_min_companies": template_min,
        "templates": len(shared),
        "template_coverage": round(sum(shared.values()) / n_texts, 4) if n_texts else 0.0,
        "top_templates": dict(list(shared.items())[:30]),
    }


def cmd_es_stats(args: argparse.Namespace) -> int:
    require_vault()
    path = latest_projection()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    print(json.dumps(stats(rows), ensure_ascii=False, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ingest_dk", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("schema", help="fetch the CVR GraphQL schema (metadata only)")
    p.add_argument("--version", default="v2")
    p.set_defaults(func=cmd_schema)
    sub.add_parser("es-count", help="CVR ES: counts for the scope queries").set_defaults(
        func=cmd_es_count
    )
    p = sub.add_parser("es-fetch", help="CVR ES: whitelisted projection of the scope")
    p.add_argument("--limit", type=int, default=0, help="stop after this many documents (0 = all)")
    p.set_defaults(func=cmd_es_fetch)
    sub.add_parser("es-stats", help="aggregate statistics over the projection").set_defaults(
        func=cmd_es_stats
    )
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
