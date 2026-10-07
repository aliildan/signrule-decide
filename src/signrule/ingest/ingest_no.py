"""Norway ingest: Enhetsregisteret (entities) + Fullmakttjenesten (signatur/prokura rules).

Usage (python -m signrule.ingest.ingest_no ...):
    entities              bulk entity list -> in-scope projection
    roles-schema          key structure of roller/totalbestand (nothing stored)
    fullmakt [--limit N]  per-orgnr signatur + prokura, cached, resumable
    stats                 aggregate counts over the cache

Output is aggregate only (counts, codes, lengths). Rule texts, names and records are never
printed (CLAUDE.md §2.3, §10).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import yaml

from signrule.common.paths import CONFIG_DIR, raw_dir, require_vault
from signrule.ingest.http import IngestError, RegisterClient
from signrule.ingest.pii import SANITIZER_VERSION, redact_strings

log = logging.getLogger("ingest_no")

ENHETSREGISTERET = "https://data.brreg.no/enhetsregisteret/api"
FULLMAKT = "https://data.brreg.no/fullmakt"
FULLMAKT_ACCEPT = "application/vnd.brreg.enhetsregisteret.fullmakt.oppslag.v1+json;charset=UTF-8"
FULLMAKT_TYPES = ("signatur", "prokura")

RAW = raw_dir("no")
ENTITIES_DIR = RAW / "enheter"
FULLMAKT_DIR = RAW / "fullmakt"


# ---- config ---------------------------------------------------------------------------------


def load_ingest_config() -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / "ingest.yaml").read_text(encoding="utf-8"))


def load_scope() -> dict[str, int]:
    """Organisation form -> fetch tier (1 = first)."""
    cfg = yaml.safe_load((CONFIG_DIR / "no_scope.yaml").read_text(encoding="utf-8"))
    blocked = set(cfg.get("exempt", [])) | set(cfg.get("excluded", []))
    scope = {form: int(tier) for tier, forms in cfg["tiers"].items() for form in forms}
    overlap = blocked & scope.keys()
    if overlap:
        raise ValueError(f"forms both in scope and exempt/excluded: {sorted(overlap)}")
    return scope


def make_client() -> RegisterClient:
    cfg = load_ingest_config()
    return RegisterClient(
        user_agent=cfg["user_agent"],
        max_rps=float(cfg.get("max_rps", 5.0)),
        max_retries=int(cfg.get("max_retries", 6)),
    )


# ---- entities -------------------------------------------------------------------------------


def project_entity(e: dict[str, Any]) -> dict[str, Any]:
    """Whitelisted entity fields. No names or addresses (ENK names are personal data)."""
    return {
        "orgnr": e.get("organisasjonsnummer"),
        "orgform": (e.get("organisasjonsform") or {}).get("kode"),
        "registered": e.get("registreringsdatoEnhetsregisteret"),
        "founded": e.get("stiftelsesdato"),
        "konkurs": e.get("konkurs"),
        "under_avvikling": e.get("underAvvikling"),
        "under_tvangsavvikling": e.get("underTvangsavviklingEllerTvangsopplosning"),
        "deleted": e.get("slettedato"),
        "nace": (e.get("naeringskode1") or {}).get("kode"),
    }


def _snapshot_tag(last_modified: str | None) -> str:
    """YYYY-MM-DD from Last-Modified (RFC 2822, or Java's `Sat Oct 03 04:26:42 CEST 2026`)."""
    if last_modified:
        try:
            return parsedate_to_datetime(last_modified).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            m = re.match(r"\w{3} (\w{3}) (\d{1,2}) [\d:]+ \S+ (\d{4})$", last_modified.strip())
            if m:
                return datetime.strptime(" ".join(m.groups()), "%b %d %Y").strftime("%Y-%m-%d")
    return datetime.now(UTC).strftime("%Y-%m-%d")


def latest_entities_snapshot() -> Path:
    snaps = sorted(ENTITIES_DIR.glob("enheter_*.jsonl.gz"))
    if not snaps:
        raise SystemExit(
            "no entities snapshot; run `python -m signrule.ingest.ingest_no entities` first"
        )
    return snaps[-1]


def iter_entities(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def cmd_entities(args: argparse.Namespace) -> int:
    require_vault()
    url = f"{ENHETSREGISTERET}/enheter/lastned"
    scope = load_scope()
    with make_client() as client:
        head = client.head(url)
        tag = _snapshot_tag(head.headers.get("last-modified"))
        out = ENTITIES_DIR / f"enheter_{tag}.jsonl.gz"
        if out.exists() and not args.refresh:
            print(f"entities snapshot {tag} already cached; skipping (use --refresh to re-fetch)")
            return 0
        mib = int(head.headers.get("content-length", 0)) >> 20
        print(f"downloading entity bulk file ({mib} MiB)...")
        resp = client.get_bytes(url)
    raw = resp.content
    digest = hashlib.sha256(raw).hexdigest()
    entities = json.loads(gzip.decompress(raw))
    del raw

    forms_all: Counter[str] = Counter()
    forms_kept: Counter[str] = Counter()
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for e in entities:
            p = project_entity(e)
            forms_all[p["orgform"] or "?"] += 1
            if p["orgform"] in scope:
                forms_kept[p["orgform"]] += 1
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    tmp.replace(out)
    manifest = {
        "url": url,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "last_modified": head.headers.get("last-modified"),
        "sha256_download": digest,
        "n_total": sum(forms_all.values()),
        "n_in_scope": sum(forms_kept.values()),
        "forms_total": dict(forms_all.most_common()),
        "forms_in_scope": dict(forms_kept.most_common()),
        "projection": "project_entity/v1",
    }
    out.with_suffix("").with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"entities: {manifest['n_total']} total, {manifest['n_in_scope']} in scope -> {out.name}")
    for form, n in forms_kept.most_common():
        print(f"  {form:5s} {n:>8d}")
    return 0


# ---- roles bulk: structure only -------------------------------------------------------------


def key_paths(obj: Any, prefix: str = "", counts: Counter[str] | None = None) -> Counter[str]:
    """Count every key path in a JSON structure. Values are never recorded."""
    counts = Counter() if counts is None else counts
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            counts[path] += 1
            key_paths(v, path, counts)
    elif isinstance(obj, list):
        for v in obj:
            counts[f"{prefix}[]"] += 1
            key_paths(v, f"{prefix}[]", counts)
    return counts


def _code_values(
    obj: Any, prefix: str = "", out: dict[str, Counter[str]] | None = None
) -> dict[str, Counter[str]]:
    """Value counts for enumeration fields only (keys named `kode`)."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if k == "kode" and isinstance(v, str):
                out.setdefault(path, Counter())[v] += 1
            else:
                _code_values(v, path, out)
    elif isinstance(obj, list):
        for v in obj:
            _code_values(v, f"{prefix}[]", out)
    return out


def cmd_roles_schema(args: argparse.Namespace) -> int:
    """Resolve CLAUDE.md §13 Q1: does roller/totalbestand carry signatur/prokura free text?

    Downloads into memory only; nothing is written to disk.
    """
    url = f"{ENHETSREGISTERET}/roller/totalbestand"
    with make_client() as client:
        resp = client.get_bytes(url)
    data = json.loads(gzip.decompress(resp.content))
    sample = data[: args.sample] if isinstance(data, list) else data
    paths = key_paths(sample)
    codes = _code_values(sample)
    n = len(data) if isinstance(data, list) else 1
    print(f"roller/totalbestand: {n} top-level records; key paths over first {len(sample)}:")
    for p, c in sorted(paths.items()):
        print(f"  {c:>9d}  {p}")
    print("enumeration values (`kode` fields):")
    for p, ctr in sorted(codes.items()):
        top = ", ".join(f"{k}={v}" for k, v in ctr.most_common(40))
        print(f"  {p}: {top}")
    hits = [
        p
        for p in paths
        if any(s in p.lower() for s in ("fritekst", "signatur", "prokura", "tekst"))
    ]
    print(f"paths mentioning fritekst/signatur/prokura/tekst: {hits or 'none'}")
    return 0


# ---- fullmakt -------------------------------------------------------------------------------


@dataclass(frozen=True)
class FullmaktFields:
    orgnr: str | None
    orgform: str | None
    rutine: str | None
    rutine_text: str | None
    regel: str | None
    regel_ident: str | None
    regel_text: str | None
    kombinasjon_status: str | None
    grunnlag: str | None
    kombinasjon_codes: tuple[str, ...]
    fritekst: str | None
    rolle_fritekst_count: int
    mulige_roller: dict[str, int]
    signatur_roller: dict[str, int]


def _get(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _role_counts(people: Iterable[dict[str, Any]] | None) -> dict[str, int]:
    return dict(Counter(_get(p, "rolle", "kode") or "?" for p in people or []))


def extract_fullmakt(body: dict[str, Any]) -> FullmaktFields:
    grunnlag = body.get("signeringsGrunnlag") or {}
    sp = grunnlag.get("signaturProkuraRoller") or {}
    mulige = _get(grunnlag, "muligeSigneringsRoller", "personRolleGrunnlag") or []
    persons = list(sp.get("personRolleGrunnlag") or []) + list(mulige)
    for k in _get(body, "signeringsKombinasjon", "kombinasjon") or []:
        persons += k.get("personRolleKombinasjon") or []
    fritekst = sp.get("signaturProkuraFritekst")
    return FullmaktFields(
        orgnr=_get(body, "enhet", "organisasjonsnummer"),
        orgform=_get(body, "enhet", "organisasjonsform", "kode"),
        rutine=_get(body, "status", "rutineStatus", "kode"),
        rutine_text=_get(body, "status", "rutineStatus", "tekstforklaring"),
        regel=_get(body, "status", "regelStatus", "kode"),
        regel_ident=_get(body, "status", "regelStatus", "regelIdent"),
        regel_text=_get(body, "status", "regelStatus", "tekstforklaring"),
        kombinasjon_status=_get(body, "status", "kombinasjonStatus", "kode"),
        grunnlag=grunnlag.get("kode"),
        kombinasjon_codes=tuple(
            k.get("kode") for k in _get(body, "signeringsKombinasjon", "kombinasjon") or []
        ),
        fritekst=fritekst if isinstance(fritekst, str) and fritekst.strip() else None,
        rolle_fritekst_count=sum(1 for p in persons if p.get("rolleFritekst")),
        mulige_roller=_role_counts(mulige),
        signatur_roller=_role_counts(sp.get("personRolleGrunnlag")),
    )


def _len_bucket(n: int) -> str:
    for edge in (20, 40, 80, 160, 320, 640):
        if n < edge:
            return f"<{edge}"
    return ">=640"


def summarize(rows: Iterable[FullmaktFields]) -> dict[str, Any]:
    """Aggregate statistics. Contains codes and counts only, never text."""
    c: dict[str, Counter[Any]] = {
        k: Counter()
        for k in (
            "orgform",
            "rutine",
            "regel",
            "regel_ident",
            "kombinasjon_status",
            "grunnlag",
            "kombinasjon_code",
            "has_fritekst",
            "fritekst_len",
            "rolle_fritekst",
            "rutine_text_na",
            "regel_text_rt",
        )
    }
    n = 0
    for r in rows:
        n += 1
        c["orgform"][r.orgform] += 1
        c["rutine"][r.rutine] += 1
        c["regel"][r.regel] += 1
        c["regel_ident"][r.regel_ident] += 1
        c["kombinasjon_status"][r.kombinasjon_status] += 1
        c["grunnlag"][r.grunnlag] += 1
        for code in r.kombinasjon_codes:
            c["kombinasjon_code"][code] += 1
        c["has_fritekst"][r.fritekst is not None] += 1
        if r.fritekst:
            c["fritekst_len"][_len_bucket(len(r.fritekst))] += 1
        c["rolle_fritekst"][r.rolle_fritekst_count > 0] += 1
        # Register-generated status explanations are fixed templates, not entity text.
        if r.rutine == "NA":
            c["rutine_text_na"][r.rutine_text] += 1
        if r.regel == "RT":
            c["regel_text_rt"][r.regel_text] += 1
    return {"n": n, **{k: dict(v.most_common()) for k, v in c.items()}}


def fullmakt_cache_path(kind: str, orgnr: str) -> Path:
    return FULLMAKT_DIR / kind / orgnr[:3] / f"{orgnr}.json"


def select_orgnrs(
    scope: dict[str, int],
    forms: set[str] | None,
    sample: int | None,
    seed: int,
    tiers: set[int] | None = None,
) -> list[str]:
    """In-scope, non-deleted orgnrs: tier 1 first, seeded shuffle within each tier.

    The shuffle makes every prefix of a (possibly interrupted) run a representative sample.
    """
    rows = [
        e
        for e in iter_entities(latest_entities_snapshot())
        if e["orgform"] in scope
        and (forms is None or e["orgform"] in forms)
        and (tiers is None or scope[e["orgform"]] in tiers)
        and not e["deleted"]
    ]
    rng = random.Random(seed)
    if sample is not None:
        rows = rng.sample(rows, min(sample, len(rows)))
    rows.sort(key=lambda e: e["orgnr"])
    rng.shuffle(rows)
    rows.sort(key=lambda e: scope[e["orgform"]])  # stable: keeps the shuffle within a tier
    return [e["orgnr"] for e in rows]


def cmd_fullmakt(args: argparse.Namespace) -> int:
    require_vault()
    scope = load_scope()
    forms = set(args.forms.split(",")) if args.forms else None
    tiers = {int(t) for t in args.tiers.split(",")} if args.tiers else None
    orgnrs = select_orgnrs(scope, forms, args.sample, args.seed, tiers)
    if args.limit:
        orgnrs = orgnrs[: args.limit]
    kinds = args.types.split(",")
    jobs = [(o, k) for o in orgnrs for k in kinds]
    print(
        f"fullmakt: {len(orgnrs)} orgnrs x {kinds} = {len(jobs)} lookups "
        f"(cached ones are skipped), {args.workers} workers",
        flush=True,
    )
    failed: list[str] = []
    done = 0
    t0 = time.monotonic()
    with make_client() as client:

        def one(job: tuple[str, str]) -> str | None:
            orgnr, kind = job
            try:
                client.get_json(
                    f"{FULLMAKT}/enheter/{orgnr}/{kind}",
                    fullmakt_cache_path(kind, orgnr),
                    headers={"Accept": FULLMAKT_ACCEPT},
                )
            except IngestError as e:
                log.warning("%s", e)
                return f"{orgnr} {kind}"
            return None

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(one, jobs):
                done += 1
                if result:
                    failed.append(result)
                if done % args.progress == 0:
                    s = client.stats
                    rate = s.requests / max(time.monotonic() - t0, 1e-9)
                    print(
                        f"  {done}/{len(jobs)} done; requests={s.requests} ({rate:.2f}/s) "
                        f"cache_hits={s.cache_hits} retries={s.retries} errors={len(failed)} "
                        f"status={s.status_counts}",
                        flush=True,
                    )
        s = client.stats
    if failed:
        err = FULLMAKT_DIR / "_errors.txt"
        err.parent.mkdir(parents=True, exist_ok=True)
        with err.open("a", encoding="utf-8") as f:
            f.write("\n".join(failed) + "\n")
    print(
        f"fullmakt finished: {done} lookups, requests={s.requests} cache_hits={s.cache_hits} "
        f"retries={s.retries} errors={len(failed)} status={s.status_counts}",
        flush=True,
    )
    return 1 if failed and len(failed) == done else 0


def iter_cached_fullmakt(kind: str) -> Iterator[FullmaktFields]:
    for path in sorted((FULLMAKT_DIR / kind).glob("*/*.json")):
        env = json.loads(path.read_text(encoding="utf-8"))
        if env.get("status_code") == 200 and env.get("body"):
            yield extract_fullmakt(env["body"])


def cmd_stats(args: argparse.Namespace) -> int:
    for kind in args.types.split(","):
        s = summarize(iter_cached_fullmakt(kind))
        print(f"== {kind}: {s.pop('n')} cached responses")
        for field, counts in s.items():
            print(f"  {field}: {json.dumps(counts, ensure_ascii=False, default=str)}")
    return 0


# ---- CLI ------------------------------------------------------------------------------------


def rescrub_cached(path: Path) -> bool:
    """Re-apply the in-text redaction to one cached envelope (birth years etc. stored before
    sanitizer v2); True if it was rewritten."""
    env = json.loads(path.read_text(encoding="utf-8"))
    if env.get("body") is None:
        return False
    body, n = redact_strings(env["body"])
    if not n:
        return False
    env["body"] = body
    env["sanitizer"] = SANITIZER_VERSION
    env["scrubbed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(env, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return True


def cmd_scrub(args: argparse.Namespace) -> int:
    require_vault()
    n = seen = 0
    for path in FULLMAKT_DIR.rglob("*.json"):
        seen += 1
        n += rescrub_cached(path)
    print(f"scrub: {n} of {seen} cached Fullmakt responses rewritten (in-text redaction)")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        prog="ingest_no", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("entities", help="bulk entity list -> in-scope projection")
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(func=cmd_entities)

    p = sub.add_parser("roles-schema", help="key structure of roller/totalbestand (nothing stored)")
    p.add_argument("--sample", type=int, default=20000)
    p.set_defaults(func=cmd_roles_schema)

    p = sub.add_parser("fullmakt", help="fetch signatur/prokura per orgnr (cached, resumable)")
    p.add_argument("--forms", help="comma-separated organisation forms (default: all in scope)")
    p.add_argument("--types", default=",".join(FULLMAKT_TYPES))
    p.add_argument("--limit", type=int, help="stop after N orgnrs")
    p.add_argument("--sample", type=int, help="random sample of N orgnrs instead of all")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--tiers", help="comma-separated scope tiers (default: all)")
    p.add_argument("--workers", type=int, default=4, help="threads sharing the global rate limit")
    p.add_argument("--progress", type=int, default=1000)
    p.set_defaults(func=cmd_fullmakt)

    sub.add_parser("scrub", help="re-apply in-text redaction to the fullmakt cache").set_defaults(
        func=cmd_scrub
    )
    p = sub.add_parser("stats", help="aggregate statistics over the fullmakt cache")
    p.add_argument("--types", default=",".join(FULLMAKT_TYPES))
    p.set_defaults(func=cmd_stats)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
