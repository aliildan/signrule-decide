"""Drop personal identifiers from register payloads before anything is written to disk.

CLAUDE.md §2.3: birth dates and personal ID numbers are dropped at ingest time, never stored.
Names are kept at this stage (they are needed to mask the same names inside rule texts) and are
replaced by [PERSON_n] tokens before data/processed/.
"""

from __future__ import annotations

import re
from typing import Any

from signrule.common.ids import fnr_valid, isikukood_valid

SANITIZER_VERSION = "strip_personal_ids/v2+redact_text"

_TRANSLIT = str.maketrans({"ø": "o", "æ": "ae", "å": "a", "ä": "a", "ö": "o", "ü": "u", "õ": "o"})

# Normalised key names (lower-case, transliterated, alphanumerics only).
PERSONAL_ID_KEYS = frozenset(
    {
        # Norway
        "fodselsnummer",
        "fodselsdato",
        "foedselsnummer",
        "foedselsdato",
        "personnummer",
        "dnummer",
        # Denmark
        "cpr",
        "cprnummer",
        "foedselsdag",
        # Austria / Germany
        "geburtsdatum",
        "geburtsdaten",
        "svnr",
        "sozialversicherungsnummer",
        # Estonia
        "isikukood",
        "synniaeg",
        "sunniaeg",
        "sunnikuupaev",
        "isikukoodhash",
        "fyysiliseisikukood",
        "fyysiliseisikusynniaeg",
        "osaomanikusynniaeg",
        # Generic
        "birthdate",
        "dateofbirth",
        "dob",
        "nationalid",
        "personalidentificationnumber",
        "ssn",
    }
)


def normalise_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower().translate(_TRANSLIT))


def strip_personal_ids(obj: Any) -> tuple[Any, int]:
    """Return a copy of `obj` without personal-ID keys, and the number of keys removed."""
    removed = 0

    def walk(x: Any) -> Any:
        nonlocal removed
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if normalise_key(str(k)) in PERSONAL_ID_KEYS:
                    removed += 1
                    continue
                out[k] = walk(v)
            return out
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    return walk(obj), removed


_DATE = re.compile(r"^(?:\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4})$")
_FNR_LIKE = re.compile(r"^\d{11}$")
_PERSON_KEYS = ("navn", "name", "fornavn", "etternavn", "vorname", "nachname", "nimi")


_ORG_KEYS = (
    "organisasjonsnummer",
    "orgnr",
    "cvrnummer",
    "cvr",
    "firmenbuchnummer",
    "registrikood",
    "ariregistrikood",
)


def _is_person_like(d: dict[str, Any]) -> bool:
    """An object with a name field that is not an organisation (companies have names too)."""
    keys = [normalise_key(str(k)) for k in d]
    if any(k.startswith(_ORG_KEYS) for k in keys):
        return False
    return any(k.startswith(_PERSON_KEYS) for k in keys)


def _is_national_id(s: str) -> bool:
    return bool(_FNR_LIKE.match(s)) and (fnr_valid(s) or isikukood_valid(s))


_ID_IN_TEXT = re.compile(r"(?<![\d-])(\d{11})(?![\d-])")
_BIRTH_IN_TEXT = re.compile(
    r"(?i)\b(sünniaeg|sünnikuupäev|sündinud|sünd\.|s\.|født|f\.|fødselsdato|geb\.|geboren"
    r"|born|d\.o\.b\.|jahrgang)\s*:?\s*"
    r"(\d{1,2}\.\s?\d{1,2}\.\s?\d{4}|\d{4}-\d{2}-\d{2}|(?:18|19|20)\d{2}\b)"
)
_EMAIL_IN_TEXT = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def redact_text(text: str) -> tuple[str, int]:
    """Replace personal IDs, birth dates (after a birth marker) and e-mail addresses inside a free
    text with [ID] / [DATE] / [EMAIL]. Returns (text, number of replacements). Names are masked
    later (normalize), with the record's own name list."""
    n = 0

    def rid(m: re.Match[str]) -> str:
        nonlocal n
        if fnr_valid(m.group(1)) or isikukood_valid(m.group(1)):
            n += 1
            return "[ID]"
        return m.group(0)

    def rdate(m: re.Match[str]) -> str:
        nonlocal n
        n += 1
        return f"{m.group(1)} [DATE]"

    def rmail(m: re.Match[str]) -> str:
        nonlocal n
        n += 1
        return "[EMAIL]"

    text = _ID_IN_TEXT.sub(rid, text)
    text = _BIRTH_IN_TEXT.sub(rdate, text)
    return _EMAIL_IN_TEXT.sub(rmail, text), n


def redact_strings(obj: Any) -> tuple[Any, int]:
    """`redact_text` on every string value (free texts carry birth years, IDs, e-mails)."""
    n = 0

    def walk(x: Any) -> Any:
        nonlocal n
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, str):
            t, k = redact_text(x)
            n += k
            return t
        return x

    return walk(obj), n


def sanitize_record(obj: Any) -> tuple[Any, int]:
    """Default cache sanitizer: personal-ID keys removed, then free texts redacted."""
    clean, removed = strip_personal_ids(obj)
    clean, redacted = redact_strings(clean)
    return clean, removed + redacted


def scan_for_personal_ids(obj: Any) -> list[str]:
    """JSON paths of values that look like personal IDs or birth dates. Never returns values.

    Catches what `strip_personal_ids` cannot know by key name: any string that is a valid
    Norwegian or Estonian national ID, and any bare date inside an object that also carries a
    person's name.
    """
    hits: list[str] = []

    def walk(x: Any, path: str) -> None:
        if isinstance(x, dict):
            person = _is_person_like(x)
            for k, v in x.items():
                p = f"{path}.{k}" if path else str(k)
                if isinstance(v, str):
                    s = v.strip()
                    if _is_national_id(s) or (person and _DATE.match(s)):
                        hits.append(p)
                else:
                    walk(v, p)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
        elif isinstance(x, str) and _is_national_id(x.strip()):
            hits.append(path)

    walk(obj, "")
    return hits
