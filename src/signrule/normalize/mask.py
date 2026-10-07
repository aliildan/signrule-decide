"""Person-name masking for rule texts (CLAUDE.md §6 step 3).

Names come from the record's own person lists (the register returns everyone with a signing
role). Full names are replaced first, then individual name tokens, by `[PERSON_n]` numbered by
first appearance. `residual_name_risk` flags texts that still look like they contain a name;
the pipeline drops those records instead of guessing.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from signrule.ontology.mapping import load_legal_vocab

_SENTINEL = "\x00{}\x00"
_SENTINEL_RE = re.compile("\x00(\\d+)\x00")
_POSSESSIVE = r"(?:'?s)?"
_LEGAL_FORMS = frozenset({"AS", "ASA", "SA", "ANS", "DA", "KS", "BA", "SE", "NUF", "IKS", "BRL"})
_WORD = re.compile(r"\[PERSON_\d+\]|[^\W\d_]+(?:['\-][^\W\d_]+)*")
_SENTENCE_END = re.compile(r"[.!?]\s+")


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[\s\-]+", name) if t]


def mask_text(text: str, names: Sequence[str], lang: str = "NO") -> str:
    if not text or not names:
        return text
    vocab = load_legal_vocab(lang)
    persons: list[str] = []
    for n in names:
        n = unicodedata.normalize("NFC", " ".join(n.split()))
        if n and n not in persons:
            persons.append(n)

    patterns: list[tuple[re.Pattern[str], int]] = []
    for i, p in sorted(enumerate(persons), key=lambda x: -len(x[1])):
        toks = _tokens(p)
        body = r"[\s\-]+".join(re.escape(t) for t in toks)
        patterns.append((re.compile(rf"(?<!\w){body}{_POSSESSIVE}(?!\w)", re.I), i))
    seen: set[str] = set()
    for i, p in enumerate(persons):
        for t in _tokens(p):
            key = t.casefold()
            if len(t) < 3 or key in vocab or key in seen:
                continue
            seen.add(key)
            patterns.append((re.compile(rf"(?<!\w){re.escape(t)}{_POSSESSIVE}(?!\w)", re.I), i))

    out = unicodedata.normalize("NFC", text)
    for pat, i in patterns:
        out = pat.sub(_SENTINEL.format(i), out)

    numbering: dict[str, int] = {}

    def renumber(m: re.Match[str]) -> str:
        k = numbering.setdefault(m.group(1), len(numbering) + 1)
        return f"[PERSON_{k}]"

    return _SENTINEL_RE.sub(renumber, out)


def residual_name_risk(text: str, lang: str = "NO") -> bool:
    """True if a capitalised, non-vocabulary word appears where a name could be.

    Ignored: the first word of each sentence, all-caps tokens (abbreviations, company names in
    capitals), `[PERSON_n]`, vocabulary words, and capitalised runs directly followed by a legal
    form (`Fiktiv Holding AS` is a company, not a person).
    """
    if not text:
        return False
    vocab = load_legal_vocab(lang)
    for sentence in _SENTENCE_END.split(unicodedata.normalize("NFC", text)):
        words = _WORD.findall(sentence)
        for j, w in enumerate(words):
            if j == 0 or w.startswith("[PERSON_") or w.isupper() or not w[0].isupper():
                continue
            if all(part.lower() in vocab for part in re.split(r"['\-]", w)):
                continue
            k = j
            while k < len(words) and words[k][:1].isupper() and not words[k].isupper():
                k += 1
            if k < len(words) and words[k] in _LEGAL_FORMS:
                continue
            return True
    return False
