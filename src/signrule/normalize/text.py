"""Text normalisation (CLAUDE.md §6 step 4): NFC, whitespace; casing and punctuation are kept."""

from __future__ import annotations

import re
import unicodedata

_SPACES = re.compile(r"[\s  -​  　﻿]+")
_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―−"})
_TRAILING = re.compile(r"[\s.]+$")


def normalize_text(s: str) -> str:
    """NFC, every Unicode space variant -> one ASCII space, trimmed. Case and punctuation kept."""
    return _SPACES.sub(" ", unicodedata.normalize("NFC", s)).strip()


def text_key(s: str | None) -> str:
    """Dedup/leakage key: normalised, casefolded, dashes unified, trailing periods dropped."""
    if not s:
        return ""
    t = normalize_text(s).casefold().translate(_DASHES)
    return _TRAILING.sub("", t)
