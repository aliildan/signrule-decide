"""National-ID shape checks shared by the ingest scanner and the publish check."""

from __future__ import annotations

_K1 = (3, 7, 6, 1, 8, 9, 4, 5, 2)
_K2 = (5, 4, 3, 2, 7, 6, 5, 4, 3, 2)


def _control(digits: list[int], weights: tuple[int, ...]) -> int | None:
    k = 11 - sum(d * w for d, w in zip(digits, weights, strict=True)) % 11
    if k == 11:
        return 0
    return None if k == 10 else k


def fnr_valid(s: str) -> bool:
    """True if `s` is 11 digits with valid Norwegian fødselsnummer/D-number control digits."""
    if len(s) != 11 or not s.isdigit():
        return False
    d = [int(c) for c in s]
    return _control(d[:9], _K1) == d[9] and _control(d[:10], _K2) == d[10]


def make_fnr(prefix9: str) -> str | None:
    """Append control digits to a 9-digit prefix; None if no valid number exists (tests only)."""
    d = [int(c) for c in prefix9]
    k1 = _control(d, _K1)
    if k1 is None:
        return None
    k2 = _control([*d, k1], _K2)
    return None if k2 is None else f"{prefix9}{k1}{k2}"


_EE_W1 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 1)
_EE_W2 = (3, 4, 5, 6, 7, 8, 9, 1, 2, 3)
_EE_CENTURY = {1: 1800, 2: 1800, 3: 1900, 4: 1900, 5: 2000, 6: 2000, 7: 2100, 8: 2100}


def _ee_check(d: list[int]) -> int:
    k = sum(a * w for a, w in zip(d, _EE_W1, strict=True)) % 11
    if k == 10:
        k = sum(a * w for a, w in zip(d, _EE_W2, strict=True)) % 11
    return 0 if k == 10 else k


def isikukood_valid(s: str) -> bool:
    """True if `s` is a well-formed Estonian isikukood (GYYMMDDSSSC: century digit, valid date,
    check digit)."""
    import datetime as _dt

    if len(s) != 11 or not s.isdigit() or int(s[0]) not in _EE_CENTURY:
        return False
    d = [int(c) for c in s]
    try:
        _dt.date(_EE_CENTURY[d[0]] + int(s[1:3]), int(s[3:5]), int(s[5:7]))
    except ValueError:
        return False
    return _ee_check(d[:10]) == d[10]


def make_isikukood(prefix10: str) -> str:
    """Append the check digit to a 10-digit prefix (tests only)."""
    return f"{prefix10}{_ee_check([int(c) for c in prefix10])}"
