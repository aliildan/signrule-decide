"""Second scrub of the masked review window (eval/review.py) on hand-written fixture strings."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from signrule.common.ids import make_isikukood

_SPEC = importlib.util.spec_from_file_location(
    "review", Path(__file__).resolve().parents[1] / "eval" / "review.py"
)
review = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(review)


def test_unknown_capitalised_words_become_x():
    s = review.Scrubber("AT")
    out = s.text("Geschäftsführer [PERSON_1]: vertritt gemeinsam mit Herrn Fixturename Beispiel")
    assert out == "Geschäftsführer [PERSON_1]: vertritt gemeinsam mit Herrn [X] [X]"
    assert s.replaced == 2


def test_tokens_vocabulary_and_lowercase_survive():
    s = review.Scrubber("NO")
    text = "Styrets leder alene eller [PERSON_2] og ett styremedlem i fellesskap. Fiktiv AS"
    assert s.text(text) == (
        "Styrets leder alene eller [PERSON_2] og ett styremedlem i fellesskap. [X] AS"
    )


def test_ids_and_birth_dates_are_redacted():
    s = review.Scrubber("NO")
    code = make_isikukood("3800101123")
    out = s.text(f"prokura til [PERSON_1] (f. 01.01.1980) {code}")
    assert "1980" not in out and code not in out and s.redacted == 2


def test_cap_is_enforced():
    import pytest

    with pytest.raises(SystemExit):
        review.main(["--jurisdiction", "at", "--split", "pilot", "--n", "31"])
