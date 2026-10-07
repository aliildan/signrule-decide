"""Tests for text normalisation (plan-03 T5) and name masking (plan-03 T4)."""

from __future__ import annotations

from signrule.normalize.mask import mask_text, residual_name_risk
from signrule.normalize.text import normalize_text, text_key
from signrule.ontology.mapping import load_no_rules

# ---- text ----------------------------------------------------------------------------------


def test_normalize_keeps_case_and_punctuation():
    assert normalize_text("  Daglig  leder\talene. ") == "Daglig leder alene."


def test_text_key_unifies_unicode_variants():
    a = "Styrets leder og nestleder i fellesskap – hver for seg."
    b = "Styrets leder og nestleder i fellesskap - hver for seg"
    assert text_key(a) == text_key(b)
    assert text_key("Åse") == text_key("Åse")  # composed vs decomposed
    assert text_key(None) == ""


# ---- masking -------------------------------------------------------------------------------


def test_full_name_masked():
    out = mask_text(
        "Fixture Person og Kari Fiktivsen hver for seg.", ["Fixture Person", "Kari Fiktivsen"]
    )
    assert out == "[PERSON_1] og [PERSON_2] hver for seg."


def test_possessive_and_hyphenated_names_masked():
    out = mask_text("Signatur: Ola Fiktiv-Navnesen's fullmakt, Navnesens.", ["Ola Fiktiv-Navnesen"])
    assert "Navnesen" not in out and "Fiktiv" not in out and "Ola" not in out


def test_single_name_tokens_masked_with_same_person_token():
    out = mask_text("Kari Fiktivsen alene. Fiktivsen kan også signere.", ["Kari Fiktivsen"])
    assert out == "[PERSON_1] alene. [PERSON_1] kan også signere."


def test_masking_preserves_role_words():
    text = "Daglig leder og Styrets leder i fellesskap. Prokura."
    assert mask_text(text, ["Leder Daglig"]) == text  # name tokens that are role words are kept


def test_residual_name_detector_flags_unknown_name():
    assert residual_name_risk("Signatur: Kari Fiktivsen alene.", "NO")
    assert residual_name_risk("Styrets leder og Fiktivsen i fellesskap.", "NO")


def test_residual_detector_quiet_on_standard_phrases():
    for spec in load_no_rules().values():
        assert not residual_name_risk(spec.description + ".", "NO"), spec.description
    assert not residual_name_risk("Daglig Leder alene. [PERSON_1] og [PERSON_2].", "NO")
    assert not residual_name_risk("Styret i FIRMA AS i fellesskap.", "NO")
