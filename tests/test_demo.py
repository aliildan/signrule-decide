"""scripts/demo.py helpers (no server needed)."""

from __future__ import annotations

import importlib.util

spec = importlib.util.spec_from_file_location("demo", "scripts/demo.py")
assert spec is not None and spec.loader is not None
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_german_demo_asks_only_questions_the_company_can_answer():
    ag = {
        "jurisdiction": "AT",
        "roles": [
            {"role": "Vorsitzender des Vorstands", "count": 1},
            {"role": "Vorstandsmitglied", "count": 2},
        ],
    }
    qs = demo.questions_for(ag, demo.SHOW_DE)
    assert "chair_alone" in qs and "chair_with_member" in qs
    assert "ceo_alone" not in qs and "two_ceos" not in qs
    gmbh = {"jurisdiction": "AT", "roles": [{"role": "Geschäftsführer", "count": 2}]}
    qs = demo.questions_for(gmbh, demo.SHOW_DE)
    assert "ceo_alone" in qs and "chair_alone" not in qs


def test_german_answers_are_formatted_in_german():
    assert demo.fmt_de({"type": "noul", "noul": 0.99}) == "ja (0,99)"
    assert demo.fmt_de({"type": "noul", "noul": 0.01}) == "nein (0,99)"
    assert demo.fmt_de({"type": "noul", "noul": 0.7, "abstain": True}) == "– (keine Antwort)"
    assert (
        demo.fmt_de({"type": "choice", "choice": "2", "probabilities": {"1": 0.1, "2": 0.9}})
        == "2 (0,90)"
    )


def test_every_german_example_is_austrian_and_hand_written():
    assert len(demo.EXAMPLES_DE) >= 8
    assert all(state["jurisdiction"] == "AT" for _, state in demo.EXAMPLES_DE)
    assert all("[PERSON_" in state.get("signature_rule", "") for _, state in demo.EXAMPLES_DE)
