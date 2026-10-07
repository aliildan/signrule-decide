"""Denmark normaliser on fictitious CVR projections (fixtures, not real companies)."""

from __future__ import annotations

from signrule.format.request import load_questions
from signrule.normalize import normalize_dk as n
from signrule.ontology.mapping import load_roles


def _fn(function: str, body: str, idx: int, etype: str = "PERSON") -> dict:
    key = "firm" if etype == "VIRKSOMHED" else "person"
    return {
        "body": body,
        "hovedtype": "LEDELSESORGAN",
        "function": function,
        "enhedstype": etype,
        key: idx,
    }


def _row(text: str | None = "Selskabet tegnes af en direktør.", **over) -> dict:
    row = {
        "cvr": "99999901",
        "legal_form": "APS",
        "formkode": 80,
        "status": "Normal",
        "rule_texts": [{"text": text, "from": "2020-01-01", "to": None}] if text else [],
        "functions": [_fn("DIREKTØR", "Direktion", 0)],
        "mask_names": ["Fixture Person"],
        "firm_names": [],
    }
    row.update(over)
    return row


def test_person_names_are_masked_and_collective_wording_kept_verbatim():
    roles = load_roles()
    sig, role_list, risk = n.render_dk(
        _row("Selskabet tegnes af direktør Fixture Person alene eller af direktionen."), roles
    )
    assert sig == "Selskabet tegnes af direktør [PERSON_1] alene eller af direktionen."
    assert role_list == [("Direktør", 1)] and risk is False


def test_other_participants_are_persons_and_companies_are_firms():
    roles = load_roles()
    row = _row(
        "Selskabet tegnes af Foreign Fixture i forening med Fixture Holding ApS.",
        functions=[
            _fn("BESTYRELSESMEDLEM", "Bestyrelse", 0, "ANDEN_DELTAGER"),
            _fn("BESTYRELSESMEDLEM", "Bestyrelse", 1, "VIRKSOMHED"),
        ],
        mask_names=[],
        firm_names=["Foreign Fixture", "Fixture Holding ApS"],
    )
    # the first projection version filed ANDEN_DELTAGER names under firm_names
    row["functions"][0] = {**row["functions"][0], "firm": 0}
    row["functions"][0].pop("person")
    sig, role_list, _ = n.render_dk(row, roles)
    assert sig == "Selskabet tegnes af [PERSON_1] i forening med [FIRMA_1]."
    assert role_list == [("Bestyrelsesmedlem", 2)]


def test_role_counts_per_holder_and_executive_titles():
    roles = load_roles()
    row = _row(
        functions=[
            _fn("ADM. DIR.", "Direktion", 0),
            _fn("økonomidirektør", "Direktion", 1),
            _fn("BESTYRELSESMEDLEM", "Bestyrelse", 0),
            _fn("FORMAND", "Bestyrelse", 2),
            _fn("SUPPLEANT", "Bestyrelse", 3),
        ],
        mask_names=["A Fixture", "B Fixture", "C Fixture", "D Fixture"],
    )
    _, role_list, _ = n.render_dk(row, roles)
    assert role_list == [
        ("Administrerende direktør", 1),
        ("Direktør", 1),
        ("Formand", 1),
        ("Bestyrelsesmedlem", 1),
        ("Suppleant", 1),
    ]


def test_latest_current_text_wins_and_no_text_is_skipped():
    roles, questions = load_roles(), load_questions()
    row = _row(
        rule_texts=[
            {
                "text": "Selskabet tegnes af to direktører i forening.",
                "from": "2010-01-01",
                "to": "2019-12-31",
            },
            {"text": "Selskabet tegnes af en direktør.", "from": "2015-01-01", "to": None},
            {"text": "Selskabet tegnes af en direktør alene.", "from": "2021-03-01", "to": None},
        ]
    )
    req = n.to_dk_request(row, roles, questions)
    assert req.request["state"]["signature_rule"] == "Selskabet tegnes af en direktør alene."
    assert (
        req.request["state"]["jurisdiction"] == "DK" and req.request["state"]["legal_form"] == "ApS"
    )
    assert req.request["questions"] == {} and req.request["_meta"]["id"] == "99999901"
    assert n.to_dk_request(_row(None), roles, questions) is None


def test_office_filter_drops_board_questions_without_a_board():
    state = {"roles": [{"role": "Direktør", "count": 1}]}
    answers = {
        "ceo_alone": True,
        "chair_alone": False,
        "two_board_members_jointly": False,
        "two_ceos": False,
    }
    assert n.applicable_dk(state, answers) == {"ceo_alone": True, "two_ceos": False}
    board = {"roles": [{"role": "Formand", "count": 1}, {"role": "Bestyrelsesmedlem", "count": 2}]}
    kept = n.applicable_dk(
        board, {"chair_alone": True, "ceo_alone": False, "two_board_members_jointly": True}
    )
    assert kept == {"chair_alone": True, "two_board_members_jointly": True}
