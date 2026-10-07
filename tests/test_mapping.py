"""Tests for the ontology loaders (plan-03 Task 2)."""

from __future__ import annotations

from signrule.ontology.cir import ROLES, Group
from signrule.ontology.mapping import load_legal_vocab, load_no_rules, load_roles

REGISTER_DESCRIPTIONS = {  # Fullmakt API docs §2.4, copied verbatim
    "R0001": "Styrets leder alene",
    "R0002": "Daglig leder alene",
    "R0003": "Styret i fellesskap",
    "R0004": "Styrets leder og ett styremedlem i fellesskap",
    "R0005": "To styremedlemmer i fellesskap",
    "R0006": "Styrets medlemmer hver for seg",
    "R0007": "Daglig leder eller styrets leder alene",
    "R0008": "Daglig leder og styrets leder i fellesskap",
    "R0009": "Deltakerne i fellesskap",
    "R0010": "Deltakerne hver for seg",
    "R0011": "Styrets leder og nestleder hver for seg",
    "R0012": "Styrets leder alene eller to styremedlemmer i fellesskap",
    "R0013": "Daglig leder alene eller styrets medlemmer hver for seg",
    "R0014": "Daglig leder og ett styremedlem i fellesskap",
    "R0015": "Styrets leder og nestleder i fellesskap",
    "R0016": "Tre styremedlemmer i fellesskap",
    "R0017": "Styrets leder og to styremedlemmer i fellesskap",
    "R0018": "Nestleder og ett styremedlem i fellesskap",
}


def test_all_r_codes_present():
    rules = load_no_rules()
    for i in range(1, 19):
        code = f"R{i:04d}"
        assert code in rules and rules[code].kind == "rule" and rules[code].alternatives


def test_role_combo_codes_present():
    rules = load_no_rules()
    for code in ("STYR", "DELT", "SIGN", "SIFE", "SIHV", "PROK", "POFE", "POHV"):
        assert code in rules
    assert rules["POFE"].mode == "joint" and rules["POHV"].mode == "sole"
    assert rules["STYR"].alternatives == (Group(collective="ALL_BOARD"),)


def test_groups_use_known_roles():
    for spec in load_no_rules().values():
        for g in spec.alternatives:
            assert all(role in ROLES for role, _ in g.roles)


def test_descriptions_match_register():
    rules = load_no_rules()
    for code, desc in REGISTER_DESCRIPTIONS.items():
        assert rules[code].description == desc


def test_roles_config():
    roles = load_roles()
    assert roles.info("NO", "DAGL").cir == "CEO"
    assert roles.info("NO", "LEDE").label == "Styrets leder"
    assert roles.info("NO", "XYZ", "Ukjent rolle").cir == "OTHER"
    assert roles.order_key("NO", "DAGL") < roles.order_key("NO", "MEDL")
    assert roles.not_applicable("NO", "AS") == frozenset()


def test_legal_vocab_has_role_words():
    vocab = load_legal_vocab("NO")
    assert {"daglig", "leder", "styrets", "prokura"} <= vocab
