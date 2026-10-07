"""Tests for dedup, splits and leakage (plan-03 T7)."""

from __future__ import annotations

from datetime import date

from signrule.normalize.dedup import (
    Member,
    dedup,
    leakage_report,
    random_split,
    temporal_split,
)


def member(
    eid: str,
    text: str,
    label: str = "sole_ceo",
    reg: date | None = date(2020, 1, 1),
    roles: int = 1,
) -> Member:
    req = {
        "state": {
            "jurisdiction": "NO",
            "legal_form": "AS",
            "signature_rule": text,
            "roles": [{"role": "Styremedlem", "count": roles}],
        },
        "questions": {"rule_type": {"type": "choice", "label": label}},
    }
    return Member(eid, req, text.casefold(), "", reg, has_text=bool(text))


def test_duplicates_grouped_and_counted():
    gs = dedup([member("1", "A alene"), member("2", "A alene"), member("3", "B alene")])
    sizes = sorted(g.n_duplicates for g in gs)
    assert sizes == [1, 2]


def test_label_conflicts_resolved_by_majority_ties_dropped():
    gs = dedup(
        [member("1", "A", "sole_ceo"), member("2", "A", "sole_ceo"), member("3", "A", "sole_chair")]
    )
    assert gs[0].request["questions"]["rule_type"]["label"] == "sole_ceo"
    assert gs[0].conflicts == {"rule_type": 1}
    tie = dedup([member("1", "B", "sole_ceo"), member("2", "B", "sole_chair")])
    assert tie == []  # its only question was a tie -> no labelled questions left


def test_random_split_never_splits_a_text_cluster():
    ms = [member(str(i), f"text {i % 7}", roles=i % 3 + 1) for i in range(60)]
    gs = dedup(ms)
    assign = random_split(gs)
    by_text: dict[str, set[str]] = {}
    for g in gs:
        by_text.setdefault(g.members[0].signature_key, set()).add(assign[g.cluster])
    assert all(len(s) == 1 for s in by_text.values())


def test_split_is_stable_when_data_grows():
    gs1 = dedup([member("1", "A"), member("2", "B")])
    gs2 = dedup([member("1", "A"), member("2", "B"), member("3", "C")])
    a1, a2 = random_split(gs1), random_split(gs2)
    assert all(a2[c] == s for c, s in a1.items())


def test_temporal_split_cutoff():
    new = date(2025, 6, 1)
    gs = dedup(
        [
            member("1", "old text", reg=date(2019, 1, 1)),
            member("2", "old text", reg=new, roles=2),  # same text, new entity -> not tested
            member("3", "new text", reg=new),
        ]
    )
    assign, seen_new = temporal_split(gs)
    by_text = {g.members[0].signature_key: assign[g.cluster] for g in gs}
    assert by_text["new text"] == "test"
    assert by_text["old text"] in ("train", "val")
    assert seen_new == 1


def test_leakage_report_detects_overlap_and_near_dups():
    gs = dedup([member("1", "styrets leder alene"), member("2", "styrets leder alene.")])
    # force the two near-identical texts into different splits
    assign = {gs[0].cluster: "train", gs[1].cluster: "test"}
    rep = leakage_report(gs, assign, threshold=0.8)
    assert rep.exact_text_overlap == 0 and rep.entity_overlap == 0
    assert rep.near_dup_rate == 1.0
