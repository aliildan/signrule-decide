"""scripts/make_tables.py: tables come from results files and are injected between markers."""

from __future__ import annotations

import importlib.util

spec = importlib.util.spec_from_file_location("make_tables", "scripts/make_tables.py")
assert spec is not None and spec.loader is not None
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)


def _pq(**acc):
    return {q: {"calibrated": {"accuracy": a, "n": n}} for q, (a, n) in acc.items()}


def test_coalition_mean_is_weighted_by_items():
    pq = _pq(two_ceos=(1.0, 30), member_alone=(0.5, 10), min_signers=(0.0, 99))
    assert mt.coalition_mean(pq) == (35 / 40, 40)


def test_blocks_are_replaced_and_check_detects_stale_text():
    doc = "intro\n<!-- table:at -->\nold\n<!-- /table -->\nend\n"
    new, changed = mt.inject(doc, {"at": "| a |\n|---|\n| 1 |"})
    assert changed and "| 1 |" in new and "old" not in new and new.endswith("end\n")
    again, changed = mt.inject(new, {"at": "| a |\n|---|\n| 1 |"})
    assert again == new and not changed


def test_percent_formatting():
    assert mt.pct(0.9934) == "99.3 %" and mt.pct(None) == "–"


def test_markdown_tables_convert_to_latex():
    md = "| Model | GF alone |\n|---|---|\n| **SignRule-Decide 4B** | 99.3 % |\n\nNote with 1.3 %."
    tex = mt.to_latex(md)
    assert "\\begin{tabular}{lr}" in tex and "\\toprule" in tex
    assert "\\textbf{SignRule-Decide 4B} & 99.3\\,\\% \\\\" in tex
    assert "Note" not in tex  # the note never sits inside the tabular (it would be scaled with it)
    assert mt.latex_note(md) == "Note with 1.3\\,\\%."


def test_headline_table_has_the_key_austrian_numbers():
    md = mt.headline()
    assert md.startswith("| Austria, 400 extracts never trained on | SignRule-Decide 4B |")
    for label in (
        "Managing director alone",
        "coalition questions",
        "Minimum signers",
        "Dangerous errors",
        "2 % risk",
    ):
        assert label in md


def test_coalition_set_matches_the_ontology_used_by_the_reports():
    from signrule.ontology.coalitions import ALL_COALITIONS

    assert set(mt.COALITIONS) == set(ALL_COALITIONS)  # 15: the pre-registered definition
