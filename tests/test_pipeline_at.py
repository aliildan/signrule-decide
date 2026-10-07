"""Austrian training-data keys and the wording-rule baseline (fixture strings only)."""

from __future__ import annotations

import numpy as np

from signrule.evaluation.harness import RulesAt
from signrule.normalize.pipeline_at import n_managing_directors, pattern_key, scan_row


def test_pattern_key_ignores_dates_only():
    a = "Geschäftsführer [PERSON_1]: vertritt seit 01.02.2020 selbständig"
    b = "Geschäftsführer [PERSON_1]: vertritt seit 31.12.1999 selbständig"
    c = "Geschäftsführer [PERSON_1]: vertritt seit 31.12.1999 gemeinsam mit einem Prokuristen"
    assert pattern_key(a) == pattern_key(b) != pattern_key(c)
    assert pattern_key(None) == ""


def test_n_managing_directors():
    st = {"roles": [{"role": "Geschäftsführer", "count": 2}, {"role": "Prokurist", "count": 1}]}
    assert n_managing_directors(st) == 2


def test_scan_row_flags_unmasked_name():
    line = (
        '{"state": {"signature_rule": "Geschäftsführer [PERSON_1]: vertritt gemeinsam mit '
        'Herrn Fixturename"}, "questions": {}}'
    )
    assert scan_row(line)["residual_name"] == 1


def _req(sig: str, prok: str | None = None, qids=("ceo_alone",)) -> dict:
    roles = [{"role": "Geschäftsführer", "count": sig.count("Geschäftsführer")}]
    if prok:
        roles.append({"role": "Prokurist", "count": prok.count("Prokurist")})
    st = {"signature_rule": sig, "procuration_rule": prok, "roles": roles}
    return {"state": st, "questions": {q: {} for q in qids}, "_meta": {"id": "x"}}


def test_rules_at_reads_selbstaendig_and_gemeinsam():
    rules = RulesAt()
    two_sole = _req(
        "Geschäftsführer [PERSON_1]: vertritt seit 01.01.2020 selbständig; "
        "Geschäftsführer [PERSON_2]: vertritt seit 01.01.2021 selbständig",
        "Prokurist [PERSON_3]: vertritt seit 01.01.2022 gemeinsam mit einem Geschäftsführer",
        qids=("ceo_alone", "prokura_present", "prokura_joint", "min_signers"),
    )
    out = rules.predict([two_sole])
    assert np.argmax(out[("x", "ceo_alone")]) == 1
    assert np.argmax(out[("x", "prokura_present")]) == 1
    assert np.argmax(out[("x", "prokura_joint")]) == 1
    assert np.argmax(out[("x", "min_signers")]) == rules.qc.keys("min_signers").index("1")


def test_rules_at_abstains_on_mixed_lines():
    mixed = _req(
        "Geschäftsführer [PERSON_1]: vertritt seit 01.01.2020 selbständig; "
        "Geschäftsführer [PERSON_2]: vertritt seit 01.01.2021 gemeinsam mit einem Prokuristen"
    )
    p = np.exp(RulesAt().predict([mixed])[("x", "ceo_alone")])
    assert np.allclose(p, 0.5)
