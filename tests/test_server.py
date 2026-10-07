"""Tests for the serving layer with a stub model (plan-09 T1-T2). No GPU, no Kev."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

from signrule.calib.calibration import AbstainPolicy
from signrule.format.request import load_questions

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("server_app", ROOT / "server/app.py")
assert _spec and _spec.loader
app_mod = importlib.util.module_from_spec(_spec)
sys.modules["server_app"] = app_mod
_spec.loader.exec_module(app_mod)

QC = load_questions()


def canonical(qid: str) -> dict:
    return {k: v for k, v in QC.questions[qid].items()}


def stub(p_ceo: float = 0.97, p_parse: float = 0.99, seen: list | None = None):
    async def answer(req):
        if seen is not None:
            seen.append(req)
        out = {}
        for qid, q in req["questions"].items():
            if q["type"] == "noul":
                out[qid] = {"type": "noul", "noul": p_parse if qid == "parseable" else p_ceo}
            else:
                keys = list(q["criteria"])
                probs = {k: (0.9 if i == 0 else 0.1 / (len(keys) - 1)) for i, k in enumerate(keys)}
                out[qid] = {"type": q["type"], "choice": keys[0], "probabilities": probs}
        return {"answers": out, "usage": {"input_tokens": 1, "output_tokens": 0}}

    return answer


POLICY = AbstainPolicy(
    temperatures={"choice": 1.0, "noul": 1.0, "score": 1.0},
    thresholds={"0.02": {"noul": 0.95, "choice": 0.95, "score": None}},
)


def run(service, body):
    return asyncio.run(service.systemone(body))


def test_names_removed_and_masked_before_model():
    seen: list = []
    svc = app_mod.Service(stub(seen=seen), POLICY)
    body = {
        "state": {
            "jurisdiction": "NO",
            "legal_form": "AS",
            "rule_text": "Kari Fiktivsen og styrets leder i fellesskap.",
            "roles": [{"role": "Daglig leder", "count": 1, "name": "Kari Fiktivsen"}],
        },
        "questions": {"ceo_alone": canonical("ceo_alone")},
    }
    run(svc, body)
    st = seen[0]["state"]
    assert st["signature_rule"] == "[PERSON_1] og styrets leder i fellesskap."
    assert "name" not in st["roles"][0] and "rule_text" not in st
    assert "parseable" not in seen[0]["questions"]  # no gate since plan-13


def test_injected_parseable_not_returned_and_confident_answer_kept():
    svc = app_mod.Service(stub(p_ceo=0.97), POLICY)
    resp = run(
        svc,
        {
            "state": {"signature_rule": "Daglig leder alene."},
            "questions": {"ceo_alone": canonical("ceo_alone")},
        },
    )
    a = resp["answers"]
    assert set(a) == {"ceo_alone"}
    assert a["ceo_alone"]["abstain"] is False and a["ceo_alone"]["calibrated"] is True


def test_low_confidence_abstains():
    svc = app_mod.Service(stub(p_ceo=0.80), POLICY)
    a = run(
        svc, {"state": {"signature_rule": "x"}, "questions": {"ceo_alone": canonical("ceo_alone")}}
    )["answers"]
    assert a["ceo_alone"]["abstain"] is True


def test_low_parseable_no_longer_gates_other_answers():
    svc = app_mod.Service(stub(p_ceo=0.99, p_parse=0.2), POLICY)
    a = run(
        svc,
        {
            "state": {"signature_rule": "x"},
            "questions": {"ceo_alone": canonical("ceo_alone"), "parseable": canonical("parseable")},
        },
    )["answers"]
    assert a["ceo_alone"]["abstain"] is False


def test_policy_per_jurisdiction_and_strictest_fallback():
    loose = AbstainPolicy(
        temperatures={"choice": 1.0, "noul": 1.0, "score": 1.0},
        thresholds={"0.02": {"noul": 0.90, "choice": 0.90, "score": None}},
    )
    strict = AbstainPolicy(
        temperatures={"choice": 1.0, "noul": 1.0, "score": 1.0},
        thresholds={"0.02": {"noul": 0.99, "choice": 0.99, "score": None}},
    )
    svc = app_mod.Service(stub(p_ceo=0.95), {"NO": loose, "AT": strict})
    q = {"ceo_alone": canonical("ceo_alone")}
    no = run(svc, {"state": {"jurisdiction": "NO", "signature_rule": "x"}, "questions": q})
    at = run(svc, {"state": {"jurisdiction": "AT", "signature_rule": "x"}, "questions": q})
    dk = run(svc, {"state": {"jurisdiction": "DK", "signature_rule": "x"}, "questions": q})
    assert no["answers"]["ceo_alone"]["abstain"] is False and no["jurisdiction_calibrated"] is True
    assert at["answers"]["ceo_alone"]["abstain"] is True
    assert dk["answers"]["ceo_alone"]["abstain"] is True and dk["jurisdiction_calibrated"] is False


def test_non_canonical_question_is_uncalibrated_and_not_gated_by_ltt():
    svc = app_mod.Service(stub(p_ceo=0.6), POLICY)
    q = {"type": "noul", "instructions": "Does the text mention a bank?"}
    a = run(svc, {"state": {"signature_rule": "x"}, "questions": {"custom": q}})["answers"]
    assert a["custom"]["calibrated"] is False and a["custom"]["abstain"] is False


def test_residual_name_risk_abstains():
    svc = app_mod.Service(stub(), POLICY)
    a = run(
        svc,
        {
            "state": {"signature_rule": "Signatur: Kari Fiktivsen alene."},
            "questions": {"ceo_alone": canonical("ceo_alone")},
        },
    )["answers"]
    assert a["ceo_alone"]["abstain"] is True


def test_temperature_applied_to_noul():
    pol = AbstainPolicy(temperatures={"noul": 2.0, "choice": 1.0, "score": 1.0}, thresholds={})
    svc = app_mod.Service(stub(p_ceo=0.9), pol)
    a = run(
        svc, {"state": {"signature_rule": "x"}, "questions": {"ceo_alone": canonical("ceo_alone")}}
    )["answers"]
    assert a["ceo_alone"]["noul"] == pytest.approx(0.75, abs=1e-6)  # 0.9^.5/(0.9^.5+0.1^.5)


def test_empty_questions_rejected():
    svc = app_mod.Service(stub(), POLICY)
    with pytest.raises(ValueError):
        run(svc, {"state": "x", "questions": {}})


def test_consistency_gate_abstains_on_contradiction():
    # rule_type -> first criteria key (sole_ceo) at 0.9, ceo_alone p_yes 0.02: contradictory
    svc = app_mod.Service(stub(p_ceo=0.02), POLICY)
    body = {
        "state": {
            "jurisdiction": "AT",
            "legal_form": "GmbH",
            "roles": [{"role": "Geschäftsführer", "count": 1}],
        },
        "questions": {"rule_type": canonical("rule_type"), "ceo_alone": canonical("ceo_alone")},
    }
    a = run(svc, body)["answers"]
    assert a["rule_type"]["choice"] == "sole_ceo" and a["ceo_alone"]["noul"] < 0.5
    for q in ("rule_type", "ceo_alone"):
        assert a[q]["abstain"] is True
        assert a[q]["abstain_reason"] in (
            "answers are mutually inconsistent",
            "confidence below the 2%-risk threshold",
        )
    assert a["ceo_alone"]["abstain_reason"] == "answers are mutually inconsistent"


def test_consistent_answers_not_gated():
    svc = app_mod.Service(stub(p_ceo=0.99), POLICY)
    body = {
        "state": {"signature_rule": "x"},
        "questions": {"rule_type": canonical("rule_type"), "ceo_alone": canonical("ceo_alone")},
    }
    a = run(svc, body)["answers"]
    assert a["ceo_alone"]["abstain"] is False


def test_austrian_multi_line_text_is_not_a_name_risk():
    svc = app_mod.Service(stub(p_ceo=0.99), POLICY)
    body = {
        "state": {
            "jurisdiction": "AT",
            "legal_form": "GmbH",
            "signature_rule": "Geschäftsführer [PERSON_1]: vertritt seit 01.03.2019 selbständig; "
            "Geschäftsführer [PERSON_2]: vertritt seit 15.06.2021 selbständig",
        },
        "questions": {"ceo_alone": canonical("ceo_alone")},
    }
    a = run(svc, body)["answers"]["ceo_alone"]
    assert a["abstain"] is False, a.get("abstain_reason")


def test_ambiguity_is_experimental_and_always_abstains():
    svc = app_mod.Service(stub(p_ceo=0.99), POLICY)
    body = {
        "state": {
            "jurisdiction": "NO",
            "legal_form": "AS",
            "signature_rule": "Daglig leder alene.",
        },
        "questions": {"ambiguity": canonical("ambiguity"), "ceo_alone": canonical("ceo_alone")},
    }
    out = run(svc, body)["answers"]
    assert out["ambiguity"]["abstain"] is True
    assert out["ambiguity"]["abstain_reason"] == "experimental question (not a decision output)"
    assert "probabilities" in out["ambiguity"]  # the distribution is still returned
    assert out["ceo_alone"]["abstain"] is False


def test_device_selection_and_load_options_per_platform():
    assert app_mod.pick_device("auto", cuda=True, mps=False) == "cuda"
    assert app_mod.pick_device("auto", cuda=False, mps=True) == "mps"
    assert app_mod.pick_device("auto", cuda=False, mps=False) == "cpu"
    assert app_mod.pick_device("cpu", cuda=True, mps=True) == "cpu"
    import torch
    from kev.checkpoint import LoadOptions

    base = LoadOptions()
    assert app_mod.serving_options("cuda", base).dtype == torch.bfloat16
    assert app_mod.serving_options("mps", base).backend == "auto"  # MLX where it pays
    assert app_mod.serving_options("cpu", base).dtype == torch.float32
    explicit = LoadOptions(backend="torch")
    assert app_mod.serving_options("mps", explicit).backend == "torch"  # a caller's choice wins
