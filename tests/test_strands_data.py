"""Strands Decider training examples (plan-17 T2). Fixtures are hand-written and fictitious."""

import json

import pytest

from signrule.format.request import STATE_KEYS, load_questions, to_request
from signrule.train import strands_data as sd

QC = load_questions()

# FICTITIOUS FIXTURE: not a register record.
STATE = {
    "jurisdiction": "AT",
    "legal_form": "GmbH",
    "signature_rule": "Geschäftsführer [PERSON_1]: vertritt seit <DATUM> gemeinsam mit einem "
    "weiteren Geschäftsführer oder einem Prokuristen",
    "roles": [{"role": "Geschäftsführer", "count": 2}, {"role": "Prokurist", "count": 1}],
}


def _request(answers: dict) -> dict:
    return to_request(STATE, answers, QC, source="fixture")


def _ollama_render(state: object, q: dict) -> tuple[str, list[tuple[int, int]]]:
    """Python port of Ollama's decision/strands.go (encodeStrands, strandsField, strandsContent)
    for the shapes we send: what the served model reads. Returns (prefix + prompt, option spans
    relative to the prompt)."""
    if isinstance(state, str):
        rendered = state.strip()
    else:
        rendered = json.dumps(state, indent=2, ensure_ascii=False)
    prefix = "<state>\n" + rendered + "\n</state>\n"
    header = {
        "noul": "Decide whether the statement is true of the state.",
        "choice": "Select exactly one option.",
        "score": "Rate the state against the ordered levels below (lowest first).",
    }[q["type"]]
    if q["type"] == "noul":
        crit = {
            "false": "the statement does not hold for this state",
            "true": "the statement holds for this state",
            **(q.get("criteria") or {}),
        }
        choices = [("false", crit["false"]), ("true", crit["true"])]
    elif q["type"] == "choice":
        choices = [(k, v or "") for k, v in q["criteria"].items()]
    else:
        choices = [(str(i), v) for i, v in enumerate(q["criteria"])]
    prompt = f'<question type="{q["type"]}">\n{header}\n{q["instructions"].strip()}\n<options>\n'
    spans = []
    for i, (value, desc) in enumerate(choices):
        start = len(prompt)
        prompt += f"{i + 1}. {value}"
        desc = " ".join(desc.split())
        if desc:
            prompt += " — " + desc
        spans.append((start, len(prompt)))
        prompt += "\n"
    prompt += "</options>\n</question>\n<answer>"
    return prefix + prompt, spans


def test_one_example_per_labelled_question() -> None:
    req = _request({"ceo_alone": False, "two_ceos": True, "min_signers": "2"})
    exs = sd.to_examples(req)
    assert [e["task"] for e in exs] == ["at:ceo_alone", "at:two_ceos", "at:min_signers"]
    assert [e["kind"] for e in exs] == ["noul", "noul", "choice"]
    assert exs[0]["label"] == 0 and exs[1]["label"] == 1
    assert exs[2]["options"][exs[2]["label"]][0] == "2"
    assert all(e["state"] == STATE for e in exs)


def test_noul_options_match_ollama_defaults() -> None:
    (ex,) = sd.to_examples(_request({"ceo_alone": True}))
    assert ex["options"] == [
        ["false", "the statement does not hold for this state"],
        ["true", "the statement holds for this state"],
    ]
    req = _request({"ceo_alone": True})
    req["questions"]["ceo_alone"]["criteria"] = {"true": "yes, alone"}
    (ex,) = sd.to_examples(req)
    assert ex["options"][1] == ["true", "yes, alone"]


def test_state_rendering_is_canonical_json() -> None:
    shuffled = {k: STATE[k] for k in reversed(list(STATE))}
    req = to_request(shuffled, {"ceo_alone": False}, QC, source="fixture")
    (ex,) = sd.to_examples(req)
    assert list(ex["state"]) == [k for k in STATE_KEYS if k in STATE]
    with pytest.raises(ValueError, match="unknown state key"):
        sd.to_examples({**req, "state": {**STATE, "owner": "x"}})


def test_score_and_choice_labels_are_canonical_indices() -> None:
    (amb,) = sd.to_examples(_request({"ambiguity": 1}))
    assert amb["kind"] == "score" and amb["label"] == 1
    assert [o[0] for o in amb["options"]] == ["0", "1", "2", "3"]
    (rt,) = sd.to_examples(_request({"rule_type": "joint_role_combo"}))
    assert [o[0] for o in rt["options"]] == list(QC.questions["rule_type"]["criteria"])


def test_training_prompt_equals_ollama_rendering() -> None:
    prompting = pytest.importorskip("strands_decider.prompting")
    fmt = pytest.importorskip("strands_decider.data.format")
    req = _request({"ceo_alone": False, "min_signers": "2", "ambiguity": 0})
    for ex in sd.to_examples(req):
        qid = ex["task"].split(":", 1)[1]
        text, rq = prompting.build_prompt(ex["state"], fmt.Example.from_dict(ex).to_question())
        served, spans = _ollama_render(req["state"], req["questions"][qid])
        assert text == served
        # both sides measure option spans within the question text (after the state prefix)
        assert list(rq.option_spans) == spans


def test_noul_defaults_equal_strands_constant() -> None:
    prompting = pytest.importorskip("strands_decider.prompting")
    assert sd.NOUL_DEFAULT_CRITERIA == prompting.NOUL_DEFAULT_CRITERIA
