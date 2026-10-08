"""strands_bench output conversion (plan-17 T6)."""

from signrule.train.strands_bench import probabilities, question_payload


def test_noul_becomes_false_true_probabilities() -> None:
    assert probabilities({"type": "noul", "noul": 0.75}) == {"false": 0.25, "true": 0.75}


def test_choice_and_score_keep_their_option_keys() -> None:
    ans = {
        "type": "choice",
        "choice": "2",
        "probabilities": {"1": 0.1, "2": 0.9},
        "confidence": 0.8,
    }
    assert probabilities(ans) == {"1": 0.1, "2": 0.9}
    score = {"type": "score", "score": 0.2, "legend": {}, "probabilities": {0: 0.8, 1: 0.2}}
    assert probabilities(score) == {"0": 0.8, "1": 0.2}


def test_question_payload_drops_labels() -> None:
    req = {
        "questions": {"ceo_alone": {"type": "noul", "instructions": "x", "label": True, "src": "s"}}
    }
    assert question_payload(req) == {"ceo_alone": {"type": "noul", "instructions": "x"}}
