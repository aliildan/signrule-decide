"""SignRule-Decide server: Jev-compatible POST /v1/systemone with masking and abstention (plan-09).

    uv run --group serve python server/app.py --run runs/no-9b-base \
        --policy results/no-random/no-9b-base.test.json --port 8300

Kev computes the probabilities (its batching server, raw temperature). This layer adds:
- name masking: `persons[].name` / `roles[].name` are removed and their occurrences in rule
  texts replaced by [PERSON_n]; residual name risk -> every answer abstains;
- per-question-type temperature and Learn-then-Test abstention, per jurisdiction (each policy
  fitted on that jurisdiction's val part); for a jurisdiction without a policy (e.g. DK, EE) the
  strictest of all policies, flagged `jurisdiction_calibrated: false`;
- the logical-consistency gate (answers no canonical rule can produce together abstain);
- experimental questions (ambiguity) always abstain, with their distribution still returned;
- `calibrated: false` for questions whose wording differs from configs/questions.yaml.
(The old parseable gate was removed in plan-13: `parseable` is no longer trained as false.)
No generated text, ever.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np

from signrule.calib.calibration import AbstainPolicy, strictest
from signrule.calib.consistency import STRUCTURAL, has_named_signatory, is_consistent
from signrule.format.request import QuestionsConfig, load_questions
from signrule.normalize.mask import mask_text, residual_name_risk

AnswerFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
TEXT_FIELDS = ("signature_rule", "procuration_rule")


def mask_request(req: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Remove name fields, mask names in texts. Returns (request, residual_name_risk)."""
    req = copy.deepcopy(req)
    state = req.get("state")
    if not isinstance(state, dict):
        text = str(state or "")
        return req, residual_name_risk(text)
    if "rule_text" in state and "signature_rule" not in state:
        state["signature_rule"] = state.pop("rule_text")
    names: list[str] = []
    for p in state.pop("persons", None) or []:
        if isinstance(p, dict) and p.get("name"):
            names.append(str(p["name"]))
    for r in state.get("roles") or []:
        if isinstance(r, dict) and r.get("name"):
            names.append(str(r.pop("name")))
    # the legal vocabulary of the request's jurisdiction (German role nouns are capitalised); a
    # jurisdiction without a vocabulary falls back to Norwegian
    jur = str(state.get("jurisdiction") or "").upper()
    vocab_lang = jur if jur in ("NO", "AT") else "NO"
    risk = False
    for f in TEXT_FIELDS:
        if state.get(f):
            state[f] = mask_text(str(state[f]), names, vocab_lang)
            risk |= residual_name_risk(state[f], vocab_lang)
    for note in state.get("role_notes") or []:
        if isinstance(note, dict) and note.get("note"):
            note["note"] = mask_text(str(note["note"]), names, vocab_lang)
    return req, risk


def _is_canonical(qid: str, q: dict[str, Any], qc: QuestionsConfig) -> bool:
    ref = qc.questions.get(qid)
    if ref is None or ref["type"] != q.get("type"):
        return False
    crit = q.get("criteria")
    if ref["type"] == "choice":
        return isinstance(crit, dict) and list(crit) == list(ref["criteria"])
    if ref["type"] == "score":
        return isinstance(crit, list) and len(crit) == len(ref["criteria"])
    return True


def _temper(p: np.ndarray, t: float) -> np.ndarray:
    z = np.power(np.clip(p, 1e-12, 1.0), 1.0 / t)
    return z / z.sum()


# Questions answered with their distribution but never as a decision (owner decision 2026-10-07:
# the ambiguity head is weak in every model; served once a model passes a pre-registered criterion).
EXPERIMENTAL = frozenset({"ambiguity"})


def postprocess(
    answers: dict[str, Any],
    request: dict[str, Any],
    policy: AbstainPolicy | None,
    qc: QuestionsConfig,
    alpha: float,
    name_risk: bool,
    asked: set[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for qid, ans in answers.items():
        q = request["questions"].get(qid, {})
        a = dict(ans)
        canonical = policy is not None and _is_canonical(qid, q, qc)
        qtype = str(a.get("type"))
        keys: list[str] = []
        if qtype == "noul":
            p = np.array([1 - float(a["noul"]), float(a["noul"])])
        else:
            keys = list(a["probabilities"])
            p = np.array([float(a["probabilities"][k]) for k in keys])
        if canonical and policy is not None:
            p = _temper(p, policy.temperatures.get(qtype, 1.0))
            if qtype == "noul":
                a["noul"] = float(p[1])
            else:
                a["probabilities"] = {k: float(v) for k, v in zip(keys, p, strict=True)}
                if qtype == "choice":
                    a["choice"] = keys[int(p.argmax())]
                if qtype == "score":
                    a["score"] = float(np.dot(p, np.arange(len(p))))
        conf = float(p.max())
        a["calibrated"] = bool(canonical)
        if name_risk:
            a["abstain"], a["abstain_reason"] = True, "possible personal name in text"
        elif canonical and policy is not None:
            a["abstain"] = policy.abstain(qtype, conf, alpha)
            if a["abstain"]:
                a["abstain_reason"] = f"confidence below the {alpha:.0%}-risk threshold"
        else:
            a["abstain"] = False
        out[qid] = a
    _apply_consistency_gate(out, request, qc)
    for qid in EXPERIMENTAL & out.keys():
        out[qid]["abstain"] = True
        out[qid]["abstain_reason"] = "experimental question (not a decision output)"
    return {qid: a for qid, a in out.items() if qid in asked}


def _apply_consistency_gate(
    answers: dict[str, Any], request: dict[str, Any], qc: QuestionsConfig
) -> None:
    """Abstain on structural answers that no canonical rule can produce together."""
    pred: dict[str, object] = {}
    for qid, a in answers.items():
        if qid not in STRUCTURAL or not a.get("calibrated"):
            continue
        pred[qid] = a["choice"] if a["type"] == "choice" else bool(a["noul"] >= 0.5)
    if is_consistent(pred, has_named_signatory(request.get("state"))):
        return
    for qid in pred:
        a = answers[qid]
        if not a.get("abstain"):
            a["abstain"], a["abstain_reason"] = True, "answers are mutually inconsistent"


class Service:
    def __init__(
        self,
        answer_fn: AnswerFn,
        policy: AbstainPolicy | dict[str, AbstainPolicy] | None,
        alpha: float = 0.02,
    ) -> None:
        self.answer_fn = answer_fn
        self.policies = policy if isinstance(policy, dict) else None
        self.policy = None if isinstance(policy, dict) else policy
        self.fallback = strictest(list(policy.values())) if isinstance(policy, dict) else policy
        self.alpha = alpha
        self.qc = load_questions()

    def policy_for(self, state: Any) -> tuple[AbstainPolicy | None, bool]:
        if self.policies is None:
            return self.policy, True
        jur = str((state or {}).get("jurisdiction", "")).upper() if isinstance(state, dict) else ""
        if jur in self.policies:
            return self.policies[jur], True
        return self.fallback, False

    async def systemone(self, body: dict[str, Any]) -> dict[str, Any]:
        req, risk = mask_request(body)
        asked = set(req.get("questions") or {})
        if not asked:
            raise ValueError("questions must be a non-empty object")
        policy, seen = self.policy_for(req.get("state"))
        resp = await self.answer_fn(req)
        resp = dict(resp)
        resp["answers"] = postprocess(
            resp["answers"], req, policy, self.qc, self.alpha, risk, asked
        )
        resp["jurisdiction_calibrated"] = seen
        resp["model"] = body.get("model") or "signrule-decide"
        return resp


def create_app(service: Service):  # noqa: ANN201
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="signrule-decide")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/systemone")
    async def systemone(body: dict[str, Any]) -> dict[str, Any]:
        try:
            return await service.systemone(body)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    return app


def load_policy(results_json: Path) -> AbstainPolicy:
    pol = json.loads(results_json.read_text())["policy"]
    return AbstainPolicy(
        temperatures=pol["temperatures"], thresholds=pol["thresholds"], delta=pol.get("delta", 0.1)
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", required=True)
    ap.add_argument(
        "--policy",
        action="append",
        required=True,
        help="JUR=results JSON (repeat per jurisdiction), or a single results JSON for all",
    )
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8300)
    a = ap.parse_args(argv)

    import os
    from dataclasses import replace

    import torch
    import uvicorn
    from kev.api import SystemOneRequest
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.serve import Server

    os.environ.setdefault(
        "KEV_TEMPERATURE", "1.0"
    )  # raw probabilities; we apply our own temperatures
    opts = replace(LoadOptions.from_env(), dtype=torch.bfloat16)
    ck = Checkpoint(a.run)
    tok, model = ck.load("cuda", opts)
    kev_server = Server(ck, tok, model, "cuda")

    async def answer(req: dict[str, Any]) -> dict[str, Any]:
        return await kev_server.answer_async(SystemOneRequest.model_validate(req))

    if len(a.policy) == 1 and "=" not in a.policy[0]:
        policy: AbstainPolicy | dict[str, AbstainPolicy] = load_policy(Path(a.policy[0]))
    else:
        policy = {j.upper(): load_policy(Path(p)) for j, p in (x.split("=", 1) for x in a.policy)}
    app = create_app(Service(answer, policy, a.alpha))
    uvicorn.run(app, host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
