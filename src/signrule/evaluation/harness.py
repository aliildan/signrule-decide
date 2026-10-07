"""Evaluation harness: items, predictors, calibrated metrics (plan-06 T2-T3, plan-08 T3-T4).

A predictor maps requests to per-question logits over that question's option keys (in the order
of `QuestionsConfig.keys`). Probability-only predictors return log-probabilities. Temperatures and
abstention thresholds are fitted on validation predictions and applied to the evaluated part.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from signrule.calib.calibration import (
    ALPHAS,
    AbstainPolicy,
    fit_temperature,
    ltt_threshold,
    softmax_t,
)
from signrule.common.paths import PROCESSED_DIR
from signrule.evaluation import metrics as M
from signrule.format.request import QuestionsConfig, load_questions
from signrule.normalize.text import text_key
from signrule.ontology.cir import (
    Group,
    ProcurationRule,
    SigningRule,
    derive_answers,
    procuration_mode,
)
from signrule.ontology.mapping import load_no_rules

Logits = dict[tuple[str, str], np.ndarray]


@dataclass(frozen=True)
class Item:
    rid: str
    group: str
    qid: str
    qtype: str
    y: int
    has_text: bool


def load_requests(jur: str, split: str, part: str) -> list[dict[str, Any]]:
    path = PROCESSED_DIR / jur / split / f"{part}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def label_index(q: dict[str, Any], keys: list[str]) -> int:
    lab = q["label"]
    if q["type"] == "noul":
        return int(bool(lab))
    if q["type"] == "score":
        return int(lab)
    return keys.index(lab)


def items_of(reqs: Iterable[dict[str, Any]], qc: QuestionsConfig) -> list[Item]:
    out = []
    for r in reqs:
        meta = r["_meta"]
        for qid, q in r["questions"].items():
            out.append(
                Item(
                    meta["id"],
                    meta["group_id"],
                    qid,
                    q["type"],
                    label_index(q, qc.keys(qid)),
                    bool(meta.get("has_text", True)),
                )
            )
    return out


class Predictor(Protocol):
    name: str

    def predict(self, reqs: list[dict[str, Any]]) -> Logits: ...


# ---- Kev predictions (from kev.benchmark output) ------------------------------------------


class KevPredictions:
    def __init__(self, name: str, files: list[Path], qc: QuestionsConfig | None = None) -> None:
        self.name = name
        self.qc = qc or load_questions()
        self._logits: Logits = {}
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                lg = rec["prediction"].get("logits") or {}
                pr = rec["prediction"].get("probabilities") or {}
                for qid in set(lg) | set(pr):
                    keys = self.qc.keys(qid) if qid in self.qc.questions else list((lg or pr)[qid])
                    # An option the model was not offered (benchmarked with an older question
                    # set, e.g. before `joint_alternatives`) gets probability 0.
                    if qid in lg:
                        vec = np.array([lg[qid].get(k, -np.inf) for k in keys], dtype=float)
                    else:
                        vec = np.log(np.clip([pr[qid].get(k, 0.0) for k in keys], 1e-12, 1))
                    self._logits[(rec["id"], qid)] = vec

    def predict(self, reqs: list[dict[str, Any]]) -> Logits:
        return {
            (r["_meta"]["id"], q): self._logits[(r["_meta"]["id"], q)]
            for r in reqs
            for q in r["questions"]
            if (r["_meta"]["id"], q) in self._logits
        }


# ---- B2 majority ---------------------------------------------------------------------------


class Majority:
    name = "majority"

    def __init__(self, train: list[dict[str, Any]], qc: QuestionsConfig | None = None) -> None:
        self.qc = qc or load_questions()
        counts: dict[str, Counter[int]] = defaultdict(Counter)
        for it in items_of(train, self.qc):
            counts[it.qid][it.y] += 1
        self.logp = {}
        for qid, c in counts.items():
            k = len(self.qc.keys(qid))
            p = np.array([c[i] + 1.0 for i in range(k)])
            self.logp[qid] = np.log(p / p.sum())

    def predict(self, reqs: list[dict[str, Any]]) -> Logits:
        return {
            (r["_meta"]["id"], q): self.logp[q]
            for r in reqs
            for q in r["questions"]
            if q in self.logp
        }


# ---- B1 rule-based re-implementation of the 18 Norwegian rules -----------------------------

_SENT = re.compile(r"[.;]\s*|\s+eller\s+(?=styrets|daglig|to |tre |nestleder|deltakerne)", re.I)


class RulesNo:
    """Matches each sentence of the text against the register's 18 rule descriptions (normalised,
    case-insensitive). Every sentence must match, else the baseline abstains (uniform) on the
    rule questions and predicts parseable = false, which is what the register interpreter does."""

    name = "rules_no"

    def __init__(self, qc: QuestionsConfig | None = None, sharp: float = 0.98) -> None:
        self.qc = qc or load_questions()
        self.rules = load_no_rules()
        self.by_text = {
            text_key(s.description): code for code, s in self.rules.items() if s.kind == "rule"
        }
        self.sharp = sharp

    def _match(self, text: str | None) -> tuple[str, list[Group]]:
        if not text:
            return "none", []
        parts = [p for p in _SENT.split(text) if p and p.strip()]
        groups: list[Group] = []
        for p in parts:
            code = self.by_text.get(text_key(p))
            if code is None:
                return "unmatched", []
            groups.extend(self.rules[code].alternatives)
        return "rule", groups

    def _answers(self, st: dict[str, Any]) -> dict[str, Any] | None:
        status, groups = self._match(st.get("signature_rule"))
        if status == "unmatched":
            return None
        s = SigningRule(
            "rule" if status == "rule" else "none",
            frozenset(groups),
            False,
            Group(collective="ALL_BOARD"),
        )
        pst, pgroups = self._match(st.get("procuration_rule"))
        prok_roles = any("prokura" in r["role"].lower() for r in st.get("roles") or [])
        if pst == "rule":
            p = ProcurationRule(True, procuration_mode(pgroups))
        elif pst == "unmatched" or prok_roles:
            p = ProcurationRule(True, None)
        else:
            p = ProcurationRule(False, None)
        return derive_answers(s, p)

    def predict(self, reqs: list[dict[str, Any]]) -> Logits:
        out: Logits = {}
        for r in reqs:
            ans = self._answers(r["state"])
            for qid in r["questions"]:
                keys = self.qc.keys(qid)
                k = len(keys)
                if ans is None and qid == "parseable":
                    target = 0
                elif ans is not None and qid in ans:
                    v = ans[qid]
                    target = int(v) if isinstance(v, bool | int) and qid != "rule_type" else None
                    if qid in ("rule_type", "min_signers"):
                        target = keys.index(str(v))
                    elif qid == "ambiguity":
                        target = int(v)
                else:
                    target = None
                if target is None:
                    p = np.full(k, 1.0 / k)
                else:
                    p = np.full(k, (1 - self.sharp) / (k - 1))
                    p[target] = self.sharp
                out[(r["_meta"]["id"], qid)] = np.log(p)
        return out


class RulesAt:
    """Wording rules for the Austrian court-coded questions (plan-11 Task 2): a function line that
    says "selbständig" acts alone, "gemeinsam" jointly. Shows how much of those questions is
    lexical. Abstains (uniform) when the lines disagree or say neither."""

    name = "rules_at"
    REPRESENTING = ("Geschäftsführer", "Vorsitzender des Vorstands", "Obmann", "Vorstandsmitglied")
    CHAIR = ("Vorsitzender des Vorstands", "Obmann")

    def __init__(self, qc: QuestionsConfig | None = None, sharp: float = 0.98) -> None:
        self.qc = qc or load_questions()
        self.sharp = sharp

    @staticmethod
    def _lines(text: str | None) -> list[tuple[str, str]]:
        out = []
        for part in (text or "").split("; "):
            head, _, body = part.partition(": ")
            role = re.sub(r"\s*\[(?:PERSON|FIRMA)_\d+\]$", "", head).strip()
            out.append((role, body))
        return [x for x in out if x[0]]

    @staticmethod
    def _mode(bodies: list[str]) -> bool | None:
        sole = ["selbständig" in b for b in bodies]
        joint = ["gemeinsam" in b for b in bodies]
        if bodies and all(s and not j for s, j in zip(sole, joint, strict=True)):
            return True
        if bodies and all(j and not s for s, j in zip(sole, joint, strict=True)):
            return False
        return None

    def _answers(self, st: dict[str, Any]) -> dict[str, Any]:
        sig, prok = self._lines(st.get("signature_rule")), self._lines(st.get("procuration_rule"))
        ans: dict[str, Any] = {}
        gf = [b for r, b in sig if r == "Geschäftsführer"]
        if (v := self._mode(gf)) is not None:
            ans["ceo_alone"] = v
        chair = [b for r, b in sig if r in self.CHAIR]
        if len(chair) == 1 and (v := self._mode(chair)) is not None:
            ans["chair_alone"] = v
        if any(r in self.REPRESENTING and "selbständig" in b for r, b in sig):
            ans["min_signers"] = "1"
        roles = [r["role"] for r in st.get("roles") or []]
        ans["prokura_present"] = bool(prok) or "Prokurist" in roles
        if prok and (v := self._mode([b for _, b in prok])) is not None:
            ans["prokura_joint"] = not v
        return ans

    def predict(self, reqs: list[dict[str, Any]]) -> Logits:
        out: Logits = {}
        for r in reqs:
            ans = self._answers(r["state"])
            for qid in r["questions"]:
                keys = self.qc.keys(qid)
                k = len(keys)
                target = None
                if qid in ans:
                    v = ans[qid]
                    target = keys.index(str(v)) if qid == "min_signers" else int(bool(v))
                if target is None:
                    p = np.full(k, 1.0 / k)
                else:
                    p = np.full(k, (1 - self.sharp) / (k - 1))
                    p[target] = self.sharp
                out[(r["_meta"]["id"], qid)] = np.log(p)
        return out


class PhraseAt(RulesAt):
    """RulesAt plus the hand-reviewed phrase table (configs/ontology/at_phrases.yaml) applied to
    the rendered lines: everything the table determines is predicted, the rest abstains. The
    baseline that shows how much of the Austrian structure a lookup table alone explains."""

    name = "phrase_at"

    def __init__(self, qc: QuestionsConfig | None = None, sharp: float = 0.98) -> None:
        super().__init__(qc, sharp)
        from signrule.ontology.mapping import load_roles

        self.label_to_office = {i.label: i.cir for i in load_roles().codes["AT"].values()}

    def _answers(self, st: dict[str, Any]) -> dict[str, Any]:
        from signrule.normalize.normalize_at import applicable_at
        from signrule.ontology.at_phrases import phrase_cir, state_lines

        ans = super()._answers(st)
        lines = state_lines(st, self.label_to_office)
        cir = phrase_cir(lines) if lines else None
        if cir is not None:
            for q, v in derive_answers(cir, None).items():
                if q not in ("prokura_present", "prokura_joint"):
                    ans.setdefault(q, v)
        return applicable_at(st, ans)

    def predict(self, reqs: list[dict[str, Any]]) -> Logits:
        out: Logits = {}
        for r in reqs:
            ans = self._answers(r["state"])
            for qid in r["questions"]:
                keys = self.qc.keys(qid)
                k = len(keys)
                target = None
                if qid in ans:
                    v = ans[qid]
                    if isinstance(v, bool):
                        target = int(v)
                    elif str(v) in keys:
                        target = keys.index(str(v))
                if target is None:
                    p = np.full(k, 1.0 / k)
                else:
                    p = np.full(k, (1 - self.sharp) / (k - 1))
                    p[target] = self.sharp
                out[(r["_meta"]["id"], qid)] = np.log(p)
        return out


# ---- B3 TF-IDF + logistic regression ------------------------------------------------------


def state_text(st: dict[str, Any]) -> str:
    roles = " ".join(f"{r['role']}={r['count']}" for r in st.get("roles") or [])
    return (
        f"[{st.get('legal_form')}] S: {st.get('signature_rule') or '-'} "
        f"P: {st.get('procuration_rule') or '-'} R: {roles}"
    )


class Tfidf:
    name = "tfidf_lr"

    def __init__(self, train: list[dict[str, Any]], qc: QuestionsConfig | None = None) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.qc = qc or load_questions()
        self.vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True
        )
        x_all = self.vec.fit_transform([state_text(r["state"]) for r in train])
        self.models: dict[str, Any] = {}
        self.const: dict[str, np.ndarray] = {}
        for qid in self.qc.questions:
            idx, ys = [], []
            for i, r in enumerate(train):
                q = r["questions"].get(qid)
                if q is not None:
                    idx.append(i)
                    ys.append(label_index(q, self.qc.keys(qid)))
            if not ys:
                continue
            k = len(self.qc.keys(qid))
            if len(set(ys)) == 1:
                p = np.full(k, 0.01 / max(k - 1, 1))
                p[ys[0]] = 0.99
                self.const[qid] = np.log(p)
                continue
            clf = LogisticRegression(C=4.0, max_iter=3000)
            clf.fit(x_all[idx], ys)
            self.models[qid] = clf

    def predict(self, reqs: list[dict[str, Any]]) -> Logits:
        x = self.vec.transform([state_text(r["state"]) for r in reqs])
        out: Logits = {}
        for qid, clf in self.models.items():
            k = len(self.qc.keys(qid))
            probs = clf.predict_proba(x)
            for i, r in enumerate(reqs):
                if qid in r["questions"]:
                    full = np.full(k, 1e-6)
                    full[clf.classes_] = probs[i]
                    out[(r["_meta"]["id"], qid)] = np.log(full / full.sum())
        for qid, lp in self.const.items():
            for r in reqs:
                if qid in r["questions"]:
                    out[(r["_meta"]["id"], qid)] = lp
        return out


# ---- evaluation -----------------------------------------------------------------------------


def fit_policy(items: list[Item], logits: Logits, delta: float = 0.1) -> AbstainPolicy:
    pol = AbstainPolicy(delta=delta)
    by_type: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        if (it.rid, it.qid) in logits:
            by_type[it.qtype].append(it)
    for qt, its in by_type.items():
        pol.temperatures[qt] = fit_temperature(
            [logits[(i.rid, i.qid)] for i in its], [i.y for i in its]
        )
    for alpha in ALPHAS:
        pol.thresholds[str(alpha)] = {}
        for qt, its in by_type.items():
            probs = [softmax_t(logits[(i.rid, i.qid)], pol.temperatures[qt]) for i in its]
            conf = np.array([p.max() for p in probs])
            correct = np.array([p.argmax() == i.y for p, i in zip(probs, its, strict=True)])
            pol.thresholds[str(alpha)][qt] = ltt_threshold(conf, correct, alpha, delta)
    return pol


def _block(
    its: list[Item], probs: list[np.ndarray], pol: AbstainPolicy | None, qtype: str
) -> dict[str, Any]:
    if not its:
        return {"n": 0}
    y = np.array([i.y for i in its])
    k = max(len(p) for p in probs)
    mat = np.array([np.pad(p, (0, k - len(p))) for p in probs])
    correct, conf = M.correct_conf(mat, y)
    out: dict[str, Any] = {
        "n": len(its),
        "accuracy": M.accuracy(mat, y),
        "accuracy_ci95": M.bootstrap_ci(
            lambda idx: float(correct[idx].mean()), [i.group for i in its], 500
        ),
        "macro_f1": M.macro_f1(mat, y),
        "nll": M.nll(mat, y),
        "brier": M.brier(mat, y),
        "ece_width": M.ece(conf, correct, 15, "width"),
        "ece_mass": M.ece(conf, correct, 15, "mass"),
        "aurc": M.aurc(conf, correct),
        "coverage_at_risk": {str(a): M.coverage_at_risk(conf, correct, a) for a in ALPHAS},
    }
    if qtype == "score":
        out["mae"] = M.mae_expected(mat, y)
        out["qwk"] = M.qwk(mat.argmax(1), y, k)
    if pol is not None:
        sel = {}
        for a in ALPHAS:
            lam = pol.thresholds.get(str(a), {}).get(qtype)
            answered = conf >= lam if lam is not None else np.zeros(len(conf), dtype=bool)
            n_ans = int(answered.sum())
            sel[str(a)] = {
                "lambda": lam,
                "coverage": n_ans / len(conf),
                "risk": float((~correct[answered]).mean()) if n_ans else None,
            }
        out["ltt"] = sel
    return out


def evaluate(items: list[Item], logits: Logits, policy: AbstainPolicy | None) -> dict[str, Any]:
    """Per question and per question type; raw and (if policy) calibrated + LTT selective."""
    items = [i for i in items if (i.rid, i.qid) in logits]
    res: dict[str, Any] = {"n_items": len(items), "per_question": {}, "per_type": {}, "strata": {}}
    groups: dict[str, list[Item]] = defaultdict(list)
    for i in items:
        groups[i.qid].append(i)

    temps = policy.temperatures if policy else {}

    def probs_for(its: list[Item], calibrated: bool) -> list[np.ndarray]:
        def t(qt: str) -> float:
            return temps.get(qt, 1.0) if calibrated else 1.0

        return [softmax_t(logits[(i.rid, i.qid)], t(i.qtype)) for i in its]

    for qid, its in sorted(groups.items()):
        qt = its[0].qtype
        res["per_question"][qid] = {
            "raw": _block(its, probs_for(its, False), None, qt),
            "calibrated": _block(its, probs_for(its, True), policy, qt) if policy else None,
        }
    by_type: dict[str, list[Item]] = defaultdict(list)
    for i in items:
        by_type[i.qtype].append(i)
    for qt, its in by_type.items():
        res["per_type"][qt] = {
            "raw": _block(its, probs_for(its, False), None, qt),
            "calibrated": _block(its, probs_for(its, True), policy, qt) if policy else None,
        }
    for name, cond in (("has_text", True), ("roles_only", False)):
        its = [i for i in items if i.has_text is cond]
        accs = {}
        for qid in sorted({i.qid for i in its}):
            sub = [i for i in its if i.qid == qid]
            p = probs_for(sub, bool(policy))
            accs[qid] = {
                "n": len(sub),
                "accuracy": float(
                    np.mean([pp.argmax() == i.y for pp, i in zip(p, sub, strict=True)])
                ),
            }
        res["strata"][name] = accs
    return res
