"""Score models and baselines on a split; write results/<split>/<model>.json (CLAUDE.md §8).

    uv run python eval/run_all.py --split random --part test --baselines majority,rules_no,tfidf_lr
    uv run python eval/run_all.py --split random --part test \
        --kev no-9b-base=runs/no-9b-base-bench-no-random-val,runs/no-9b-base-bench-no-random-test

For `--kev NAME=VALDIR,EVALDIR`, VALDIR/EVALDIR are `kev.benchmark` output directories (raw logits
are read from predictions.jsonl). Temperatures and LTT thresholds are always fitted on `val` and
applied to `--part`. Test parts need `--allow-test`. Every number in docs comes from these files.

Cross-jurisdiction: `--target at:pilot` scores the target's `test` part with predictors trained and
calibrated on the source jurisdiction (train/val of --jurisdiction/--split; no target calibration).
Results go to results/<jur>-<split>-to-<tjur>-<tsplit>/, with a `by_stratum` breakdown when the
target requests carry `_meta.stratum`. The frozen `pilot` split needs no --allow-test.

A jurisdiction no model was calibrated on (Denmark): `--strictest-with at` fits a second policy on
that jurisdiction's val part and scores with the strictest merge of both (as the server does for an
unseen register); for `--kev` give the extra val benchmark after EVALDIR (NAME=VAL,EVAL,EXTRAVAL).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from signrule.calib.calibration import AbstainPolicy, softmax_t, strictest
from signrule.common.paths import PROCESSED_DIR, REPO_ROOT, SPLITS_DIR
from signrule.evaluation.harness import (
    Item,
    KevPredictions,
    Majority,
    PhraseAt,
    Predictor,
    RulesAt,
    RulesNo,
    Tfidf,
    evaluate,
    fit_policy,
    items_of,
    load_requests,
)
from signrule.format.request import QuestionsConfig, load_questions

RESULTS = REPO_ROOT / "results"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _git() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(args: argparse.Namespace) -> int:
    qc = load_questions()
    jur = args.jurisdiction
    tjur, tsplit, tpart = jur, args.split, args.part
    if args.target:
        tjur, tsplit = args.target.split(":", 1)
        tpart = "test"
    if tpart == "test" and tsplit != "pilot" and not args.allow_test:
        raise SystemExit("test parts need --allow-test (read once per released model)")
    train = load_requests(jur, args.split, "train")
    val = load_requests(jur, args.split, "val")
    target = load_requests(tjur, tsplit, tpart)
    val_items, target_items = items_of(val, qc), items_of(target, qc)
    extra = list(args.strictest_with or [])
    args.extra_cal = []
    for j in extra:
        reqs = load_requests(j, args.split, "val")
        args.extra_cal.append((reqs, items_of(reqs, qc)))
    cal = "+".join([jur, *extra])
    args.tag = f"{cal}-{args.split}" + (f"-to-{tjur}-{tsplit}" if args.target else "")
    args.target_parts = (tjur, tsplit, tpart)
    args.strata = {
        r["_meta"]["id"]: r["_meta"]["stratum"] for r in target if r["_meta"].get("stratum")
    }

    predictors = build_predictors(args, train, qc)
    out_dir = RESULTS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for pred in predictors:
        _score_one(pred, args, jur, val, target, val_items, target_items, out_dir)
    return 0


def build_predictors(
    args: argparse.Namespace, train: list[dict[str, Any]], qc: QuestionsConfig
) -> list[Predictor]:
    predictors: list[Predictor] = []
    for b in filter(None, (args.baselines or "").split(",")):
        if b == "clef":
            from signrule.evaluation.clef import ClefPredictor

            predictors.append(ClefPredictor(qc))
            continue
        if b.startswith("enc:"):
            from signrule.evaluation import encoders

            # enc:NAME, enc:NAME@MAXLEN or enc:NAME@MAXLEN/BATCH (long Austrian states need more
            # than 256 tokens; a large encoder at 512 needs a smaller batch on one GPU)
            name, _, size = b[4:].partition("@")
            max_len, _, batch = size.partition("/")
            kw = {"max_len": int(max_len)} if max_len else {}
            if batch:
                kw["batch"] = int(batch)
            predictors.append(encoders.EncoderPredictor(name, train, qc, **kw))
            continue
        predictors.append(
            {
                "majority": lambda: Majority(train, qc),
                "rules_no": lambda: RulesNo(qc),
                "rules_at": lambda: RulesAt(qc),
                "phrase_at": lambda: PhraseAt(qc),
                "tfidf_lr": lambda: Tfidf(train, qc),
            }[b]()
        )
    for spec in args.kev or []:
        name, dirs = spec.split("=", 1)
        # VALDIR,EVALDIR[,EXTRA_VALDIR...] (extra val benchmarks for --strictest-with)
        files = [Path(d) / "predictions.jsonl" for d in dirs.split(",")]
        predictors.append(KevPredictions(name, files, qc))
    return predictors


def calibration_policy(
    pred: Predictor, cal_sets: list[tuple[list[dict[str, Any]], list[Item]]]
) -> AbstainPolicy:
    """Temperatures + LTT thresholds fitted on each calibration set (val parts); several sets
    (a target jurisdiction no policy was fitted on) -> their strictest merge."""
    policies = [fit_policy(items, pred.predict(reqs)) for reqs, items in cal_sets]
    return policies[0] if len(policies) == 1 else strictest(policies)


def _score_one(
    pred: Predictor,
    args: argparse.Namespace,
    jur: str,
    val: list[dict[str, Any]],
    target: list[dict[str, Any]],
    val_items: list[Item],
    target_items: list[Item],
    out_dir: Path,
) -> None:
    """Fit temperatures + LTT thresholds on val, evaluate `target`, write the results JSON."""
    policy = calibration_policy(pred, [(val, val_items), *getattr(args, "extra_cal", [])])
    target_logits = pred.predict(target)
    res = evaluate(target_items, target_logits, policy)
    if args.strata:
        res["by_stratum"] = {
            st: evaluate(
                [it for it in target_items if args.strata.get(it.rid) == st], target_logits, policy
            )
            for st in sorted(set(args.strata.values()))
        }
    tjur, tsplit, tpart = args.target_parts
    res.update(
        {
            "model": pred.name,
            "jurisdiction": jur,
            "calibration": [jur, *(args.strictest_with or [])],
            "policy_merge": "strictest" if args.strictest_with else "single",
            "split": args.split,
            "part": tpart,
            "target": {"jurisdiction": tjur, "split": tsplit, "part": tpart},
            "policy": {
                "temperatures": policy.temperatures,
                "thresholds": policy.thresholds,
                "delta": policy.delta,
            },
            "split_manifest_sha256": _sha(SPLITS_DIR / f"{jur}_{args.split}.json"),
            "data_sha256": _sha(PROCESSED_DIR / tjur / tsplit / f"{tpart}.jsonl"),
            "git_commit": _git(),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    path = out_dir / f"{pred.name}.{tpart}.json"
    path.write_text(json.dumps(res, indent=1))
    write_item_sidecar(pred.name, args, jur, target_items, target_logits, policy)
    pt = res["per_type"]
    summary = ", ".join(
        f"{qt} acc={v['calibrated']['accuracy']:.3f} ece={v['calibrated']['ece_mass']:.3f}"
        for qt, v in sorted(pt.items())
    )
    print(f"{pred.name:>22} [{tpart}] {summary} -> {path.relative_to(REPO_ROOT)}")


def write_item_sidecar(
    name: str,
    args: argparse.Namespace,
    jur: str,
    items: list[Item],
    logits: dict,
    policy: AbstainPolicy,
) -> Path:
    """Item-level calibrated correctness for paired comparisons. Private (runs/), never results/:
    ids, labels and confidences only, no text."""
    d = REPO_ROOT / "runs" / "eval" / getattr(args, "tag", f"{jur}-{args.split}")
    d.mkdir(parents=True, exist_ok=True)
    tpart = getattr(args, "target_parts", (jur, args.split, args.part))[2]
    path = d / f"{name}.{tpart}.items.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            z = logits.get((it.rid, it.qid))
            if z is None:
                continue
            p = softmax_t(z, policy.temperatures.get(it.qtype, 1.0))
            row = {
                "rid": it.rid,
                "group": it.group,
                "qid": it.qid,
                "qtype": it.qtype,
                "y": it.y,
                "pred": int(p.argmax()),
                "conf": float(p.max()),
                "p_gold": float(p[it.y]),
                "has_text": it.has_text,
            }
            f.write(json.dumps(row) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jurisdiction", default="no")
    ap.add_argument("--split", default="random")
    ap.add_argument("--part", default="val", choices=["val", "test"])
    ap.add_argument("--baselines", default="")
    ap.add_argument("--kev", action="append", help="NAME=VALDIR,EVALDIR (kev.benchmark outputs)")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--target", help="JUR:SPLIT — score another jurisdiction's test part")
    ap.add_argument(
        "--strictest-with",
        action="append",
        help="JUR: also calibrate on this jurisdiction's val part, score with the strictest merge",
    )
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
