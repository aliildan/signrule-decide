"""B7: Cloudflare Clef-Flash zero-shot (CLAUDE.md §7.5 item 5), run locally.

Uses the release's own `joint_schema_model` code from the Hugging Face snapshot (Apache-2.0).
Requests are sent without labels; per-question logits are mapped to our option-key order.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from signrule.evaluation.harness import Logits
from signrule.format.request import QuestionsConfig, load_questions

_NOUL_ALIASES = {
    "false": "false",
    "true": "true",
    "no": "false",
    "yes": "true",
    "0": "false",
    "1": "true",
}


class ClefPredictor:
    name = "clef-flash-zeroshot"

    def __init__(
        self, qc: QuestionsConfig | None = None, batch: int = 8, device: str = "cuda"
    ) -> None:
        import torch
        from huggingface_hub import snapshot_download

        self.qc = qc or load_questions()
        self.batch = batch
        self.device = torch.device(device)
        path = snapshot_download("Cloudflare/clef-flash")
        if path not in sys.path:
            sys.path.insert(0, path)
        import joint_schema_model as jsm  # type: ignore[import-not-found]

        self.jsm = jsm
        self.model, self.processor = jsm.load_release_model(path, device=device)

    def _request(self, r: dict[str, Any]) -> dict[str, Any]:
        return {
            "state": r["state"],
            "questions": {
                qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
                for qid, q in r["questions"].items()
            },
        }

    def predict(self, reqs: list[dict[str, Any]]) -> Logits:
        import torch

        tok = self.processor.tokenizer
        out: Logits = {}
        for b in range(0, len(reqs), self.batch):
            chunk = reqs[b : b + self.batch]
            encoded = [
                self.jsm.encode_record(tok, self._request(r), processor=self.processor)
                for r in chunk
            ]
            batch = self.jsm.collate_records(encoded, tok.pad_token_id, self.device)
            with torch.inference_mode():
                logits = self.model(batch)
            for r, enc, rec_logits in zip(chunk, encoded, logits, strict=True):
                for question, ql in zip(enc.questions, rec_logits, strict=True):
                    qid = question.question_id
                    if qid not in self.qc.questions:
                        continue
                    keys = self.qc.keys(qid)
                    ids = [str(o) for o in question.option_ids]
                    if self.qc.questions[qid]["type"] == "noul":
                        ids = [_NOUL_ALIASES.get(i.lower(), i) for i in ids]
                    vals = ql.float().cpu().numpy()
                    pos = {k: i for i, k in enumerate(ids)}
                    if set(keys) <= set(pos):
                        out[(r["_meta"]["id"], qid)] = np.array(
                            [vals[pos[k]] for k in keys], dtype=float
                        )
        return out
