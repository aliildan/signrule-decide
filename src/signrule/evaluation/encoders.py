"""B4: multilingual encoder with one linear head per question (CLAUDE.md §7.5 item 3).

Same inputs (the rendered state), same splits and the same label masking as the decision model:
questions absent from a record contribute no loss. Logits per question are returned to the
harness, which calibrates them exactly like every other predictor.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch import nn

from signrule.evaluation.harness import Logits, label_index, state_text
from signrule.format.request import QuestionsConfig, load_questions


class _MultiHead(nn.Module):
    def __init__(self, encoder: nn.Module, hidden: int, sizes: dict[str, int]) -> None:
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict({q: nn.Linear(hidden, k) for q, k in sizes.items()})
        self.drop = nn.Dropout(0.1)

    def forward(self, **enc: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = self.drop((h * mask).sum(1) / mask.sum(1).clamp(min=1))
        return {q: head(pooled) for q, head in self.heads.items()}


class EncoderPredictor:
    def __init__(
        self,
        model_id: str,
        train: list[dict[str, Any]],
        qc: QuestionsConfig | None = None,
        *,
        epochs: int = 4,
        lr: float = 3e-5,
        head_lr: float = 1e-3,
        batch: int = 32,
        max_len: int = 256,
        seed: int = 13,
        device: str = "cuda",
        bf16: bool = True,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.name = "enc_" + model_id.split("/")[-1]
        self.qc = qc or load_questions()
        self.max_len = max_len
        self.batch = batch
        self.device = device
        # DeBERTa-v3 is numerically unstable under half precision (NaN losses); train it in fp32.
        self.bf16 = bf16 and "deberta" not in model_id.lower()
        torch.manual_seed(seed)
        random.seed(seed)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        # transformers 5 loads checkpoints in their stored dtype (mDeBERTa: fp16); heads are fp32.
        encoder = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        sizes = {q: len(self.qc.keys(q)) for q in self.qc.questions}
        self.model = _MultiHead(encoder, encoder.config.hidden_size, sizes).to(device)
        params = [
            {"params": self.model.encoder.parameters(), "lr": lr},
            {"params": self.model.heads.parameters(), "lr": head_lr},
        ]
        opt = torch.optim.AdamW(params, weight_decay=0.01)
        steps = epochs * ((len(train) + batch - 1) // batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[lr, head_lr], total_steps=max(steps, 2), pct_start=0.1
        )
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(epochs):
            order = list(range(len(train)))
            random.shuffle(order)
            for b in range(0, len(order), batch):
                rows = [train[i] for i in order[b : b + batch]]
                enc = self._encode(rows)
                with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=self.bf16):
                    out = self.model(**enc)
                loss = torch.zeros((), device=device)
                for qid, logits in out.items():
                    idx = [i for i, r in enumerate(rows) if qid in r["questions"]]
                    if not idx:
                        continue
                    y = torch.tensor(
                        [label_index(rows[i]["questions"][qid], self.qc.keys(qid)) for i in idx],
                        device=device,
                    )
                    loss = loss + loss_fn(logits[idx].float(), y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
        self.model.eval()

    def _encode(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        enc = self.tok(
            [state_text(r["state"]) for r in rows],
            truncation=True,
            max_length=self.max_len,
            padding=True,
            return_tensors="pt",
        )
        return {
            k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")
        }

    @torch.no_grad()
    def predict(self, reqs: list[dict[str, Any]]) -> Logits:
        out: Logits = {}
        for b in range(0, len(reqs), 64):
            rows = reqs[b : b + 64]
            with torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16, enabled=self.bf16):
                logits = self.model(**self._encode(rows))
            for qid, lg in logits.items():
                arr = lg.float().cpu().numpy()
                for i, r in enumerate(rows):
                    if qid in r["questions"]:
                        out[(r["_meta"]["id"], qid)] = np.asarray(arr[i], dtype=float)
        return out
