# SignRule-Decide

**Who can sign for this company? Typed, calibrated answers from Austrian register extracts —
locally, without sending data anywhere.**

[Licence: Apache-2.0](LICENSE) · Python 3.12 · 4B parameters · one GPU · [open weights](https://huggingface.co/aildan/signrule-decide-4b)

---

Every KYB onboarding needs the answer to one question: *who may bind this company?* The Austrian
Firmenbuch publishes the answer as German free text for every managing director, board member,
partner and Prokurist — and no machine interpretation. Today an analyst reads it by hand.

SignRule-Decide reads the register text and the registered roles and returns the decision, not a
summary:

> **Geschäftsführer [PERSON_1]:** vertritt seit 01.03.2019 gemeinsam mit einem weiteren
> Geschäftsführer oder einem Prokuristen · *(same for [PERSON_2])* · **Prokurist [PERSON_3]:**
> Gesamtprokura gemeinsam mit einem Geschäftsführer

| Question | Answer | Confidence |
|---|---|---|
| Can one Geschäftsführer act alone? | **no** | 1.00 |
| Can two Geschäftsführer act together? | **yes** | 1.00 |
| Can a Geschäftsführer act with a Prokurist? | **yes** | 1.00 |
| Minimum number of signatures | **2** | 1.00 |
| Is the procuration joint (Gesamtprokura)? | **yes** | 1.00 |

More examples, in German: [`results/demo/demo-de.md`](results/demo/demo-de.md).

## Why SignRule-Decide

- **Decisions, not text.** Every answer is a probability over fixed options (yes/no, a number of
  signers, a rule type). Nothing is generated, so nothing is invented.
- **Knows when to stay silent.** Answers are calibrated per question type, and the server abstains
  when a question cannot be answered within the risk level you choose (1, 2 or 5 %).
- **Private by design.** It never needs names: callers pass roles and counts, names are masked as
  `[PERSON_n]`, and the model runs on your own hardware.
- **One model, one question set, several registers.** Austria and Norway today, Denmark as a beta.
- **Open.** Code, weights and every evaluation number are public (Apache-2.0); the data can be
  re-fetched from the official registers with the included scripts.

## Accuracy at a glance

Austria, reference set: 400 free-text extracts that no model was trained on. Every number is
generated from [`results/`](results/) by `scripts/make_tables.py`.

<!-- table:at-gold -->
| Austria, reference set (400) | GF alone | Chair alone | Coalitions (11) | Min. signers | Rule type |
|---|---|---|---|---|---|
| **SignRule-Decide 4B** | 100.0 % | 100.0 % | 99.3 % | 97.7 % | 94.6 % |
| mmBERT-base, fine-tuned (1,024 tokens) | 100.0 % | 100.0 % | 98.8 % | 96.9 % | 94.3 % |
| XLM-R-large, fine-tuned (512 tokens) | 73.5 % | 95.8 % | 92.4 % | 63.5 % | 54.8 % |
| Phrase table (rules from the training labels) | 100.0 % | 100.0 % | 91.4 % | 80.2 % | 75.7 % |
| Keyword rules (selbständig / gemeinsam) | 100.0 % | 100.0 % | 54.3 % | 35.0 % | 11.1 % |

SignRule-Decide 4B on the same items: dangerous yes/no errors ("can sign" when the answer is "cannot") 12 of 2178 (0.6 %); at a 2 % risk target it answers 99.9 % of the questions with an observed risk of 1.3 %.
<!-- /table -->

How the reference set was made — read this before quoting the numbers: each extract was answered
independently by **two different AI assistants**, only answers both agree on are kept, and a person
spot-checked a sample. The numbers therefore measure agreement with that consensus, not with fully
human labels (see [Evaluation data](#evaluation-data)).

A small encoder (mmBERT-base, about 300M parameters) fine-tuned on the same Austrian labels comes
close on this set. SignRule-Decide makes fewer structural errors and ranks its own confidence
better, which is what the abstention relies on. If you need only Austria and only these questions,
a fine-tuned encoder is a valid, cheaper choice.

## Supported registers

| Register | Status | Evidence |
|---|---|---|
| **Austria** — Firmenbuch | **Validated** | official court codes in training; reference set of 400 above |
| **Norway** — Brønnøysundregistrene | **Validated** | the register's own interpreter as training labels; 250 texts the interpreter could *not* read |
| **Denmark** — CVR | **Beta** (never trained on) | 300-item reference set, zero-shot; see limitations |
| Anything else | Untested | — |

<!-- table:other-registers -->
| Reference set | CEO alone | Coalitions | Min. signers | Rule type | Register in training? |
|---|---|---|---|---|---|
| Norway, texts the official interpreter could not read (250) | 100.0 % | 99.0 % | 98.0 % | 93.0 % | trained |
| Denmark (300) | 89.2 % | 92.5 % | 86.4 % | 62.4 % | **never seen** |
<!-- /table -->

The second Austrian batch focuses on the structures that are hardest to read:

<!-- table:at-structures -->
| Structure (second batch, 200) | Coalitions | Min. signers | Dangerous errors |
|---|---|---|---|
| Vorstand (AG, Genossenschaft, Privatstiftung) | 99.7 % | 99.1 % | 3 / 924 |
| Partnerships (OG, KG) | 97.7 % | 95.8 % | 3 / 176 |
| GmbH with several Geschäftsführer | 99.4 % | 100.0 % | 1 / 165 |
<!-- /table -->

Register-labelled test parts (official codes as labels; wording shared with training, so near
ceiling):

<!-- table:in-distribution -->
| Register-labelled test (codes / interpreter) | Coalitions | Min. signers | Rule type |
|---|---|---|---|
| Norway, held-out texts | 100.0 % | 99.7 % | 99.8 % |
| Austria, held-out patterns | 100.0 % | 100.0 % | 99.8 % |
| Austria, frozen pilot (2,500 companies) | 100.0 % | 100.0 % | 100.0 % |
<!-- /table -->

## Quick start

```bash
git clone https://github.com/aliildan/signrule-decide && cd signrule-decide && uv sync
hf download aildan/signrule-decide-4b --local-dir runs/signrule-decide-4b
uv run python server/app.py --run runs/signrule-decide-4b \
  --policy NO=runs/signrule-decide-4b/calibration/NO.json \
  --policy AT=runs/signrule-decide-4b/calibration/AT.json      # /v1/systemone on localhost:8300
```

Weights: [huggingface.co/aildan/signrule-decide-4b](https://huggingface.co/aildan/signrule-decide-4b).
The model does not generate text, so it cannot run in Ollama or llama.cpp; use this server.

Ask questions with the canonical wordings from `configs/questions.yaml`:

```python
import httpx, yaml

questions = yaml.safe_load(open("configs/questions.yaml"))
ask = {q: questions[q] for q in ("ceo_alone", "ceo_with_prokurist", "min_signers")}
state = {
    "jurisdiction": "AT",
    "legal_form": "GmbH",
    "signature_rule": "Geschäftsführer [PERSON_1]: vertritt seit 01.03.2019 gemeinsam mit einem "
                      "weiteren Geschäftsführer oder einem Prokuristen",
    "roles": [{"role": "Geschäftsführer", "count": 2}, {"role": "Prokurist", "count": 1}],
}
r = httpx.post("http://localhost:8300/v1/systemone", json={"state": state, "questions": ask})
for q, a in r.json()["answers"].items():
    print(q, a.get("noul", a.get("choice")), "abstain" if a["abstain"] else "")
```

Each answer carries `noul` (probability of "yes") or `choice` + `probabilities`, `calibrated`
(true for the canonical wordings) and `abstain` with a reason. The API follows the System One
schema (`/v1/systemone`), so existing Jev/Kev clients work unchanged.

## Data

About a million official company registrations were read; because register texts repeat
heavily, they reduce to about 31 thousand distinct training cases (text and roles).

<!-- table:data -->
| Register | Read from the register | Kept | Unique (text, roles) | Train / val / test |
|---|---|---|---|---|
| Norway | 637,492 companies (1,274,984 API responses) | 597,487 | 31,583 | 25,394 / 2,729 / 3,460 |
| Austria | 239,474 company extracts | 206,553 | 7,710 patterns | 6,197 / 744 / 769 (+ 2,500 pilot) |
| Denmark | 143,646 companies | 143,311 | 14,147 | evaluation only |
| **Total** | **1,020,612 companies** |  |  | **31,591 training cases**, 2 epochs, 37.7 M tokens |
<!-- /table -->

## Hardware

Trained and served on one consumer GPU; no cloud.

<!-- table:compute -->
| Compute | Value |
|---|---|
| GPU | 1 × NVIDIA GeForce RTX 5090 (31.8 GB) |
| CPU / RAM | Intel(R) Core(TM) i9-14900KF / 92 GB |
| Training (LoRA, bf16) | 14.6 h, peak GPU memory 20.2 GB, 7,836 optimizer steps |
| Serving (one request, 8.5 questions) | p50 160 ms, p95 175 ms, 6.7 requests/s |
<!-- /table -->

Serving latency was measured with an earlier checkpoint of the same 4B architecture.

## How it works

- **Backbone:** `Qwen/Qwen3.5-4B-Base` (Apache-2.0), LoRA adapter, and a
  [Kev](https://github.com/jaredpalmer/kev) pointer head: each answer option is scored against the
  question; a softmax over the options is the answer.
- **Labels from the registers.** Norway's register interprets its own signing rules (rule codes),
  Austria's court codes every person's power (alone / jointly); for joint powers a hand-written,
  reviewed table of the register's standard wording names the partner. No synthetic examples and no
  language-model labels in training.
- **One representation for every register:** alternatives (OR) of groups (AND) of offices; every
  answer is derived from it.
- **Calibration:** one temperature per question type and Learn-then-Test thresholds per register,
  fitted on validation data only; an unknown register gets the strictest combination.

## Evaluation data

| Set | Labels | Size |
|---|---|---|
| Register-labelled test parts | official codes (Norway interpreter, Austrian court codes) | thousands of texts, see the tables |
| Reference sets (Austria, Norway, Denmark) | two different AI assistants answered independently; kept where both agree; human spot checks | 400 / 250 / 300 extracts |

Hypotheses and decision rules were written down before the results were read; the release model was
chosen by the project owner although one criterion (on the experimental ambiguity score) was not
met. Both are documented in the [model card](model_card.md).

## Limitations and responsible use

- **Not legal advice.** A register can be out of date, and articles of association can contain
  powers the register text does not show. Keep a human in the loop for every binding decision.
- **Per-person powers:** when holders of one office have different powers, the office-level question
  ("can a managing director act alone?") is open by our convention; the model tends to answer "yes"
  if any holder may act alone.
- **Partnerships** (OG/KG) are the weakest Austrian structure.
- **Denmark (beta):** *"direktionen"* is often read as a collective body, and the abstention
  thresholds do not transfer to a register the model has never seen.
- The **ambiguity** score is experimental and never served as a decision.
- There is **no fully human-verified evaluation subset**; the hard-case numbers are agreement with
  the consensus of two AI assistants, spot-checked by a person.

## Reproduce

The data is public but not redistributed here. The scripts fetch it from the official sources
(rate-limited, cached, descriptive User-Agent):

| Register | Source | Licence | Script |
|---|---|---|---|
| Norway | Brønnøysundregistrene, Enhetsregisteret + Fullmakttjenesten | NLOD | `python -m signrule.ingest.ingest_no` |
| Austria | Firmenbuch HVD via JustizOnline (API key) | CC BY 4.0 | `python -m signrule.ingest.ingest_at` |
| Denmark | CVR system-til-system (user account) | Danish public-data terms | `python -m signrule.ingest.ingest_dk` |

Then `make data && make data-check`, `python -m signrule.normalize.pipeline_at run && … check`,
`python -m signrule.train.kev_wrapper train --config configs/train/4b-noat-v2.yaml` (≈ 15 h on one
RTX 5090), `kev_wrapper bench`, and `eval/run_all.py`. The reference sets contain register texts and
are not published; their agreement statistics are in `results/gold/`.

## Licence and attribution

Code and model weights: Apache-2.0. Contains data from Brønnøysundregistrene (NLOD); Firmenbuch –
Bundesministerium für Justiz / JustizOnline (HVD), CC BY 4.0; Det Centrale Virksomhedsregister
(CVR), Erhvervsstyrelsen — "Indeholder data, som benyttes i henhold til vilkår for brug af danske
offentlige data". No register data is redistributed. Built with the help of an AI coding assistant.
