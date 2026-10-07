---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B-Base
language: [de, nb, nn, da]
library_name: kev
tags: [kyb, legal, company-register, signing-rules, calibrated-classification, austria, firmenbuch]
---

# SignRule-Decide 4B

A typed decision model for company signing and representation rules as they appear in official
commercial registers. Given the register text, the legal form and the registered roles (names
masked), it answers fixed questions with calibrated probabilities and abstains when a question
cannot be answered within the chosen risk level. It never generates text.

**Primary use: the Austrian Firmenbuch**, which publishes the *Vertretungsbefugnis* wording for
every function holder but no machine interpretation. Norway (whose register ships an official
interpreter) is the main source of training labels; Denmark is a zero-shot test.

## How to use

```bash
git clone https://github.com/aliildan/signrule-decide && cd signrule-decide && uv sync
hf download aildan/signrule-decide-4b --local-dir runs/signrule-decide-4b
uv run python server/app.py --run runs/signrule-decide-4b \
  --policy NO=runs/signrule-decide-4b/calibration/NO.json \
  --policy AT=runs/signrule-decide-4b/calibration/AT.json      # /v1/systemone on :8300
```

The base model `Qwen/Qwen3.5-4B-Base` is downloaded on first start. Files: `adapter_model.safetensors`
(LoRA), `head.pt` (pointer head, PyTorch state), `calibration/NO.json` and `calibration/AT.json`
(temperatures and abstention thresholds fitted on each register's validation part), tokenizer and
training configuration. Request format and examples: the GitHub README and `results/demo/demo-de.md`.
Ollama (v0.35+) serves decision models over the same `/v1/systemone` API for the Clef, Laya and
Strands Decider architectures; this checkpoint is in Kev's format, which Ollama does not load yet.
Use the reference server; the requests are identical.

## Model details

| | |
|---|---|
| Backbone | `Qwen/Qwen3.5-4B-Base` (Apache-2.0), frozen, bf16 |
| Adapter | LoRA r = 16, α = 32 on attention, MLP and Gated DeltaNet projections |
| Readout | Kev pointer head (one slot per option, softmax over options) |
| Training | 2 epochs, AdamW, lr 5e-5 (adapter) / 1e-4 (head), shared-prefix forward, seed 13 |
| Questions | `configs/questions.yaml`: office questions, 11 coalition questions, min. signers, rule type, procuration, parseable, ambiguity (experimental) |
| Output | per question: option probabilities, calibrated with a per-type temperature; `abstain` from Learn-then-Test thresholds (α = 1 / 2 / 5 %) per register |
| Licence | Apache-2.0 |

## Compute

<!-- table:compute -->
| Compute | Value |
|---|---|
| GPU | 1 × NVIDIA GeForce RTX 5090 (31.8 GB) |
| CPU / RAM | Intel(R) Core(TM) i9-14900KF / 92 GB |
| Training (LoRA, bf16) | 14.6 h, peak GPU memory 20.2 GB, 7,836 optimizer steps |
| Serving (one request, 8.5 questions) | p50 160 ms, p95 175 ms, 6.7 requests/s |
<!-- /table -->

Serving latency was measured with an earlier checkpoint of the same 4B architecture.

## Intended use

Decision support in KYB / onboarding: pre-filling "who may bind this company?" from a register
extract, with a human reviewing the result. The calibrated probability and the abstention flag are
part of the answer and must be shown to the reviewer.

**Out of scope:** legal advice; registers other than Austria, Norway and Denmark without a new
evaluation; powers that are not in the register text (articles of association, internal limits);
any use that sends personal data to the model (names must be masked before the request).

## Training data and labels

All examples are real register texts with labels from the registers themselves — no synthetic
examples, no labels from language models.

- **Norway** (Brønnøysundregistrene, NLOD): signing rules of all in-scope entities from
  Fullmakttjenesten; labels are the register interpreter's rule codes, plus texts that consist word
  for word of the register's own rule descriptions (labelled as their union).
- **Austria** (Firmenbuch HVD, CC BY 4.0): per-person representation codes of the court (E/G) and,
  for joint powers, the partner structure from a hand-written, reviewed exact-match table of the
  register's standard wording. A `phrase_at` baseline (the same table used as a predictor) is
  reported next to every Austrian number.
- Deduplicated by text and roles; splits by text cluster (Norway) and by date-normalised pattern
  (Austria), so no identical text is in training and test. Names → `[PERSON_n]`, companies →
  `[FIRMA_n]`; birth dates and personal IDs dropped at ingest.

<!-- table:data -->
| Register | Companies read | Kept | Unique | Train / val / test |
|---|---|---|---|---|
| Norway | 637,492 | 597,487 | 31,583 | 25,394 / 2,729 / 3,460 |
| Austria | 239,474 | 206,553 | 7,710 | 6,197 / 744 / 769 |
| Denmark | 143,646 | 143,311 | 14,147 | evaluation only |
| **Total** | **1,020,612** |  |  | **31,591** training cases |

Companies read: Norway with 1,274,984 API responses (signing and procuration); Austria company extracts. Unique = distinct (text, roles) groups; Austria: date-normalised patterns, plus a frozen pilot of 2,500 companies. Training: 2 epochs, 37.7 M tokens.
<!-- /table -->

## Evaluation

**Reference sets (read before quoting):** for every extract, two *different* AI assistants
answered independently following written guidelines; only answers both agree on are kept
(disagreements: documented adjudication), and a person spot-checked a sample. The numbers therefore
measure agreement with an AI consensus, not with fully human labels; Cohen's κ between the two
assistants is in `results/gold/`. Items were never seen in training. Austria: 400 free-text
extracts in two batches, the second focused on boards and partnerships. **Register-labelled test
parts** use the registers' own codes as labels. Numbers are generated from `results/`.

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

<!-- table:at-structures -->
| Structure (second batch, 200) | Coalitions | Min. signers | Dangerous errors |
|---|---|---|---|
| Vorstand (AG, Genossenschaft, Privatstiftung) | 99.7 % | 99.1 % | 3 / 924 |
| Partnerships (OG, KG) | 97.7 % | 95.8 % | 3 / 176 |
| GmbH with several Geschäftsführer | 99.4 % | 100.0 % | 1 / 165 |
<!-- /table -->

<!-- table:other-registers -->
| Reference set | CEO alone | Coalitions | Min. signers | Rule type | In training? |
|---|---|---|---|---|---|
| Norway (250, beyond the interpreter) | 100.0 % | 99.0 % | 98.0 % | 93.0 % | trained |
| Denmark (300) | 89.2 % | 92.5 % | 86.4 % | 62.4 % | **never seen** |
<!-- /table -->

<!-- table:in-distribution -->
| Register-labelled test (codes / interpreter) | Coalitions | Min. signers | Rule type |
|---|---|---|---|
| Norway, held-out texts | 100.0 % | 99.7 % | 99.8 % |
| Austria, held-out patterns | 100.0 % | 100.0 % | 99.8 % |
| Austria, frozen pilot (2,500 companies) | 100.0 % | 100.0 % | 100.0 % |
<!-- /table -->

A fine-tuned encoder (mmBERT-base) trained on the same Austrian labels is close to this model on
the Austrian gold set (table); this model ranks its own confidence better (lower risk at the
same coverage), which the abstention relies on, and serves several registers with one question set.

Benchmarks encode every request as the server does (states up to 64k tokens). An earlier benchmark
setting skipped Austrian states over 384 tokens; all numbers above include them.

## Limitations and risks

- **Errors that matter most are rare but not zero:** answering "can sign" where the answer is
  "cannot" (see the dangerous-error line above). Keep a human in the loop.
- **Per-person powers:** when holders of one office have different powers, the office-level
  question is open by our convention; the model tends to answer "yes" if one holder may act alone.
- **Denmark (zero-shot):** "direktionen" is often read as a collective body; abstention thresholds
  fitted on Norway and Austria do not transfer to an unseen register (observed risk above the
  target). Treat Danish answers as beta.
- **Ambiguity score:** experimental, never served as a decision.
- **No fully human-verified evaluation subset:** the hard-case numbers are agreement with the
  consensus of two AI assistants (human spot-checked), not with independent human annotators.
- **Pre-registration:** hypotheses and decision rules were written before results were read
  (`results/` holds every outcome). The release model was chosen by the project owner although one
  pre-registered criterion (no drop on any question vs. the previous model; it failed on the
  experimental ambiguity score) was not met; both models' results are published.

## Use of AI tools

The reference-set answers were produced by AI assistants (see Evaluation). The code, documentation and
this card were written with the help of an AI coding assistant; every number comes from `results/`.
Training labels never come from language models.

## Privacy

The model never needs names: callers pass roles and counts; the reference server masks names that
are passed anyway and abstains when a text still looks like it contains one. No register data,
training data or gold annotations are distributed.

## Citation

The paper:

```bibtex
@misc{ildan2026whomaysign,
  author    = {Ildan, Ali},
  title     = {Who May Sign? Typed, Calibrated Decisions on Company Representation Rules in the Austrian Commercial Register},
  year      = {2026},
  publisher = {Zenodo},
  note      = {Preprint},
  doi       = {10.5281/zenodo.23224942},
  url       = {https://doi.org/10.5281/zenodo.23224942}
}
```

The code and model:

```bibtex
@software{ildan2026signrule,
  author    = {Ildan, Ali},
  title     = {SignRule-Decide: typed, calibrated decisions on company representation rules},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23224891},
  url       = {https://github.com/aliildan/signrule-decide}
}
```

## Attribution

Contains data from Brønnøysundregistrene (NLOD); Firmenbuch – Bundesministerium für Justiz /
JustizOnline (HVD), CC BY 4.0; Det Centrale Virksomhedsregister (CVR), Erhvervsstyrelsen —
"Indeholder data, som benyttes i henhold til vilkår for brug af danske offentlige data".
