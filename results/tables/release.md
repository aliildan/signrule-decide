## headline

| Austria, 400 extracts never trained on | SignRule-Decide 4B |
|---|---|
| Office and coalition questions (15 yes/no, 2,178 answers) | 99.3 % |
| Minimum signers (389 extracts) | 97.7 % |
| Rule type (14 patterns) | 94.6 % |
| Dangerous errors ("can sign" when it cannot) | 0.6 % (12 of 2,178; 10 with confidence ≥ 0.9) |
| At a 2 % risk target | answers 99.9 %, observed risk 1.3 % |

## at-gold

| Austria, reference set (400) | GF alone | Chair alone | Coalitions (15) | Min. signers | Rule type |
|---|---|---|---|---|---|
| **SignRule-Decide 4B** | 100.0 % | 100.0 % | 99.3 % | 97.7 % | 94.6 % |
| mmBERT-base, fine-tuned (1,024 tokens) | 100.0 % | 100.0 % | 98.8 % | 96.9 % | 94.3 % |
| XLM-R-large, fine-tuned (512 tokens) | 73.5 % | 95.8 % | 89.3 % | 63.5 % | 54.8 % |
| Phrase table (rules from the training labels) | 100.0 % | 100.0 % | 92.1 % | 80.2 % | 75.7 % |
| Keyword rules (selbständig / gemeinsam) | 100.0 % | 100.0 % | 59.1 % | 35.0 % | 11.1 % |

SignRule-Decide 4B on the same items: dangerous yes/no errors ("can sign" when the answer is "cannot") 12 of 2178 (0.6 %); at a 2 % risk target it answers 99.9 % of the questions with an observed risk of 1.3 %.

## at-structures

| Structure (second batch, 200) | Coalitions | Min. signers | Dangerous errors |
|---|---|---|---|
| Vorstand (AG, Genossenschaft, Privatstiftung) | 99.7 % | 99.1 % | 3 / 924 |
| Partnerships (OG, KG) | 97.7 % | 95.8 % | 3 / 176 |
| GmbH with several Geschäftsführer | 99.4 % | 100.0 % | 1 / 165 |

## other-registers

| Reference set | CEO alone | Coalitions | Min. signers | Rule type | In training? |
|---|---|---|---|---|---|
| Norway (250, beyond the interpreter) | 100.0 % | 99.0 % | 98.0 % | 93.0 % | trained |
| Denmark (300) | 89.2 % | 92.5 % | 86.4 % | 62.4 % | **never seen** |

## in-distribution

| Register-labelled test (codes / interpreter) | Coalitions | Min. signers | Rule type |
|---|---|---|---|
| Norway, held-out texts | 100.0 % | 99.7 % | 99.8 % |
| Austria, held-out patterns | 100.0 % | 100.0 % | 99.8 % |
| Austria, frozen pilot (2,500 companies) | 100.0 % | 100.0 % | 100.0 % |

## compute

| Compute | Value |
|---|---|
| GPU | 1 × NVIDIA GeForce RTX 5090 (31.8 GB) |
| CPU / RAM | Intel(R) Core(TM) i9-14900KF / 92 GB |
| Training (LoRA, bf16) | 14.6 h, peak GPU memory 20.2 GB, 7,836 optimizer steps |
| Serving (one request, 8.5 questions) | p50 160 ms, p95 175 ms, 6.7 requests/s |

## data

| Register | Companies read | Kept | Unique | Train / val / test |
|---|---|---|---|---|
| Norway | 637,492 | 597,487 | 31,583 | 25,394 / 2,729 / 3,460 |
| Austria | 239,474 | 206,553 | 7,710 | 6,197 / 744 / 769 |
| Denmark | 143,646 | 143,311 | 14,147 | evaluation only |
| **Total** | **1,020,612** |  |  | **31,591** training cases |

Companies read: Norway with 1,274,984 API responses (signing and procuration); Austria company extracts. Unique = distinct (text, roles) groups; Austria: date-normalised patterns, plus a frozen pilot of 2,500 companies. Training: 2 epochs, 37.7 M tokens.

## prereg

|  | Pre-registered hypothesis | Outcome | Evidence |
|---|---|---|---|
| H1 | The best decision LLM beats the best fine-tuned encoder on the Norwegian test (paired CI lower bound > 0) | **not met** | 9B - mmBERT-base +0.30 pp, CI95 [-0.06, +0.70] |
| H2 | After temperature scaling on validation, test ECE <= 0.05 (yes/no, choice) | **met** | ECE 0.002 / 0.007 |
| H3 | Learn-then-Test: yes/no coverage >= 50 % at alpha 2 % with test risk <= alpha | **met** | coverage 100 %, risk 0.6 % |
| H4 | Temporal test within 3 pp of the random test on four main questions | **met (weak)** | 73-87 temporal items |
| H5 | Norway-only model, zero-shot on court-coded Austrian companies: CEO alone and procuration present >= 90 % | **met** | 90.4 % / 100 % |
| H6 | Beats TF-IDF and the encoder on Austrian CEO alone by >= 10 pp | **partly** | +84 pp vs TF-IDF; encoder not run then |
| H7 | Norwegian thresholds keep risk <= 5 % on Austria at alpha 2 % | **met (boundary)** | 5.0 % |
| H8-H10 | Norway + Austria training (codes only) vs Norway only | **superseded** | replaced by A'/B'/C' (coalition questions) before any training |
| H11 | Austrian reference set: coalitions >= 95 %, derived min signers >= 90 %, derived rule type >= 85 % | **not met** | 99.4 % / 78.3 % / 84.8 % (direct heads 96.8 % / 96.7 %, secondary) |
| H12 | At alpha 2 %: >= 70 % of Austrian coalition items answered with risk <= 5 % | **met** | coverage 100 %, risk 0.6 % |
| H13 | Norway + Austria beats Norway only by >= 10 pp on Austrian coalitions; Norway unchanged | **met** | +17.7 pp; worst Norwegian change 0.00 pp |
| H14 | Beats the phrase table on (coalitions + derived min signers) / 2 | **not met** | 88.9 % vs 90.6 % (direct head 98.1 %, secondary) |
| H15 | Norwegian reference set rule type >= 75 %, and no question drops > 1 pp vs C' | **not met** | 93.0 % (C' 62.3 %); Austrian ambiguity -12.6 pp, one two-board-members item -3.6 pp |
| H16 | Austrian reference set: derived min signers >= 90 %, derived rule type >= 85 %, coalitions >= 95 % | **not met** | 77.7 % / 88.0 % / 99.2 % |
| H17 | Danish reference set, never trained on: coalitions >= 90 %, min signers >= 85 % | **met** | 92.5 % / 86.4 % |
| H18 | Strictest merged policy keeps risk <= 5 % at alpha 5 % on Denmark | **not met** | risk 14.3 % |
| H21 | Boards and partnerships: coalitions >= 95 %, min signers >= 90 %, dangerous yes/no errors <= 1 % | **met** | 99.4 % / 98.5 % / 0.55 % |
| H22 | Risk <= 2 % at alpha 2 % on answered structural questions | **met** | 1.3 % at 99.9 % coverage |

## ollama

| Reference sets | Reference server (Kev) | Ollama model (Strands Decider) |
|---|---|---|
| Austria: office and coalition questions | 99.3 % | 99.5 % |
| Austria: minimum signers | 97.7 % | 96.9 % |
| Austria: rule type | 94.6 % | 94.3 % |
| Austria: dangerous errors ("can sign" when it cannot) | 12 of 2,178 | 6 of 2,178 |
| Norway (beyond the interpreter): rule type | 93.0 % | 95.5 % |
| Denmark (never trained on): coalitions | 92.5 % | 93.2 % |

The Ollama model's numbers are measured on its PyTorch weights. Served by Ollama (q8), it gives the same decision on 99.9 % of the 4,308 Austrian answers, mean probability difference 0.0005, largest 0.10 (on answers the model itself was unsure about).
