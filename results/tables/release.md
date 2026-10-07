## at-gold

| Austria, reference set (400) | GF alone | Chair alone | Coalitions (11) | Min. signers | Rule type |
|---|---|---|---|---|---|
| **SignRule-Decide 4B** | 100.0 % | 100.0 % | 99.3 % | 97.7 % | 94.6 % |
| mmBERT-base, fine-tuned (1,024 tokens) | 100.0 % | 100.0 % | 98.8 % | 96.9 % | 94.3 % |
| XLM-R-large, fine-tuned (512 tokens) | 73.5 % | 95.8 % | 92.4 % | 63.5 % | 54.8 % |
| Phrase table (rules from the training labels) | 100.0 % | 100.0 % | 91.4 % | 80.2 % | 75.7 % |
| Keyword rules (selbständig / gemeinsam) | 100.0 % | 100.0 % | 54.3 % | 35.0 % | 11.1 % |

SignRule-Decide 4B on the same items: dangerous yes/no errors ("can sign" when the answer is "cannot") 12 of 2178 (0.6 %); at a 2 % risk target it answers 99.9 % of the questions with an observed risk of 1.3 %.

## at-structures

| Structure (second batch, 200) | Coalitions | Min. signers | Dangerous errors |
|---|---|---|---|
| Vorstand (AG, Genossenschaft, Privatstiftung) | 99.7 % | 99.1 % | 3 / 924 |
| Partnerships (OG, KG) | 97.7 % | 95.8 % | 3 / 176 |
| GmbH with several Geschäftsführer | 99.4 % | 100.0 % | 1 / 165 |

## other-registers

| Reference set | CEO alone | Coalitions | Min. signers | Rule type | Register in training? |
|---|---|---|---|---|---|
| Norway, texts the official interpreter could not read (250) | 100.0 % | 99.0 % | 98.0 % | 93.0 % | trained |
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

| Register | Read from the register | Kept | Unique (text, roles) | Train / val / test |
|---|---|---|---|---|
| Norway | 637,492 companies (1,274,984 API responses) | 597,487 | 31,583 | 25,394 / 2,729 / 3,460 |
| Austria | 239,474 company extracts | 206,553 | 7,710 patterns | 6,197 / 744 / 769 (+ 2,500 pilot) |
| Denmark | 143,646 companies | 143,311 | 14,147 | evaluation only |
| **Total** | **1,020,612 companies** |  |  | **31,591 training cases**, 2 epochs, 37.7 M tokens |
