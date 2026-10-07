# SignRule-Decide — common commands (CLAUDE.md §10). Data commands print aggregates only.
UV := uv run
CFG ?= configs/train/base.yaml
RUN ?=
SPLIT ?= random

.PHONY: env hooks test lint publish-check vault ingest-no data data-check baselines train bench eval serve ingest-dk ingest-at calib

env:            ## sync deps and verify torch sees sm_120
	uv sync && $(UV) python scripts/check_env.py

hooks:          ## install git hooks (pre-commit publish check, pre-push main guard)
	git config core.hooksPath scripts/hooks

test:
	$(UV) pytest

lint:
	$(UV) ruff check . && $(UV) ruff format --check . && $(UV) pyright

publish-check:  ## scan the public allow-list for data, secrets, PII
	python3 scripts/publish_check.py --publish

vault:          ## mount the encrypted vault (after reboot)
	scripts/setup_data_root.sh mount

ingest-no:      ## Norway: resumable background fetch (systemd user unit)
	scripts/run_ingest_no.sh

data:           ## cache -> masked, labelled, deduplicated, split requests
	$(UV) python -m signrule.normalize.pipeline_no run

data-check:     ## leakage + PII scan + unmapped share; stamps the data for training
	$(UV) python -m signrule.normalize.pipeline_no check --near-dups

baselines:      ## majority, rules, tf-idf, encoders on val and test
	$(UV) python eval/run_all.py --split $(SPLIT) --part val --baselines majority,rules_no,tfidf_lr,enc:jhu-clsp/mmBERT-base
	$(UV) python eval/run_all.py --split $(SPLIT) --part test --allow-test --baselines majority,rules_no,tfidf_lr,enc:jhu-clsp/mmBERT-base

train:          ## train with Kev under the project guards
	$(UV) python -m signrule.train.kev_wrapper train --config $(CFG)

bench:          ## kev.benchmark (raw logits) on val and test for RUN=runs/<name>
	$(UV) python -m signrule.train.kev_wrapper bench --run $(RUN) --split $(SPLIT) --part val --raw
	$(UV) python -m signrule.train.kev_wrapper bench --run $(RUN) --split $(SPLIT) --part test --raw

eval:           ## calibrated metrics + LTT for RUN=runs/<name> (after `make bench`)
	$(UV) python eval/run_all.py --split $(SPLIT) --part test --allow-test \
	  --kev $(notdir $(RUN))=$(RUN)-bench-no-$(SPLIT)-val,$(RUN)-bench-no-$(SPLIT)-test

calib: eval

serve:          ## /v1/systemone on localhost:8300 for RUN=runs/<name> (NO + AT policies from its val-fitted results)
	$(UV) python server/app.py --run $(RUN) --port 8300 \
	  --policy NO=results/no-$(SPLIT)/$(notdir $(RUN)).test.json \
	  --policy AT=results/at-$(SPLIT)/$(notdir $(RUN)).test.json

ingest-dk:
	@echo "planned in plan-04 (needs CVR access)"; exit 2

ingest-at:
	@echo "planned in plan-05 (needs JustizOnline API key)"; exit 2
