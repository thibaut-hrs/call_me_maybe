UV = uv
PROJECT = src/call_me_maybe.py

UV_CACHE_DIR := $(CURDIR)/.uv-cache
HF_HOME := $(CURDIR)/.huggingface
HF_HUB_CACHE := $(HF_HOME)/hub

export UV_CACHE_DIR
export HF_HOME
export HF_HUB_CACHE

install:
	$(UV) sync

run:
	$(UV) run python $(PROJECT)

lint:
	$(UV) run flake8 --exclude=.venv/,llm_sdk/ .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

test:
	$(UV) run python src/experiments/experiment.py

check-cache:
	$(UV) run python -c "import os; print(os.getenv('HF_HOME')); print(os.getenv('HF_HUB_CACHE'))"
