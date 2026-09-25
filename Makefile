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
	$(UV) run python -m src

lint:
	$(UV) run flake8 --exclude=.venv/,llm_sdk/,.huggingface,.mypy_cache,.uv-cache .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

debug:
	$(UV) run python -m pdb $(PROJECT)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

check-cache:
	$(UV) run python -c "import os; print(os.getenv('HF_HOME')); print(os.getenv('HF_HUB_CACHE'))"

uninstall: clean
	rm -rf .huggingface .uv-cache .venv

.PHONY: install run lint debug clean check-cache
