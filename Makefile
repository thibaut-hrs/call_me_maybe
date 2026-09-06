UV = uv
PROJECT = call_me_maybe.py

install:
	$(UV) sync

run:
	$(UV) run python $(PROJECT)

lint:
	$(UV) run flake8 --exclude=.venv/,llm_sdk/ .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

test:
	$(UV) run python src/experiments/experiment.py