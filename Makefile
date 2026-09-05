UV = uv
PROJECT = call_me_maybe.py

install:
	$(UV) sync

run:
	$(UV) run python $(PROJECT)