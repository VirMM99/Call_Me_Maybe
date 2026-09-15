PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
# MAIN ?= Call_Me_Maybe/main.py

.PHONY: install run debug clean lint lint-strict

# Creates a virtual environment and installs the project dependencies.
# A venv is used because campus Python installations are often
# "externally managed" (PEP 668) and reject global pip installs.
install:
	uv sync

# Runs the main script with the default
run:
	uv run python -m src

# Runs the main script under pdb (debug mode)
debug:
	uv run python -m pdb -m src 

# Removes temporary files, caches and the virtual environment
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	rm -rf .mypy_cache .pytest_cache .ruff_cache build dist *.egg-info

# Mandatory lint: flake8 + mypy with the exact flags required by the subject
lint:
	$(VENV)/bin/flake8 . --exclude=.venv,venv,env,__pycache__,.git
	$(VENV)/bin/mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude=.venv,venv,env,__pycache__,.git

# Optional stricter lint (mypy --strict). mypy is run from inside Call_Me_Maybe/
# because the modules use flat imports (e.g. "from hub import Hub") that
# Python only resolves because the script directory is on sys.path.
lint-strict:
	$(VENV)/bin/flake8 . --exclude=.venv,venv,env,__pycache__,.git
	../$(VENV)/bin/mypy . --strict
