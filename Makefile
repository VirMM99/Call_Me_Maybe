MYPY_FLAGS := --python-version 3.10 \
            --warn-return-any \
            --warn-unused-ignores \
            --ignore-missing-imports \
            --disallow-untyped-defs \
            --check-untyped-defs


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
    find . -type d -name "__pycache__" -exec rm -rf {} +
    rm -rf .mypy_cache

# Mandatory lint: flake8 + mypy with the exact flags required by the subject
lint:
	uv run flake8 .
	uv run mypy . $(MYPY_FLAGS)
# Optional stricter lint (mypy --strict)
lint-strict:
	uv run flake8 .
	uv run mypy . --strict

.PHONY: install run debug clean lint lint-strict