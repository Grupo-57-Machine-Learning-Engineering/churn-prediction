.PHONY: install lint format test cov pre-commit clean

install:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

cov:
	uv run pytest --cov=src --cov-report=term-missing

pre-commit:
	uv run pre-commit run --all-files

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
