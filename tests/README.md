# Tests

This directory contains platform contract and cross-project integration tests.
Project-owned behavior is tested beside each maintained project under
`projects/<name>/tests/`.

## Running Tests

```bash
# Complete platform and recipe suite
uv run pytest

# Lightweight tests that do not require the NeMo dependency stack
uv run pytest -m "not requires_nemo"

# Parallel execution
uv run pytest -n auto

# One test module
uv run pytest tests/test_core.py

# Run with coverage
uv run pytest --cov=nvflow --cov-report=html
```

## Test Structure

```text
tests/
├── fixtures/                    # Stable command/config snapshots
├── test_architecture_*.py       # Package and dependency boundaries
├── test_project_*.py            # Project/recipe contracts and packaging
├── test_workflow_*.py           # Workflow loading and validation
└── test_*.py                    # Platform integration and utility tests

projects/<name>/tests/           # Domain behavior owned by that project
```

Project-environment tests use temporary caches and verify each project resolves
without installing sibling projects. Normal unit tests disable transparent CLI
handoff so in-process fakes remain deterministic.

## Writing Tests

### Unit Tests

Test individual components in isolation:

```python
def test_my_function():
    result = my_function(input_data)
    assert result == expected_output
```

### Full-dependency tests

Tests requiring the installed NeMo-Skills/NeMo-Run stack use the only registered
marker:

```python
@pytest.mark.requires_nemo
def test_workflow_plan():
    ...
```

Pytest uses strict marker validation. Add any new marker to `pyproject.toml`
before using it.

### Fixtures

Use fixtures for common setup:

```python
@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_with_fixture(sample_data):
    assert sample_data["key"] == "value"
```

## Test Coverage

Coverage identifies untested behavior but is not a substitute for contract,
snapshot, and boundary assertions. View the generated report:
```bash
uv run pytest
# Open htmlcov/index.html in browser
```

## Continuous Integration

CI separates lightweight and full-dependency lanes. Pre-commit runs formatting,
linting, type checking, license checks, and dependency-policy validation.
