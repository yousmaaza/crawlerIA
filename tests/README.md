# Testing Strategy for Multimodal RAG System

This directory contains tests for the Multimodal RAG system. The tests are organized into unit tests and integration tests.

## Test Categories

- **Unit Tests**: Tests for individual components with mocks for external dependencies
- **Integration Tests**: Tests that verify how components work together or with external services
- **Crawler Tests**: Tests specific to the crawler module

## Running Tests

### Using pytest directly

```bash
# Run all tests
pytest

# Run only unit tests
pytest -m unit

# Run only crawler tests
pytest -m crawler

# Run only integration tests
pytest -m integration

# Run tests with verbose output
pytest -v

# Run a specific test file
pytest tests/test_crawler.py
```

### Using the run_tests.py script

We provide a convenience script to run the tests:

```bash
# Run unit tests (default)
python run_tests.py

# Run all tests
python run_tests.py --all

# Run only unit tests
python run_tests.py --unit

# Run only integration tests
python run_tests.py --integration

# Run only crawler tests
python run_tests.py --crawler

# Run with verbose output
python run_tests.py --verbose
```

## Requirements for Integration Tests

Some integration tests require API keys to be set in the environment:

- `FIRECRAWL_API_KEY`: Required for crawler integration tests
- `COLIVARA_API_KEY`: Required for document processor integration tests

If these keys are not set, the corresponding tests will be skipped.

## Test Data

Tests that require test data will create and clean up after themselves. Integration tests will create files in the `data/screenshots`, `data/pdfs`, and other directories as needed. 

The tests are designed to clean up after themselves, but if they fail in unexpected ways, you may need to manually delete test files.

## Adding New Tests

When adding new tests:

1. Place unit tests in files named `test_*.py`
2. Use appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)
3. Consider using `pytest.mark.skipif` for tests that require external dependencies
4. Ensure tests clean up after themselves in teardown methods

## Mocking External Services

For unit tests, we mock external services like FirecrawlApp and ColiVara using the `unittest.mock` module. See existing tests for examples of how to mock these services.
