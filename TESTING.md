# Noeta Testing Guide

Complete guide for running tests in the Noeta DSL project.

---

## Quick Start

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v
```

---

## Basic Commands

### Run All Tests
```bash
pytest tests/
```
Runs all tests in the `tests/` directory.

### Run Tests with Verbose Output
```bash
pytest tests/ -v
```
Shows detailed output for each test (test names, pass/fail status).

### Run Tests Quietly
```bash
pytest tests/ -q
```
Minimal output - only shows summary at the end.

---

## Running Specific Tests

### Run Single Test File
```bash
# Run only lexer tests
pytest tests/test_lexer.py

# Run only parser tests
pytest tests/test_parser.py

# Run only semantic tests
pytest tests/test_semantic.py

# Run only integration tests
pytest tests/test_integration.py

# Run only code generator tests
pytest tests/test_codegen.py

# Run only error tests
pytest tests/test_errors.py
```

### Run Specific Test Class
```bash
# Run all tests in a specific class
pytest tests/test_lexer.py::TestLexerBasicTokenization

# Run semantic undefined dataset tests
pytest tests/test_semantic.py::TestSemanticUndefinedDataset
```

### Run Specific Test Function
```bash
# Run single test function
pytest tests/test_lexer.py::TestLexerBasicTokenization::test_tokenize_simple_load

# Run semantic suggestion test
pytest tests/test_semantic.py::TestSemanticDatasetSuggestions::test_suggests_similar_dataset
```

### Run Tests Matching Pattern
```bash
# Run all tests with "semantic" in the name
pytest tests/ -k semantic

# Run all tests with "error" in the name
pytest tests/ -k error

# Run all tests with "integration" in the name
pytest tests/ -k integration

# Run tests NOT matching a pattern
pytest tests/ -k "not integration"
```

---

## Coverage Reports

### Show Coverage in Terminal
```bash
pytest tests/ --cov=. --cov-report=term-missing
```
Displays code coverage percentage and shows which lines are not covered.

### Generate HTML Coverage Report
```bash
pytest tests/ --cov=. --cov-report=html
```
Creates an HTML report in `htmlcov/` directory.
Open `htmlcov/index.html` in your browser to view detailed coverage.

### Coverage for Specific Module
```bash
# Coverage for lexer only
pytest tests/test_lexer.py --cov=noeta_lexer --cov-report=term-missing

# Coverage for semantic analyzer only
pytest tests/test_semantic.py --cov=noeta_semantic --cov-report=term-missing
```

---

## Advanced Options

### Stop at First Failure
```bash
pytest tests/ -x
```
Stops running tests after the first failure. Useful for debugging.

### Run Only Failed Tests from Last Run
```bash
pytest tests/ --lf
```
Re-runs only the tests that failed last time. Saves time during debugging.

### Run Tests in Parallel (Faster)
```bash
pytest tests/ -n auto
```
Runs tests in parallel using all available CPU cores. Requires `pytest-xdist`.

### Show Detailed Error Traces
```bash
# Short traces (default)
pytest tests/ --tb=short

# Long traces (full details)
pytest tests/ --tb=long

# No traces (only summary)
pytest tests/ --tb=no

# Only one line per failure
pytest tests/ --tb=line
```

### Print Statements During Tests
```bash
pytest tests/ -s
```
Shows `print()` statements from tests (normally hidden).

### Collect Tests Without Running
```bash
pytest tests/ --collect-only
```
Lists all tests that would be run without actually running them.

---

## Debugging Tests

### Debug Single Test with Output
```bash
pytest tests/test_semantic.py::TestSemanticUndefinedDataset::test_undefined_dataset_in_select -v -s
```
Shows verbose output and print statements for debugging.

### Run with Python Debugger
```bash
pytest tests/test_lexer.py --pdb
```
Drops into Python debugger (pdb) on test failures.

### Run with Warnings Shown
```bash
pytest tests/ -W all
```
Shows all Python warnings during test execution.

---

## Test Selection by Markers

### Run Only Unit Tests
```bash
pytest tests/ -m unit
```

### Run Only Integration Tests
```bash
pytest tests/ -m integration
```

### Run Smoke Tests (Quick Tests)
```bash
pytest tests/ -m smoke
```

### Skip Slow Tests
```bash
pytest tests/ -m "not slow"
```

---

## Continuous Integration

### Full Test Suite with Coverage
```bash
pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html
```
Recommended for CI/CD pipelines.

### Quick Validation
```bash
pytest tests/ -x --tb=short
```
Fast feedback - stops at first failure with short error traces.

---

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures (11 fixtures)
├── test_lexer.py            # Lexer tests (60 tests)
├── test_parser.py           # Parser tests (75 tests)
├── test_codegen.py          # Code generator tests (21 tests)
├── test_semantic.py         # Semantic analyzer tests (48 tests)
├── test_integration.py      # Integration tests (55 tests)
└── test_errors.py           # Error handling tests (37 tests)

Total: 208 tests
```

---

## Common Workflows

### During Development
```bash
# Quick check while coding
pytest tests/test_semantic.py -x -q

# Run related tests
pytest tests/ -k "semantic or integration" -v
```

### Before Committing
```bash
# Run all tests with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

### Debugging a Failure
```bash
# Run only the failing test with full output
pytest tests/test_lexer.py::TestLexerErrors::test_invalid_character -v -s --tb=long

# Re-run last failures
pytest tests/ --lf -v
```

### Performance Check
```bash
# Show slowest 10 tests
pytest tests/ --durations=10

# Run in parallel for speed
pytest tests/ -n auto
```

---

## Understanding Test Output

### Passed Test
```
tests/test_lexer.py::TestLexerBasic::test_tokenize_simple_load PASSED [10%]
```
✅ Test passed successfully.

### Failed Test
```
tests/test_lexer.py::TestLexerBasic::test_tokenize_simple_load FAILED [10%]
```
❌ Test failed - scroll down for error details.

### Skipped Test
```
tests/test_lexer.py::TestLexerBasic::test_tokenize_simple_load SKIPPED [10%]
```
⏭️ Test was skipped (usually due to @pytest.skip decorator).

### Error Test
```
tests/test_lexer.py::TestLexerBasic::test_tokenize_simple_load ERROR [10%]
```
💥 Test had an error during setup/teardown.

---

## Summary Statistics

At the end of each test run, you'll see:
```
======================== 150 passed, 58 failed in 1.78s ========================
```

This shows:
- **150 passed**: Tests that succeeded
- **58 failed**: Tests that failed
- **1.78s**: Total execution time

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'pytest'"
**Solution**: Install pytest dependencies
```bash
pip install pytest pytest-cov pytest-xdist
```

### "No tests collected"
**Solution**: Make sure you're in the project root directory
```bash
cd /path/to/noeta
pytest tests/
```

### Tests are slow
**Solution**: Run in parallel
```bash
pytest tests/ -n auto
```

### Need to see print statements
**Solution**: Use -s flag
```bash
pytest tests/ -s
```

### Too much output
**Solution**: Use quiet mode
```bash
pytest tests/ -q --tb=no
```

---

## Quick Reference Table

| Command | Description | Use Case |
|---------|-------------|----------|
| `pytest tests/` | Run all tests | Standard test run |
| `pytest tests/ -v` | Verbose output | See each test result |
| `pytest tests/ -q` | Quiet output | Minimal output |
| `pytest tests/ -x` | Stop at first failure | Quick debugging |
| `pytest tests/ -s` | Show print statements | Debug with prints |
| `pytest tests/ --lf` | Run last failures | Fix failures iteratively |
| `pytest tests/ -n auto` | Parallel execution | Speed up tests |
| `pytest tests/ -k pattern` | Run matching tests | Test subset |
| `pytest tests/ --cov=.` | Show coverage | Check test coverage |
| `pytest tests/ --tb=short` | Short error traces | Less verbose errors |
| `pytest tests/ --collect-only` | List tests | See available tests |
| `pytest tests/ --durations=10` | Show slowest tests | Find slow tests |

---

## Example Commands for Common Scenarios

### "I just changed the lexer"
```bash
pytest tests/test_lexer.py -v
```

### "I want to know coverage"
```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

### "A semantic test is failing"
```bash
pytest tests/test_semantic.py::TestSemanticUndefinedDataset -v -s
```

### "I want a quick sanity check"
```bash
pytest tests/ -x -q
```

### "I need the full picture"
```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

### "Only integration tests"
```bash
pytest tests/test_integration.py -v
```

---

## Test Configuration

Tests are configured via `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers --tb=short --color=yes
```

---

## Resources

- **pytest documentation**: https://docs.pytest.org/
- **Coverage.py documentation**: https://coverage.readthedocs.io/
- **Project tests**: `tests/` directory
- **Test fixtures**: `tests/conftest.py`

---

**Last Updated**: December 17, 2025
**Total Tests**: 208 (150 passing, 58 failing, 72% pass rate)
