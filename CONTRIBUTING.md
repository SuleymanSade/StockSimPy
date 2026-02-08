# Contributing to stocksimpy

Thanks for your interest in contributing! stocksimpy is a beginner-friendly Python library for loading market data, running strategies, and backtesting. Contributions of all kinds — bug reports, documentation, new features, or tests — are welcome.

## 1. Code of Conduct

Please follow the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) in all interactions.

## 2. Setting Up the Development Environment

1. Clone the repository:

    ```bash
    git clone https://github.com/suleymansade/stocksimpy.git
    cd stocksimpy
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

3. Install stocksimpy in editable mode with development dependencies:

    ```bash
    pip install -e ".[dev]"
    ```

4. Install pre-commit hooks:

    ```bash
    pre-commit install
    ```

## 3. Code Style

### Formatting

* Black handles automatic formatting.
* Line length: 88 characters.
* All commits must pass pre-commit formatting hooks.

### Typing

* All public functions must include type hints.

### Docstrings

* Use **NumPy-style docstrings** for all public functions, classes, and methods.
* Docstrings must include:
  * Parameters
  * Returns
  * Raises (if applicable)
  * Examples (if helpful)
* Follow the official guide for formatting: [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)

### Imports

* Absolute imports preferred.
* Group imports as: standard library → third-party → internal modules.

## 4. Project Structure

```bash
src/stocksimpy/ # Main code
            core/ # Backtesting, data loading
            addons/ # Extra resources, strategies, indicators
            utils/ # Tools for analysis, visualizing, performance
docs/
    source/ # documentation, sphinx docs
tests/
    # all the unit tests
...
```

## 5. Adding Features

* Open an issue first for major changes.
* Include a clear explanation, API design, and examples.
* Pull Requests must include tests, documentation updates, and clear commit messages.

## 6. Testing

* We use ``pytest``
* Run all tests with

```bash
pytest
```

## 7. Documentation

* You can find the latest stable docs [here](https://stocksimpy.readthedocs.io/en/latest/index.html)
* Docs live under docs/source/.
* Update docs for any public API you change.
* Build docs locally:

```bash
cd docs
make html # Alternatively you can use: 
# sphinx-build -b html source build\html
```

## 8. Pull Requests

Before submitting:

* Ensure pytest passes.
* Ensure pre-commit hooks pass.
* Include updated documentation.
* Link your PR to a related issue if applicable.

## 9. Reporting Bugs

* Include a minimal reproducible example.
* Describe expected vs actual behavior.
* Include OS, Python version, and stocksimpy version.

## 10. Feature Suggestions

* Propose features via GitHub issues.
* Avoid large PRs without prior discussion.
