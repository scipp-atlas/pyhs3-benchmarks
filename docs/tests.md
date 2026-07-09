# Testing

The project includes a comprehensive unit test suite covering benchmark execution, validation, plotting utilities, helper functions, and command-line interfaces.

All tests are executed through the project's Pixi environment, ensuring a consistent and reproducible software stack across development machines and continuous integration.

---

# Running the Test Suite

To execute the complete test suite, run

```bash
pixi run test
```

or equivalently

```bash
pixi run pytest
```

---

# Running Coverage

To generate a terminal coverage report, run

```bash
pixi run coverage
```

which executes

```bash
pytest --cov=src --cov-report=term-missing
```

To generate an HTML coverage report,

```bash
pixi run pytest \
    --cov=src \
    --cov-report=html
```

The report will be written to

```text
htmlcov/
└── index.html
```

---

# Running Individual Test Files

Any benchmark can be tested independently.

Examples:

```bash
pixi run pytest tests/test_run_workspace_loading.py
```

```bash
pixi run pytest tests/test_run_model_creation.py
```

```bash
pixi run pytest tests/test_run_log_prob_compilation.py
```

```bash
pixi run pytest tests/test_run_compiled_evaluation.py
```

```bash
pixi run pytest tests/test_run_pdf_evaluation.py
```

```bash
pixi run pytest tests/test_run_nll_scan.py
```

```bash
pixi run pytest tests/test_run_cross_nll_scan.py
```

---

# Running Individual Tests

Pytest also allows execution of a single test function.

For example,

```bash
pixi run pytest \
    tests/test_run_workspace_loading.py \
    -k test_run_single_benchmark
```

or

```bash
pixi run pytest \
    tests/test_run_workspace_loading.py::test_run_single_benchmark
```

---

# Test Organization

The test suite mirrors the benchmark implementation.

```text
tests/
├── test_run_workspace_loading.py
├── test_run_model_creation.py
├── test_run_log_prob_construction.py
├── test_run_log_prob_compilation.py
├── test_run_compiled_evaluation.py
├── test_run_pdf_evaluation.py
├── test_run_nll_scan.py
├── test_run_memory_scaling.py
├── test_run_model_complexity_scaling.py
├── test_run_graph_canonicalization.py
├── test_run_graph_optimization.py
├── test_run_cross_nll_scan.py
├── test_run_cross_scalar_pdf_evaluation.py
├── test_run_all_benchmarks.py
└── test_utils.py
```

Each benchmark module has a corresponding test module covering:

- benchmark configuration validation;
- successful benchmark execution;
- failure handling;
- JSON output generation;
- plotting utilities;
- command-line interface behavior;
- numerical validation where applicable.

---

# Mock-based Testing

Most unit tests use lightweight mocks instead of real HS3 workspaces.

This approach provides several advantages:

- deterministic execution;
- fast test runtime;
- no dependency on generated benchmark inputs;
- reliable execution in continuous integration environments.

Integration tests requiring real benchmark workspaces are intentionally kept to a minimum.

---

# Development Checks

Several additional development tasks are available through Pixi.

Run the complete validation suite before opening a pull request:

```bash
pixi run ci
```

This performs:

- Python source compilation;
- Ruff linting;
- Ruff formatting checks.

To run the linter only,

```bash
pixi run lint
```

To automatically format the source code,

```bash
pixi run format
```

To verify formatting without modifying files,

```bash
pixi run format-check
```

If pre-commit hooks are used, they can be installed with

```bash
pixi run install-hooks
```

and executed manually with

```bash
pixi run pre-commit
```

---

# Continuous Integration

The GitHub Actions workflow automatically executes the project's validation pipeline for every push and pull request.

Running

```bash
pixi run ci
```

and

```bash
pixi run test
```

locally before submitting changes is therefore strongly recommended.
