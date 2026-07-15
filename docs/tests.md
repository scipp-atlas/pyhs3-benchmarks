# Testing

On this page, you will learn how to run the test suite, measure test coverage, and validate repository changes before contributing.

The project includes automated tests covering benchmark execution, shared infrastructure, plotting utilities, helper modules, and command-line interfaces. All tests are executed through the Pixi environment to ensure consistent behavior across development machines and continuous integration.

---

# Running the Test Suite

Run the complete test suite with

```bash
pixi run test
```

or equivalently

```bash
pixi run pytest
```

---

# Measuring Test Coverage

Generate a terminal coverage report:

```bash
pixi run coverage
```

which executes

```bash
pytest --cov=src --cov-report=term-missing
```

To generate an HTML report instead:

```bash
pixi run pytest \
    --cov=src \
    --cov-report=html
```

Coverage reports are written to

```text
htmlcov/
└── index.html
```

---

# Running Individual Test Files

Benchmark test modules can be executed independently.

For example,

```bash
pixi run pytest tests/test_run_workspace_loading.py
```

or

```bash
pixi run pytest tests/test_run_model_creation.py
```

The same pattern applies to all benchmark-specific test modules.

---

# Running Individual Tests

Execute a single test using either a name filter

```bash
pixi run pytest \
    tests/test_run_workspace_loading.py \
    -k test_run_single_benchmark
```

or the fully qualified test name

```bash
pixi run pytest \
    tests/test_run_workspace_loading.py::test_run_single_benchmark
```

---

# Test Organization

The test suite mirrors the structure of the benchmark implementations.

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

Benchmark tests typically cover

- benchmark configuration;
- successful execution;
- failure handling;
- JSON report generation;
- plotting utilities;
- command-line interfaces;
- numerical validation where applicable.

---

# Mock-Based Testing

Most unit tests rely on lightweight mocks rather than real HS3 workspaces.

This approach provides

- deterministic execution;
- fast test runtime;
- minimal external dependencies;
- reliable continuous integration.

Integration tests using real benchmark workspaces are kept to a minimum.

---

# Development Checks

Before opening a pull request, run

```bash
pixi run ci
```

This executes

- Python source compilation;
- Ruff linting;
- formatting checks.

Individual tasks are also available:

```bash
pixi run lint
```

```bash
pixi run format
```

```bash
pixi run format-check
```

If pre-commit hooks are used, install them with

```bash
pixi run install-hooks
```

and execute them with

```bash
pixi run pre-commit
```

---

# Continuous Integration

The GitHub Actions workflow automatically validates the repository for every push and pull request.

Running

```bash
pixi run ci
```

and

```bash
pixi run test
```

locally before submitting changes is strongly recommended.

---

# Related Documentation

See also

- **Development**
- **Repository Structure**
- **Benchmark Methodology**
- **Getting Started**
- **API Reference**
