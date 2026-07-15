# Repository Structure

On this page, you will learn how the **PyHS3 Benchmarks** repository is organized and where to find the main components of the benchmarking framework.

The repository is organized around a reproducible benchmarking workflow, with separate directories for benchmark implementations, benchmark inputs, generated outputs, documentation, and automated tests.

The most important directories are

```text
docs/        Documentation
inputs/      Benchmark workspaces
src/         Benchmark implementations
results/     Generated benchmark results
tests/       Automated tests
```

---

# Repository Layout

```text
pyhs3-benchmarks/
│
├── docs/
├── external/
├── inputs/
├── results/
├── src/
├── tests/
│
├── pixi.toml
├── pyproject.toml
└── README.md
```

The repository separates source code, benchmark inputs, generated outputs, and documentation, making benchmark campaigns reproducible and easy to navigate.

---

# Source Code

```text
src/
```

The `src` directory contains the implementation of the benchmarking framework.

Major components include

- workflow benchmark implementations;
- cross-framework benchmark implementations;
- the benchmark matrix runner;
- benchmark stage definitions;
- plotting utilities;
- shared helper functions.

Typical executable entry points include

```text
run_workspace_loading.py
run_model_creation.py
run_pdf_evaluation.py
run_compiled_evaluation.py
run_cross_nll_scan.py
run_all_benchmarks.py
```

Most benchmark suites are implemented as standalone executable modules while sharing common infrastructure for configuration, reporting, plotting, and execution.

---

# Benchmark Inputs

```text
inputs/
```

The `inputs` directory contains the benchmark workspaces used throughout the repository.

These include

- HS3 workspaces;
- matching ROOT workspaces for cross-framework benchmarks;
- models with different channel counts;
- benchmark configurations for scalability studies.

The same benchmark inputs are reused across benchmark campaigns to ensure reproducibility.

For details about available benchmark workspaces, see **Benchmark Workspaces**.

---

# Benchmark Results

```text
results/
```

Benchmark execution stores generated reports under `results/`.

This directory contains

- benchmark JSON reports;
- benchmark campaign summaries;
- execution logs;
- benchmark metadata.

See **Outputs** for the complete report format and generated artifacts.

---

# Documentation

```text
docs/
```

The documentation directory contains

- installation and usage guides;
- benchmark methodology;
- benchmark reference pages;
- cross-framework documentation;
- workspace documentation.

Generated documentation figures are stored under

```text
docs/assets/plots/
```

to keep documentation synchronized with benchmark outputs.

---

# Tests

```text
tests/
```

The repository includes automated tests for

- benchmark execution;
- benchmark infrastructure;
- plotting utilities;
- shared helper modules;
- numerical validation.

See **Development** for contributor guidelines and testing recommendations.

---

# Repository Design

The repository follows a modular organization.

Each directory has a single responsibility:

- `src/` implements benchmark execution.
- `inputs/` stores benchmark workspaces.
- `results/` stores generated benchmark outputs.
- `docs/` documents the benchmarking framework.
- `tests/` validates repository functionality.

This separation simplifies maintenance while making the repository easier to extend.

---

# Related Documentation

See also

- **Getting Started**
- **Installation**
- **Benchmark Methodology**
- **Development**
- **Outputs**
- **Benchmark Workspaces**
