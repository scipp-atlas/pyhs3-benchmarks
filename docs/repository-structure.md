# Repository Structure

The **PyHS3 Benchmarks** repository is organized around a reproducible benchmarking workflow.

Rather than separating code by implementation details, the repository groups components according to their role in the benchmarking process—from benchmark inputs, through execution, to result generation and documentation.

The most important directories are

```text
docs/        Documentation
inputs/      Benchmark workspaces
src/         Benchmark implementations
results/     Generated benchmark results
tests/       Automated tests
```

Additional directories provide generated reports, published documentation, and external dependencies.

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

---

# Source Code

```text
src/
```

The `src` directory contains the implementation of the benchmarking framework.

Rather than a traditional Python package with many submodules, each benchmark is implemented as a dedicated executable module.

The directory includes

- workflow benchmark implementations;
- cross-framework benchmark implementations;
- the shared benchmark runner;
- benchmark stage definitions;
- plotting utilities;
- shared helper functions.

Typical benchmark entry points include

```text
run_workspace_loading.py
run_model_creation.py
run_pdf_evaluation.py
run_compiled_evaluation.py
run_cross_nll_scan.py
run_all_benchmarks.py
```

Each benchmark can be executed independently from the command line while sharing common infrastructure for reporting, plotting, and configuration.

---

# Benchmark Inputs

```text
inputs/
```

The repository includes a collection of benchmark workspaces covering multiple statistical models and complexity levels.

Benchmark inputs include

- HS3 workspaces;
- corresponding ROOT workspaces used for cross-framework validation;
- benchmark models with varying numbers of channels;
- multiple model configurations for scaling studies.

Most benchmark suites operate directly on these workspaces, making benchmark results reproducible across different environments.

---

# Benchmark Results

```text
results/
```

Benchmark execution produces structured outputs organized by benchmark name.

Typical outputs include

- benchmark result JSON files;
- benchmark matrix summaries;
- benchmark logs;
- execution metadata.

These machine-readable outputs form the basis for visualization, regression tracking, and cross-framework comparison.

---

# Documentation

```text
docs/
```

The documentation contains

- installation and usage guides;
- benchmark methodology;
- benchmark reference pages;
- cross-framework validation documentation;
- workspace documentation;
- generated benchmark figures.

Documentation figures are stored under

```text
docs/assets/plots/
```

allowing figures and documentation to remain synchronized.

---

# Tests

```text
tests/
```

The repository includes automated tests covering

- benchmark execution;
- benchmark runners;
- plotting utilities;
- shared infrastructure;
- numerical validation.

Where appropriate, benchmark outputs are compared against reference values to ensure correctness and reproducibility.

---

# External Dependencies

```text
external/
```

The repository contains external software required for selected benchmark suites.

Currently this directory contains an xRooFit checkout used for cross-framework benchmarking against ROOT-based statistical workflows.

Keeping external dependencies separate from the main source tree simplifies maintenance while preserving reproducibility.

---

# Generated Documentation

```text
site/
```

The `site` directory contains the generated static documentation.

It is produced automatically from the Markdown documentation and associated assets and is intended for deployment rather than manual editing.

---

# Typical Repository Workflow

The repository is organized around the following workflow.

```text
Benchmark Workspace
        │
        ▼
Benchmark Execution
        │
        ▼
JSON Results
        │
        ▼
Plot Generation
        │
        ▼
Documentation
```

Each stage produces inputs for the next, resulting in a fully reproducible benchmarking pipeline.

---

# Design Philosophy

The repository emphasizes

- reproducibility;
- modular benchmark implementations;
- reusable benchmarking infrastructure;
- consistent reporting;
- automated visualization.

Each benchmark focuses on measuring a single stage of the statistical workflow, while shared functionality—including reporting, plotting, workspace handling, and command-line interfaces—is reused across the project.

This architecture simplifies maintenance and makes it straightforward to extend the benchmark suite without changing the underlying infrastructure.

---

# Related Documentation

Continue with

- **Getting Started**
- **Benchmark Methodology**
- **Benchmark Suite**
- **Benchmark Results**
- **Benchmark Workspaces**
