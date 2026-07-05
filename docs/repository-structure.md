# Repository Structure

This document describes the overall organization of the **PyHS3 Benchmarks** repository and explains the purpose of each major component.

The repository is organized around a simple workflow:

- **inputs** provide benchmark data;
- **src** implements the benchmark suite;
- **results** store benchmark outputs;
- **docs** contains documentation and generated figures;
- **tests** validate benchmark correctness.

This separation keeps the project modular, reproducible, and easy to extend.

---

# Repository Layout

```text
pyhs3-benchmarks/
│
├── docs/
├── inputs/
├── results/
├── src/
├── tests/
│
├── pyproject.toml
├── pixi.toml
└── README.md
```

Each directory is described below.

---

# Source Code

```text
src/
```

The `src` directory contains the implementation of the benchmarking framework.

It includes

- workflow benchmarks;
- cross-framework benchmarks;
- benchmark runner;
- plotting utilities;
- workspace generators;
- shared benchmarking utilities.

Most users interact with these modules through their command-line interfaces rather than importing them directly.

---

# Benchmark Inputs

```text
inputs/
```

The `inputs/` directory contains all benchmark datasets used throughout the repository.

These include

- benchmark HS3 workspaces;
- synthetic scalar PDF workspaces;
- synthetic binned likelihood models;
- ROOT workspaces used for xRooFit comparisons.

Most workflow benchmarks operate on the benchmark workspace collection described in the **Benchmark Workspaces** documentation.

---

# Benchmark Results

```text
results/
```

Benchmark execution produces structured JSON reports.

A typical directory structure is

```text
results/

    workspace_loading/

        workspace_loading_result.json

    model_creation/

        model_creation_result.json

    ...

    matrix_summary.json
```

The JSON reports are the primary machine-readable outputs of the benchmarking framework and serve as the basis for visualization and automated analysis.

---

# Documentation

```text
docs/
```

The documentation contains

- installation guides;
- benchmarking methodology;
- workflow benchmark documentation;
- cross-framework benchmark documentation;
- workspace documentation;
- generated benchmark figures.

Generated plots used by the documentation are stored in

```text
docs/assets/plots/
```

Keeping figures alongside the documentation simplifies publishing and maintenance.

---

# Tests

```text
tests/
```

The `tests/` directory contains unit and integration tests covering

- benchmark implementations;
- workspace generation;
- shared utilities;
- validation logic.

Where possible, benchmark results are compared against reference implementations to ensure numerical correctness.

---

# Typical Benchmark Workflow

A typical benchmark session follows the workflow below.

```text
Benchmark Workspace
        │
        ▼
Benchmark
        │
        ▼
JSON Report
        │
        ▼
Plot Generation
        │
        ▼
Documentation
```

This workflow is identical whether benchmarks are executed individually or through the benchmark runner.

---

# Benchmark Runner

The repository provides a unified benchmark runner capable of executing one or more benchmark suites across multiple workspaces.

Typical responsibilities include

- executing benchmark suites;
- coordinating repeated measurements;
- collecting benchmark reports;
- generating matrix summaries;
- producing comparison plots.

Using a common runner ensures that all benchmark suites follow the same execution model and reporting conventions.

---

# Design Philosophy

The repository follows a modular design.

Each benchmark focuses on measuring a single stage of the statistical workflow, while common functionality—such as reporting, plotting, workspace handling, and command-line interfaces—is shared across the project.

This architecture simplifies maintenance, encourages code reuse, and makes it straightforward to introduce new benchmark suites without modifying the existing infrastructure.

---

# Related Documentation

See also

- **Getting Started**
- **Benchmark Suite**
- **Benchmark Methodology**
- **Benchmark Results**
- **Benchmark Workspaces**
