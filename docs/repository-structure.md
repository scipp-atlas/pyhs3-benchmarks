# Repository Structure

This document describes the overall organization of the **PyHS3 Benchmarks** repository and explains the role of each major component.

The project is organized around a simple principle:

- **inputs** define benchmark data;
- **benchmarks** perform measurements;
- **results** store benchmark outputs;
- **plots** visualize benchmark results;
- **documentation** describes the benchmarking framework.

---

# Repository Layout

```text
pyhs3-benchmarks/
│
├── docs/
├── src/
├── inputs/
├── results/
├── plots/
├── profiling/
├── tests/
│
├── README.md
├── pyproject.toml
└── pixi.toml
```

Each directory serves a specific purpose and is described below.

---

# Source Code

```text
src/
```

The `src` directory contains the benchmarking implementation.

It includes

- workflow benchmarks;
- cross-framework benchmarks;
- plotting utilities;
- workspace generators;
- shared benchmarking utilities.

Most users interact with the repository exclusively through the command-line interfaces provided by these modules.

---

# Benchmark Inputs

```text
inputs/
```

The `inputs` directory stores benchmark datasets and statistical workspaces.

Depending on the benchmark, these include

- Alexx benchmark workspaces;
- synthetic scalar PDF workspaces;
- synthetic binned likelihood models;
- ROOT workspaces used for xRooFit comparisons.

Each benchmark documents the expected workspace format separately.

---

# Benchmark Results

```text
results/
```

Every benchmark produces structured JSON output.

Results generated with the matrix runner follow the directory structure

```text
results/

    <benchmark>/

        <workspace>/

            repeat_000/

                <benchmark>_result.json
```

This layout makes it possible to benchmark large collections of workspaces while keeping individual runs independent and reproducible.

The generated JSON files serve as the primary input for reporting and visualization.

---

# Plots

```text
plots/
```

Generated figures are stored separately from numerical benchmark results.

Typical outputs include

- execution time comparisons;
- scaling plots;
- memory usage;
- cross-framework comparisons;
- benchmark overview reports.

Separating plots from benchmark results allows figures to be regenerated without rerunning the benchmarks.

---

# Documentation

```text
docs/
```

The documentation is organized into topic-oriented sections.

Major categories include

- getting started;
- benchmark methodology;
- workflow benchmarks;
- cross-framework benchmarks;
- workspaces;
- outputs;
- development.

This organization keeps individual pages focused while avoiding an oversized README.

---

# Profiling

```text
profiling/
```

The profiling directory contains utilities and experiments used to investigate performance bottlenecks beyond the standard benchmark suite.

These tools are primarily intended for development and optimization.

---

# Tests

```text
tests/
```

Unit and integration tests validate benchmark correctness, workspace generation, and supporting utilities.

Where possible, benchmarks are validated against reference implementations to ensure numerical consistency.

---

# Typical Benchmark Workflow

A typical benchmark session follows the workflow below.

```text
Input Workspace
        │
        ▼
Benchmark
        │
        ▼
JSON Result
        │
        ▼
Overview Builder
        │
        ▼
Plots
        │
        ▼
Documentation
```

The same workflow is used for both individual benchmark executions and large-scale matrix runs.

---

# Matrix Runner

The repository includes a unified matrix runner capable of executing benchmark suites over collections of workspaces.

Typical responsibilities include

- discovering input workspaces;
- executing benchmark suites;
- collecting benchmark outputs;
- handling repeated runs;
- generating benchmark summaries.

Using a single runner ensures that all benchmarks follow the same execution model and produce results with a consistent directory structure.

---

# Design Philosophy

The repository is intentionally modular.

Each benchmark is responsible only for measuring one aspect of the statistical workflow.

Common functionality—including workspace discovery, benchmarking utilities, plotting infrastructure, and reporting—is shared across the project.

This separation simplifies maintenance, encourages reuse, and makes it straightforward to introduce new benchmark suites without modifying the existing architecture.
