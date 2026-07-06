# PyHS3 Benchmarks

Welcome to the documentation for **PyHS3 Benchmarks**, the benchmarking and validation suite for the PyHS3 ecosystem.

This project provides a reproducible framework for evaluating every major stage of the statistical inference workflow, from loading serialized HS3 workspaces to executing compiled likelihood evaluations. In addition to profiling PyHS3 itself, the repository includes cross-framework comparisons with RooFit, xRooFit, and pyhf to validate numerical consistency and measure relative performance.

The benchmarking suite is designed for both **development** and **research**:

- evaluate performance of individual workflow stages;
- identify computational bottlenecks;
- measure memory usage and scalability;
- compare different statistical frameworks;
- validate numerical agreement between implementations;
- detect performance regressions over time.

---

# Documentation

The documentation is organized by topic.

## Getting Started

If you are new to the project, begin here.

- Getting Started
- Installation
- Repository Structure

---

## Benchmarking

Documentation for the PyHS3 workflow benchmarks.

These benchmarks measure the cost of each major stage of the statistical model lifecycle, including workspace loading, model creation, graph optimization, compilation, likelihood evaluation, and scaling.

Topics include:

- Workspace Loading
- Model Creation
- Log-Probability Construction
- Log-Probability Compilation
- Graph Canonicalization
- Graph Optimization
- Compiled Evaluation
- PDF Evaluation
- NLL Scan
- Memory Scaling
- Model Complexity Scaling

---

## Cross-Framework Comparisons

Cross-framework benchmarks evaluate both numerical agreement and runtime performance across multiple statistical toolkits.

Current comparisons include:

- PyHS3 vs RooFit
- PyHS3 vs xRooFit
- Scalar PDF Evaluation
- Vectorized PDF Evaluation
- Binned Likelihood Evaluation
- NLL Scan
- Model Complexity Scaling

---

## Workspaces

This section describes the benchmark inputs used throughout the project.

Topics include:

- Benchmark workspaces
- Synthetic scalar PDF workspaces
- Synthetic binned likelihood models
- ROOT workspaces
- Workspace generation

---

## Outputs

Benchmark execution produces structured JSON reports together with publication-quality plots.

The documentation explains:

- benchmark outputs;
- directory layout;
- generated figures;
- summary reports;
- benchmark overview plots.

---

## Development

Information for contributors.

Topics include:

- adding new benchmarks;
- registering benchmarks;
- extending the matrix runner;
- benchmarking conventions;
- validation guidelines.

---

# Benchmark Workflow

The benchmarking pipeline follows the complete lifecycle of an HS3 statistical model.

```text
HS3 Workspace
      │
      ▼
Workspace Loading
      │
      ▼
Model Creation
      │
      ▼
Log-Probability Construction
      │
      ▼
Graph Canonicalization
      │
      ▼
Graph Optimization
      │
      ▼
Compilation
      │
      ▼
Compiled Evaluation
      │
      ├──────────────► PDF Evaluation
      │
      ├──────────────► NLL Scan
      │
      ├──────────────► Memory Profiling
      │
      └──────────────► Cross-Framework Benchmarks
```

---

# Quick Start

Install the project:

```bash
pixi install
```

Run a benchmark:

```bash
pixi run python -m src.run_all_workspace_benchmark_matrix \
    --workspace-dir inputs/all_workspaces \
    --benchmarks workspace_loading
```

Generate the benchmark overview:

```bash
pixi run python -m src.plot_benchmark_overview \
    --results-dir results/all_full_matrix \
    --plot-dir plots/all_full_matrix/benchmark_overview
```

For detailed instructions, continue with the Getting Started guide.
