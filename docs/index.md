# PyHS3 Benchmarks

Welcome to the documentation for **PyHS3 Benchmarks**, a benchmarking and validation framework for the PyHS3 ecosystem.

The repository provides a reproducible environment for measuring the performance of statistical inference workflows built on HS3 workspaces. It combines stage-by-stage performance benchmarking, cross-framework numerical validation, automated reporting, and publication-quality visualizations in a single benchmarking suite.

PyHS3 Benchmarks is designed for both **development** and **research**, making it possible to

- profile every major stage of the statistical model lifecycle;
- identify computational bottlenecks;
- measure runtime and memory consumption;
- compare multiple statistical frameworks using equivalent benchmark inputs;
- validate numerical agreement between implementations;
- monitor performance regressions across repository revisions.

---

# Key Features

The benchmarking framework provides:

- **Workflow benchmarks** covering the complete HS3 model lifecycle, from workspace loading to compiled likelihood evaluation.
- **Cross-framework comparisons** between PyHS3 and other statistical inference frameworks using apples-to-apples benchmark methodology.
- **Automated benchmark execution** through a unified benchmark runner.
- **Structured JSON reports** suitable for downstream analysis and regression tracking.
- **Publication-quality plots** generated directly from benchmark results.
- **Scalable benchmark campaigns** supporting collections of benchmark workspaces with varying model complexity.

---

# Documentation Guide

The documentation is organized into several sections.

## Getting Started

Start here if you are new to the project.

Learn how to install the repository, execute your first benchmark, and understand the generated outputs.

Recommended reading:

- Getting Started
- Installation
- Repository Structure

---

## Benchmark Suite

The benchmark suite measures the complete statistical model lifecycle, including

- workspace loading;
- model creation;
- log-probability construction;
- graph optimization;
- compilation;
- compiled evaluation;
- PDF evaluation;
- NLL scans;
- memory and scalability analysis.

Each benchmark documents its methodology, execution procedure, generated outputs, and interpretation of the results.

---

## Cross-Framework Validation

Cross-framework benchmarks compare PyHS3 with other statistical inference frameworks while ensuring that identical statistical models, datasets, and benchmark configurations are used.

Current comparisons include

- PyHS3 vs RooFit;
- PyHS3 vs xRooFit;
- scalar PDF evaluation;
- ΔNLL scans.

---

## Benchmark Workspaces

The repository uses a common benchmark workspace collection across nearly all benchmark suites.

This section explains

- workspace naming conventions;
- benchmark workspace collections;
- model complexity;
- ROOT counterparts used for cross-framework validation.

---

## Results and Outputs

Benchmark execution produces

- structured JSON reports;
- benchmark summaries;
- publication-quality figures;
- overview reports for complete benchmark campaigns.

These outputs provide the basis for performance analysis, numerical validation, and long-term regression tracking.

---

## Development

Contributor documentation includes

- benchmark implementation guidelines;
- benchmark methodology;
- matrix runner architecture;
- repository organization;
- extension of existing benchmark suites.

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
      ├──────────────► Memory Scaling
      │
      └──────────────► Cross-Framework Benchmarks
```

Each benchmark isolates one stage of this workflow whenever possible, making it easier to identify performance bottlenecks and evaluate optimization strategies.

---

# Quick Start

Install the project:

```bash
pixi install
```

Run a workflow benchmark:

```bash
pixi run python -m src.run_workspace_loading \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --n-runs 10 \
    --plot
```

Continue with the **Getting Started** guide for a complete introduction to the benchmarking workflow.
