# PyHS3 Benchmarks

On this page, you will learn what **PyHS3 Benchmarks** provides, how the documentation is organized, and how to run your first benchmark.

**PyHS3 Benchmarks** is a benchmarking and validation framework for the PyHS3 ecosystem. It provides a reproducible environment for evaluating statistical inference workflows built on HS3 workspaces through workflow benchmarks, cross-framework validation, automated reporting, and publication-quality visualizations.

The project helps you

-   identify computational bottlenecks;
-   measure runtime and memory consumption;
-   compare statistical frameworks using equivalent benchmark inputs;
-   validate numerical agreement between implementations;
-   monitor performance regressions across repository revisions.

For details on how benchmarks are executed and measured, see **Benchmark Methodology**.

---

# Key Features

PyHS3 Benchmarks provides

-   **Workflow benchmarks** covering the HS3 model lifecycle, from workspace loading to compiled likelihood evaluation.
-   **Cross-framework comparisons** between PyHS3 and other statistical inference frameworks.
-   **Automated benchmark execution** through a shared benchmark runner.
-   **Structured JSON reports** for downstream analysis and regression tracking.
-   **Publication-quality plots** generated directly from benchmark results.
-   **Scalable benchmark campaigns** using benchmark workspaces with varying model complexity.

---

# Documentation Guide

## Getting Started

Start here if you are new to the project.

This section explains how to install the project, run your first benchmark, and understand the repository layout.

Recommended reading:

-   Getting Started
-   Installation
-   Repository Structure

---

## Benchmark Suite

Browse the benchmark documentation for individual workflow stages, including

-   workspace loading;
-   model creation;
-   log-probability construction;
-   graph optimization;
-   compilation;
-   compiled evaluation;
-   PDF evaluation;
-   NLL scans;
-   memory and scalability analysis.

Each page focuses on a single benchmark, while the shared methodology is documented separately.

---

## Cross-Framework Validation

Cross-framework benchmarks compare equivalent statistical computations across supported frameworks.

Current comparisons include

-   PyHS3 vs RooFit
-   scalar PDF evaluation;
-   ΔNLL scans.

---

## Benchmark Workspaces

This section documents the benchmark workspaces used throughout the repository, including workspace collections, naming conventions, model complexity, and ROOT counterparts used for cross-framework validation.

---

## Results and Outputs

Benchmark execution produces

-   structured JSON reports;
-   benchmark summaries;
-   publication-quality figures;
-   overview reports for complete benchmark campaigns.

See the **Outputs** page for details on generated artifacts.

---

## Development

Contributor documentation covers

-   benchmark implementation;
-   repository organization;
-   benchmark matrix runner;
-   testing;
-   profiling.

---

# Benchmark Workflow

The benchmarking pipeline follows the complete lifecycle of an HS3 statistical model.

``` text
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

Each benchmark focuses on one stage of this workflow. The complete measurement strategy is described in **Benchmark Methodology**.

---

# Quick Start

Install the project:

``` bash
pixi install
```

Run your first benchmark:

``` bash
pixi run python -m src.run_workspace_loading \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
    --n-runs 10 \
    --plot
```

Continue with **Getting Started** for installation details, benchmark configuration, and additional examples.
