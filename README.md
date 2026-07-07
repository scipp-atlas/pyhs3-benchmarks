# PyHS3 Benchmarks

[![Documentation](https://img.shields.io/badge/docs-Zensical-blue)](#documentation)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Pixi](https://img.shields.io/badge/environment-pixi-8A2BE2)](https://pixi.sh/)

> Benchmarking, validation, and profiling suite for PyHS3.

PyHS3 Benchmarks provides a reproducible benchmarking framework for evaluating the performance of the PyHS3 statistical inference library.

The repository covers the complete statistical inference workflow—from workspace loading and model construction to compiled likelihood evaluation—and includes cross-framework benchmarks comparing PyHS3 with ROOT-based statistical frameworks using statistically equivalent benchmark workspaces.

The project is designed to support

- performance analysis;
- optimization;
- numerical validation;
- scalability studies;
- regression tracking.

---

# Documentation

Comprehensive project documentation is provided through the **Zensical** documentation site included in this repository.

Clone the repository

```bash
git clone https://github.com/scipp-atlas/pyhs3-benchmarks.git

cd pyhs3-benchmarks
```

Install the project environment

```bash
pixi install
```

Start the documentation server

```bash
pixi run zensical serve
```

After the server starts, open the local URL displayed in the terminal (typically `http://localhost:3000`).

The documentation includes

- installation and setup;
- repository overview;
- benchmark methodology;
- benchmark workflows;
- benchmark workspace design;
- workflow benchmarks;
- cross-framework benchmarks;
- development guide;
- API overview.

---

# Repository Overview

```text
docs/          Zensical documentation
src/           Benchmark implementations
inputs/        Benchmark workspaces
results/       Benchmark reports
plots/         Plot generation utilities
tests/         Automated tests
```

---

# Benchmark Dataset

Benchmark inputs are generated automatically using the companion repository

https://github.com/scipp-atlas/workspace-scripts

The generation process produces paired **HS3** and **ROOT** workspaces from identical statistical models. These workspaces provide reproducible benchmark inputs and enable apples-to-apples comparisons across statistical frameworks.

---

# Quick Start

Run an individual benchmark

```bash
pixi run python -m src.run_workspace_loading
```

Run the complete benchmark suite

```bash
pixi run python -m src.run_all_benchmarks
```

---

# Project Goals

The repository aims to

- benchmark every major stage of the PyHS3 inference pipeline;
- evaluate runtime and memory performance;
- validate numerical agreement across statistical frameworks;
- support optimization and regression tracking;
- provide reproducible benchmark datasets and publication-quality figures.
