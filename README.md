# PyHS3 Benchmarks

A dedicated benchmarking and validation repository for PyHS3.

This repository contains benchmarking infrastructure developed as part of the PyHS3 benchmarking and optimization effort. It is used to evaluate the performance, scalability, and numerical correctness of PyHS3 across a variety of workflows, workspace configurations, and validation scenarios.

The repository is maintained separately from the main PyHS3 codebase to allow independent development of benchmarking, validation, profiling, and performance-tracking tools.

## Goals

The primary goals of this repository are:

* measure PyHS3 runtime and memory usage
* benchmark different stages of the PyHS3 workflow
* study scaling behavior with increasing model complexity
* validate numerical agreement across implementations
* compare PyHS3 against other statistical frameworks
* support profiling and optimization studies
* track performance changes across PyHS3 versions

## Benchmark Categories

### Core PyHS3 Benchmarks

Benchmarks covering the main PyHS3 workflow:

* workspace loading
* model creation
* log-probability construction
* graph canonicalization
* graph optimization
* graph compilation
* compiled evaluation
* NLL scans

### Scaling Benchmarks

Benchmarks studying performance trends as problem size increases:

* memory scaling
* model complexity scaling
* workspace size scaling

### Cross-Framework Benchmarks

Comparisons between PyHS3 and external statistical frameworks:

* RooFit
* pyhf
* zfit
* numba-stats

### Validation Benchmarks

Numerical validation studies designed to detect regressions and verify agreement between implementations.

### Optimization Benchmarks

Before/after comparisons used to quantify the impact of performance improvements and code optimizations.

## Repository Structure

```text
benchmarking/
├── inputs/      # benchmark inputs and generated workspaces
├── scripts/     # benchmark and validation scripts
├── results/     # raw benchmark outputs
├── plots/       # generated figures
└── reports/     # benchmark summaries and reports
```

## Development Status

This repository is under active development.

The current focus is on building a comprehensive benchmark suite that will support future profiling, optimization, validation, and performance-tracking efforts for PyHS3.
