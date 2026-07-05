# Benchmark Methodology

This document describes the benchmarking methodology used throughout the PyHS3 Benchmarks project.

Although individual benchmarks measure different stages of the statistical inference workflow, they all follow the same execution model and reporting conventions. Using a unified methodology ensures that benchmark results are reproducible, comparable, and suitable for long-term performance tracking.

---

# Design Principles

Every benchmark in this repository follows four guiding principles.

## Isolation

Each benchmark measures a single stage of the statistical workflow whenever possible.

For example:

- Workspace Loading measures only workspace deserialization.
- Model Creation measures model construction independently of workspace loading.
- Compiled Evaluation measures numerical execution independently of compilation.
- Memory Scaling isolates memory consumption during selected workflow stages.

Isolating benchmark stages makes it significantly easier to identify performance bottlenecks.

---

## Reproducibility

All benchmark inputs are deterministic.

Workspace generators use fixed random seeds where applicable, ensuring that identical benchmark configurations always produce identical benchmark inputs.

As a result, performance differences between benchmark runs are primarily caused by implementation changes rather than input variability.

---

## Automation

All benchmark suites are designed to execute automatically.

The repository provides a unified matrix runner capable of

- discovering benchmark workspaces;
- executing benchmark suites;
- collecting benchmark outputs;
- organizing benchmark results;
- generating benchmark summaries.

This allows complete benchmark campaigns to be executed using a single command.

---

## Extensibility

Benchmark implementations share a common execution model.

Adding a new benchmark typically requires only

- implementing the benchmark itself;
- registering it in the matrix runner;
- documenting the benchmark.

Existing reporting and plotting infrastructure can then be reused without modification.

---

# Benchmark Lifecycle

Every benchmark follows the same high-level execution pipeline.

```text
Input Workspace
        │
        ▼
Load Benchmark Input
        │
        ▼
Prepare Benchmark State
        │
        ▼
Warm-up (optional)
        │
        ▼
Repeated Measurements
        │
        ▼
Result Aggregation
        │
        ▼
JSON Report
        │
        ▼
Plot Generation
```

Each stage is described below.

---

# Warm-up

Some benchmarks perform one or more warm-up iterations before collecting measurements.

Warm-up executions are not included in the reported statistics.

Their purpose is to eliminate one-time initialization costs such as

- JIT compilation;
- graph initialization;
- memory allocation;
- cache population.

This produces more representative timing measurements for repeated execution.

---

# Repeated Measurements

Timing benchmarks are executed multiple times.

Repeated execution reduces measurement noise introduced by

- operating system scheduling;
- background processes;
- cache effects;
- temporary resource contention.

Most timing benchmarks therefore report aggregated statistics rather than a single execution time.

---

# Numerical Validation

Several benchmark suites compare multiple statistical frameworks.

These comparisons measure both

- execution performance;
- numerical agreement.

Validation ensures that benchmark optimizations do not change the underlying statistical model.

Depending on the benchmark, validation may compare

- PDF values;
- negative log-likelihood values;
- likelihood scan shapes;
- compiled versus interpreted execution.

---

# Benchmark Outputs

Each benchmark produces a structured JSON report containing

- benchmark metadata;
- benchmark configuration;
- execution status;
- timing statistics;
- memory statistics (when applicable);
- validation results (when applicable).

JSON outputs provide a stable interface for downstream visualization and automated analysis.

---

# Plot Generation

Most benchmark suites optionally generate publication-quality figures.

Typical visualizations include

- execution time comparisons;
- scaling behaviour;
- memory consumption;
- framework comparisons;
- benchmark summaries.

Plots are generated from benchmark outputs and therefore can be recreated without rerunning the benchmark itself.

---

# Matrix Benchmarking

Large benchmark campaigns are executed using the matrix runner.

Rather than running a benchmark on a single workspace, the matrix runner executes the same benchmark over an entire collection of workspaces while preserving a consistent directory structure for all outputs.

This approach enables systematic evaluation across models of different complexity without requiring manual orchestration.

---

# Performance Metrics

Different benchmark suites report different performance metrics.

Common metrics include

| Metric | Description |
|---------|-------------|
| Wall time | Total execution time measured by the benchmark |
| Mean execution time | Average over repeated runs |
| Standard deviation | Variability between repeated executions |
| Peak memory | Maximum resident memory usage |
| Validation status | Numerical agreement across implementations |

Each benchmark page documents any additional benchmark-specific metrics.

---

# Reproducibility Recommendations

For consistent benchmark results, it is recommended to

- use identical benchmark inputs;
- execute benchmarks on an otherwise idle system;
- avoid mixing debug and optimized builds;
- compare results generated with identical benchmark configurations.

Following these recommendations minimizes measurement variability and improves comparability between benchmark campaigns.
