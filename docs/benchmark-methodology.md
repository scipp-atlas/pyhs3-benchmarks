# Benchmark Methodology

This document describes the methodology used throughout the **PyHS3 Benchmarks** project.

Although individual benchmark suites evaluate different stages of the statistical inference workflow, they all follow the same measurement strategy, reporting conventions, and validation principles. A unified methodology ensures that benchmark results are reproducible, directly comparable, and suitable for long-term performance analysis.

---

# Goals

The benchmark suite is designed to answer several complementary questions about the performance of PyHS3.

Rather than identifying a single "fastest" implementation, the benchmarks are intended to

- identify computational bottlenecks throughout the statistical workflow;
- evaluate the impact of implementation changes and optimizations;
- measure runtime and memory consumption;
- validate numerical agreement between equivalent implementations;
- monitor performance regressions across repository revisions.

Each benchmark therefore measures one well-defined aspect of the workflow while sharing a common execution and reporting infrastructure.

---

# Design Principles

Every benchmark in this repository follows the same guiding principles.

## Isolation

Each benchmark measures a single stage of the statistical workflow whenever possible.

For example,

- **Workspace Loading** measures only workspace deserialization;
- **Model Creation** measures model construction independently of loading;
- **Compiled Evaluation** measures numerical execution independently of compilation;
- **Memory Scaling** isolates memory consumption during selected workflow stages.

Separating workflow stages makes it easier to identify performance bottlenecks and evaluate optimization strategies without introducing unrelated sources of variability.

---

## Reproducibility

Benchmark inputs are deterministic.

Benchmark workspaces are fixed, benchmark configurations are explicitly defined, and repeated executions use identical inputs.

As a result, differences between benchmark runs primarily reflect implementation changes rather than variations in benchmark data.

---

## Automation

All benchmark suites are designed to execute automatically.

The shared benchmark runner is responsible for

- executing benchmark suites;
- organizing benchmark outputs;
- collecting benchmark statistics;
- generating reports;
- producing publication-quality figures.

This enables complete benchmark campaigns to be executed using a consistent workflow.

---

## Extensibility

Every benchmark follows the same execution model and reporting conventions.

Adding a new benchmark typically requires only

- implementing the benchmark;
- registering it with the benchmark runner;
- documenting its methodology.

Existing reporting, plotting, and output infrastructure can then be reused without modification.

---

# Benchmark Lifecycle

Every benchmark follows the same high-level execution pipeline.

```text
Benchmark Workspace
        │
        ▼
Load Benchmark Input
        │
        ▼
Prepare Benchmark State
        │
        ▼
Optional Warm-up
        │
        ▼
Repeated Measurements
        │
        ▼
Aggregate Results
        │
        ▼
Generate JSON Report
        │
        ▼
Generate Plots
```

Although individual benchmark suites may omit or extend certain stages, this overall workflow is shared throughout the repository.

---

# Measurement Strategy

Performance measurements are collected using a consistent execution strategy.

## Warm-up

Some benchmark suites perform one or more warm-up iterations before measurements begin.

Warm-up executions are excluded from the reported statistics.

Their purpose is to remove one-time initialization costs such as

- JIT compilation;
- graph initialization;
- cache population;
- initial memory allocation.

This produces measurements that more accurately represent repeated execution.

---

## Repeated Measurements

Timing benchmarks are executed multiple times.

Repeated execution reduces measurement noise introduced by

- operating system scheduling;
- temporary resource contention;
- cache effects;
- background processes.

Reported timing values therefore represent aggregated statistics rather than a single execution.

---

## Aggregation

Benchmark outputs summarize repeated executions using aggregate statistics appropriate for the measured quantity.

Depending on the benchmark, reported values may include

- mean execution time;
- standard deviation;
- throughput;
- peak memory usage;
- current memory usage.

---

# Performance Metrics

Different benchmark suites emphasize different performance characteristics.

Commonly reported metrics include

| Metric | Description |
|---------|-------------|
| Wall time | Total benchmark execution time |
| Mean execution time | Average over repeated executions |
| Throughput | Number of evaluations completed per unit time |
| Current RSS delta | Increase in resident memory during execution |
| Peak RSS delta | Maximum additional resident memory used |
| Validation status | Numerical agreement with reference implementation |

Individual benchmark pages describe any benchmark-specific metrics that they report.

---

# Numerical Validation

Several benchmark suites evaluate not only performance but also numerical correctness.

Depending on the benchmark, validation may compare

- PDF values;
- negative log-likelihood values;
- likelihood scan shapes;
- compiled and interpreted execution;
- reference framework outputs.

Numerical validation ensures that performance optimizations preserve the underlying statistical model and do not alter computational results.

---

# Cross-Framework Comparisons

Cross-framework benchmarks are designed to compare equivalent statistical computations rather than different analysis workflows.

Whenever possible, comparisons use

- identical statistical models;
- identical datasets;
- identical parameter values;
- equivalent benchmark configurations.

This apples-to-apples methodology ensures that observed performance differences reflect implementation characteristics rather than differences in benchmark inputs.

Some statistical frameworks expose different APIs or computational models. Where an exact one-to-one comparison is not possible, the corresponding benchmark documentation explicitly describes the assumptions and limitations of the comparison.

---

# Benchmark Outputs

Each benchmark produces a structured JSON report containing benchmark metadata, execution statistics, benchmark configuration, and validation results where applicable.

These reports provide a stable interface for

- visualization;
- regression tracking;
- automated analysis;
- downstream reporting.

Most benchmark suites also generate publication-quality figures directly from the JSON outputs, allowing plots to be regenerated without repeating benchmark execution.

---

# Reproducibility Recommendations

For meaningful comparisons across benchmark campaigns, it is recommended to

- use identical benchmark workspaces;
- execute benchmarks with the same benchmark configuration;
- compare results produced by equivalent software environments;
- avoid unnecessary background workload during benchmark execution;
- compare optimized builds with optimized builds and debug builds with debug builds.

Following these recommendations minimizes measurement variability and improves long-term comparability of benchmark results.

---

# Related Documentation

For additional information, see

- **Benchmark Runner** for the shared execution infrastructure.
- **Benchmark Results** for the generated reports and figures.
- **Benchmark Workflow** for the overall benchmarking pipeline.
- **Cross-Framework Validation** for framework-specific comparison methodology.
