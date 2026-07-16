# Benchmark Methodology

On this page, you will learn how benchmarks in **PyHS3 Benchmarks** are executed, measured, validated, and reported. Although benchmark suites target different stages of the statistical inference workflow, they all follow the same methodology to ensure reproducible and comparable results.

---

## Goals

The benchmark suite is designed to

-   identify computational bottlenecks throughout the statistical workflow;
-   evaluate the impact of implementation changes and optimizations;
-   measure runtime and memory consumption;
-   validate numerical agreement between equivalent implementations;
-   monitor performance regressions across repository revisions.

Each benchmark measures one well-defined aspect of the workflow while sharing a common execution and reporting infrastructure.

---

## Design Principles

Every benchmark in this repository follows the same guiding principles.

### Isolation

Each benchmark isolates a single stage of the statistical workflow, making it easier to identify bottlenecks and evaluate optimization strategies.

For example,

-   **Workspace Loading** measures only workspace deserialization;
-   **Model Creation** measures model construction independently of loading;
-   **Compiled Evaluation** measures numerical execution independently of compilation;
-   **Memory Scaling** isolates memory consumption during selected workflow stages.

### Reproducibility

Benchmark inputs are deterministic. Fixed workspaces, explicit benchmark configurations, and identical inputs across repeated runs ensure that differences primarily reflect implementation changes rather than variations in benchmark data.

### Automation

All benchmark suites run through the shared benchmark runner, which executes benchmarks, collects statistics, generates reports, and produces publication-quality figures using a consistent workflow.

### Extensibility

Benchmarks share the same execution model and reporting conventions. Adding a new benchmark typically requires implementing the benchmark,
registering it with the benchmark runner, and documenting its methodology.

---

## Benchmark Lifecycle

Every benchmark follows the same high-level execution pipeline.

``` text
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

## Measurement Strategy

Performance measurements are collected using a consistent execution strategy.

### Warm-up

Some benchmark suites perform one or more warm-up iterations before measurements begin. Warm-up executions are excluded from the reported
statistics to remove one-time initialization costs such as JIT compilation, graph initialization, cache population, and initial memory allocation.

### Repeated Measurements

Timing benchmarks are executed multiple times to reduce measurement noise caused by operating system scheduling, temporary resource contention, cache effects, and background processes.

### Aggregation

Repeated measurements are summarized using statistics appropriate for the benchmark, including:

-   mean execution time;
-   standard deviation;
-   throughput;
-   peak memory usage;
-   current memory usage.

---

## Performance Metrics

Different benchmark suites emphasize different performance characteristics.

| Metric | Description |
|--------|-------------|
| Wall time | Total benchmark execution time |
| Mean execution time | Average over repeated executions |
| Throughput | Number of evaluations completed per unit time |
| Current RSS delta | Increase in resident memory during execution |
| Peak RSS delta | Maximum additional resident memory used |
| Validation status | Numerical agreement with reference implementation |

Individual benchmark pages describe any benchmark-specific metrics they report.

---

## Numerical Validation

Several benchmark suites measure both performance and numerical correctness. Depending on the benchmark, validation may compare:

-   PDF values;
-   negative log-likelihood values;
-   likelihood scan shapes;
-   compiled and interpreted execution;
-   reference framework outputs.

These checks ensure that performance optimizations preserve the underlying statistical model.

---

## Cross-Framework Comparisons

Cross-framework benchmarks compare equivalent statistical computations rather than different analysis workflows. Whenever possible, they use identical statistical models, datasets, parameter values, and benchmark configurations so that observed performance differences reflect implementation characteristics rather than benchmark inputs.

Framework-specific assumptions or limitations are documented on the corresponding benchmark pages.

---

## Benchmark Outputs

Each benchmark produces a structured JSON report containing benchmark metadata, execution statistics, benchmark configuration, and validation results where applicable. Most benchmark suites also generate publication-quality figures from these reports. See the **Outputs** page for details.

---

## Reproducibility Recommendations

For meaningful comparisons across benchmark campaigns, it is recommended to

-   use identical benchmark workspaces;
-   execute benchmarks with the same benchmark configuration;
-   compare results produced by equivalent software environments;
-   avoid unnecessary background workload during benchmark execution;
-   compare optimized builds with optimized builds and debug builds with
    debug builds.

---

## Related Documentation

For additional information, see

-   **Benchmark Runner** for the shared execution infrastructure.
-   **Outputs** for generated reports and figures.
-   **Workspace Lifecycle** for the overall benchmarking pipeline.
-   **Cross-Framework** for framework-specific comparison methodology.
