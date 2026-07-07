# Benchmark Results

Every benchmark executed within the PyHS3 Benchmarks repository produces a collection of structured results.

These results provide considerably more than simple timing measurements. They capture benchmark configuration, execution statistics, memory usage, validation status, and benchmark-specific metrics, forming the basis for performance analysis, regression tracking, and cross-framework comparisons.

Because every benchmark follows the same reporting conventions, results from different benchmark suites can be interpreted consistently and compared throughout the repository.

---

# Overview

A benchmark execution produces information describing

- what was benchmarked;
- how the benchmark was executed;
- whether execution completed successfully;
- what performance characteristics were observed;
- whether the produced results passed validation.

Although individual benchmark suites measure different quantities, the overall structure of benchmark results remains consistent.

---

# Result Hierarchy

Benchmark results are naturally organized into several levels.

```text
Benchmark Campaign
        │
        ▼
Benchmark Suite
        │
        ▼
Workspace Result
        │
        ▼
Performance Metrics
        │
        ▼
Validation Results
```

Each level adds additional context while preserving a common reporting structure.

---

# Benchmark Metadata

Every benchmark result contains descriptive information that identifies the benchmark execution.

Typical metadata includes

- benchmark name;
- benchmark workspace;
- execution status;
- benchmark configuration;
- execution timestamp (when available).

This metadata ensures that benchmark results remain traceable and reproducible.

---

# Performance Measurements

The primary purpose of benchmark execution is to measure performance.

Depending on the benchmark suite, reported measurements may include

- wall time;
- average execution time;
- throughput;
- current RSS memory increase;
- peak RSS memory increase;
- benchmark-specific performance metrics.

The exact quantities depend on the workflow stage being measured.

For example,

- **Workspace Loading** reports deserialization time and memory usage.
- **Compiled Evaluation** emphasizes execution throughput.
- **Memory Scaling** focuses on memory consumption.
- **Model Complexity Scaling** evaluates performance trends across increasingly complex statistical models.

---

# Validation Results

Performance measurements are meaningful only when benchmark execution produces correct results.

For this reason, benchmark suites perform validation before recording benchmark statistics.

Validation may include

- successful benchmark execution;
- finite numerical values;
- successful graph construction;
- successful compilation;
- numerical agreement with reference implementations.

Benchmark-specific validation procedures are documented on the corresponding benchmark pages.

---

# Aggregated Statistics

Individual benchmark executions are rarely interpreted in isolation.

Instead, benchmark suites summarize repeated executions using aggregate statistics such as

- mean;
- median;
- standard deviation;
- throughput;
- peak memory usage.

Aggregated statistics reduce measurement noise while providing a more representative view of benchmark performance.

The methodology used to compute these statistics is described in **Benchmark Methodology**.

---

# Comparing Results

One of the primary goals of the repository is to compare benchmark results across

- different benchmark workspaces;
- different workflow stages;
- different repository revisions;
- different statistical frameworks.

Meaningful comparisons require

- identical benchmark inputs;
- equivalent benchmark configuration;
- reproducible execution environments.

Following these principles ensures that observed differences reflect implementation changes rather than variations in benchmark setup.

---

# Regression Tracking

Benchmark results provide a foundation for long-term performance monitoring.

Comparing results across repository revisions makes it possible to

- identify performance regressions;
- evaluate optimization strategies;
- verify expected performance improvements;
- detect unexpected changes in runtime or memory consumption.

Because benchmark reports follow a common structure, these comparisons can be automated.

---

# Cross-Framework Validation

Several benchmark suites compare PyHS3 against external statistical frameworks.

In these cases, benchmark results include both

- performance measurements;
- numerical agreement.

This combination makes it possible to evaluate not only execution speed but also whether different implementations produce equivalent statistical results.

Where exact one-to-one comparisons are not possible, benchmark-specific documentation describes the corresponding limitations.

---

# Interpreting Benchmark Results

Benchmark results should always be interpreted within the context of the measured workflow stage.

For example,

- longer execution times do not necessarily indicate poorer overall performance if they correspond to one-time setup operations;
- increased memory consumption may be expected for larger statistical models;
- throughput measurements should be interpreted separately from initialization costs.

Individual benchmark pages provide detailed guidance for interpreting their benchmark-specific metrics.

---

# Relationship to Benchmark Outputs

Benchmark results describe the information produced by benchmark execution.

The physical artifacts generated by the repository—including JSON reports, summary files, and publication-quality figures—are documented separately in **Outputs**.

Keeping benchmark results conceptually separate from generated files allows the repository to distinguish between measured performance and the artifacts used to store or visualize it.

---

# Related Documentation

See also

- **Benchmark Methodology** for the measurement strategy used throughout the repository.
- **Benchmark Workflow** for the benchmark execution process.
- **Benchmark Runner** for the infrastructure coordinating benchmark campaigns.
- **Outputs** for generated reports, figures, and benchmark artifacts.
- **Cross-Framework Validation** for framework-specific benchmark interpretation.
