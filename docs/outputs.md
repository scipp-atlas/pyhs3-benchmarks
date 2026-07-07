# Benchmark Results

Every benchmark execution produces a collection of structured artifacts that document the outcome of the benchmark.

These artifacts support several purposes throughout the repository:

- performance analysis;
- numerical validation;
- visualization;
- regression tracking;
- automated reporting.

Because all benchmark suites follow the same reporting conventions, benchmark results remain consistent and comparable across the repository.

---

# Generated Artifacts

A benchmark execution typically produces two categories of outputs:

- **structured benchmark reports** containing machine-readable measurements;
- **publication-quality figures** generated from those reports.

These outputs are stored separately.

```text
results/            Machine-readable benchmark reports

docs/assets/plots/  Generated benchmark figures
```

Separating numerical results from visualizations makes it possible to regenerate figures without rerunning potentially expensive benchmark computations.

---

# Benchmark Reports

The primary output of every benchmark is a structured JSON report.

Although individual benchmark suites report different quantities, benchmark reports typically contain

- benchmark metadata;
- benchmark configuration;
- execution status;
- timing measurements;
- memory measurements;
- validation results (where applicable);
- benchmark-specific metrics.

Because every benchmark follows the same reporting conventions, JSON reports provide a stable interface for automated analysis and custom tooling.

---

# Results Directory

Benchmark reports are organized by benchmark suite.

For example,

```text
results/

    workspace_loading/

        workspace_loading_result.json

    model_creation/

        model_creation_result.json

    compiled_evaluation/

        compiled_evaluation_result.json

    ...

    matrix_summary.json
```

Each benchmark stores its own results independently, allowing benchmark suites to be executed separately while preserving a consistent repository structure.

---

# Matrix Summary

Executing multiple benchmark suites using

```bash
pixi run python -m src.run_all_benchmarks
```

also generates

```text
results/
└── matrix_summary.json
```

The matrix summary provides a high-level overview of the benchmark campaign, including

- executed benchmark suites;
- processed benchmark workspaces;
- execution status;
- failed benchmark runs;
- locations of generated reports.

This file serves as the primary entry point for automated reporting and large benchmark campaigns.

---

# Generated Figures

Most benchmark suites can generate publication-quality figures directly from their JSON reports.

Generated figures are stored under

```text
docs/
└── assets/
    └── plots/
```

Each benchmark maintains its own figure directory.

For example,

```text
docs/
└── assets/
    └── plots/
        └── workspace_loading/
            ├── workspace_loading_wall_time.png
            ├── workspace_loading_current_rss_delta.png
            └── workspace_loading_peak_rss_delta.png
```

Typical visualizations include

- execution time comparisons;
- memory profiles;
- throughput measurements;
- scalability studies;
- framework comparisons;
- likelihood scan visualizations.

Because figures are generated from structured benchmark reports, they can always be reproduced without repeating benchmark execution.

---

# Benchmark Status

Every benchmark records its execution status as part of the generated report.

Typical values include

- `success`
- `failed`

Recording unsuccessful benchmark executions simplifies debugging while allowing automated tooling to ignore failed runs during performance analysis.

---

# Interpreting Benchmark Results

Most benchmark suites execute multiple repeated measurements rather than reporting a single execution.

Depending on the benchmark, reported statistics may include

- mean execution time;
- median execution time;
- standard deviation;
- throughput;
- peak memory usage;
- current memory usage.

Using aggregated statistics reduces measurement noise and provides more reliable performance estimates.

The exact metrics reported by each benchmark are documented on the corresponding benchmark page.

---

# Reproducibility

Benchmark results are designed to be reproducible.

Executing the same benchmark with identical inputs and configuration should produce equivalent numerical outputs, while measured execution times may vary slightly depending on hardware and system load.

For this reason, every benchmark report records both the benchmark configuration and the measured statistics.

---

# Typical Result Lifecycle

Benchmark outputs follow a consistent lifecycle.

```text
Execute Benchmark
        │
        ▼
Generate JSON Report
        │
        ▼
Generate Figures
        │
        ▼
Analyze Performance
        │
        ▼
Compare Benchmark Campaigns
```

This separation between benchmark execution, report generation, and visualization simplifies automated analysis and long-term performance tracking.

---

# Common Use Cases

Benchmark outputs support a wide range of development and analysis tasks, including

- detecting performance regressions;
- evaluating optimization strategies;
- validating numerical agreement;
- studying scalability;
- generating publication-quality figures;
- producing automated benchmark summaries.

The common reporting format allows these analyses to be performed consistently across every benchmark suite.

---

# Related Documentation

For additional information, see

- **Benchmark Methodology**
- **Benchmark Runner**
- **Benchmark Workflow**
- **Benchmark Suite**
- **Repository Structure**
