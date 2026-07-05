# Benchmark Results

Every benchmark execution produces structured outputs that can be used for analysis, visualization, validation, and long-term performance tracking.

All benchmark outputs are deterministic and organized using a consistent directory structure, making it straightforward to compare benchmark campaigns across different repository revisions.

---

# Output Directories

Benchmark outputs are organized into two primary locations.

```text
results/
docs/assets/plots/
```

The `results/` directory stores structured benchmark reports, while `docs/assets/plots/` contains generated figures used throughout the documentation.

Keeping numerical results separate from visualizations allows plots to be regenerated without rerunning benchmarks.

---

# Results Directory

Individual benchmark executions produce benchmark-specific JSON reports.

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

Each benchmark stores its own machine-readable report containing the measurements collected during execution.

When benchmarks are executed through the benchmark runner, a `matrix_summary.json` file is also generated, summarizing the complete benchmark campaign.

---

# Benchmark Reports

Each benchmark generates a structured JSON report.

Although the exact contents depend on the benchmark, reports typically include

- benchmark metadata;
- benchmark configuration;
- execution status;
- timing statistics;
- memory statistics (when applicable);
- validation information (when applicable);
- benchmark-specific measurements.

These JSON reports are the primary machine-readable outputs of the repository and can be used for automated analysis or custom visualization.

---

# Matrix Summary

Executing benchmarks through

```bash
pixi run python -m src.run_all_benchmarks
```

produces

```text
results/
└── matrix_summary.json
```

The summary contains information such as

- executed benchmark suites;
- processed workspaces;
- execution status;
- failed benchmark runs;
- locations of generated benchmark reports.

This file provides a convenient entry point for automated reporting and regression tracking.

---

# Generated Figures

Many benchmarks can generate publication-quality figures during execution.

Generated plots are written to

```text
docs/
└── assets/
    └── plots/
```

Each benchmark stores its figures in its own directory.

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

Typical figures include

- execution time comparisons;
- memory usage comparisons;
- scalability plots;
- framework comparisons;
- likelihood scan visualizations.

Since plots are generated directly from benchmark results, they can always be recreated without rerunning expensive benchmark computations.

---

# Repeated Measurements

Most timing benchmarks execute multiple repeated measurements.

Rather than reporting a single execution time, benchmark reports typically include

- mean execution time;
- median execution time;
- standard deviation.

These aggregated statistics reduce measurement noise and provide a more reliable estimate of benchmark performance.

---

# Benchmark Status

Every benchmark execution records its completion status.

Typical values include

- `success`
- `failed`

Failed benchmark runs remain part of the JSON report, making it possible to diagnose execution problems while excluding unsuccessful runs from comparison plots.

---

# Reproducibility

Benchmark outputs are designed to be reproducible.

Running the same benchmark with the same inputs and configuration should produce equivalent numerical results, while execution times may vary slightly depending on hardware and system load.

For this reason, benchmark reports always include both the benchmark configuration and the measured statistics.

---

# Typical Workflow

A typical benchmarking workflow is

```text
Run Benchmark
      │
      ▼
JSON Report
      │
      ▼
Generate Plots
      │
      ▼
Analyze Results
      │
      ▼
Documentation
```

Separating benchmark execution from visualization makes it possible to regenerate plots at any time without repeating benchmark execution.

---

# Using Benchmark Results

Benchmark reports support several common use cases, including

- performance regression detection;
- optimization studies;
- scalability analysis;
- cross-framework comparisons;
- publication-quality figures;
- automated reporting.

Because every benchmark follows the same reporting conventions, analysis tools can process benchmark results consistently across the entire repository.

---

# Related Documentation

See also

- **Benchmark Methodology**
- **Benchmark Suite**
- **Workspace Loading**
- **Repository Structure**
