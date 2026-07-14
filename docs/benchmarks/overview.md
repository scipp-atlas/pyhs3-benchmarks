# Benchmark Overview

## Overview

`plot_benchmark_overview.py` generates high-level summary figures from
the benchmark matrix produced by `run_all_benchmarks.py`. Rather than
re-running benchmarks, it collects benchmark outputs from
`results/benchmark_matrix`, extracts the metrics of interest, and
produces publication-ready summary plots.

The script provides a unified view of:

-   overall benchmark performance;
-   timing breakdown by benchmark stage;
-   memory usage by benchmark stage;
-   cross-framework scalar PDF comparisons;
-   cross-framework pointwise ΔNLL comparisons;
-   cross-framework HistFactory likelihood comparisons.

------------------------------------------------------------------------

# Workflow

``` text
run_all_benchmarks.py
        │
        ▼
results/benchmark_matrix/
        │
        ▼
plot_benchmark_overview.py
        │
        ▼
docs/assets/images/plots/benchmark_overview/
```

The plotting script automatically discovers benchmark outputs and skips
unavailable benchmarks, allowing the overview to remain valid even when
only a subset of the benchmark suite has been executed.

------------------------------------------------------------------------

# Command line interface

``` bash
pixi run python -m src.plot_benchmark_overview \
    --results-dir results/benchmark_matrix \
    --plot-dir docs/assets/images/plots/benchmark_overview \
    --plots all
```

## Arguments

  Argument          Description
  ----------------- -------------------------------------------------------
  `--results-dir`   Root directory containing benchmark results.
  `--plot-dir`      Output directory for generated figures.
  `--plots`         Comma-separated list of overview figures to generate.

------------------------------------------------------------------------

# Supported plot groups

-   performance_summary
-   stage_timing
-   stage_memory
-   cross_framework_summary

`cross_framework_summary` produces three figures:

1.  Cross-framework Scalar PDF summary
2.  Cross-framework Pointwise NLL summary
3.  Cross-framework HistFactory likelihood summary

------------------------------------------------------------------------

# Automatically discovered benchmark outputs

The script searches the benchmark matrix and loads metrics from
benchmark JSON files, including runtime, peak memory, compiled
evaluation latency, PDF evaluation, ΔNLL scan timings, and
cross-framework validation results.

Missing benchmarks are ignored rather than treated as errors.

------------------------------------------------------------------------

# Generated figures

## 1. Benchmark performance summary

![](../assets/plots/benchmark_overview/benchmark_overview_performance_summary.png)

This dashboard provides a compact comparison of the principal
performance metrics across the benchmark suite:

-   setup time;
-   compiled evaluation latency;
-   PDF evaluation latency;
-   scalar cross-framework PDF evaluation;
-   NLL scan latency;
-   cross-framework ΔNLL evaluation.

It is intended as the first high-level performance overview.

------------------------------------------------------------------------

## 2. Stage timing breakdown

![](../assets/plots/benchmark_overview/benchmark_overview_stage_timing.png)

Shows the contribution of each benchmark stage:

-   workspace loading;
-   model creation;
-   log-probability construction;
-   log-probability compilation;
-   compiled evaluation;
-   PDF evaluation;
-   NLL scan.

This figure identifies which stages dominate total runtime.

------------------------------------------------------------------------

## 3. Stage memory breakdown

![](../assets/plots/benchmark_overview/benchmark_overview_stage_memory.png)

Displays peak RSS growth attributed to each benchmark stage.

Compilation is typically the dominant memory consumer.

------------------------------------------------------------------------

## 4. Cross-framework Scalar PDF summary

![](../assets/plots/benchmark_overview/benchmark_overview_cross_framework_scalar_pdf.png)

Compares scalar PDF evaluation latency across supported frameworks
(PyHS3, RooFit, and other available engines).

This benchmark measures a single PDF evaluation and is independent of
likelihood scans.

------------------------------------------------------------------------

## 5. Cross-framework Pointwise NLL summary

![](../assets/plots/benchmark_overview/benchmark_overview_cross_framework_pointwise_nll.png)

Compares complete pointwise NLL evaluations.

Unlike scalar PDF evaluation, this benchmark evaluates the entire
negative log-likelihood for one parameter point.

------------------------------------------------------------------------

## 6. Cross-framework HistFactory likelihood summary

![](../assets/plots/benchmark_overview/benchmark_overview_cross_framework_histfactory_likelihood.png)

Summarizes the paired HistFactory benchmark introduced for
engine-to-engine comparison between PyHS3 and pyhf.

Characteristics:

-   identical statistical models;
-   identical expected event counts;
-   ΔNLL agreement validated numerically;
-   warm steady-state evaluation only;
-   apples-to-apples engine comparison.

This benchmark intentionally uses simple paired HistFactory models and
should not be interpreted as a replacement for the RooFit/xRooFit
benchmarks, which evaluate more complex workspaces.

------------------------------------------------------------------------

# Internal architecture

The overview generator performs four steps:

1.  Discover benchmark outputs.
2.  Parse benchmark-specific JSON formats.
3.  Convert metrics into a common internal representation.
4.  Produce publication-quality figures.

Adding a new benchmark generally requires only extending the parser and
registering its metrics.

------------------------------------------------------------------------

# Notes

-   The overview aggregates existing benchmark outputs and never
    executes benchmarks.
-   Missing benchmark results are skipped automatically.
-   Cross-framework figures intentionally summarize different benchmark
    families and should not be interpreted as a single ranking across
    incompatible statistical models.
-   HistFactory comparisons are limited to paired compatible models,
    while RooFit comparisons evaluate substantially more complex
    workspaces.
