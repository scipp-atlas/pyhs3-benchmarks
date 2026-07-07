# Benchmark Matrix Runner

The benchmark matrix runner provides a unified interface for executing benchmark campaigns across multiple benchmark suites, workspace configurations, and execution modes.

Instead of invoking individual benchmark scripts manually, the matrix runner coordinates benchmark execution, manages outputs, collects results, and generates a consolidated execution summary.

The script serves as the primary entry point for large-scale benchmark campaigns.

---

# Overview

The benchmark matrix runner

- discovers benchmark workspaces;
- selects benchmark suites;
- builds benchmark execution commands;
- executes benchmark scripts;
- captures logs;
- records execution status;
- generates summary reports.

It provides a consistent execution workflow regardless of the selected benchmark suite.

---

# Supported Benchmark Categories

The runner currently supports

| Category | Description |
|----------|-------------|
| Workflow Benchmarks | Workspace loading, model creation, graph construction, compilation, evaluation, and NLL scans |
| Memory Benchmarks | Memory profiling and scaling studies |
| Model Complexity Benchmarks | Scaling with increasing statistical model complexity |
| Cross-Framework Benchmarks | PyHS3 vs ROOT comparisons |
| Overview Generation | Aggregate benchmark summary plots |

The available benchmark suites are defined internally through the `BenchmarkSpec` registry.

---

# Workspace Discovery

Unless explicitly specified, benchmark workspaces are discovered automatically from the `inputs/` directory.

Workspace selection can be customized using

- `--workspaces`
- `--workspace-dir`
- `--workspace-glob`
- `--workspace-regex`
- `--exclude-workspaces`

This makes it possible to benchmark

- individual workspaces;
- subsets of the benchmark dataset;
- complete benchmark campaigns.

---

# Benchmark Selection

Individual benchmark suites can be selected using

```bash
--benchmarks workspace_loading pdf_evaluation
```

Alternatively, benchmark groups can be executed

```bash
--groups pyhs3
```

Available groups include

- `pyhs3`
- `cross`
- `scalar`
- `overview`

---

# Typical Usage

Run all benchmarks

```bash
pixi run python -m src.run_all_benchmarks
```

Run selected benchmark suites

```bash
pixi run python -m src.run_all_benchmarks \
    --benchmarks workspace_loading pdf_evaluation
```

Benchmark specific workspaces

```bash
pixi run python -m src.run_all_benchmarks \
    --workspaces \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/30ch_bkgGenericPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json
```

Generate plots together with benchmark execution

```bash
pixi run python -m src.run_all_benchmarks --plot
```

Preview commands without executing benchmarks

```bash
pixi run python -m src.run_all_benchmarks --dry-run
```

---

# Output Organization

The runner automatically creates a structured output directory.

Each benchmark execution stores

- benchmark results;
- stdout logs;
- stderr logs;
- generated figures;
- execution summaries.

At the end of the benchmark campaign, the runner writes

- a JSON summary;
- a failed benchmark report.

This organization makes benchmark campaigns reproducible and simplifies debugging.

---

# Automatic ROOT Pairing

For benchmarks requiring ROOT workspaces, the runner automatically searches for matching ROOT counterparts corresponding to the selected HS3 workspaces.

Benchmarks that require paired workspaces are skipped if no matching ROOT workspace is available.

---

# Repeatability

The runner supports repeated benchmark execution through the `--repeat` option.

Each repetition is stored independently, making it possible to

- estimate measurement variability;
- compare repeated executions;
- evaluate benchmark stability.

---

# Failure Handling

Several options simplify large benchmark campaigns.

- `--fail-fast`
- `--timeout-seconds`
- `--dry-run`

Execution status is recorded for every benchmark independently, allowing partially successful benchmark campaigns to be analyzed without rerunning completed benchmarks.

---

# Why Use the Matrix Runner?

Running benchmark suites individually is practical during development.

For reproducible benchmark campaigns, however, the matrix runner provides

- consistent benchmark configuration;
- automatic workspace discovery;
- unified output management;
- centralized execution logging;
- reproducible benchmark summaries.

For this reason, the matrix runner is the recommended entry point for executing benchmark campaigns throughout the repository.

---

# Related Documentation

- Benchmark Workflow
- Benchmark Results
- Outputs
- Benchmark Workspaces
