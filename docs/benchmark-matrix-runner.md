# Benchmark Matrix Runner

The benchmark matrix runner provides a unified interface for executing benchmark campaigns across multiple benchmark suites, workspace configurations, and execution modes.

Instead of invoking individual benchmark scripts manually, the matrix runner coordinates benchmark selection, command construction, execution, output management, logging, and consolidated reporting.

The script serves as the primary entry point for reproducible benchmark campaigns across the repository.

---

# Overview

The benchmark matrix runner

- discovers benchmark workspaces;
- selects individual benchmarks or benchmark groups;
- groups benchmarks by execution mode;
- builds benchmark-specific command lines;
- executes benchmark scripts in subprocesses;
- captures standard output and standard error;
- records execution status and duration;
- organizes benchmark results and figures;
- writes a consolidated campaign summary.

It provides a consistent execution workflow regardless of whether a benchmark operates on individual workspaces, matched JSON/ROOT pairs, or its own internally managed input dataset.

---

# Supported Benchmark Categories

The runner currently supports the following benchmark categories.

| Category | Description |
|----------|-------------|
| Workflow Benchmarks | Workspace loading, model creation, graph construction, compilation, evaluation, and NLL scans |
| Memory Benchmarks | Memory profiling and scaling studies |
| Model Complexity Benchmarks | Scaling with increasing statistical model complexity |
| Cross-Framework Benchmarks | PyHS3, pyhf, RooFit, and xRooFit comparison benchmarks |
| Scalar Benchmarks | Cross-framework scalar PDF evaluation |
| Overview Generation | Aggregate benchmark summary plots |

The available benchmark suites are defined internally through the `BenchmarkSpec` registry.

---

# Benchmark Registry

Every benchmark is registered through a `BenchmarkSpec` entry.

A benchmark specification defines

- benchmark name;
- benchmark group;
- execution kind;
- Python module;
- whether the workspace matrix is used;
- whether a matching ROOT workspace is required;
- whether the benchmark runs once per campaign.

This registry allows the matrix runner to construct the appropriate command automatically for each benchmark suite.

---

# Benchmark Execution Modes

Benchmark suites are executed using one of four execution modes.

| Mode | Description |
|------|-------------|
| `multi_workspace` | Executes once for each selected workspace, or once for the complete workspace batch when comparison plots are requested |
| `single_workspace` | Executes a framework-comparison benchmark separately for each selected workspace |
| `json_root_pair` | Executes a benchmark requiring matched JSON and ROOT workspaces |
| `run_once` | Executes once per benchmark campaign using its own input configuration |

The execution mode is defined by the `BenchmarkSpec` registry and determines how the benchmark command is constructed.

---

## Multi-workspace execution

Most PyHS3 workflow benchmarks use the `multi_workspace` mode.

Without `--plot`, each selected workspace is executed in a separate subprocess.

With `--plot`, supported benchmarks receive all selected workspaces in one subprocess so that comparison figures can be generated from a common result set.

---

## Single-workspace execution

Single-workspace benchmarks are executed independently for every selected workspace.

This mode is used when a benchmark expects exactly one workspace argument while still participating in the workspace matrix.

---

## JSON/ROOT-pair execution

Some cross-framework benchmarks require both

- an HS3 JSON workspace;
- a statistically corresponding ROOT workspace.

The runner resolves the matching ROOT file automatically and skips the benchmark if the pair is unavailable.

---

## Run-once execution

Some benchmark suites operate on their own predefined datasets instead of the general workspace matrix.

Examples include

- `cross_model_complexity_scaling`;
- `cross_binned_likelihood`;
- `cross_vectorized_pdf_evaluation`;
- `cross_scalar_pdf_evaluation`;
- `benchmark_overview`.

These benchmarks execute exactly once per repetition and manage their own inputs internally.

---

# Workspace Discovery

Unless explicitly specified, benchmark workspaces are discovered automatically from the `inputs/` directory.

Workspace selection can be customized using

- `--workspaces`;
- `--workspace-dir`;
- `--workspace-glob`;
- `--workspace-regex`;
- `--exclude-workspaces`;
- `--limit`.

This makes it possible to benchmark

- individual workspaces;
- filename-based subsets;
- regular-expression-based subsets;
- the complete benchmark dataset.

Only JSON files are included in the workspace matrix.

---

# Benchmark Selection

Individual benchmark suites can be selected using

```bash
pixi run python -m src.run_all_benchmarks \
  --benchmarks workspace_loading pdf_evaluation
```

Benchmark groups can be selected using

```bash
pixi run python -m src.run_all_benchmarks \
  --groups pyhs3
```

Available groups include

- `pyhs3`;
- `cross`;
- `scalar`;
- `overview`.

Benchmarks can also be removed from a larger selection using

```bash
--exclude-benchmarks
```

---

# Typical Usage

## Run all benchmarks

```bash
pixi run python -m src.run_all_benchmarks
```

## Run selected benchmark suites

```bash
pixi run python -m src.run_all_benchmarks \
  --benchmarks workspace_loading pdf_evaluation
```

## Run all cross-framework benchmarks

```bash
pixi run python -m src.run_all_benchmarks \
  --groups cross
```

## Run the PyHS3/pyhf binned-likelihood benchmark

```bash
pixi run python -m src.run_all_benchmarks \
  --benchmarks cross_binned_likelihood
```

## Benchmark specific workspaces

```bash
pixi run python -m src.run_all_benchmarks \
  --workspaces \
    inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
    inputs/30ch_bkgGenericPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json
```

## Generate plots together with benchmark execution

```bash
pixi run python -m src.run_all_benchmarks --plot
```

## Preview commands without executing benchmarks

```bash
pixi run python -m src.run_all_benchmarks --dry-run
```

## Repeat the complete benchmark campaign

```bash
pixi run python -m src.run_all_benchmarks \
  --repeat 3
```

---

# Cross-Framework Binned-Likelihood Configuration

The `cross_binned_likelihood` benchmark is a `run_once` benchmark that uses paired simple HistFactory/HS3 workspaces from `inputs/pyhf`.

Its matrix-runner options include

- `--pyhf-input-dir`;
- `--pyhf-mu-min`;
- `--pyhf-mu-max`;
- `--pyhf-mu-points`;
- `--pyhf-repeats`;
- `--pyhf-warmups`;
- `--pyhf-scaling-repeats`;
- `--pyhf-scaling-bins`;
- `--pyhf-rtol`;
- `--pyhf-atol`.

For example,

```bash
pixi run python -m src.run_all_benchmarks \
  --benchmarks cross_binned_likelihood \
  --pyhf-mu-points 201 \
  --pyhf-repeats 5000 \
  --pyhf-warmups 100 \
  --pyhf-scaling-repeats 1000 \
  --pyhf-scaling-bins 2 4 8 16 32 64 128 256 512 1024 2048
```

The benchmark is executed once per campaign repetition rather than once per general workspace.

---

# Automatic ROOT Pairing

For benchmarks requiring ROOT workspaces, the runner automatically searches for matching ROOT counterparts.

If `--root-workspace-dir` is provided, the matching file is resolved as

```text
<root-workspace-dir>/<json-workspace-stem>.root
```

Otherwise, the runner searches next to the JSON workspace.

Benchmarks requiring paired workspaces are skipped when no matching ROOT workspace is available.

The skipped execution is recorded in the campaign summary.

---

# Plot Generation

When `--plot` is enabled, benchmark modules that support plotting receive an explicit plot directory.

For multi-workspace benchmarks, plotting may switch execution from per-workspace mode to a single batch subprocess so that comparison figures can be generated from all selected workspaces together.

Run-once benchmarks manage their own figures within the output directory assigned by the matrix runner.

Not every benchmark produces figures, so plot availability depends on the selected benchmark suite.

---

# Output Organization

The runner creates a structured hierarchy for results, logs, and figures.

## Workspace-based benchmarks

Workspace-specific benchmark runs are stored as

```text
results/benchmark_matrix/
└── <benchmark>/
    └── <workspace>/
        └── repeat_000/
            ├── <benchmark>_result.json
            ├── stdout.txt
            └── stderr.txt
```

Corresponding figures are stored under

```text
plots/benchmark_matrix/
└── <benchmark>/
    └── <workspace>/
        └── repeat_000/
```

---

## Run-once benchmarks

Benchmarks that manage their own input datasets are stored under a `global` workspace key.

```text
results/benchmark_matrix/
└── <benchmark>/
    └── global/
        └── repeat_000/
            ├── results.json
            ├── stdout.txt
            └── stderr.txt
```

Corresponding figures are stored under

```text
plots/benchmark_matrix/
└── <benchmark>/
    └── global/
        └── repeat_000/
```

The `global` directory indicates that the benchmark was not executed for one specific workspace from the general workspace matrix.

---

## Batch plotting runs

When a multi-workspace benchmark is executed in batch mode for plotting, results are stored under

```text
results/benchmark_matrix/
└── <benchmark>/
    └── global/
        └── repeat_000/
```

while comparison figures are stored under the benchmark-level plot directory.

---

# Execution Logs

Every subprocess receives dedicated log files.

- `stdout.txt` contains standard output.
- `stderr.txt` contains warnings and error output.

This allows individual failures to be diagnosed without rerunning the entire campaign.

The matrix runner also records the exact command used for each execution.

---

# Campaign Summary

After every benchmark execution, the runner updates a consolidated JSON summary.

By default, the summary is written to

```text
results/benchmark_matrix/matrix_summary.json
```

It contains

- total execution count;
- successful executions;
- failed executions;
- timeouts;
- unexpected errors;
- dry-run records;
- skipped benchmarks;
- the complete command and output metadata for every execution.

A separate

```text
failed_summary.txt
```

file contains a concise report for failed, timed-out, or errored executions.

---

# Repeatability

The runner supports repeated benchmark execution through

```bash
--repeat
```

Each repetition is stored independently using

```text
repeat_000
repeat_001
repeat_002
...
```

This makes it possible to

- estimate measurement variability;
- compare repeated executions;
- evaluate benchmark stability;
- retain complete logs for every repetition.

---

# Failure Handling

Several options simplify large benchmark campaigns.

## Fail fast

```bash
--fail-fast
```

Stops the campaign after the first failed execution.

Skipped benchmarks do not trigger fail-fast termination.

---

## Timeout

```bash
--timeout-seconds
```

Limits the runtime of every benchmark subprocess independently.

Timed-out executions are recorded in the campaign summary.

---

## Dry run

```bash
--dry-run
```

Builds and records benchmark commands without executing them.

Dry-run mode is useful for verifying workspace selection, benchmark selection, command-line arguments, output paths, and ROOT pairing.

---

# Isolated Subprocess Execution

Every benchmark command is executed through a separate subprocess.

This provides

- independent standard-output and standard-error capture;
- timeout handling;
- explicit return-code tracking;
- reduced interference between benchmark implementations;
- reproducible command records.

The matrix runner itself does not interpret benchmark-specific numerical results. It records whether the benchmark subprocess completed successfully and leaves numerical validation to the individual benchmark module.

---

# Why Use the Matrix Runner?

Running benchmark suites individually is practical during development.

For reproducible benchmark campaigns, the matrix runner provides

- consistent benchmark configuration;
- automatic workspace discovery;
- benchmark-specific command construction;
- support for multiple execution modes;
- automatic JSON/ROOT pairing;
- unified output management;
- centralized execution logging;
- repeated benchmark campaigns;
- consolidated success and failure summaries.

For these reasons, the matrix runner is the recommended entry point for repository-wide benchmark campaigns.

---

# Related Documentation

See also

- **Benchmark Methodology**
- **Benchmark Workspaces**
- **Cross-Framework Benchmarks**
- **Benchmark Results**
- **Outputs**
