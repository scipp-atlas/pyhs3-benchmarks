# Benchmark Matrix Runner

On this page, you will learn how the benchmark matrix runner executes and manages benchmark campaigns across the repository.

The benchmark matrix runner is the recommended entry point for running reproducible benchmark campaigns. Instead of invoking individual benchmark scripts manually, it coordinates benchmark selection, workspace discovery, command construction, execution, logging, and output organization.

---

## Overview

The benchmark matrix runner

- discovers benchmark workspaces;
- selects benchmark suites or benchmark groups;
- constructs benchmark-specific commands;
- executes benchmarks in isolated subprocesses;
- organizes benchmark outputs and figures;
- records execution status and logs;
- generates consolidated campaign summaries.

For details on how benchmarks are measured and validated, see **Benchmark Methodology**.

---

## Supported Benchmark Categories

The runner supports the following benchmark categories.

| Category | Description |
|----------|-------------|
| Workflow Benchmarks | Workspace loading, model creation, compilation, evaluation, and NLL scans |
| Memory Benchmarks | Memory profiling and scaling studies |
| Model Complexity Benchmarks | Scaling with increasing model complexity |
| Cross-Framework Benchmarks | PyHS3, pyhf, RooFit, and xRooFit comparisons |
| Scalar Benchmarks | Scalar PDF evaluation |
| Overview Generation | Aggregate benchmark summary plots |

Available benchmark suites are registered internally through `BenchmarkSpec`.

---

## Benchmark Registry

Each benchmark is registered through a `BenchmarkSpec` entry defining

- benchmark name;
- benchmark group;
- execution mode;
- Python module;
- workspace requirements;
- whether matching ROOT workspaces are required;
- whether the benchmark runs once per campaign.

The registry allows the runner to build the correct command automatically for every benchmark suite.

---

## Execution Modes

Benchmark suites are executed using one of four execution modes.

| Mode | Description |
|------|-------------|
| `multi_workspace` | Executes once per workspace or once per workspace batch when generating comparison plots |
| `single_workspace` | Executes independently for each selected workspace |
| `json_root_pair` | Executes benchmarks requiring matched JSON and ROOT workspaces |
| `run_once` | Executes once per campaign using internally managed inputs |

The execution mode determines how benchmark commands are constructed and scheduled.

---

## Workspace Discovery

By default, benchmark workspaces are discovered automatically from `inputs/`.

Workspace selection can be customized using

- `--workspaces`;
- `--workspace-dir`;
- `--workspace-glob`;
- `--workspace-regex`;
- `--exclude-workspaces`;
- `--limit`.

Only JSON workspaces participate in the workspace matrix.

---

## Benchmark Selection

Run selected benchmark suites:

```bash
pixi run python -m src.run_all_benchmarks \
  --benchmarks workspace_loading pdf_evaluation
```

Run benchmark groups:

```bash
pixi run python -m src.run_all_benchmarks \
  --groups cross
```

Exclude benchmarks from a larger selection:

```bash
--exclude-benchmarks
```

---

## Common Usage

Run all benchmarks:

```bash
pixi run python -m src.run_all_benchmarks
```

Generate plots:

```bash
pixi run python -m src.run_all_benchmarks --plot
```

Preview commands without executing benchmarks:

```bash
pixi run python -m src.run_all_benchmarks --dry-run
```

Repeat a complete benchmark campaign:

```bash
pixi run python -m src.run_all_benchmarks \
  --repeat 3
```

---

## Benchmark-Specific Options

Some benchmark suites accept additional benchmark-specific options.

For example, `cross_binned_likelihood` supports configuration of the HistFactory scan through dedicated `--pyhf-*` arguments, which are forwarded directly by the matrix runner.

```bash
pixi run python -m src.run_all_benchmarks \
  --benchmarks cross_binned_likelihood \
  --pyhf-mu-points 201 \
  --pyhf-repeats 5000
```

---

## Automatic ROOT Pairing

Benchmarks requiring ROOT workspaces automatically search for matching ROOT counterparts.

If `--root-workspace-dir` is specified, matching files are resolved from that directory. Otherwise, the runner searches alongside the JSON workspace.

Benchmarks without matching ROOT workspaces are skipped and recorded in the campaign summary.

---

## Output Organization

The matrix runner organizes benchmark results, logs, and figures using a consistent directory structure.

### Workspace-based benchmarks

```text
results/benchmark_matrix/
└── <benchmark>/
    └── <workspace>/
        └── repeat_000/
            ├── <benchmark>_result.json
            ├── stdout.txt
            └── stderr.txt
```

```text
plots/benchmark_matrix/
└── <benchmark>/
    └── <workspace>/
        └── repeat_000/
```

### Run-once benchmarks

```text
results/benchmark_matrix/
└── <benchmark>/
    └── global/
        └── repeat_000/
            ├── results.json
            ├── stdout.txt
            └── stderr.txt
```

```text
plots/benchmark_matrix/
└── <benchmark>/
    └── global/
        └── repeat_000/
```

Batch plotting runs also use the `global` directory while generating comparison plots across multiple workspaces.

---

## Execution Logs

Each benchmark subprocess produces

- `stdout.txt`
- `stderr.txt`

The runner also records the executed command and execution status for every benchmark.

---

## Campaign Summary

After each campaign, the runner updates

```text
results/benchmark_matrix/matrix_summary.json
```

The summary includes execution status, timing information, skipped benchmarks, failures, and command metadata.

Failed or timed-out executions are additionally summarized in

```text
failed_summary.txt
```

---

## Failure Handling

Large benchmark campaigns are supported through

- `--fail-fast`
- `--timeout-seconds`
- `--dry-run`

These options simplify debugging, validation, and long-running benchmark executions.

---

## Why Use the Matrix Runner?

The benchmark matrix runner provides

- automatic workspace discovery;
- benchmark-specific command construction;
- multiple execution modes;
- automatic JSON/ROOT pairing;
- centralized logging;
- repeated benchmark campaigns;
- consistent output organization;
- consolidated execution summaries.

For repository-wide benchmark campaigns, it is the recommended execution interface.

---

## Related Documentation

See also

- **Getting Started**
- **Benchmark Methodology**
- **Benchmark Workspaces**
- **Outputs**
- **Cross-Framework Benchmarks**
- **API Reference**
