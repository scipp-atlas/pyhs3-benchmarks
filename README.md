# PyHS3 Benchmarks

A dedicated benchmarking and validation repository for PyHS3.

This repository contains benchmark, validation, and profiling infrastructure developed as part of the PyHS3 benchmarking and optimization effort. It is used to evaluate performance, scalability, memory usage, and numerical correctness across a variety of PyHS3 workflows and workspace configurations.

The repository is maintained separately from the main PyHS3 codebase to allow independent development of benchmarking, validation, profiling, and performance-tracking tools.

---

# Goals

The primary goals of this repository are:

* measure PyHS3 runtime and memory usage
* benchmark different stages of the PyHS3 workflow
* study scaling behavior with increasing model complexity
* validate numerical agreement across implementations
* compare PyHS3 against other statistical frameworks
* support profiling and optimization studies
* track performance changes across PyHS3 versions

---

# Benchmark Configuration

Current benchmark baseline:

```text
PyHS3 branch: main
PyHS3 commit: 326aadd
```

---

# Environment Setup

This repository assumes an existing PyHS3 development environment.

Example:

```bash
conda activate iris-hep
```

Verify that benchmarks are using the expected local PyHS3 checkout:

```bash
python - <<'PY'
import pyhs3
print(pyhs3.__file__)
PY
```

Expected output should point to a local PyHS3 source tree.

---

# Repository Structure

```text
inputs/
├── benchmark inputs
├── generated workspaces
└── validation examples

src/
├── benchmark scripts
├── validation scripts
├── plotting utilities
└── shared helpers

results/
└── raw benchmark outputs

plots/
└── generated benchmark figures

reports/
└── benchmark summaries and reports
```

---

# Benchmark Categories

## Core PyHS3 Benchmarks

Benchmarks covering the primary PyHS3 workflow:

```text
Workspace.load()
    ↓
Workspace.model()
    ↓
model.log_prob
    ↓
graph compilation
    ↓
compiled evaluation
```

Current coverage:

| Benchmark                    | Status         |
| ---------------------------- | -------------- |
| Workspace Loading            | ✅ Implemented  |
| Model Creation               | 🚧 In Progress |
| Log Probability Construction | 🚧 In Progress |
| Log Probability Compilation  | 🚧 In Progress |
| Compiled Evaluation          | 🚧 In Progress |
| NLL Scans                    | 🚧 Planned     |

---

## Scaling Benchmarks

Benchmarks studying performance as problem size increases.

Planned:

* memory scaling
* model complexity scaling
* workspace size scaling

---

## Cross-Framework Benchmarks

Comparisons against external statistical frameworks.

Planned:

* RooFit
* pyhf
* zfit
* numba-stats

These benchmarks are currently experimental and may require additional software installations.

---

## Validation Benchmarks

Numerical validation studies designed to detect regressions and verify agreement between implementations.

Examples:

* NLL agreement checks
* parameter consistency checks
* workspace validation
* regression testing

---

## Optimization Benchmarks

Before/after comparisons used to quantify the impact of performance improvements and code optimizations.

Metrics include:

* wall time
* memory usage
* numerical agreement

---

# Workspace Loading Benchmark

## Purpose

Measures the cost of loading an HS3 workspace into PyHS3.

This benchmark evaluates:

```text
Workspace.load(...)
```

and validates that the resulting workspace contains the required top-level HS3 components.

---

## Validation Checks

The benchmark verifies that:

* the workspace loads successfully
* distributions are present
* likelihoods are present
* data objects are present

Unsupported or malformed inputs are expected to fail validation.

---

## Supported Inputs

Validated HS3 workspaces:

* `simple_workspace.json`
* `simple_workspace_nonp.json`
* `simple_workspace_generic.json`
* `simple_workspace_generic_nonp.json`
* `simplemodel_correlated_background_hs3.json`

---

## Command Examples

Run the default workspace:

```bash
python src/run_workspace_loading.py
```

Run a specific workspace:

```bash
python src/run_workspace_loading.py \
    --workspaces inputs/simple_workspace.json
```

Run multiple workspaces:

```bash
python src/run_workspace_loading.py \
    --workspaces \
        inputs/simple_workspace.json \
        inputs/simple_workspace_nonp.json \
        inputs/simple_workspace_generic.json
```

Generate plots:

```bash
python src/run_workspace_loading.py \
    --n-runs 5 \
    --plot
```

Run with custom output location:

```bash
python src/run_workspace_loading.py \
    --n-runs 10 \
    --output-dir results/workspace_loading \
    --output-name workspace_loading_custom.json
```

---

## Arguments

| Argument        | Type        | Default                           | Description                                            |
| --------------- | ----------- | --------------------------------- | ------------------------------------------------------ |
| `--workspaces`  | `Path` list | `DEFAULT_WORKSPACE`               | One or more HS3 workspace JSON files to benchmark.     |
| `--n-runs`      | `int`       | `DEFAULT_N_RUNS`                  | Number of repeated timing measurements per workspace. |
| `--output-dir`  | `Path`      | `results/workspace_loading`       | Directory where benchmark JSON results are saved.      |
| `--output-name` | `str`       | `workspace_loading_result.json`   | Name of the JSON output file.                          |
| `--plot`        | flag        | `False`                           | Create comparison plots.                               |
| `--plot-dir`    | `Path`      | `plots/workspace_loading`         | Directory where generated plots are saved.             |
| `--plot-name`   | `str`       | `workspace_loading_wall_time.png` | Name of the wall-time plot output file.                |

---

## Outputs

The workspace loading benchmark writes results to:

```text
results/workspace_loading/
└── workspace_loading_result.json
```

Generated plots are written to:

```text
plots/workspace_loading/
├── workspace_loading_wall_time.png
├── workspace_loading_current_rss_delta.png
└── workspace_loading_peak_rss_delta.png
```

---

## Generated Metrics

For each workspace the benchmark records:

* wall time samples
* mean wall time
* median wall time
* wall time standard deviation
* current RSS memory usage before and after loading
* current RSS delta
* peak RSS memory usage before and after loading
* peak RSS delta
* number of distributions
* number of likelihoods
* number of data objects
* number of domains
* number of parameter points
* HS3 metadata information

---

## Memory Measurement Notes

Memory metrics are measured in a fresh subprocess for each workspace.

* `current_rss_delta_mb` reports the process RSS increase after a single workspace load.
* `peak_rss_delta_mb` reports the increase in maximum RSS observed during that subprocess.

These metrics are process-level measurements and should be interpreted as approximations of memory usage rather than exact object-level memory consumption.

---

## Example Plots

### Wall Time

![Workspace Loading Wall Time](plots/workspace_loading/workspace_loading_wall_time.png)

### Current RSS Delta

![Workspace Loading Current RSS Delta](plots/workspace_loading/workspace_loading_current_rss_delta.png)

---

# Benchmark Outputs

Each benchmark may generate:

## Results

```text
results/<benchmark_name>/
```

Typical outputs:

* JSON summaries
* timing measurements
* memory measurements
* validation information

## Plots

```text
plots/<benchmark_name>/
```

Typical outputs:

* runtime comparisons
* scaling curves
* memory usage trends
* validation summaries

---

# Future Work

Planned work includes:

* complete benchmark migration into this repository
* benchmark orchestration through `run_all_benchmarks.py`
* Scalene profiling integration
* before/after optimization comparisons
* automated regression detection
* GitHub Actions smoke tests
* performance tracking across PyHS3 releases

---

# Development Status

This repository is under active development.

Current milestone:

* migrate benchmark infrastructure into a dedicated repository
* validate benchmark correctness
* complete benchmark suite implementation by the end of June 2026

After benchmarking is complete, the focus will shift to:

* Scalene profiling
* bottleneck identification
* PyHS3 optimization studies
* performance regression tracking
