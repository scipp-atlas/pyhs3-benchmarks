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

Benchmark numbers should always be interpreted relative to the PyHS3 version used during benchmark execution.

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
jaxify(...)
    ↓
compiled evaluation
```

Current coverage:

| Benchmark                    | Status         |
| ---------------------------- | -------------- |
| Workspace Loading            | Implemented |
| Model Creation               | Implemented |
| Log Probability Construction | Implemented |
| Log Probability Compilation  | Implemented |
| Compiled Evaluation          | Implemented |
| NLL Scans                    | 📋 Planned     |

---

## Stable Benchmarks

The following benchmarks have been validated and are currently considered stable:

* Workspace Loading
* Model Creation
* Log Probability Construction
* Log Probability Compilation
* Compiled Evaluation

Each benchmark includes:

* timing measurements
* memory measurements
* validation checks
* JSON outputs
* automated plots

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

# Benchmark Defaults

Shared benchmark defaults are defined in `src/config.py`.

Current defaults:

| Setting           | Value                        |
| ----------------- | ---------------------------- |
| DEFAULT_TARGET    | `L_ch0`                      |
| DEFAULT_MODE      | `FAST_RUN`                   |
| DEFAULT_N_RUNS    | `5`                          |
| DEFAULT_WORKSPACE | `simple_workspace_nonp.json` |

These defaults are used unless explicitly overridden through command-line arguments.

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

---

## Supported Inputs

Validated HS3 workspaces:

* `simple_workspace.json`
* `simple_workspace_nonp.json`
* `simple_workspace_generic.json`
* `simple_workspace_generic_nonp.json`
* `simplemodel_correlated_background_hs3.json`

---

## Outputs

Results:

```text
results/workspace_loading/
└── workspace_loading_result.json
```

Plots:

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
* current RSS memory usage
* peak RSS memory usage
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

These metrics are process-level measurements and should be interpreted as approximations rather than exact object-level memory consumption.

---

## Example Plots

### Wall Time

![Workspace Loading Wall Time](plots/workspace_loading/workspace_loading_wall_time.png)

### Current RSS Delta

![Workspace Loading Current RSS Delta](plots/workspace_loading/workspace_loading_current_rss_delta.png)

---

# Model Creation Benchmark

## Purpose

Measures the cost of creating a PyHS3 model from an already-loaded workspace.

This benchmark evaluates:

```text
Workspace.model(...)
```

Workspace loading is treated as setup and is intentionally excluded from the timed section.

---

## Validation Checks

The benchmark verifies that the created model:

* is successfully created
* exposes `log_prob`
* exposes `data`
* exposes `free_params`
* reports the number of free parameters

---

## Outputs

Results:

```text
results/model_creation/
└── model_creation_result.json
```

Plots:

```text
plots/model_creation/
├── model_creation_wall_time.png
├── model_creation_current_rss_delta.png
└── model_creation_peak_rss_delta.png
```

---

## Generated Metrics

For each workspace/target/mode combination the benchmark records:

* wall time samples
* mean wall time
* median wall time
* wall time standard deviation
* current RSS memory usage
* peak RSS memory usage
* model type
* availability of `log_prob`
* availability of `data`
* availability of `free_params`
* number of free parameters

---

## Memory Measurement Notes

Memory metrics are measured in a fresh subprocess for each workspace/target/mode combination.

Timing and memory are measured separately:

* timing uses repeated `Workspace.model(...)` calls
* memory uses one isolated `Workspace.model(...)` call

This avoids reporting memory accumulation from repeated graph creation as the memory cost of a single model construction.

---

## Example Plots

### Wall Time

![Model Creation Wall Time](plots/model_creation/model_creation_wall_time.png)

### Current RSS Delta

![Model Creation Current RSS Delta](plots/model_creation/model_creation_current_rss_delta.png)

---

# Log Probability Construction Benchmark

## Purpose

Measures the cost of constructing the symbolic PyTensor log-probability graph from an already-created PyHS3 model.

This benchmark evaluates:

```text
model.log_prob
```

Workspace loading and model creation are treated as setup and are intentionally excluded from the timed section.

This benchmark measures symbolic graph construction only.

The benchmark does not include:

* workspace loading
* model creation
* graph compilation
* numerical evaluation

These stages are benchmarked separately.

---

## Validation Checks

The benchmark verifies that the constructed log-probability object:

* is successfully created
* is a valid PyTensor `TensorVariable`
* has a valid name
* has a valid dtype
* has a valid dimensionality
* can be used in subsequent compilation benchmarks

---

## Outputs

Results:

```text
results/log_prob_construction/
└── log_prob_construction_result.json
```

Plots:

```text
plots/log_prob_construction/
├── log_prob_construction_wall_time.png
├── log_prob_construction_current_rss_delta.png
└── log_prob_construction_peak_rss_delta.png
```

---

## Generated Metrics

For each workspace/target/mode combination the benchmark records:

* wall time samples
* mean wall time
* median wall time
* wall time standard deviation
* current RSS memory usage
* peak RSS memory usage
* log-probability object type
* log-probability object name
* log-probability dtype
* log-probability dimensionality
* compilation readiness flag

---

## Memory Measurement Notes

Memory metrics are measured in a fresh subprocess for each workspace/target/mode combination.

Timing and memory are measured separately:

* timing uses repeated `model.log_prob` construction
* memory uses one isolated `model.log_prob` construction

This avoids reporting memory accumulation from repeated graph construction as the memory cost of a single log-probability graph creation.

---

## Example Plots

### Wall Time

![Log Probability Construction Wall Time](plots/log_prob_construction/log_prob_construction_wall_time.png)

### Current RSS Delta

![Log Probability Construction Current RSS Delta](plots/log_prob_construction/log_prob_construction_current_rss_delta.png)

---

# Log Probability Compilation Benchmark

## Purpose

Measures the cost of compiling a symbolic PyTensor log-probability graph into a JAX-executable graph.

This benchmark evaluates:

```text
jaxify(log_prob)
```

Workspace loading, model creation, and log-probability construction are treated as setup and are intentionally excluded from the timed section.

---

## Validation Checks

The benchmark verifies that the compiled object:

* is successfully created
* is a valid `JaxifiedGraph`
* exposes the expected inputs
* can be executed successfully
* returns a finite numerical result

---

## Outputs

Results:

```text
results/log_prob_compilation/
└── log_prob_compilation_result.json
```

Plots:

```text
plots/log_prob_compilation/
├── log_prob_compilation_wall_time.png
├── log_prob_compilation_current_rss_delta.png
└── log_prob_compilation_peak_rss_delta.png
```

---

## Generated Metrics

For each workspace/target/mode combination the benchmark records:

* wall time samples
* mean wall time
* median wall time
* wall time standard deviation
* current RSS memory usage
* peak RSS memory usage
* compiled graph type
* number of compiled inputs
* compiled input names
* validation output value
* finite-result validation status

---

## Memory Measurement Notes

Compilation memory usage includes JAX/XLA initialization overhead and should be interpreted as compilation-time process memory consumption rather than model memory footprint.

---

## Example Plots

### Wall Time

![Log Probability Compilation Wall Time](plots/log_prob_compilation/log_prob_compilation_wall_time.png)

### Current RSS Delta

![Log Probability Compilation Current RSS Delta](plots/log_prob_compilation/log_prob_compilation_current_rss_delta.png)

---

# Compiled Evaluation Benchmark

## Purpose

Measures the execution cost of evaluating an already-compiled JAX graph.

This benchmark evaluates:

```text
compiled(...)
```

Workspace loading, model creation, graph construction, and graph compilation are treated as setup and are intentionally excluded from the timed section.

---

## Validation Checks

The benchmark verifies that:

* evaluation succeeds
* outputs are finite
* repeated evaluations are numerically stable
* repeated evaluations produce consistent results

---

## Outputs

Results:

```text
results/compiled_evaluation/
└── compiled_evaluation_result.json
```

Plots:

```text
plots/compiled_evaluation/
├── compiled_evaluation_average_time.png
└── compiled_evaluation_throughput.png
```

---

## Generated Metrics

For each workspace/target/mode/evaluation-count combination the benchmark records:

* total wall time
* average wall time per evaluation
* throughput (evaluations per second)
* current RSS memory usage
* peak RSS memory usage
* number of evaluations
* output stability metrics
* maximum absolute deviation
* finite-result validation status

---

## Memory Measurement Notes

Memory measurements are performed separately from timing measurements.

Memory is evaluated using a single compiled graph execution to avoid reporting accumulated memory from repeated evaluations.

For most small validation workspaces, memory deltas are expected to be close to zero.

---

## Example Plots

### Average Runtime Per Evaluation

![Compiled Evaluation Average Runtime](plots/compiled_evaluation/compiled_evaluation_average_time.png)

### Throughput

![Compiled Evaluation Throughput](plots/compiled_evaluation/compiled_evaluation_throughput.png)

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

# Profiling Roadmap

Profiling is intentionally postponed until the benchmark suite is complete.

Planned profiling work:

* Scalene integration
* hotspot identification
* memory profiling
* optimization candidate analysis

Profiling results will be added after the benchmark migration phase is complete.

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
