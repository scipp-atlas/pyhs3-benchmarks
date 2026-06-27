# PyHS3 Benchmarks

A dedicated benchmarking, validation, and profiling repository for PyHS3.

This repository provides a collection of reproducible benchmarks for measuring the performance, scalability, memory usage, and numerical correctness of PyHS3 across different stages of the statistical inference workflow.

The benchmark suite is maintained independently from the main PyHS3 repository to allow continuous performance tracking, regression detection, optimization studies, and cross-framework comparisons.

---

# Goals

The repository is designed to:

- benchmark every major stage of the PyHS3 workflow
- measure execution time and memory usage
- study scaling with increasing model complexity
- validate numerical correctness
- compare PyHS3 against external statistical frameworks
- support profiling and optimization work
- monitor performance changes across PyHS3 versions

---

# Repository Structure

```text
inputs/
├── benchmark workspaces
├── generated scalar PDF workspaces
└── generated binned likelihood models

src/
├── benchmark scripts
├── workspace generators
├── plotting utilities
└── shared helper functions

results/
└── JSON benchmark outputs

plots/
└── generated benchmark figures

reports/
└── benchmark summaries
```

---

# Benchmark Inputs

The repository contains generators for reproducible benchmark inputs.

## Scalar PDF Workspaces

Generate scalar PDF benchmark workspaces:

```bash
python src/generate_scalar_pdf_workspaces.py
```

Validate generated workspaces:

```bash
python src/generate_scalar_pdf_workspaces.py --validate
```

Generated files:

```text
inputs/scalar_pdf_workspaces/
├── normal_pdf_workspace.json
├── poisson_pdf_workspace.json
└── exponential_pdf_workspace.json
```

These workspaces contain a single probability distribution and are intended for isolated PDF evaluation benchmarks.

Supported distributions:

- Gaussian
- Poisson
- Exponential

The generator is deterministic and produces identical outputs across repeated runs.

---

## Binned Likelihood Models

Generate benchmark likelihood models:

```bash
python src/generate_binned_likelihood_models.py
```

Validate generated PyHS3 workspaces:

```bash
python src/generate_binned_likelihood_models.py --validate
```

Generated files:

```text
inputs/binned_likelihood_models/
├── common_3bins.json
├── common_30bins.json
├── common_300bins.json
├── pyhf_3bins.json
├── pyhf_30bins.json
├── pyhf_300bins.json
├── pyhs3_3bins.json
├── pyhs3_30bins.json
└── pyhs3_300bins.json
```

The `common_*` files store shared signal, background, and observation arrays.

The `pyhf_*` and `pyhs3_*` workspaces are generated from identical inputs to enable fair cross-framework comparisons.

Current benchmark sizes:

```text
3 bins
30 bins
300 bins
```

The generator uses deterministic random seeds, ensuring reproducible benchmark inputs.

---

# Environment Setup

The benchmark suite assumes an existing PyHS3 development environment.

Activate your environment:

```bash
conda activate iris-hep
```

Install a local editable copy of PyHS3:

```bash
pip install -e /path/to/pyhs3
```

Example:

```bash
pip install -e /mnt/h/iris-hep/coding/pyhs3
```

Verify that benchmarks use the expected checkout:

```bash
python - <<'PY'
import pyhs3
print(pyhs3.__file__)
PY
```

Example output:

```text
/mnt/h/iris-hep/coding/pyhs3/src/pyhs3/__init__.py
```

Using an editable installation ensures benchmark results always correspond to the currently checked-out PyHS3 source tree.

---

# Available Benchmarks

| Benchmark | Purpose |
|------------|---------|
| `run_workspace_loading.py` | Benchmark `Workspace.load()` |
| `run_model_creation.py` | Benchmark `Workspace.model()` |
| `run_log_prob_construction.py` | Benchmark symbolic `model.log_prob` construction |
| `run_log_prob_compilation.py` | Benchmark `jaxify(model.log_prob)` |
| `run_compiled_evaluation.py` | Benchmark compiled graph execution |
| `run_pdf_evaluation.py` | Benchmark repeated `model.pdf()` evaluation |
| `run_nll_scan.py` | Benchmark repeated NLL scans |
| `run_memory_scaling.py` | Measure memory usage across workflow stages |
| `run_model_complexity_scaling.py` | Study scaling with increasing workspace complexity |
| `run_graph_canonicalization.py` | Benchmark PyTensor canonicalization |
| `run_graph_optimization.py` | Benchmark PyTensor graph optimization |
| `run_all_benchmarks.py` | Execute the complete benchmark suite |
| `plot_benchmark_overview.py` | Generate overview plots from benchmark results |

---

# Benchmark Defaults

Shared defaults are defined in `src/config.py`.

| Setting | Default |
|----------|---------|
| Workspace | `inputs/simple_workspace_nonp.json` |
| Target | `L_ch0` |
| Mode | `FAST_RUN` |
| Number of runs | `5` |

Unless overridden on the command line, all benchmarks use these defaults.

# Workspace Loading Benchmark

## Purpose

Measures the cost of loading an HS3 workspace into PyHS3.

The benchmark evaluates

```python
Workspace.load(...)
```

without performing any model construction, graph generation, or compilation.

This benchmark represents the first stage of the PyHS3 workflow.

---

## Command

```bash
python src/run_workspace_loading.py \
    --workspaces inputs/simple_workspace.json \
    --n-runs 20 \
    --plot
```

---

## Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--workspaces` | One or more HS3 workspaces to benchmark | `inputs/simple_workspace_nonp.json` |
| `--n-runs` | Number of timing repetitions | `5` |
| `--plot` | Generate benchmark plots | disabled |

---

## Validation

The benchmark verifies that

- the workspace loads successfully
- metadata is available
- distributions are present
- likelihoods are present
- data objects are present

---

## Outputs

Results are written to

```text
results/workspace_loading/
└── workspace_loading_result.json
```

Generated plots

```text
plots/workspace_loading/
├── workspace_loading_wall_time.png
├── workspace_loading_current_rss_delta.png
└── workspace_loading_peak_rss_delta.png
```

---

## Reported Metrics

For each workspace the benchmark records

- wall time samples
- mean wall time
- median wall time
- wall time standard deviation
- current RSS
- peak RSS
- number of distributions
- number of likelihoods
- number of data objects
- number of domains
- number of parameter points
- HS3 metadata version

---

## Memory Measurement

Memory is measured in a fresh subprocess.

The reported RSS values correspond to a single workspace load and are intended as process-level approximations rather than exact object memory usage.

---

## Example Plots

### Wall Time

![Workspace Loading Wall Time](plots/workspace_loading/workspace_loading_wall_time.png)

### Current RSS Delta

![Workspace Loading Current RSS Delta](plots/workspace_loading/workspace_loading_current_rss_delta.png)

### Peak RSS Delta

![Workspace Loading Peak RSS Delta](plots/workspace_loading/workspace_loading_peak_rss_delta.png)

---

# Model Creation Benchmark

## Purpose

Measures the cost of constructing a PyHS3 model from an already loaded workspace.

The benchmark evaluates

```python
Workspace.model(...)
```

Workspace loading is intentionally excluded from the timed section.

---

## Command

```bash
python src/run_model_creation.py \
    --workspaces inputs/simple_workspace.json \
    --targets L_ch0 \
    --modes FAST_RUN \
    --n-runs 20 \
    --plot
```

---

## Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--workspaces` | HS3 workspaces | `inputs/simple_workspace_nonp.json` |
| `--targets` | Analysis target | `L_ch0` |
| `--modes` | Model creation mode | `FAST_RUN` |
| `--n-runs` | Number of timing repetitions | `5` |
| `--plot` | Generate plots | disabled |

---

## Validation

The benchmark verifies that

- model creation succeeds
- a valid PyHS3 model object is returned

---

## Outputs

```text
results/model_creation/
└── model_creation_result.json
```

Plots

```text
plots/model_creation/
├── model_creation_wall_time.png
├── model_creation_current_rss_delta.png
└── model_creation_peak_rss_delta.png
```

---

## Reported Metrics

For each benchmark configuration the following metrics are recorded

- wall time samples
- mean wall time
- median wall time
- wall time standard deviation
- current RSS
- peak RSS
- model type

---

## Memory Measurement

Timing and memory measurements are intentionally separated.

Timing uses repeated `Workspace.model(...)` calls.

Memory measurements execute a single isolated model construction to avoid measuring memory accumulation from repeated graph creation.

---

## Example Plots

### Wall Time

![Model Creation Wall Time](plots/model_creation/model_creation_wall_time.png)

### Current RSS Delta

![Model Creation Current RSS Delta](plots/model_creation/model_creation_current_rss_delta.png)

### Peak RSS Delta

![Model Creation Peak RSS Delta](plots/model_creation/model_creation_peak_rss_delta.png)

---

# Log Probability Construction Benchmark

## Purpose

Measures the cost of constructing the symbolic log-probability expression.

The benchmark evaluates

```python
model.log_prob
```

This stage creates the symbolic PyTensor computation graph but does not perform compilation or execution.

---

## Command

```bash
python src/run_log_prob_construction.py \
    --workspaces inputs/simple_workspace.json \
    --targets L_ch0 \
    --modes FAST_RUN \
    --n-runs 20 \
    --plot
```

---

## Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--workspaces` | HS3 workspaces | `inputs/simple_workspace_nonp.json` |
| `--targets` | Analysis target | `L_ch0` |
| `--modes` | Execution mode | `FAST_RUN` |
| `--n-runs` | Timing repetitions | `5` |
| `--plot` | Generate plots | disabled |

---

## Validation

The benchmark verifies that

- the symbolic graph is successfully constructed
- a valid `TensorVariable` is returned
- graph metadata is available

---

## Outputs

```text
results/log_prob_construction/
└── log_prob_construction_result.json
```

Plots

```text
plots/log_prob_construction/
├── log_prob_construction_wall_time.png
├── log_prob_construction_current_rss_delta.png
└── log_prob_construction_peak_rss_delta.png
```

---

## Reported Metrics

- wall time samples
- mean wall time
- median wall time
- wall time standard deviation
- current RSS
- peak RSS
- tensor type
- tensor dtype
- tensor dimensionality

---

## Memory Measurement

Memory is measured using a single construction of the symbolic graph.

Repeated timing runs create fresh models to ensure every timing sample measures a complete graph construction.

---

## Example Plots

### Wall Time

![Log Probability Construction Wall Time](plots/log_prob_construction/log_prob_construction_wall_time.png)

### Current RSS Delta

![Log Probability Construction Current RSS Delta](plots/log_prob_construction/log_prob_construction_current_rss_delta.png)

### Peak RSS Delta

![Log Probability Construction Peak RSS Delta](plots/log_prob_construction/log_prob_construction_peak_rss_delta.png)

---

# Log Probability Compilation Benchmark

## Purpose

Measures the cost of compiling a symbolic log-probability graph into an executable JAX function.

The benchmark evaluates

```python
jaxify(model.log_prob)
```

Compilation is measured independently from graph construction and graph execution.

---

## Command

```bash
python src/run_log_prob_compilation.py \
    --workspaces inputs/simple_workspace.json \
    --targets L_ch0 \
    --modes FAST_RUN \
    --n-runs 20 \
    --plot
```

---

## Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--workspaces` | HS3 workspaces | `inputs/simple_workspace_nonp.json` |
| `--targets` | Analysis target | `L_ch0` |
| `--modes` | Model execution mode | `FAST_RUN` |
| `--n-runs` | Number of timing repetitions | `5` |
| `--plot` | Generate plots | disabled |

---

## Validation

The benchmark verifies that

- graph compilation succeeds
- a valid `JaxifiedGraph` is produced
- the compiled graph executes successfully
- the output is finite

---

## Outputs

```text
results/log_prob_compilation/
└── log_prob_compilation_result.json
```

Plots

```text
plots/log_prob_compilation/
├── log_prob_compilation_wall_time.png
├── log_prob_compilation_current_rss_delta.png
└── log_prob_compilation_peak_rss_delta.png
```

---

## Reported Metrics

The benchmark reports

- wall time samples
- mean wall time
- median wall time
- wall time standard deviation
- current RSS
- peak RSS
- compiled graph type
- compiled input names
- validation output

---

## Memory Measurement

Memory measurements are performed using a single compilation.

Repeated timing measurements execute independent compilations to avoid measuring memory accumulation.

---

## Example Plots

### Wall Time

![Log Probability Compilation Wall Time](plots/log_prob_compilation/log_prob_compilation_wall_time.png)

### Current RSS Delta

![Log Probability Compilation Current RSS Delta](plots/log_prob_compilation/log_prob_compilation_current_rss_delta.png)

### Peak RSS Delta

![Log Probability Compilation Peak RSS Delta](plots/log_prob_compilation/log_prob_compilation_peak_rss_delta.png)

---

# Compiled Evaluation Benchmark

## Purpose

Measures the execution performance of an already compiled log-probability graph.

The benchmark evaluates repeated calls to the compiled JAX function while excluding graph construction and compilation costs.

---

## Command

```bash
python src/run_compiled_evaluation.py \
    --workspaces inputs/simple_workspace.json \
    --targets L_ch0 \
    --modes FAST_RUN \
    --n-evaluations 10000 \
    --plot
```

---

## Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--workspaces` | HS3 workspaces | `inputs/simple_workspace_nonp.json` |
| `--targets` | Analysis target | `L_ch0` |
| `--modes` | Execution mode | `FAST_RUN` |
| `--n-evaluations` | Number of repeated evaluations | `100` |
| `--plot` | Generate plots | disabled |

---

## Validation

The benchmark verifies that

- compiled execution succeeds
- all outputs are finite
- repeated evaluations are numerically stable

---

## Outputs

```text
results/compiled_evaluation/
└── compiled_evaluation_result.json
```

Plots

```text
plots/compiled_evaluation/
├── compiled_evaluation_average_runtime_seconds_per_evaluation.png
├── compiled_evaluation_throughput_evaluations_per_second.png
├── compiled_evaluation_current_rss_delta.png
└── compiled_evaluation_peak_rss_delta.png
```

---

## Reported Metrics

The benchmark records

- total runtime
- average runtime per evaluation
- throughput
- current RSS
- peak RSS
- first output
- last output
- output stability

---

## Memory Measurement

Memory measurements are performed independently from timing.

Timing reflects only repeated execution of the compiled graph.

---

## Example Plots

### Average Runtime

![Compiled Evaluation Runtime](plots/compiled_evaluation/compiled_evaluation_average_runtime_seconds_per_evaluation.png)

### Throughput

![Compiled Evaluation Throughput](plots/compiled_evaluation/compiled_evaluation_throughput_evaluations_per_second.png)

### Current RSS Delta

![Compiled Evaluation Current RSS](plots/compiled_evaluation/compiled_evaluation_current_rss_delta.png)

### Peak RSS Delta

![Compiled Evaluation Peak RSS](plots/compiled_evaluation/compiled_evaluation_peak_rss_delta.png)

---

# PDF Evaluation Benchmark

## Purpose

Measures repeated evaluation of a probability density function using

```python
model.pdf(...)
```

This benchmark is intended to isolate PDF evaluation performance independently from likelihood construction and compilation.

---

## Command

```bash
python src/run_pdf_evaluation.py \
    --workspaces inputs/scalar_pdf_workspaces/normal_pdf_workspace.json \
    --distribution pdf \
    --n-evaluations 10000 \
    --plot
```

---

## Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--workspaces` | Scalar benchmark workspaces | generated workspaces |
| `--distribution` | Distribution name | `sig_ch0` |
| `--n-evaluations` | Number of repeated evaluations | `100` |
| `--plot` | Generate plots | disabled |

---

## Validation

The benchmark verifies that

- PDF evaluation succeeds
- outputs are finite
- repeated evaluations are numerically stable

---

## Outputs

```text
results/pdf_evaluation/
└── pdf_evaluation_result.json
```

Plots

```text
plots/pdf_evaluation/
├── pdf_evaluation_cold_start_time_seconds.png
├── pdf_evaluation_average_runtime_seconds_per_evaluation.png
├── pdf_evaluation_throughput_evaluations_per_second.png
├── pdf_evaluation_current_rss_delta.png
└── pdf_evaluation_peak_rss_delta.png
```

---

## Reported Metrics

The benchmark records

- cold-start latency
- average runtime
- throughput
- current RSS
- peak RSS
- output stability
- reference output

---

## Memory Measurement

Cold-start execution is measured separately from repeated evaluations.

Memory measurements are collected independently from timing measurements.

---

## Example Plots

### Cold Start

![Cold Start](plots/pdf_evaluation/pdf_evaluation_cold_start_time_seconds.png)

### Average Runtime

![Average Runtime](plots/pdf_evaluation/pdf_evaluation_average_runtime_seconds_per_evaluation.png)

### Throughput

![Throughput](plots/pdf_evaluation/pdf_evaluation_throughput_evaluations_per_second.png)

### Current RSS Delta

![Current RSS](plots/pdf_evaluation/pdf_evaluation_current_rss_delta.png)

### Peak RSS Delta

![Peak RSS](plots/pdf_evaluation/pdf_evaluation_peak_rss_delta.png)

---

# NLL Scan Benchmark

## Purpose

Measures repeated negative log-likelihood scans over a selected parameter.

The benchmark evaluates repeated execution of the compiled likelihood over a user-defined scan range.

---

## Command

```bash
python src/run_nll_scan.py \
    --workspaces inputs/simple_workspace.json \
    --targets L_ch0 \
    --scan-parameter mu_sig \
    --scan-min 0 \
    --scan-max 5 \
    --n-scan-points 1001 \
    --plot
```

---

## Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--workspaces` | HS3 workspaces | `inputs/simple_workspace_nonp.json` |
| `--targets` | Analysis target | `L_ch0` |
| `--scan-parameter` | Parameter to scan | `mu_sig` |
| `--scan-min` | Lower scan boundary | `0` |
| `--scan-max` | Upper scan boundary | `5` |
| `--n-scan-points` | Number of scan points | `101` |
| `--plot` | Generate plots | disabled |

---

## Validation

The benchmark verifies that

- all scan values are finite
- the minimum likelihood is identified
- the scan completes successfully

---

## Outputs

```text
results/nll_scan/
└── nll_scan_result.json
```

Plots

```text
plots/nll_scan/
├── nll_scan_runtime_per_scan_point_seconds.png
├── nll_scan_throughput_scan_points_per_second.png
├── nll_scan_current_rss_delta.png
└── nll_scan_peak_rss_delta.png
```

---

## Reported Metrics

The benchmark records

- runtime per scan point
- scan throughput
- current RSS
- peak RSS
- minimum NLL
- minimum scan parameter
- NLL range

---

## Memory Measurement

The scan executes using a precompiled graph.

Memory measurements therefore isolate scan execution rather than graph construction or compilation.

---

## Example Plots

### Runtime per Scan Point

![Runtime](plots/nll_scan/nll_scan_runtime_per_scan_point_seconds.png)

### Scan Throughput

![Throughput](plots/nll_scan/nll_scan_throughput_scan_points_per_second.png)

### Current RSS Delta

![Current RSS](plots/nll_scan/nll_scan_current_rss_delta.png)

### Peak RSS Delta

![Peak RSS](plots/nll_scan/nll_scan_peak_rss_delta.png)

---

# Memory Scaling Benchmark

## Purpose

Measures memory usage across multiple stages of the PyHS3 workflow.

Unlike the individual benchmarks, this benchmark executes several workflow stages sequentially and summarizes their memory consumption.

Supported workflow stages include

- Workspace Loading
- Model Creation
- Log Probability Construction
- Log Probability Compilation
- Compiled Evaluation
- PDF Evaluation
- NLL Scan

---

## Command

```bash
python src/run_memory_scaling.py \
    --workspaces inputs/simple_workspace.json \
    --stages all \
    --plot
```

---

## Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--workspaces` | HS3 workspaces | `inputs/simple_workspace_nonp.json` |
| `--targets` | Analysis target | `L_ch0` |
| `--modes` | Execution mode | `FAST_RUN` |
| `--stages` | Workflow stages to benchmark | `all` |
| `--n-runs` | Timing repetitions | `5` |
| `--n-evaluations` | Evaluation repetitions | `100` |
| `--distribution` | PDF distribution | `sig_ch0` |
| `--scan-parameter` | NLL scan parameter | `mu_sig` |
| `--scan-min` | Scan lower bound | `0` |
| `--scan-max` | Scan upper bound | `5` |
| `--n-scan-points` | Number of scan points | `101` |
| `--plot` | Generate plots | disabled |

---

## Validation

The benchmark verifies

- every selected benchmark completes successfully
- all RSS measurements are available
- stage summaries are generated correctly

---

## Outputs

```text
results/memory_scaling/
└── memory_scaling_result.json
```

Plots

```text
plots/memory_scaling/
├── memory_scaling_current_rss_delta_mb.png
└── memory_scaling_peak_rss_delta_mb.png
```

---

## Reported Metrics

The benchmark reports

- current RSS increase
- peak RSS increase
- cumulative RSS
- maximum peak RSS
- stage-by-stage memory usage

---

## Example Plots

### Current RSS Delta

![Current RSS](plots/memory_scaling/memory_scaling_current_rss_delta_mb.png)

### Peak RSS Delta

![Peak RSS](plots/memory_scaling/memory_scaling_peak_rss_delta_mb.png)

---

# Model Complexity Scaling Benchmark

## Purpose

Measures how benchmark performance changes as workspace complexity increases.

Multiple workspaces of different sizes are benchmarked using identical benchmark settings.

This benchmark is intended to study scaling behaviour rather than absolute runtime.

---

## Command

```bash
python src/run_model_complexity_scaling.py \
    --workspaces \
        inputs/simple_workspace.json \
        inputs/simple_workspace_generic.json \
    --stages all \
    --plot
```

---

## Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--workspaces` | Input workspaces | required |
| `--targets` | Analysis target | `L_ch0` |
| `--modes` | Execution mode | `FAST_RUN` |
| `--stages` | Workflow stages | `all` |
| `--n-runs` | Timing repetitions | `5` |
| `--n-evaluations` | Evaluation repetitions | `100` |
| `--distribution` | PDF distribution | `sig_ch0` |
| `--scan-parameter` | Scan parameter | `mu_sig` |
| `--scan-min` | Lower scan limit | `0` |
| `--scan-max` | Upper scan limit | `5` |
| `--n-scan-points` | Number of scan points | `101` |
| `--plot` | Generate plots | disabled |

---

## Validation

The benchmark verifies

- all selected stages complete successfully
- scaling summary is generated
- CSV report is produced

---

## Outputs

```text
results/model_complexity_scaling/
└── model_complexity_scaling_result.json
```

Reports

```text
reports/model_complexity_scaling/
└── model_complexity_scaling_summary.csv
```

Plots

```text
plots/model_complexity_scaling/
├── model_complexity_scaling_wall_time.png
├── model_complexity_scaling_peak_rss.png
├── model_complexity_scaling_stage_timing.png
└── model_complexity_scaling_stage_memory.png
```

---

## Reported Metrics

The benchmark reports

- workspace size
- timing for every workflow stage
- memory usage for every workflow stage
- total setup time
- evaluation runtime
- scan runtime

---

## Example Plots

### Wall Time

![Wall Time](plots/model_complexity_scaling/model_complexity_scaling_wall_time.png)

### Peak RSS

![Peak RSS](plots/model_complexity_scaling/model_complexity_scaling_peak_rss.png)

### Stage Timing

![Stage Timing](plots/model_complexity_scaling/model_complexity_scaling_stage_timing.png)

### Stage Memory

![Stage Memory](plots/model_complexity_scaling/model_complexity_scaling_stage_memory.png)

---

# Graph Canonicalization Benchmark

## Purpose

Measures the cost of PyTensor graph canonicalization.

The benchmark evaluates the canonicalization rewrites applied before graph optimization.

---

## Command

```bash
python src/run_graph_canonicalization.py \
    --workspaces inputs/simple_workspace.json \
    --plot
```

---

## Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--workspaces` | HS3 workspaces | `inputs/simple_workspace_nonp.json` |
| `--targets` | Analysis target | `L_ch0` |
| `--modes` | Execution mode | `FAST_RUN` |
| `--n-runs` | Timing repetitions | `5` |
| `--plot` | Generate plots | disabled |

---

## Validation

The benchmark verifies

- canonicalization succeeds
- graph remains valid
- graph statistics are recorded

---

## Outputs

```text
results/graph_canonicalization/
└── graph_canonicalization_result.json
```

Plots

```text
plots/graph_canonicalization/
├── graph_canonicalization_wall_time.png
├── graph_canonicalization_current_rss_delta.png
└── graph_canonicalization_peak_rss_delta.png
```

---

## Reported Metrics

- wall time
- current RSS
- peak RSS
- graph inputs
- graph outputs
- apply nodes before optimization
- apply nodes after optimization

---

## Example Plots

### Wall Time

![Wall Time](plots/graph_canonicalization/graph_canonicalization_wall_time.png)

### Current RSS

![Current RSS](plots/graph_canonicalization/graph_canonicalization_current_rss_delta.png)

### Peak RSS

![Peak RSS](plots/graph_canonicalization/graph_canonicalization_peak_rss_delta.png)

---

# Graph Optimization Benchmark

## Purpose

Measures the execution cost of PyTensor JAX graph optimization.

This benchmark is performed after graph canonicalization and before compilation.

---

## Command

```bash
python src/run_graph_optimization.py \
    --workspaces inputs/simple_workspace.json \
    --plot
```

---

## Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--workspaces` | HS3 workspaces | `inputs/simple_workspace_nonp.json` |
| `--targets` | Analysis target | `L_ch0` |
| `--modes` | Execution mode | `FAST_RUN` |
| `--n-runs` | Timing repetitions | `5` |
| `--plot` | Generate plots | disabled |

---

## Validation

The benchmark verifies

- optimization succeeds
- optimized graph is valid
- graph statistics are recorded

---

## Outputs

```text
results/graph_optimization/
└── graph_optimization_result.json
```

Plots

```text
plots/graph_optimization/
├── graph_optimization_wall_time.png
├── graph_optimization_current_rss_delta.png
└── graph_optimization_peak_rss_delta.png
```

---

## Reported Metrics

- wall time
- current RSS
- peak RSS
- graph inputs
- graph outputs
- apply nodes before optimization
- apply nodes after optimization

---

## Example Plots

### Wall Time

![Wall Time](plots/graph_optimization/graph_optimization_wall_time.png)

### Current RSS

![Current RSS](plots/graph_optimization/graph_optimization_current_rss_delta.png)

### Peak RSS

![Peak RSS](plots/graph_optimization/graph_optimization_peak_rss_delta.png)

---

# Running the Complete Benchmark Suite

The repository provides a convenience script for executing the complete benchmark suite.

Rather than launching each benchmark individually, the runner automatically executes all supported benchmark scripts using a predefined benchmark configuration.

---

## Command

```bash
python src/run_all_benchmarks.py
```

---

## Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--preset` | Benchmark preset (`smoke`, `default`, `full`) | `default` |
| `--workspaces` | Input workspaces | built-in workspace list |
| `--target` | Analysis target | `L_ch0` |
| `--mode` | Execution mode | `FAST_RUN` |
| `--plot` | Generate plots | enabled for `default` and `full` |
| `--dry-run` | Print commands without executing them | disabled |

---

## Benchmark Presets

### Smoke

Designed for quick validation during development.

| Setting | Value |
|----------|------:|
| Runs | 1 |
| Evaluations | 1 |
| Scan points | 11 |
| Plot generation | No |

Run:

```bash
python src/run_all_benchmarks.py --preset smoke
```

---

### Default

Balanced benchmark configuration intended for routine benchmarking.

| Setting | Value |
|----------|------:|
| Runs | 20 |
| Evaluations | 1000 |
| Scan points | 1001 |
| Plot generation | Yes |

Run:

```bash
python src/run_all_benchmarks.py --preset default
```

---

### Full

Comprehensive benchmark configuration intended for performance studies and release benchmarking.

| Setting | Value |
|----------|------:|
| Runs | 200 |
| Evaluations | 10000 |
| Scan points | 5001 |
| Plot generation | Yes |

Run:

```bash
python src/run_all_benchmarks.py --preset full
```

---

## Executed Benchmarks

The benchmark runner executes

- Workspace Loading
- Model Creation
- Log Probability Construction
- Log Probability Compilation
- Compiled Evaluation
- PDF Evaluation
- NLL Scan
- Memory Scaling
- Model Complexity Scaling
- Graph Canonicalization
- Graph Optimization

Each benchmark stores its own JSON results and benchmark plots inside the corresponding `results/` and `plots/` directories.

---

## Dry Run

To preview all benchmark commands without executing them:

```bash
python src/run_all_benchmarks.py --dry-run
```

This is useful for verifying benchmark configuration before launching long benchmark runs.

---

# Benchmark Overview

After running one or more benchmarks, the repository can generate summary plots across all benchmark results.

---

## Command

```bash
python src/plot_benchmark_overview.py
```

---

## Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--results-dir` | Benchmark result directory | `results/` |
| `--plots` | Overview plots to generate | `all` |

---

## Available Overview Plots

The following plots are currently supported.

| Plot | Description |
|------|-------------|
| `status` | Benchmark success/failure summary |
| `wall_time` | Average benchmark wall time |
| `evaluation_time` | Average evaluation runtime |
| `scan_time` | Runtime per NLL scan point |
| `setup_time` | Setup cost across workflow stages |
| `peak_rss` | Peak RSS memory usage |
| `total_peak_rss` | Total peak RSS across benchmarks |
| `stage_timing` | Runtime broken down by workflow stage |
| `stage_memory` | Memory usage broken down by workflow stage |

Generate every overview plot:

```bash
python src/plot_benchmark_overview.py --plots all
```

Generate selected plots only:

```bash
python src/plot_benchmark_overview.py \
    --plots wall_time peak_rss stage_memory
```

---

## Outputs

Overview figures are written to

```text
plots/benchmark_overview/
```

Depending on the selected plots, generated figures may include

```text
benchmark_status.png
benchmark_wall_time.png
benchmark_evaluation_time.png
benchmark_scan_time.png
benchmark_setup_time.png
benchmark_peak_rss.png
benchmark_total_peak_rss.png
benchmark_stage_timing.png
benchmark_stage_memory.png
```

---

## Example Plots

### Benchmark Status

![Benchmark Status](plots/benchmark_overview/benchmark_status.png)

### Wall Time

![Benchmark Wall Time](plots/benchmark_overview/benchmark_wall_time.png)

### Evaluation Runtime

![Evaluation Runtime](plots/benchmark_overview/benchmark_evaluation_time.png)

### Scan Runtime

![Scan Runtime](plots/benchmark_overview/benchmark_scan_time.png)

### Setup Time

![Setup Time](plots/benchmark_overview/benchmark_setup_time.png)

### Peak RSS

![Peak RSS](plots/benchmark_overview/benchmark_peak_rss.png)

### Total Peak RSS

![Total Peak RSS](plots/benchmark_overview/benchmark_total_peak_rss.png)

### Stage Timing

![Stage Timing](plots/benchmark_overview/benchmark_stage_timing.png)

### Stage Memory

![Stage Memory](plots/benchmark_overview/benchmark_stage_memory.png)

---

# Benchmark Workflow

The benchmark suite follows the typical PyHS3 execution pipeline.

```text
Workspace.load()
        │
        ▼
Workspace.model()
        │
        ▼
model.log_prob
        │
        ▼
jaxify(...)
        │
        ▼
Compiled evaluation
        │
        ▼
NLL scans
```

Individual benchmarks isolate each stage of this workflow, while the scaling benchmarks combine multiple stages to study overall performance characteristics.

---

# Output Files

Every benchmark produces

- JSON results
- optional benchmark plots
- validation information
- timing measurements
- memory measurements

Scaling benchmarks additionally generate CSV summary reports.

---

# Reproducibility

All benchmark input generators are deterministic.

The benchmark suite is designed so that

- identical benchmark inputs are generated across repeated runs
- benchmark configurations are fully reproducible
- benchmark results are stored as JSON
- plots can be regenerated from stored benchmark results

This makes the repository suitable for continuous benchmarking, regression testing, and performance tracking across PyHS3 releases.

# Future Work

Planned benchmark extensions include

- RooFit comparisons
- pyhf comparisons
- zfit comparisons
- numba-stats comparisons
- additional scaling studies
- automated benchmark reports
- continuous performance tracking across PyHS3 commits
