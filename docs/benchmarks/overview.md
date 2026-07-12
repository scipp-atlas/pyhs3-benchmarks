# Benchmark Overview

This page summarizes the complete benchmark suite and provides a high-level overview of the performance characteristics measured across all implemented benchmarks.

Unlike the individual benchmark pages, which focus on a single experiment, the overview aggregates results from multiple benchmark categories and highlights setup costs, runtime behaviour, memory usage, and cross-framework comparisons.

Generate the overview using:

```bash
pixi run python -m src.plot_benchmark_overview \
    --results-dir results/docs_examples \
    --plot-dir docs/assets/plots/benchmark_overview \
    --plots all
```

---

## Command-line Arguments

The benchmark overview script supports the following command-line arguments.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--results-dir` | `Path` | `results/` | Root directory containing benchmark result JSON files. The script recursively searches for files ending with `_result.json`. |
| `--plot-dir` | `Path` | `docs/assets/plots/benchmark_overview/` | Directory where the generated overview plots will be saved. |
| `--plots` | `str ...` | `all` | Overview plots to generate. Supported values are `performance_summary`, `setup_summary`, `evaluation_summary`, `scan_summary`, `stage_timing`, `stage_memory`, `diagnostics`, `cross_framework_summary`, or `all`. |
| `--benchmarks` | `str ...` | all benchmarks | Restrict the overview to specific benchmark names. |
| `--workspaces` | `str ...` | all workspaces | Include only selected workspaces. |
| `--targets` | `str ...` | all targets | Filter results by target distribution. |
| `--modes` | `str ...` | all modes | Filter by PyTensor compilation mode. |
| `--n-runs` | `int ...` | all | Filter by the number of timing repetitions. |
| `--n-evaluations` | `int ...` | all | Filter evaluation benchmarks by evaluation count. |
| `--n-scan-points` | `int ...` | all | Filter NLL scan benchmarks by scan resolution. |
| `--include-failed` | flag | disabled | Include failed benchmark runs. |
| `--strict` | flag | disabled | Stop immediately if malformed result files are encountered. |

---

## Benchmark performance summary

![Benchmark performance summary](../assets/plots/benchmark_overview/benchmark_overview_performance_summary.png)

This figure provides a compact overview of the most important benchmark metrics collected across the project.

It summarizes:

- workspace setup latency;
- compiled evaluation performance;
- scalar PDF evaluation performance;
- pointwise NLL evaluation performance.

Rather than replacing the individual benchmark reports, this figure provides a convenient high-level summary of the complete benchmark suite.

---

## Stage timing breakdown

![Stage timing breakdown](../assets/plots/benchmark_overview/benchmark_overview_stage_timing.png)

This figure decomposes the total initialization time into the individual execution stages:

- workspace loading;
- model creation;
- log-probability graph construction;
- JAX compilation.

The figure illustrates where the initialization cost is spent before steady-state evaluations become possible.

Across all tested workspaces, model creation and JAX compilation dominate the overall startup latency.

---

## Stage memory breakdown

![Stage memory breakdown](../assets/plots/benchmark_overview/benchmark_overview_stage_memory.png)

This figure shows the increase in peak RSS memory during the initialization pipeline.

Most additional memory is allocated during JAX compilation, while the remaining stages contribute comparatively little to the total memory footprint.

Together with the timing breakdown, this figure identifies which initialization stages dominate both runtime and memory consumption.

---

## Cross-framework Scalar PDF summary

![Cross-framework Scalar PDF summary](../assets/plots/benchmark_overview/benchmark_overview_cross_framework_scalar_pdf.png)

This overview summarizes the apples-to-apples scalar PDF benchmark.

For every workspace, the figure compares:

- **pyHS3 non-compiled (PyTensor)**
- **pyHS3 compiled (JAX)**
- **RooFit**

Only the **varying observable** benchmark configuration is included. This prevents RooFit from returning cached values and ensures that every framework evaluates the PDF for changing observable values.

The comparison therefore reflects the true steady-state cost of individual scalar PDF evaluations.

---

## Cross-framework Pointwise NLL summary

![Cross-framework Pointwise NLL summary](../assets/plots/benchmark_overview/benchmark_overview_cross_framework_pointwise_nll.png)

This figure summarizes the point-by-point negative log-likelihood (NLL) benchmark.

Each bar represents one complete NLL evaluation performed using:

- identical datasets;
- identical parameter values;
- identical model configurations.

For compiled pyHS3, the JAX compilation phase is intentionally excluded from the timed region, so only steady-state evaluation performance is compared.

Across all tested workspaces, the compiled implementation consistently outperforms the non-compiled PyTensor implementation, with speedups ranging from approximately **2.5× to nearly 18×**, depending on the model complexity.

RooFit remains highly competitive on some larger workspaces, while compiled pyHS3 approaches or exceeds RooFit performance on others.

---

## Notes

- The overview automatically discovers benchmark result files by recursively searching `--results-dir` for files ending in `_result.json`.
- By default, only successful benchmark runs are included in the generated plots.
- Cross-framework summaries include only numerically validated apples-to-apples comparisons.
- Multiple filters (`--benchmarks`, `--workspaces`, `--targets`, etc.) may be combined to generate overview reports for specific subsets of benchmark results.
- Unless `--strict` is enabled, malformed or incomplete result files are skipped automatically so that a single invalid benchmark does not prevent generation of the remaining overview figures.

---

## Notes on xRooFit

A dedicated xRooFit benchmark was also investigated.

However, for the generated benchmark workspaces,

```cpp
xRooNode(...).nll(...)
```

did not successfully construct a valid NLL object, preventing a fully apples-to-apples comparison against xRooFit's own NLL implementation.

Consequently, the current cross-framework comparisons use **RooFit** as the ROOT reference implementation. Support for dedicated xRooFit NLL benchmarks can be added once compatible workspaces become available.
