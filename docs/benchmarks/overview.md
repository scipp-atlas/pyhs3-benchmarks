# Benchmark Overview

This benchmark overview summarizes PyHS3 performance measurements on generated workspaces and the available cross-framework comparisons.

Run:

```bash
pixi run python -m src.plot_benchmark_overview \
    --results-dir results/docs_examples \
    --plot-dir docs/assets/plots/benchmark_overview \
    --plots all
```

## Benchmark performance summary

![Benchmark performance summary](docs/assets/plots/benchmark_overview/benchmark_overview_performance_summary.png)

## Stage timing breakdown

![Stage timing breakdown](docs/assets/plots/benchmark_overview/benchmark_overview_stage_timing.png)

## Stage memory breakdown

![Stage memory breakdown](docs/assets/plots/benchmark_overview/benchmark_overview_stage_memory.png)

## Cross-framework runtime comparison

![Cross-framework runtime comparison](docs/assets/plots/benchmark_overview/benchmark_overview_cross_framework_summary.png)

## Interpretation

The overview shows that PyHS3 setup time is mostly dominated by model creation and log-probability compilation. Once the model is built, repeated warm evaluation is much faster.

The memory breakdown shows that peak RSS is also dominated by the compilation stage.

The cross-framework plots include only successful and numerically validated apples-to-apples comparisons.

## Notes on xRooFit

A dedicated xRooFit benchmark was investigated, but for the current generated ROOT workspaces, `xRooNode(...).nll(...)` returned a null NLL object. Therefore, a fully apples-to-apples benchmark against xRooFit’s own NLL machinery could not yet be constructed.

A pointwise ROOT/RooFit comparison is possible, but it should not be presented as benchmarking xRooFit-specific NLL algorithms.
