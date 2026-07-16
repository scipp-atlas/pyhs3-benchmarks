# Graph Optimization

On this page, you will learn what the **Graph Optimization** benchmark measures, how to run it, and how to interpret its results.

The **Graph Optimization** benchmark measures the time and memory required to optimize a symbolic PyTensor `FunctionGraph` using the JAX optimizer.

Workspace loading, model creation, symbolic log-probability construction, and `FunctionGraph` creation are treated as setup steps and are excluded from the reported measurements. Compilation into executable code is benchmarked separately.

---

## What This Benchmark Measures

The benchmark measures only the execution of the graph optimization stage.

For each benchmark configuration, it reports

- mean wall time;
- median wall time;
- standard deviation;
- current RSS memory increase;
- peak RSS memory increase;
- graph validation statistics.

During validation, the benchmark records

- graph inputs;
- graph outputs;
- ApplyNodes before optimization;
- ApplyNodes after optimization.

Only graph optimization is included in the reported timings.

Details of the measurement methodology are described in **Benchmark Methodology**.

---

## Benchmark Workflow

```text
Workspace
      │
      ▼
Workspace.load(...)
      │
      ▼
Workspace.model(...)
      │
      ▼
model.log_prob
      │
      ▼
FunctionGraph(...)
      │
      ▼
JAX Graph Optimizer
      │
      ├────────► Graph Validation
      ├────────► Timing Statistics
      └────────► Memory Statistics
      │
      ▼
JSON Report
      │
      ▼
Comparison Plots (optional)
```

Only the optimization pass contributes to the reported benchmark results.

---

## When to Use This Benchmark

This benchmark is useful for

- measuring graph rewrite performance;
- comparing optimization costs across benchmark workspaces;
- evaluating memory usage during optimization;
- detecting optimizer regressions;
- measuring graph simplification before compilation.

---

## Running the Benchmark

### Run directly

```bash
pixi run python -m src.run_graph_optimization \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --targets L_ch0 \
    --modes FAST_RUN \
    --n-runs 200 \
    --output-dir results/graph_optimization \
    --plot \
    --plot-dir docs/assets/plots/graph_optimization
```

### Run through the Benchmark Matrix Runner

```bash
pixi run python -m src.run_all_benchmarks \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --benchmarks graph_optimization \
    --targets L_ch0 \
    --modes FAST_RUN \
    --n-runs 200 \
    --plot
```

---

## Command-line Arguments

| Argument | Description |
|----------|-------------|
| `--workspaces` | Workspace files to benchmark. |
| `--targets` | Model targets passed to `Workspace.model(...)`. |
| `--modes` | PyTensor compilation modes. |
| `--n-runs` | Number of repeated optimization measurements. |
| `--output-dir` | Directory for benchmark reports. |
| `--output-name` | Output JSON filename. |
| `--plot` | Generate comparison plots. |
| `--plot-dir` | Directory for generated figures. |

Common benchmark arguments and execution behavior are described in **Benchmark Methodology**.

---

## Generated Outputs

The benchmark produces

```text
results/
└── graph_optimization/
    └── graph_optimization_result.json
```

and, when plotting is enabled,

```text
docs/
└── assets/
    └── plots/
        └── graph_optimization/
            ├── graph_optimization_wall_time.png
            ├── graph_optimization_current_rss_delta.png
            └── graph_optimization_peak_rss_delta.png
```

The report structure and output conventions are documented in **Benchmark Results**.

---

## Results

### Wall-Time Comparison

![Graph optimization wall time](../assets/plots/graph_optimization/graph_optimization_wall_time.png)

Graph optimization completes in approximately **390–663 ms** across the benchmark workspace collection.

All benchmark workspaces complete optimization in well under one second, indicating that symbolic graph rewriting introduces only moderate overhead before compilation.

---

### Current RSS Memory

![Graph optimization current RSS](../assets/plots/graph_optimization/graph_optimization_current_rss_delta.png)

Graph optimization increases resident memory by approximately **15–16 MB** across all benchmark workspaces.

The memory footprint is highly consistent regardless of model complexity.

---

### Peak RSS Memory

![Graph optimization peak RSS](../assets/plots/graph_optimization/graph_optimization_peak_rss_delta.png)

Peak RSS closely follows current RSS, differing by less than **0.4 MB** for every benchmark workspace.

This indicates that graph optimization performs very few large temporary allocations.

---

### Graph Validation

Besides timing and memory measurements, the benchmark validates the optimized graph by reporting

- graph inputs;
- graph outputs;
- ApplyNodes before optimization;
- ApplyNodes after optimization.

Across the benchmark dataset, optimization consistently reduces the number of ApplyNodes before compilation, demonstrating that the optimizer successfully simplifies the symbolic computation graph.

---

## Implementation Notes

The benchmark includes several implementation choices that improve measurement quality.

- Workspace loading is excluded from the reported timings.
- Model creation, symbolic graph construction, and `FunctionGraph` creation are treated as setup.
- Each benchmark optimizes a freshly constructed graph.
- Graph validation is performed before results are recorded.

The general benchmark methodology is documented in **Benchmark Methodology**.

---

## Limitations

This benchmark measures only symbolic graph optimization.

It does **not** measure

- workspace loading;
- model creation;
- symbolic graph construction;
- graph compilation;
- compiled evaluation;
- PDF evaluation;
- likelihood evaluation.

These workflow stages are benchmarked separately.

---

## Related Documentation

See also

- **Log-Probability Construction**
- **Log-Probability Compilation**
- **Compiled Evaluation**
- **Benchmark Methodology**
- **Benchmark Results**
- **Workspace Lifecycle**
