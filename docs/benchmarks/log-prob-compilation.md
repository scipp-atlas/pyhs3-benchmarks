# Log-Probability Compilation

On this page, you will learn what the **Log-Probability Compilation** benchmark measures, how to run it, and how to interpret its results.

The **Log-Probability Compilation** benchmark measures the time and memory required to compile an already constructed symbolic log-probability graph into an executable JAX function.

Workspace loading, model creation, and symbolic graph construction are treated as setup steps and are excluded from the reported measurements. Numerical evaluation is benchmarked separately.

---

## What This Benchmark Measures

The benchmark measures only the execution of

```python
compiled = compile_log_prob(log_prob)
```

For each benchmark configuration, it reports

- mean wall time;
- median wall time;
- standard deviation;
- current RSS memory increase;
- peak RSS memory increase;
- compilation validation status.

The benchmark measures JAX compilation only. Symbolic graph construction and numerical execution are intentionally excluded.

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
compile_log_prob(...)
      │
      ├────────► Compilation Validation
      ├────────► Timing Statistics
      └────────► Memory Statistics
      │
      ▼
JSON Report
      │
      ▼
Comparison Plots (optional)
```

Only JAX compilation contributes to the reported benchmark results.

---

## When to Use This Benchmark

This benchmark is useful for

- measuring JAX compilation overhead;
- comparing compilation performance across benchmark workspaces;
- evaluating memory usage during compilation;
- detecting compilation regressions;
- separating compilation costs from graph construction and execution.

---

## Running the Benchmark

### Run directly

```bash
pixi run python -m src.run_log_prob_compilation \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --targets L_ch0 \
    --modes FAST_RUN \
    --n-runs 30 \
    --output-dir results/docs_examples/log_prob_compilation \
    --plot \
    --plot-dir docs/assets/plots/log_prob_compilation
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
    --benchmarks log_prob_compilation \
    --targets L_ch0 \
    --modes FAST_RUN \
    --n-runs 30 \
    --plot
```

---

## Command-line Arguments

| Argument | Description |
|----------|-------------|
| `--workspaces` | Workspace files to benchmark. |
| `--targets` | Model targets passed to `Workspace.model(...)`. |
| `--modes` | PyTensor compilation modes. |
| `--n-runs` | Number of repeated compilation measurements. |
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
└── log_prob_compilation/
    └── log_prob_compilation_result.json
```

and, when plotting is enabled,

```text
docs/
└── assets/
    └── plots/
        └── log_prob_compilation/
            ├── log_prob_compilation_wall_time.png
            ├── log_prob_compilation_current_rss_delta.png
            └── log_prob_compilation_peak_rss_delta.png
```

The report structure and output conventions are documented in **Benchmark Results**.

---

## Results

### Wall-Time Comparison

![Log probability compilation wall time](../assets/plots/log_prob_compilation/log_prob_compilation_wall_time.png)

Compilation requires approximately **0.40–0.68 s** across the benchmark workspace collection.

The fastest observed configuration is the **10-channel** workspace without nuisance parameters (~398 ms), while the single-channel benchmark requires approximately **680 ms**. The remaining workspaces compile in roughly **535–585 ms**.

Compared with symbolic graph construction, compilation is substantially more expensive because JAX transforms the symbolic graph into executable machine code.

---

### Current RSS Memory

![Log probability compilation current RSS](../assets/plots/log_prob_compilation/log_prob_compilation_current_rss_delta.png)

Compilation consistently allocates approximately **143–146 MB** of additional resident memory.

The memory footprint is largely independent of workspace complexity, indicating that compilation overhead is dominated by the compilation process itself.

---

### Peak RSS Memory

![Log probability compilation peak RSS](../assets/plots/log_prob_compilation/log_prob_compilation_peak_rss_delta.png)

Peak RSS closely follows the current RSS measurements.

The benchmark shows a nearly constant compilation memory footprint across all benchmark workspaces.

---

## Implementation Notes

The benchmark includes several implementation choices that improve measurement quality.

- Workspace loading is excluded from the reported timings.
- Model creation and symbolic graph construction are treated as setup steps.
- Each timing iteration compiles a freshly constructed graph.
- The compiled function is validated before results are recorded.

The general benchmark methodology is documented in **Benchmark Methodology**.

---

## Limitations

This benchmark measures only JAX compilation of the symbolic log-probability graph.

It does **not** measure

- workspace loading;
- model creation;
- symbolic graph construction;
- compiled execution;
- PDF evaluation;
- likelihood evaluation.

These workflow stages are benchmarked separately.

---

## Related Documentation

See also

- **Log-Probability Construction**
- **Compiled Evaluation**
- **Benchmark Methodology**
- **Benchmark Results**
- **Workspace Lifecycle**
