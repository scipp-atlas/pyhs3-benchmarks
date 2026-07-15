# Model Creation

On this page, you will learn what the **Model Creation** benchmark measures, how to run it, and how to interpret its results.

The **Model Creation** benchmark measures the time and memory required to construct a PyHS3 `Model` object from an already loaded HS3 workspace.

Unlike the **Workspace Loading** benchmark, workspace deserialization is excluded from the reported measurements. This benchmark measures only the cost of calling `Workspace.model(...)`, which prepares the statistical model for subsequent graph construction, compilation, likelihood evaluation, and fitting.

---

# What This Benchmark Measures

For each benchmark configuration, the benchmark reports

- mean wall time;
- median wall time;
- standard deviation;
- current RSS memory increase;
- peak RSS memory increase;
- model validation status.

Workspace loading is treated as a setup step and is excluded from all reported measurements.

Details of the measurement methodology are described in **Benchmark Methodology**.

---

# Benchmark Workflow

```text
Workspace
      │
      ▼
Workspace.load(...)
      │
      ▼
Workspace.model(...)
      │
      ├────────► Model Validation
      ├────────► Timing Statistics
      └────────► Memory Statistics
      │
      ▼
JSON Report
      │
      ▼
Comparison Plots (optional)
```

The workspace is loaded once before benchmarking begins. Only model construction is included in the reported timings.

---

# When to Use This Benchmark

This benchmark is useful for

- measuring model construction overhead;
- comparing initialization performance across workspaces;
- evaluating scaling with model complexity;
- detecting performance regressions;
- estimating memory requirements before graph construction.

---

# Running the Benchmark

## Run directly

```bash
pixi run python -m src.run_model_creation \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --targets L_ch0 \
    --modes FAST_RUN \
    --n-runs 30 \
    --output-dir results/docs_examples/model_creation \
    --plot \
    --plot-dir docs/assets/plots/model_creation
```

## Run through the Benchmark Matrix Runner

```bash
pixi run python -m src.run_all_benchmarks \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --benchmarks model_creation \
    --targets L_ch0 \
    --modes FAST_RUN \
    --n-runs 30 \
    --plot
```

---

# Command-line Arguments

| Argument | Description |
|----------|-------------|
| `--workspaces` | Workspace files to benchmark. |
| `--targets` | Model targets passed to `Workspace.model(...)`. |
| `--modes` | PyTensor compilation modes. |
| `--n-runs` | Number of repeated timing measurements. |
| `--output-dir` | Directory for benchmark reports. |
| `--output-name` | Output JSON filename. |
| `--plot` | Generate comparison plots. |
| `--plot-dir` | Directory for generated figures. |
| `--plot-name` | Filename of the wall-time comparison plot. |

Common benchmark arguments and execution behavior are described in **Benchmark Methodology**.

---

# Generated Outputs

The benchmark produces

```text
results/
└── model_creation/
    └── model_creation_result.json
```

and, when plotting is enabled,

```text
docs/
└── assets/
    └── plots/
        └── model_creation/
            ├── model_creation_wall_time.png
            ├── model_creation_current_rss_delta.png
            └── model_creation_peak_rss_delta.png
```

The report structure and output conventions are documented in **Benchmark Results**.

---

# Results

## Wall-Time Comparison

![Model creation wall time](../assets/plots/model_creation/model_creation_wall_time.png)

This benchmark measures the time required to construct a `Model` object from an already loaded workspace.

Model creation scales from approximately **122 ms** for the single-channel workspace to approximately **3.26 s** for the 30-channel benchmark workspace.

Compared with workspace loading, this stage is substantially more expensive because PyHS3 constructs the complete symbolic statistical model before later workflow stages.

Execution time generally increases with workspace complexity, although larger workspaces exhibit greater variability due to the increased complexity of the generated computational graph.

---

## Peak RSS Memory

![Peak RSS memory](../assets/plots/model_creation/model_creation_peak_rss_delta.png)

Peak RSS measures the maximum resident memory reached during model construction.

Memory usage increases steadily with workspace complexity, reflecting the additional symbolic graphs, parameters, and intermediate structures created while building the statistical model.

---

## Current RSS Memory

![Current RSS memory](../assets/plots/model_creation/model_creation_current_rss_delta.png)

Current RSS measures resident memory immediately after model creation completes.

The close agreement between current and peak RSS indicates that most allocated memory belongs to the constructed model rather than temporary allocations released after construction.

---

# Implementation Notes

The benchmark includes several implementation choices that improve measurement quality.

- Workspace loading is excluded from the reported timings.
- Timing and memory measurements are collected independently.
- Each timing measurement constructs a fresh model.
- Memory measurements use isolated model construction.
- Model validation is performed before results are recorded.

The general benchmark methodology is documented in **Benchmark Methodology**.

---

# Limitations

This benchmark measures only statistical model construction.

It does **not** measure

- workspace loading;
- log-probability construction;
- graph canonicalization;
- graph optimization;
- graph compilation;
- likelihood evaluation;
- PDF evaluation;
- fitting.

These workflow stages are benchmarked separately.

---

# Related Documentation

See also

- **Workspace Loading**
- **Log-Probability Construction**
- **Benchmark Methodology**
- **Benchmark Results**
- **Workspace Lifecycle**
