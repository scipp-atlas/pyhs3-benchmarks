# Workspace Loading

On this page, you will learn what the **Workspace Loading** benchmark measures, how to run it, and how to interpret its results.

The **Workspace Loading** benchmark measures the time and memory required to deserialize an HS3 workspace into an in-memory `Workspace` object.

Unlike later workflow benchmarks, it measures only the deserialization stage. Model creation, graph construction, compilation, numerical evaluation, and fitting are benchmarked separately.

---

# What This Benchmark Measures

For each workspace, the benchmark reports

- mean wall time;
- median wall time;
- standard deviation;
- current RSS memory increase;
- peak RSS memory increase;
- workspace validation status.

Timing statistics are collected over repeated executions, while memory measurements are obtained from a clean workspace load. Details of the measurement methodology are described in **Benchmark Methodology**.

---

# Benchmark Workflow

For every workspace, the benchmark performs the following steps.

```text
Workspace JSON
       │
       ▼
Validate Input
       │
       ▼
Measure Initial Memory
       │
       ▼
Workspace.load(...)
       │
       ├────────► Validation
       │
       ├────────► Timing Statistics
       │
       └────────► Memory Statistics
       │
       ▼
JSON Report
       │
       ▼
Comparison Plots (optional)
```

Each workspace is benchmarked independently to minimize interference between benchmark runs.

---

# When to Use This Benchmark

This benchmark is useful for

- comparing workspace loading performance;
- measuring startup overhead;
- evaluating memory consumption during deserialization;
- detecting loading regressions;
- comparing workspaces of different complexity.

---

# Running the Benchmark

## Run directly

```bash
pixi run python -m src.run_workspace_loading \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --n-runs 30 \
    --output-dir results/docs_examples/workspace_loading \
    --plot \
    --plot-dir docs/assets/plots/workspace_loading
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
    --benchmarks workspace_loading \
    --n-runs 30 \
    --plot
```

---

# Command-line Arguments

| Argument | Description |
|----------|-------------|
| `--workspaces` | One or more workspace files to benchmark. |
| `--n-runs` | Number of repeated timing measurements. |
| `--output-dir` | Directory for benchmark reports. |
| `--output-name` | Output JSON filename. |
| `--plot` | Generate comparison plots. |
| `--plot-dir` | Directory for generated figures. |
| `--plot-name` | Wall-time plot filename. |

Common benchmark arguments and execution behavior are described in **Benchmark Methodology**.

---

# Generated Outputs

The benchmark produces

```text
results/
└── workspace_loading/
    └── workspace_loading_result.json
```

and, when plotting is enabled,

```text
docs/
└── assets/
    └── plots/
        └── workspace_loading/
            ├── workspace_loading_wall_time.png
            ├── workspace_loading_current_rss_delta.png
            └── workspace_loading_peak_rss_delta.png
```

The report structure and output conventions are documented in **Benchmark Results**.

---

# Results

## Wall-Time Comparison

![Workspace loading wall time](../assets/plots/workspace_loading/workspace_loading_wall_time.png)

The wall-time comparison shows the average loading time for each benchmark workspace.

For the example benchmark dataset,

- the single-channel workspace loads in approximately **2.6 ms**;
- the 3-channel workspace in approximately **3.4 ms**;
- the 5- and 10-channel workspaces in approximately **5–6 ms**;
- the 30-channel workspace in approximately **15 ms**.

Loading time increases with workspace complexity because larger workspaces contain more statistical objects that must be reconstructed during deserialization.

Error bars represent one standard deviation across repeated benchmark executions.

---

## Peak RSS Memory

![Peak RSS memory](../assets/plots/workspace_loading/workspace_loading_peak_rss_delta.png)

Peak RSS measures the maximum resident memory reached while loading each workspace.

The benchmark shows a gradual increase from approximately **2.3 MB** for the smallest workspace to roughly **3.5 MB** for the largest benchmark workspace, reflecting the additional temporary memory required to reconstruct larger statistical models.

---

## Current RSS Memory

![Current RSS memory](../assets/plots/workspace_loading/workspace_loading_current_rss_delta.png)

Current RSS measures resident memory immediately after successful loading.

The close agreement between current and peak RSS indicates that most allocated memory remains associated with the loaded workspace rather than temporary intermediate allocations.

---

# Implementation Notes

The benchmark includes several implementation choices that improve measurement quality.

- Each workspace is benchmarked in a separate Python process.
- Workspace validation is performed before results are recorded.
- Timing and memory measurements are collected independently.

The general benchmark methodology is documented in **Benchmark Methodology**.

---

# Limitations

This benchmark measures only HS3 workspace deserialization.

It does **not** measure

- model creation;
- graph construction;
- graph optimization;
- compilation;
- probability density evaluation;
- likelihood evaluation;
- fitting.

These stages are covered by dedicated benchmark pages.

---

# Related Documentation

See also

- **Benchmark Methodology**
- **Benchmark Results**
- **Benchmark Matrix Runner**
- **Workspace Lifecycle**
- **Model Creation**
