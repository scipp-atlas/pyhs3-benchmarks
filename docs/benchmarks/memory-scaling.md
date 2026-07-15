# Memory Scaling

On this page, you will learn how memory usage is distributed across the PyHS3 workflow and which workflow stages dominate the overall memory footprint.

The **Memory Scaling** benchmark measures memory consumption for each major workflow stage using isolated benchmark processes. Running every stage independently allows memory allocations to be attributed to individual stages without interference from previous computations.

---

# What This Benchmark Measures

For each workflow stage, the benchmark reports

- current RSS increase;
- peak RSS increase;
- current RSS after execution;
- peak RSS after execution.

The following workflow stages are benchmarked independently:

1. Workspace Loading
2. Model Creation
3. Log-Probability Construction
4. Log-Probability Compilation
5. Compiled Evaluation
6. PDF Evaluation
7. NLL Scan

Measurement methodology and execution strategy are described in **Benchmark Methodology**.

---

# Benchmark Workflow

```text
Workspace
      │
      ▼
Execute Workflow Stage
      │
      ▼
Measure Current RSS
      │
      ▼
Measure Peak RSS
      │
      ▼
Validate Results
      │
      ▼
JSON Report
      │
      ▼
Comparison Plots (optional)
```

Each workflow stage executes in a separate Python process to isolate its memory footprint.

---

# When to Use This Benchmark

This benchmark is useful for

- identifying memory-intensive workflow stages;
- comparing memory usage across different workflows;
- measuring compilation overhead;
- tracking memory regressions;
- evaluating memory scaling with increasing model complexity.

---

# Running the Benchmark

## Run directly

```bash
pixi run python -m src.run_memory_scaling \
    --workspaces \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --targets L_ch0 \
    --modes FAST_RUN \
    --distribution sig_ch0 \
    --n-evaluations 100 \
    --scan-parameter mu_sig \
    --scan-min 0.0 \
    --scan-max 5.0 \
    --n-scan-points 101 \
    --plot \
    --plot-dir docs/assets/plots/memory_scaling
```

## Run through the Benchmark Matrix Runner

```bash
pixi run python -m src.run_all_benchmarks \
    --workspaces \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --benchmarks memory_scaling \
    --targets L_ch0 \
    --modes FAST_RUN \
    --distribution sig_ch0 \
    --n-evaluations 100 \
    --scan-parameter mu_sig \
    --scan-min 0.0 \
    --scan-max 5.0 \
    --n-scan-points 101 \
    --plot
```

---

# Command-line Arguments

| Argument | Description |
|----------|-------------|
| `--workspaces` | Workspace files to benchmark. |
| `--targets` | Model targets. |
| `--modes` | PyTensor compilation modes. |
| `--stages` | Workflow stages to profile. |
| `--n-runs` | Number of repeated timing measurements. |
| `--n-evaluations` | Number of repeated evaluations. |
| `--distribution` | Distribution used during PDF evaluation. |
| `--scan-parameter` | Parameter scanned during the NLL benchmark. |
| `--scan-min` | Lower scan bound. |
| `--scan-max` | Upper scan bound. |
| `--n-scan-points` | Number of scan points. |
| `--output-dir` | Directory for benchmark reports. |
| `--output-name` | Output JSON filename. |
| `--plot` | Generate comparison figures. |
| `--plot-dir` | Directory for generated plots. |

Common benchmark arguments and execution behavior are described in **Benchmark Methodology**.

---

# Generated Outputs

The benchmark produces

```text
results/
└── memory_scaling/
    └── memory_scaling_result.json
```

and, when plotting is enabled,

```text
docs/
└── assets/
    └── plots/
        └── memory_scaling/
```

The report structure and output conventions are documented in **Benchmark Results**.

---

# Results

## Current RSS Increase

![](../assets/plots/memory_scaling/memory_scaling_current_rss_delta.png)

Current RSS remains small for most workflow stages.

The dominant allocation occurs during **Log-Probability Compilation**, where JAX compilation allocates approximately **140 MB** of additional memory. Workspace loading, model creation, compiled evaluation, PDF evaluation, and NLL scanning contribute comparatively little to overall memory growth.

---

## Peak RSS Increase

![](../assets/plots/memory_scaling/memory_scaling_peak_rss_delta.png)

Peak RSS closely follows the current RSS measurements.

Again, compilation is responsible for nearly all additional memory allocation, while the remaining workflow stages exhibit only modest increases.

---

## Peak RSS After Each Workflow Stage

![](../assets/plots/memory_scaling/memory_scaling_peak_rss_after.png)

The highest memory footprint is observed immediately after log-probability compilation.

Subsequent compiled evaluations, PDF evaluations, and NLL scans reuse the compiled representation without introducing substantial additional allocations.

---

# Key Observations

The benchmark highlights several important characteristics of the current implementation.

- Memory consumption is dominated by log-probability compilation.
- Workspace loading and model construction have relatively small memory footprints.
- Runtime evaluation stages reuse the compiled graph with minimal additional allocations.
- Compilation represents the primary one-time memory cost of the workflow.

---

# Limitations

This benchmark measures memory usage for individual workflow stages executed in isolation.

It is intended for comparative analysis of memory behavior rather than detailed runtime performance. Timing-focused measurements are documented in the corresponding benchmark pages.

---

# Related Documentation

See also

- **Benchmark Methodology**
- **Benchmark Results**
- **Workspace Lifecycle**
- **Log-Probability Compilation**
- **Compiled Evaluation**
- **Model Complexity Scaling**
