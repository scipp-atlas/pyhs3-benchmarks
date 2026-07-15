# NLL Scan

On this page, you will learn what the **NLL Scan** benchmark measures, how to run it, and how to interpret its results.

The **NLL Scan** benchmark measures the performance of evaluating a compiled negative log-likelihood over a parameter scan.

Unlike the **PDF Evaluation** benchmark, which repeatedly evaluates a single probability density, this benchmark measures repeated evaluations of the compiled log-probability while varying one parameter of interest (`mu_sig`) across a predefined scan range.

---

# What This Benchmark Measures

For each benchmark configuration, the benchmark reports

- total scan runtime;
- runtime per scan point;
- current RSS memory increase;
- peak RSS memory increase;
- numerical validation status.

The benchmark also verifies that

- every NLL value is finite;
- the requested number of scan points is produced;
- the minimum NLL value is identified successfully.

Details of the measurement methodology are described in **Benchmark Methodology**.

---

# Benchmark Workflow

```text
Workspace
      │
      ▼
Load Workspace
      │
      ▼
Create Model
      │
      ▼
Compile Log-Probability
      │
      ▼
Generate Scan Points
      │
      ▼
Evaluate NLL
      │
      ├────────► Runtime
      ├────────► Runtime per Point
      ├────────► Memory
      └────────► Validation
      │
      ▼
JSON Report
      │
      ▼
Comparison Plots (optional)
```

Workspace loading, model construction, and graph compilation are setup stages and are excluded from the reported scan timing.

---

# When to Use This Benchmark

This benchmark is useful for

- evaluating likelihood scan performance;
- studying scaling with scan resolution;
- measuring runtime per scan point;
- detecting performance regressions;
- evaluating memory usage during parameter scans.

---

# Running the Benchmark

## Run directly

```bash
pixi run python -m src.run_nll_scan \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --targets L_ch0 \
    --modes FAST_RUN \
    --scan-parameter mu_sig \
    --scan-min 0.0 \
    --scan-max 2.0 \
    --n-scan-points 2 10 100 1000 10000 100000 \
    --output-dir results/nll_scan \
    --plot \
    --plot-dir docs/assets/plots/nll_scan
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
    --benchmarks nll_scan \
    --targets L_ch0 \
    --modes FAST_RUN \
    --scan-parameter mu_sig \
    --scan-min 0.0 \
    --scan-max 2.0 \
    --n-scan-points 2 10 100 1000 10000 100000 \
    --plot
```

---

# Command-line Arguments

| Argument | Description |
|----------|-------------|
| `--workspaces` | Workspace files to benchmark. |
| `--targets` | Model targets used for the scan. |
| `--modes` | PyTensor compilation modes. |
| `--scan-parameter` | Parameter varied during the scan. |
| `--scan-min` | Lower bound of the scan range. |
| `--scan-max` | Upper bound of the scan range. |
| `--n-scan-points` | Numbers of scan points to benchmark. |
| `--output-dir` | Directory for benchmark reports. |
| `--output-name` | Output JSON filename. |
| `--plot` | Generate comparison plots. |
| `--plot-dir` | Directory for generated figures. |

Common benchmark arguments and execution behavior are described in **Benchmark Methodology**.

---

# Generated Outputs

The benchmark produces

```text
results/
└── nll_scan/
    └── nll_scan_result.json
```

and, when plotting is enabled,

```text
docs/
└── assets/
    └── plots/
        └── nll_scan/
```

The report structure and output conventions are documented in **Benchmark Results**.

---

# Results

## Total Runtime

![](../assets/plots/nll_scan/nll_scan_total_runtime.png)

Total runtime scales approximately linearly with the number of scan points because each additional point requires one additional evaluation of the compiled likelihood.

The benchmark demonstrates predictable scaling across all benchmark workspaces.

---

## Runtime per Scan Point

![](../assets/plots/nll_scan/nll_scan_runtime_per_point.png)

Runtime per scan point remains nearly constant over several orders of magnitude in scan size.

This indicates that the computational cost of evaluating a single likelihood point is largely independent of the total scan length.

---

## Current RSS Memory

![](../assets/plots/nll_scan/nll_scan_current_rss_delta.png)

Current RSS shows little additional memory usage for small scans.

Memory increases become noticeable only for very large scan configurations due to larger output arrays.

---

## Peak RSS Memory

![](../assets/plots/nll_scan/nll_scan_peak_rss_delta.png)

Peak RSS closely follows the behavior of current RSS.

Even for the largest scans, memory growth remains modest relative to the overall workload.

---

# Implementation Notes

The benchmark includes several implementation choices that improve measurement quality.

- Workspace loading is excluded from the reported timings.
- Model construction and graph compilation are treated as setup.
- Only repeated evaluations of the compiled likelihood are measured.
- Numerical validation is performed before results are recorded.

The general benchmark methodology is documented in **Benchmark Methodology**.

---

# Limitations

This benchmark measures only repeated evaluation of a compiled negative log-likelihood across a parameter scan.

It does **not** measure

- workspace loading;
- model creation;
- graph construction;
- graph optimization;
- graph compilation.

These stages are benchmarked separately.

---

# Related Documentation

See also

- **Benchmark Methodology**
- **Benchmark Results**
- **Benchmark Matrix Runner**
- **Workspace Lifecycle**
- **Compiled Evaluation**
- **PDF Evaluation**
