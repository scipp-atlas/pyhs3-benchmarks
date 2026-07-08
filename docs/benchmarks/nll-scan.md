# NLL Scan

The NLL scan benchmark measures the performance of evaluating a negative log-likelihood over a parameter scan.

Unlike the PDF evaluation benchmark, this benchmark repeatedly evaluates the compiled log-probability while varying a single parameter of interest (`mu_sig`) across a predefined scan range.

For each workspace and scan resolution, the benchmark measures

- total scan runtime;
- runtime per scan point;
- current RSS increase;
- peak RSS increase;
- validation of the produced NLL curve.

---

# What is measured

For each benchmark run, the following operations are performed:

1. load the workspace;
2. construct the statistical model;
3. build and compile the log-probability graph;
4. generate uniformly spaced scan points;
5. evaluate the NLL at every scan point.

The benchmark reports only the scan performance after compilation.

---

# Running the benchmark

## Individual benchmark

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

## Using the benchmark runner

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

---

# Command-line Arguments

The benchmark supports the following command-line arguments.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--workspaces` | `Path ...` | `DEFAULT_WORKSPACE` | One or more HS3 workspace JSON files to benchmark. Each workspace is benchmarked independently. |
| `--targets` | `str ...` | `DEFAULT_TARGET` | One or more model targets (for example, analysis or likelihood names) used to construct the statistical model before performing the scan. |
| `--modes` | `str ...` | `DEFAULT_MODE` | One or more PyTensor compilation modes used when building the compiled log-probability function. |
| `--scan-parameter` | `str` | `mu_sig` | Name of the model parameter to scan. Its value is varied uniformly across the specified scan range while all other parameters remain fixed. |
| `--scan-min` | `float` | `0.0` | Lower bound of the scan range (inclusive). |
| `--scan-max` | `float` | `5.0` | Upper bound of the scan range (inclusive). Must be greater than `--scan-min`. |
| `--n-scan-points` | `int ...` | `101` | Number of uniformly spaced scan points. A separate benchmark is executed for each specified scan resolution. |
| `--output-dir` | `Path` | `results/nll_scan/` | Directory where the benchmark JSON results will be written. |
| `--output-name` | `str` | `nll_scan_result.json` | Name of the JSON file containing the benchmark results. |
| `--plot` | flag | disabled | Generate comparison plots after the benchmark completes. |
| `--plot-dir` | `Path` | `docs/assets/plots/nll_scan/` | Directory where generated benchmark plots will be stored. |

## Notes

- At least one workspace, target, mode, and scan resolution must be provided.
- A separate benchmark is executed for every combination of workspace, target, mode, and number of scan points.
- `--scan-min` must be smaller than `--scan-max`.
- Every value supplied to `--n-scan-points` must be at least **2**.
- Workspace loading, model construction, log-probability graph creation, and graph compilation are treated as setup steps and are excluded from the reported scan timing.
- The benchmark measures only repeated evaluations of the compiled log-probability across the requested parameter grid.

---

# Benchmark outputs

The benchmark produces

- total scan runtime;
- runtime per scan point;
- current RSS increase;
- peak RSS increase;
- validation information;
- JSON result file;
- optional plots.

---

# Validation

Each benchmark run verifies that

- all computed NLL values are finite;
- the requested number of scan points is produced;
- the minimum NLL value is successfully identified;
- the corresponding scan parameter value is reported;
- the complete scan values and NLL values are stored in the output JSON.

---

# Example results

## Total runtime

![](../assets/plots/nll_scan/nll_scan_total_runtime.png)

As expected, the total runtime scales approximately linearly with the number of scan points since every additional point requires one additional evaluation of the compiled likelihood.

The benchmark demonstrates stable scaling across all tested workspaces, with the largest scans requiring proportionally longer execution times while preserving predictable performance characteristics.

---

## Runtime per scan point

![](../assets/plots/nll_scan/nll_scan_runtime_per_point.png)

The runtime per scan point remains nearly constant over several orders of magnitude in scan size.

This indicates that the computational cost of evaluating an individual likelihood point is essentially independent of the total scan length, confirming good scaling of the evaluation pipeline.

---

## Current RSS increase

![](../assets/plots/nll_scan/nll_scan_current_rss_delta.png)

For small scans, essentially no additional memory is allocated.

Beginning around 1000 scan points, several workspaces allocate a few additional megabytes while constructing larger scan outputs, after which memory usage remains relatively stable.

---

## Peak RSS increase

![](../assets/plots/nll_scan/nll_scan_peak_rss_delta.png)

Peak RSS follows the same behavior as the current RSS measurements.

Most workspaces exhibit negligible peak memory growth for small scans, with only moderate increases appearing for the largest scan configurations.

---

# Interpretation

The benchmark shows that NLL scanning is computationally well behaved.

The runtime grows linearly with the number of scan points, while the cost per evaluation remains nearly constant across different scan sizes. Memory usage is minimal for small scans and increases only modestly for very large scans, making the implementation suitable for large-scale likelihood scans.
