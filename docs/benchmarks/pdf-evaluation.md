# PDF Evaluation Benchmark

Measures the performance of repeated `model.pdf(...)` evaluation for a selected probability distribution.

The benchmark separates the first (cold-start) evaluation from repeated warm evaluations to distinguish initialization overhead from steady-state execution.

## What is measured

For every benchmark configuration the following metrics are collected:

- cold-start evaluation time;
- average warm evaluation time;
- warm throughput;
- current RSS memory delta;
- peak RSS memory delta;
- output stability.

The benchmark also verifies that

- every PDF value is finite;
- repeated evaluations produce numerically stable results.

---

## Running

Example:

```bash
pixi run python -m src.run_pdf_evaluation \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --targets L_ch0 \
    --modes FAST_RUN \
    --distributions sig_ch0 \
    --n-evaluations 1 10 100 1000 10000 \
    --output-dir results/docs_examples/pdf_evaluation \
    --plot \
    --plot-dir docs/assets/plots/pdf_evaluation
```

---

---

## Command-line Arguments

The benchmark supports the following command-line arguments.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--workspaces` | `Path ...` | `DEFAULT_WORKSPACE` | One or more HS3 workspace JSON files to benchmark. Each workspace is loaded once before PDF evaluation begins. |
| `--targets` | `str ...` | `DEFAULT_TARGET` | One or more model targets (for example, analysis or likelihood names) used when constructing the PyHS3 model. |
| `--modes` | `str ...` | `DEFAULT_MODE` | One or more PyTensor compilation modes passed to `Workspace.model(...)`. Each mode is benchmarked independently. |
| `--distributions` | `str ...` | `sig_ch0` | One or more probability distributions to evaluate using `model.pdf(...)`. Each distribution is benchmarked separately. |
| `--n-evaluations` | `int ...` | `1 10 100 1000 10000` | Numbers of repeated warm PDF evaluations to perform. A separate benchmark result is generated for each evaluation count. |
| `--output-dir` | `Path` | `results/pdf_evaluation/` | Directory where the benchmark JSON results will be written. |
| `--output-name` | `str` | `pdf_evaluation_result.json` | Name of the JSON file containing the benchmark results. |
| `--plot` | flag | disabled | Generate comparison plots after the benchmark completes. |
| `--plot-dir` | `Path` | `docs/assets/plots/pdf_evaluation/` | Directory where generated benchmark plots will be saved. |

## Notes

- At least one workspace, target, mode, distribution, and evaluation count must be provided.
- A separate benchmark is executed for every combination of workspace, target, mode, distribution, and number of evaluations.
- The benchmark measures the first `model.pdf(...)` call separately from repeated warm evaluations in order to distinguish initialization overhead from steady-state performance.
- Workspace loading and model creation are treated as setup steps and are excluded from the reported timing measurements.
- Memory plots are generated only when at least one benchmark reports a non-zero RSS increase.

---

## Results

The benchmark writes

```
pdf_evaluation_result.json
```

containing one result for every combination of

- workspace;
- target;
- execution mode;
- distribution;
- number of evaluations.

Each result contains

- cold-start timing;
- warm timing;
- throughput;
- memory statistics;
- numerical validation results.

---

## Plots

### Cold-start time

Shows the execution time of the first `model.pdf(...)` call.

![PDF cold start](../assets/plots/pdf_evaluation/pdf_evaluation_cold_start_time_grouped.png)

---

### Average warm evaluation time

Average wall time per evaluation after the initial cold-start call.

![Average warm evaluation time](../assets/plots/pdf_evaluation/pdf_evaluation_average_time_lines.png)

---

### Warm throughput

Number of PDF evaluations executed per second during repeated execution.

![Warm throughput](../assets/plots/pdf_evaluation/pdf_evaluation_throughput_lines.png)

---

### Current RSS delta

Memory growth measured using current resident set size.

The plot is generated only when at least one benchmark exhibits a non-zero RSS increase.

![Current RSS delta](../assets/plots/pdf_evaluation/pdf_evaluation_current_rss_delta_grouped.png)
