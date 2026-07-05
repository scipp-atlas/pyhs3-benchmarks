## Model Complexity Scaling

The `model_complexity_scaling` benchmark evaluates how benchmark metrics change as the complexity of the statistical model increases. Unlike the stage-by-stage benchmark, this benchmark compares multiple representative workspaces of different sizes and structures, allowing trends in setup time, evaluation performance, and memory usage to be observed.

### Workspaces

The benchmark was run on five representative HS3 workspaces:

- `1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x`
- `3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x`
- `5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x`
- `10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x`
- `30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x`

Each benchmark executes the complete workflow:

1. Workspace loading
2. Model creation
3. Log-probability construction
4. Log-probability compilation
5. Compiled log-probability evaluation
6. PDF evaluation
7. NLL scan

---

### Total setup time

The total setup time is the sum of:

- workspace loading,
- model creation,
- log-probability construction,
- log-probability compilation.

It represents the one-time initialization cost before repeated model evaluations can be performed.

![Total setup time](../assets/plots/model_complexity_scaling/model_complexity_total_setup_time.png)

As expected, larger workspaces require longer initialization times. The 30-channel workspace has the highest setup cost because of its substantially larger computational graph.

---

### Compiled evaluation time

This metric measures the average execution time of one compiled log-probability evaluation after JAX compilation has completed.

![Compiled evaluation time](../assets/plots/model_complexity_scaling/model_complexity_compiled_evaluation_time.png)

Despite significant differences in model size, compiled evaluation time remains close to 3–4 ms per evaluation, demonstrating that JAX compilation keeps execution performance relatively stable across different model complexities.

---

### PDF evaluation time

This benchmark measures the average runtime of evaluating a single probability density function.

![PDF evaluation time](../assets/plots/model_complexity_scaling/model_complexity_pdf_evaluation_time.png)

PDF evaluation remains extremely fast for all tested workspaces, with runtimes on the order of only a few microseconds. The small variation indicates that individual PDF evaluations contribute negligibly to the overall benchmark runtime.

---

### NLL scan time

This benchmark measures the average runtime required to evaluate one point during a negative log-likelihood scan.

![NLL scan time](../assets/plots/model_complexity_scaling/model_complexity_nll_scan_time.png)

NLL scan performance also remains remarkably stable across the tested workspaces, with runtimes of roughly 3–5 ms per scan point.

---

### Peak memory usage

This metric reports the total peak RSS increase accumulated during benchmark execution.

![Peak RSS delta](../assets/plots/model_complexity_scaling/model_complexity_peak_rss_delta.png)

Peak memory consumption changes only slightly across the selected workspaces. Most of the memory allocation occurs during JAX compilation, while increasing the number of channels has only a modest effect on the total peak RSS observed during the benchmark.

---

### Summary

The benchmark demonstrates several important characteristics of the current implementation:

- setup time increases noticeably as model complexity grows;
- compiled evaluation performance remains nearly constant across representative workspaces;
- PDF evaluation is consistently very fast;
- NLL scan performance shows little dependence on model size;
- peak memory usage is dominated by the compilation stage rather than by the number of channels.

Overall, the benchmark indicates that increasing workspace complexity primarily affects model initialization, while runtime evaluation performance remains relatively stable after compilation.
