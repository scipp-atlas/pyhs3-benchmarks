# Log-Probability Construction

Constructing the symbolic log-probability graph is the first computational step after model creation.

This benchmark measures the cost of accessing `model.log_prob`, which builds the symbolic PyTensor likelihood expression without compiling or evaluating it.

The benchmark isolates graph construction from both model creation and graph compilation.

---

# What is measured

The benchmark measures only the execution of

```python
log_prob = model.log_prob
```

The following operations are performed before timing begins:

- loading the workspace;
- creating the statistical model.

The following operations are **not** included:

- graph compilation;
- likelihood evaluation;
- optimization passes;
- numerical execution.

---

# Running the benchmark

## Individual benchmark

```bash
pixi run python -m src.run_log_prob_construction \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --targets L_ch0 \
    --modes FAST_RUN \
    --n-runs 30 \
    --output-dir results/log_prob_construction \
    --plot \
    --plot-dir docs/assets/plots/log_prob_construction
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
    --benchmarks log_prob_construction \
    --targets L_ch0 \
    --modes FAST_RUN \
    --n-runs 30 \
    --plot
```

---

# Benchmark outputs

The benchmark produces

- wall time measurements;
- current RSS increase;
- peak RSS increase;
- validation information describing the constructed symbolic graph;
- JSON result file;
- optional plots.

---

# Validation

Each benchmark run verifies that

- `model.log_prob` is successfully constructed;
- the returned object is a PyTensor `TensorVariable`;
- the graph has a valid dtype and dimensionality;
- the graph is ready for compilation.

---

# Example results

## Wall time

![](../assets/plots/log_prob_construction/log_prob_construction_wall_time.png)

Graph construction is inexpensive, requiring approximately **5–10 ms** across the tested workspaces.

The 10-channel workspace is noticeably faster (~5 ms), while the remaining workspaces consistently complete in about 9.7 ms.

---

## Current RSS increase

![](../assets/plots/log_prob_construction/log_prob_construction_current_rss_delta.png)

Constructing the symbolic graph allocates almost no additional memory.

The measured RSS increase remains below **0.05 MB** for every workspace.

---

## Peak RSS increase

![](../assets/plots/log_prob_construction/log_prob_construction_peak_rss_delta.png)

Peak memory usage is effectively unchanged during graph construction.

Only the smallest workspace exhibits a measurable increase (approximately **0.125 MB**), while all remaining workspaces show no observable peak RSS growth.

---

# Interpretation

This benchmark demonstrates that symbolic log-probability construction is both fast and memory efficient.

Since only the computational graph is created, the cost remains nearly independent of workspace complexity and is substantially smaller than later stages such as graph compilation or likelihood evaluation.
