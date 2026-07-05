# Graph Canonicalization Benchmark

## Overview

This benchmark measures the cost of applying PyTensor's **canonicalization rewrites** to the computational graph produced from a pyHS3 likelihood model.

Canonicalization is one of the earliest optimization stages performed by PyTensor before function compilation. During this pass, algebraically equivalent expressions are rewritten into a canonical form, redundant operations are simplified, and graph structure is normalized to enable later optimization passes.

Unlike the previous benchmark (`log_prob_construction`), this benchmark operates directly on an already constructed `FunctionGraph`. It isolates the performance of the canonicalization stage itself.

For every benchmark run the following metrics are collected:

- graph canonicalization wall time,
- current RSS memory increase,
- peak RSS memory increase,
- reduction in the number of Apply nodes after canonicalization.

---

## Running the benchmark

### Standard benchmark

```bash
pixi run python -m src.run_graph_canonicalization \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --targets L_ch0 \
    --modes FAST_RUN \
    --n-runs 30 \
    --output-dir results/docs_examples/graph_canonicalization \
    --plot \
    --plot-dir docs/assets/plots/graph_canonicalization
```

### Using the benchmark runner

```bash
pixi run python -m src.run_all_benchmarks \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --benchmarks graph_canonicalization \
    --targets L_ch0 \
    --modes FAST_RUN \
    --n-runs 30 \
    --plot
```

---

## Implementation

The benchmark performs the following steps:

1. Load the workspace.
2. Construct the symbolic log-probability graph.
3. Build a fresh `FunctionGraph`.
4. Measure memory before canonicalization.
5. Apply the PyTensor `canonicalize` rewrite database.
6. Measure execution time.
7. Measure memory after canonicalization.
8. Record the reduction in Apply nodes.

Each timing benchmark rebuilds the graph from scratch to ensure independent measurements.

---

## Results

### Wall time

![Graph canonicalization wall time](assets/plots/graph_canonicalization/graph_canonicalization_wall_time.png)

Canonicalization requires approximately **313–528 ms**, depending on the workspace.

| Workspace | Mean wall time (ms) |
|-----------|--------------------:|
| 1 channel | **527.5 ± 4.8** |
| 3 channels | **451.1 ± 5.4** |
| 5 channels | **477.0 ± 5.0** |
| 10 channels | **313.5 ± 3.0** |
| 30 channels | **453.1 ± 5.1** |

The 10-channel workspace completes canonicalization fastest because its graph contains substantially fewer Apply nodes after disabling nuisance parameters.

Overall, canonicalization contributes only a few hundred milliseconds to the compilation pipeline.

---

### Current RSS increase

![Current RSS](assets/plots/graph_canonicalization/graph_canonicalization_current_rss_delta.png)

Canonicalization allocates very little additional memory.

| Workspace | Current RSS increase (MB) |
|-----------|--------------------------:|
| 1 channel | **13.87** |
| 3 channels | **11.32** |
| 5 channels | **13.56** |
| 10 channels | **13.27** |
| 30 channels | **12.68** |

Across all benchmark configurations the additional resident memory remains close to **12–14 MB**.

---

### Peak RSS increase

![Peak RSS](assets/plots/graph_canonicalization/graph_canonicalization_peak_rss_delta.png)

Peak RSS follows nearly the same trend as current RSS.

| Workspace | Peak RSS increase (MB) |
|-----------|-----------------------:|
| 1 channel | **13.61** |
| 3 channels | **11.08** |
| 5 channels | **13.72** |
| 10 channels | **12.98** |
| 30 channels | **12.62** |

No benchmark exhibits excessive temporary allocations during canonicalization.

---

## Graph simplification

Besides timing and memory, this benchmark validates that canonicalization successfully simplifies the computational graph.

Typical reductions observed during benchmarking include:

| Workspace | Apply nodes before | Apply nodes after | Reduction |
|-----------|-------------------:|------------------:|----------:|
| 1 channel | 101 | 51 | **−50** |
| 3 channels | 94 | 48 | **−46** |
| 5 channels | 97 | 48 | **−49** |
| 10 channels | 80 | 29 | **−51** |
| 30 channels | 94 | 48 | **−46** |

Across all tested workspaces, canonicalization removes roughly **half of the Apply nodes**, confirming that the rewrite pass effectively simplifies the symbolic computation graph before later optimization stages.

---

## Summary

The graph canonicalization benchmark demonstrates that PyTensor's canonicalization stage is lightweight.

Across representative pyHS3 workspaces:

- execution time remains below **0.55 s**,
- memory overhead stays around **12–14 MB**,
- peak memory closely matches current RSS,
- approximately **50% of Apply nodes** are eliminated during graph simplification.

These results indicate that canonicalization introduces only a modest compilation cost while substantially simplifying the symbolic graph for subsequent optimization passes.
