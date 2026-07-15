# Benchmark Suite

On this page, you will learn how the benchmark suite is organized and where to find documentation for each benchmark category.

The PyHS3 Benchmarks repository provides a comprehensive benchmark suite covering the complete lifecycle of statistical models, from workspace loading to compiled evaluation and cross-framework comparisons.

All benchmarks follow the common methodology described in **Benchmark Methodology** and use the benchmark workspace collection documented in **Benchmark Workspaces**, unless stated otherwise.

---

# Benchmark Categories

The documentation is organized into three main sections.

## Workflow Benchmarks

Workflow benchmarks measure individual stages of the PyHS3 execution pipeline.

Each benchmark isolates one workflow stage, making it possible to understand initialization costs, execution performance, memory usage, and scalability independently.

## Cross-Framework Benchmarks

Cross-framework benchmarks compare PyHS3 with equivalent implementations in other statistical frameworks.

These benchmarks evaluate both numerical agreement and execution performance using equivalent statistical models.

## Benchmark Results

Benchmark reports can be combined into summary figures that compare workflow stages, memory usage, scalability, and cross-framework performance.

---

# Benchmark Pipeline

The workflow benchmarks follow the statistical model lifecycle.

```text
HS3 Workspace
      │
      ▼
Workspace Loading
      │
      ▼
Model Creation
      │
      ▼
Log-Probability Construction
      │
      ▼
Graph Canonicalization
      │
      ▼
Graph Optimization
      │
      ▼
Log-Probability Compilation
      │
      ▼
Compiled Evaluation
      │
      ├────────────► PDF Evaluation
      │
      ├────────────► NLL Scan
      │
      ├────────────► Memory Scaling
      │
      └────────────► Cross-Framework Benchmarks
```

Each workflow stage has a dedicated benchmark page describing what is measured, how to run the benchmark, and how to interpret the results.

---

# Workflow Benchmarks

| Benchmark | Purpose |
|------------|---------|
| Workspace Loading | Measure HS3 workspace deserialization. |
| Model Creation | Measure statistical model construction. |
| Log-Probability Construction | Measure symbolic graph construction. |
| Graph Canonicalization | Measure canonicalization rewrites. |
| Graph Optimization | Measure graph optimization passes. |
| Log-Probability Compilation | Measure JAX compilation. |
| Compiled Evaluation | Measure execution of compiled log-probability graphs. |
| PDF Evaluation | Measure repeated PDF evaluation. |
| NLL Scan | Measure repeated likelihood scans. |
| Memory Scaling | Compare memory usage across workflow stages. |
| Model Complexity Scaling | Compare performance as model complexity increases. |

---

# Cross-Framework Benchmarks

| Benchmark | Purpose |
|------------|---------|
| PyHS3 vs xRooFit | Compare end-to-end workflow performance. |
| Scalar PDF Evaluation | Compare scalar probability density evaluation. |
| Vectorized PDF Evaluation | Compare vectorized PDF evaluation. |
| Binned Likelihood Evaluation | Compare HistFactory likelihood evaluation. |
| NLL Scan | Compare likelihood scan performance. |
| Model Complexity Scaling | Compare scalability across statistical frameworks. |

---

# Running Benchmarks

Benchmark campaigns are typically executed using the Benchmark Matrix Runner.

Run a single benchmark:

```bash
pixi run python -m src.run_all_benchmarks \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --benchmarks workspace_loading
```

Run the complete benchmark suite:

```bash
pixi run python -m src.run_all_benchmarks \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --benchmarks all
```

Additional execution options are documented in **Benchmark Matrix Runner**.

---

# Where to Go Next

Continue with the following documentation sections:

- **Benchmark Methodology** — common benchmarking principles and measurement strategy.
- **Benchmark Workspaces** — benchmark datasets and workspace catalog.
- **Workflow Benchmarks** — detailed documentation for each workflow stage.
- **Cross-Framework Benchmarks** — comparisons with RooFit, pyhf, and xRooFit.
- **Benchmark Results** — interpreting benchmark reports and summary figures.
