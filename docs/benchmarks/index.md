# Benchmark Suite

The PyHS3 Benchmarks repository provides a comprehensive collection of benchmarks covering the complete statistical model lifecycle.

Rather than focusing on a single operation, the benchmark suite evaluates every major stage involved in preparing, optimizing, and evaluating statistical models represented using the HS3 format.

All benchmark suites follow the common benchmarking methodology described in the **Benchmark Methodology** guide and operate on a common collection of benchmark workspaces unless stated otherwise.

---

# Benchmark Categories

The benchmark suite is organized into three major categories.

## Workflow Benchmarks

Workflow benchmarks measure the cost of individual stages in the statistical model lifecycle.

These benchmarks isolate specific operations, making it possible to identify performance bottlenecks, compare implementations, and evaluate optimization opportunities throughout the model execution pipeline.

## Cross-Framework Benchmarks

Cross-framework benchmarks compare PyHS3 against other statistical inference frameworks.

These benchmarks evaluate both numerical agreement and execution performance across multiple implementations.

## Reporting

Every benchmark produces one or more of the following artifacts:

- structured JSON reports;
- visualization plots (optional);
- benchmark metadata;
- validation summaries (where applicable).

Large benchmark campaigns additionally generate benchmark overview reports summarizing the entire execution.

---

# Benchmark Pipeline

The benchmark suite follows the typical statistical model lifecycle.

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

The pipeline reflects the order in which statistical models are typically prepared, optimized, compiled, and evaluated during an analysis.

---

# Workflow Benchmarks

| Benchmark | Measures | Input | Outputs |
|------------|----------|-------|---------|
| Workspace Loading | HS3 workspace deserialization | One or more HS3 workspaces | JSON report, wall-time plot, RSS memory plots |
| Model Creation | Statistical model construction | One or more HS3 workspaces | JSON report, timing plots |
| Log-Probability Construction | Computational graph construction | One or more HS3 workspaces | JSON report, timing plots |
| Log-Probability Compilation | Graph compilation for execution | One or more HS3 workspaces | JSON report, timing plots |
| Graph Canonicalization | Graph normalization | One or more HS3 workspaces | JSON report |
| Graph Optimization | Graph optimization passes | One or more HS3 workspaces | JSON report |
| Compiled Evaluation | Execution of compiled likelihoods | One or more HS3 workspaces | JSON report, timing plots |
| PDF Evaluation | Probability density evaluation | One or more HS3 workspaces | JSON report, timing plots |
| NLL Scan | Negative log-likelihood scans | One or more HS3 workspaces | JSON report, scan plots |
| Memory Scaling | Memory consumption versus model size | Multiple HS3 workspaces | JSON report, scaling plots |
| Model Complexity Scaling | Runtime scaling with model complexity | Multiple HS3 workspaces | JSON report, scaling plots |

---

# Cross-Framework Benchmarks

| Benchmark | Frameworks | Purpose |
|------------|------------|---------|
| PyHS3 vs xRooFit | PyHS3, xRooFit | Compare end-to-end workflow performance |
| Scalar PDF Evaluation | PyHS3, RooFit, PyHF | Compare scalar PDF evaluation performance |
| Vectorized PDF Evaluation | PyHS3, RooFit | Compare vectorized PDF evaluation performance |
| Binned Likelihood Evaluation | PyHS3, RooFit, PyHF | Compare binned likelihood evaluation |
| NLL Scan | PyHS3, RooFit | Compare likelihood scan performance |
| Model Complexity Scaling | Multiple frameworks | Compare scalability across increasingly complex statistical models |

---

# Running Benchmarks

All benchmark suites can be executed through the benchmark runner.

For example, run a single benchmark:

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

or execute the complete benchmark suite:

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

Each benchmark page documents benchmark-specific command-line options, generated artifacts, output formats, and result interpretation.

---

# Related Documentation

The documentation is organized as follows:

- **Benchmark Methodology** — common benchmarking principles, measurement strategy, and execution methodology.
- **Workflow Benchmarks** — detailed documentation for each benchmark in the statistical model lifecycle.
- **Cross-Framework Benchmarks** — framework comparison methodology and benchmark results.

Individual benchmark pages provide detailed descriptions of:

- benchmark objectives;
- execution workflow;
- command-line interface;
- generated outputs;
- JSON result format;
- performance plots;
- implementation details;
- benchmark limitations.
