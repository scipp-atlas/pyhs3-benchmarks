# Benchmark Workspaces

The benchmark workspace collection is the canonical dataset used throughout the PyHS3 Benchmarks repository.

Unless stated otherwise, every workflow benchmark, scaling benchmark, and cross-framework benchmark operates on this collection. Using a common benchmark dataset ensures that benchmark results are directly comparable and that performance differences reflect implementation changes rather than differences in benchmark inputs.

---

# Overview

The benchmark workspace collection contains statistical models spanning a broad range of configurations and complexities.

The workspaces vary in

- number of analysis channels;
- background parameterization;
- signal model;
- signal shape configuration;
- nuisance parameter configuration;
- constraint model;
- signal yield.

Together, these configurations provide representative benchmark inputs covering a wide spectrum of statistical models.

---

# Workspace Naming Convention

Each workspace filename completely describes the statistical model it contains.

The naming convention follows the grammar

```text
<channels>_<background>_<signal>_<shape>_<nuisance>_<constraints>_<yield>
```

For example,

```text
10ch_bkgGenExp_sigGeneric_shapeFloat_npOn_constrGauss_yield1x
```

is interpreted as

| Component | Description |
|-----------|-------------|
| `10ch` | Ten analysis channels |
| `bkgGenExp` | Generated exponential background model |
| `sigGeneric` | Generic signal model |
| `shapeFloat` | Floating signal-shape parameters |
| `npOn` | Nuisance parameters enabled |
| `constrGauss` | Gaussian constraints |
| `yield1x` | Nominal signal yield |

This naming convention allows the statistical model configuration to be understood directly from the filename without opening the workspace.

---

# Workspace Complexity

The benchmark collection contains models of increasing complexity.

Typical channel counts include

- 1 channel;
- 3 channels;
- 5 channels;
- 10 channels;
- 15 channels;
- 20 channels;
- 25 channels;
- 30 channels.

Increasing the number of channels generally increases the size of the statistical model and is therefore particularly useful for scalability studies and performance regression testing.

---

# ROOT Counterparts

Each HS3 benchmark workspace has a corresponding ROOT workspace representing the same statistical model.

These ROOT workspaces are primarily used by the cross-framework benchmarks comparing PyHS3 with xRooFit.

Using equivalent statistical models across frameworks ensures that observed performance differences originate from implementation rather than model definition.

---

# Benchmark Coverage

The benchmark workspace collection is used throughout the repository.

| Benchmark Category | Uses Benchmark Workspaces |
|--------------------|:-------------------------:|
| Workflow Benchmarks | ✅ |
| Scaling Benchmarks | ✅ |
| Memory Benchmarks | ✅ |
| Cross-Framework Benchmarks | ✅ |
| PyHS3 vs xRooFit | ✅ |

---

# Running Benchmarks

The benchmark runner accepts one or more workspaces.

For example,

```bash
pixi run python -m src.run_all_benchmarks \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --benchmarks workspace_loading
```

To execute the complete benchmark suite,

```bash
pixi run python -m src.run_all_benchmarks \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --benchmarks all
```

Additional workspaces can be included simply by extending the `--workspaces` argument.

---

# Design Philosophy

The benchmark workspace collection serves as the canonical benchmark dataset for the repository.

Using a single, well-defined collection

- simplifies benchmarking;
- enables reproducible performance evaluation;
- allows direct comparison between benchmark suites;
- provides consistent inputs for cross-framework validation.

Whenever possible, new benchmark suites should operate on this collection to preserve consistency across the repository.

---

# Related Documentation

For additional information, see

- **Workspace Collection**
- **Benchmark Suite**
- **Benchmark Methodology**
- **Cross-Framework Benchmarks**
