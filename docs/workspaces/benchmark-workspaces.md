# Benchmark Workspaces

The benchmark workspace collection is the canonical dataset used throughout the **PyHS3 Benchmarks** repository.

Unless explicitly stated otherwise, every workflow benchmark, scaling benchmark, memory benchmark, and cross-framework benchmark operates on this collection. Using a common benchmark dataset ensures that benchmark results remain directly comparable and that observed performance differences originate from implementation changes rather than differences in statistical models.

The benchmark workspaces are **not created manually**. They are generated using the **workspace-scripts** repository:

https://github.com/scipp-atlas/workspace-scripts

The generation process is fully reproducible and produces statistically equivalent **HS3** and **ROOT** workspaces that serve as the common benchmark inputs across the repository.

---

# Benchmark Dataset Philosophy

The benchmark dataset is built around a simple design principle.

Every workspace originates from a common baseline statistical model, with each workspace variant modifying **exactly one characteristic** of that baseline whenever possible.

Rather than comparing unrelated statistical models, this approach isolates the performance impact of individual modeling choices while keeping the remainder of the statistical model unchanged.

This makes benchmark results significantly easier to interpret and provides a reliable foundation for both scalability studies and cross-framework comparisons.

---

# Baseline Workspace

The baseline model represents a simple simultaneous likelihood suitable for benchmarking the complete statistical workflow.

The baseline configuration consists of

- three analysis channels;
- a RooExponential background model;
- a RooGaussian signal model;
- a shared signal width nuisance parameter;
- a Gaussian constraint;
- approximately thirty events per channel.

Every benchmark workspace is derived from this baseline by modifying one aspect of the statistical model.

---

# Workspace Variants

Different workspace variants isolate different modeling choices.

## Background Models

Background variants evaluate the effect of different background parameterizations.

Typical variants include

| Variant | Description |
|---------|-------------|
| `bkgRooExp` | Native RooExponential background model |
| `bkgGeneric` | Exponential background implemented as a RooGenericPdf / HS3 generic distribution |
| `bkgGenericPoly` | Polynomial background parameterization |
| `bkgGenericFixShape` | Background shape fixed during inference |
| `bkgGenericNoNP` | Generic background without the width nuisance parameter |

---

## Signal Models

Signal variants modify only the signal parameterization.

Typical variants include

| Variant | Description |
|---------|-------------|
| `sigGauss` | Native RooGaussian signal model |
| `sigGeneric` | Signal implemented as a RooGenericPdf / HS3 generic distribution |

---

## Shape Configuration

Signal shape parameters can either participate in the fit or remain fixed.

| Variant | Description |
|---------|-------------|
| `shapeFloat` | Signal shape parameters float during optimization |
| `shapeFixed` | Signal shape parameters remain fixed |

---

## Nuisance Parameters

Different nuisance-parameter configurations allow benchmarking the cost of systematic uncertainties.

| Variant | Description |
|---------|-------------|
| `npOn` | Width nuisance parameter enabled |
| `npOff` | Width nuisance parameter removed |

---

## Constraint Models

Different auxiliary constraint terms can be applied to nuisance parameters.

| Variant | Description |
|---------|-------------|
| `constrGauss` | Gaussian constraint |
| `constrPoisson` | Poisson constraint |
| `constrNone` | No auxiliary constraint |

---

## Signal Yield

The benchmark dataset also varies the expected signal statistics.

Typical configurations include

| Variant | Description |
|---------|-------------|
| `yield0p1x` | One tenth of the nominal signal yield |
| `yield1x` | Nominal signal yield |
| `yield10x` | Ten times the nominal signal yield |
| `yield100x` | One hundred times the nominal signal yield |

Changing the event yield makes it possible to study how numerical performance changes as the statistical power of the dataset increases.

---

## Channel Count

Workspace complexity is primarily controlled through the number of simultaneous analysis channels.

Typical benchmark configurations include

- 1 channel;
- 3 channels;
- 5 channels;
- 10 channels;
- 15 channels;
- 20 channels;
- 25 channels;
- 30 channels.

Increasing the number of channels increases the size and complexity of the statistical model, making these workspaces particularly useful for scalability studies.

---

# Workspace Naming Convention

Every benchmark workspace filename completely describes the statistical model it contains.

The naming convention follows the pattern

```text
<channels>_<background>_<signal>_<shape>_<nuisance>_<constraint>_<yield>
```

For example,

```text
10ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield1x
```

can be interpreted as

| Component | Meaning |
|-----------|---------|
| `10ch` | Ten simultaneous analysis channels |
| `bkgRooExp` | RooExponential background model |
| `sigGeneric` | Generic signal implementation |
| `shapeFloat` | Floating signal shape parameters |
| `npOn` | Nuisance parameter enabled |
| `constrGauss` | Gaussian auxiliary constraint |
| `yield1x` | Nominal signal yield |

Because every component of the filename describes one property of the statistical model, workspace configurations can be understood directly from their filenames without opening the workspace.

---

# ROOT Counterparts

Each HS3 benchmark workspace has a corresponding ROOT workspace generated from the same statistical model.

These ROOT workspaces are primarily used by the cross-framework benchmark suite to compare PyHS3 with ROOT-based statistical frameworks such as xRooFit.

Using equivalent statistical models across frameworks ensures that cross-framework benchmarks remain **apples-to-apples**, with observed differences reflecting implementation characteristics rather than differences in model construction.

---

# Benchmark Coverage

The benchmark workspace collection is used throughout the repository.

| Benchmark Category | Uses Benchmark Workspaces |
|--------------------|:-------------------------:|
| Workflow Benchmarks | ✓ |
| Scaling Benchmarks | ✓ |
| Memory Benchmarks | ✓ |
| Cross-Framework Benchmarks | ✓ |
| PyHS3 vs xRooFit | ✓ |

Unless a benchmark explicitly documents otherwise, it is expected to operate on this benchmark dataset.

---

# Using Benchmark Workspaces

One or more workspaces can be supplied to the benchmark runner.

For example,

```bash
pixi run python -m src.run_all_benchmarks \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/30ch_bkgGenericPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --benchmarks workspace_loading
```

Additional benchmark workspaces can be included simply by extending the `--workspaces` argument.

---

# Design Principles

The benchmark workspace collection has been designed to

- provide reproducible benchmark inputs;
- isolate the impact of individual modeling choices;
- support scalability studies;
- enable apples-to-apples cross-framework comparisons;
- serve as a common benchmark dataset across the repository.

Using a single, well-defined benchmark dataset greatly simplifies benchmark interpretation while ensuring that benchmark results remain comparable across different workflow stages and software frameworks.

---

# Related Documentation

See also

- **Workspace Lifecycle**
- **Benchmark Methodology**
- **Benchmark Workflow**
- **Cross-Framework Benchmarks**
- **Benchmark Suite**
