# Benchmark Workspaces

On this page, you will learn how the benchmark dataset is organized, how workspace filenames encode statistical models, and which workspaces are included in the nominal benchmark collection.

Benchmark workspaces are generated using the **workspace-scripts** repository:

https://github.com/scipp-atlas/workspace-scripts

The repository uses a curated benchmark dataset that serves as the common input for nearly every benchmark suite.

---

## Dataset Design

All benchmark workspaces originate from a common baseline statistical model.

Rather than comparing unrelated statistical models, workspace variants modify selected model properties while keeping the remaining model configuration unchanged. This allows benchmark results to be interpreted in terms of specific modeling choices rather than changes to the overall statistical model.

For details on how these workspaces are processed inside PyHS3, see **Workspace Lifecycle**.

---

## Nominal Benchmark Workspace Collection

The table below lists the nominal benchmark workspaces used throughout this documentation and benchmark suite.

| Workspace ID | Filename | Format | Channels | Events / Bins | Signal | Background | Constraints | Used for | Supported Frameworks |
|---------------|----------|--------|---------:|---------------|--------|------------|-------------|----------|----------------------|
| WS-01 | `1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json` | HS3 JSON | 1 | — | Gaussian | RooExponential | Gaussian | Baseline workflow benchmarks | PyHS3 |
| WS-02 | `3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json` | HS3 JSON | 3 | — | Generic | Generic Polynomial | Gaussian | Generic model benchmarks | PyHS3 |
| WS-03 | `5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json` | HS3 JSON | 5 | — | Generic | RooExponential | Gaussian | Yield-scaling and workflow benchmarks | PyHS3 |
| WS-04 | `10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json` | HS3 JSON | 10 | — | Generic | RooExponential | Gaussian (NP disabled) | Nuisance-parameter studies | PyHS3 |
| WS-05 | `30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json` | HS3 JSON | 30 | — | Generic | Generic Polynomial | Gaussian | Large-scale and scalability benchmarks | PyHS3 |

These workspaces constitute the **nominal benchmark dataset** used throughout the documentation.

Additional workspace variants can be generated with the **workspace-scripts** repository but are not part of the standard benchmark collection described here.

---

## Additional Workspace Formats

The primary benchmark dataset consists of HS3 workspaces stored in

```text
inputs/
```

The repository also contains

```text
inputs/
└── pyhf/
```

which stores **pyhf** workspaces used by the **Cross-Framework Binned Likelihood** benchmark.

These workspaces represent the same statistical models in pyhf format and are generated using the **workspace-scripts** repository.

When available, equivalent ROOT workspaces are used by the cross-framework benchmark suite to compare PyHS3 with ROOT-based statistical frameworks.

---

## Workspace Variants

Benchmark workspaces may vary the following model properties.

### Background Models

| Variant | Description |
|----------|-------------|
| `bkgRooExp` | RooExponential background model |
| `bkgGeneric` | Generic exponential background |
| `bkgGenPoly` | Generic polynomial background |

### Signal Models

| Variant | Description |
|----------|-------------|
| `sigGauss` | Gaussian signal model |
| `sigGeneric` | Generic signal implementation |

### Shape Configuration

| Variant | Description |
|----------|-------------|
| `shapeFloat` | Floating signal shape parameters |
| `shapeFixed` | Fixed signal shape parameters |

### Nuisance Parameters

| Variant | Description |
|----------|-------------|
| `npOn` | Nuisance parameter enabled |
| `npOff` | Nuisance parameter disabled |

### Constraint Models

| Variant | Description |
|----------|-------------|
| `constrGauss` | Gaussian constraint |
| `constrPoisson` | Poisson constraint |
| `constrNone` | No auxiliary constraint |

### Signal Yield

| Variant | Description |
|----------|-------------|
| `yield0p1x` | One tenth of nominal yield |
| `yield1x` | Nominal yield |
| `yield10x` | Ten times nominal yield |
| `yield100x` | One hundred times nominal yield |

---

## Workspace Naming Convention

Benchmark workspace filenames follow the pattern

```text
<channels>_<background>_<signal>_<shape>_<nuisance>_<constraint>_<yield>
```

For example,

```text
10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x
```

is interpreted as

| Component | Meaning |
|----------|---------|
| `10ch` | Ten analysis channels |
| `bkgRooExp` | RooExponential background |
| `sigGeneric` | Generic signal model |
| `shapeFloat` | Floating signal shape |
| `npOff` | Nuisance parameter disabled |
| `constrGauss` | Gaussian constraint |
| `yield1x` | Nominal signal yield |

---

## Using Benchmark Workspaces

Run benchmarks on one or more selected workspaces:

```bash
pixi run python -m src.run_all_benchmarks \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --benchmarks workspace_loading
```

Workspace discovery, filtering, and benchmark campaigns are described in **Benchmark Matrix Runner**.

---

## Maintaining the Workspace Catalog

Whenever benchmark workspaces are added or updated, the catalog should also be updated.

For each workspace, document

- filename;
- format;
- channels;
- signal and background models;
- constraint configuration;
- benchmark purpose;
- supported frameworks.

This table serves as the canonical reference for the nominal benchmark dataset.

---

## Related Documentation

See also

- **Workspaces**
- **Workspace Lifecycle**
- **Benchmark Matrix Runner**
- **Cross-Framework Benchmarks**
- **Benchmark Methodology**
